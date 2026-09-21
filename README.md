# focus-llama

> **A [`llama.cpp`](https://github.com/ggml-org/llama.cpp) fork for Dynamic Attention Masking: the model declares, in its own output, which parts of the KV cache the next tokens may attend to, and the engine enforces it at decode time.**

**Status: experimental / work in progress.** Nothing below the "Implemented" heading is promised until it is checked off in the roadmap.

---

## What this is

`focus-llama` is a research fork of `llama.cpp` for experimenting with **Declarative Attention (DA)**, a protocol from
*Language Models Can Control Their Own Attention* (Ho et al., 2026, [arXiv:2609.02737](https://arxiv.org/abs/2609.02737)).

In DA, an off-the-shelf model is prompted to split its chain-of-thought into spans with a declared attention scope:

| Mode | Tag | Attends to |
|------|-----|-----------|
| global | `<global> ... </global>` | Scaffold + **all** context chunks (navigation) |
| focus  | `<focus magic_chunks="N"> ... </focus>` | Scaffold + **only the named chunk(s)** + the response so far |
| local  | `<local> ... </local>` | Scaffold + the response so far (no context chunks) |

The *scaffold* (system preamble, question, instruction) is attended in every mode. The engine parses these tags from the generated stream, like tool calls, and restricts attention accordingly. No auxiliary scorer and no training are needed; the paper shows this works zero-shot on sufficiently large models.

This fork provides the missing engine side for `llama.cpp`: a DA stream parser, a mode-aware KV/attention controller, and instrumentation to measure what is actually attended.

## Why a fork

Stock `llama.cpp` cannot do this out of the box:

- `llama-server` never manipulates the KV cache mid-generation. Range-level control needs a custom decode loop.
- `llama.cpp` uses a contiguous cell-based KV cache, not paged blocks. The paper's reference implementation (vLLM) skips work by rewriting the block table; there is no equivalent hook here.
- Masking a region only *logically* removes it. Unless the attended region is physically compacted or the kernel skips masked tiles, the KV read volume does not shrink.

## Design

```
 prompt builder            decode loop (this repo)                     KV control
┌──────────────┐   ┌───────────────────────────────────┐   ┌──────────────────────────┐
│ chunker      │   │ llama_decode(batch)               │   │ backend A: seq_rm        │
│  ~2K-token   │──▶│ sample token                      │──▶│   (logical mask, exact)  │
│  magic chunks│   │ DA state machine                  │   │ backend B: two streams   │
│ tool-use     │   │   parse tag on closing '>'        │   │   (logical read-set)     │
│ transcript   │   │   emit mode transition            │   │ backend C: kernel-level  │
└──────────────┘   │ apply mode -> KV control          │   │   block/tile skipping    │
                   │ log attended tokens per step      │   └──────────────────────────┘
                   └───────────────────────────────────┘
```

Key points:

- **Parsing lives outside `llama_decode`.** Sampling already returns to the caller every token, so the state machine runs in the caller's loop and applies the new mode before the next batch. Chunk ids should be constrained (e.g. with a GBNF grammar) so a malformed `magic_chunks` cannot be emitted.
- **The sequence is never edited in place.** Positions are preserved, so removing a region is equivalent to masking it with `-inf` (no RoPE renumbering).
- **Isolation from upstream.** DA code lives in its own directory with minimal hooks into core, to keep rebasing on upstream `llama.cpp` cheap.

### KV control backends

| Backend | Mechanism | Speed-up | Cost |
|---------|-----------|----------|------|
| **A. `seq_rm`** | Remove non-focus ranges, restore by re-prefill | None guaranteed (semantic equivalence only) | Re-prefill on every focus → global switch |
| **B. Two streams** | Stream 0 holds the full context permanently; stream 1 holds the kept (scaffold + focus) chunks + response, copied with `seq_cp` (a cell retag in the unified pool, no data copy) | Logical read-set reduction - the attended tokens shrink, and the FA kernels' masked-chunk skipping (Metal 32-cell decode / 64-block prefill, CPU per-position) reduces the physical KV load at chunk granularity. B's unique value: the original stream is preserved, so a global return needs no re-prefill | ~2x KV metadata (unified pool) |
| **C. Kernel skipping** | Skip fully-masked tiles/blocks in the flash-attention kernels | Potentially in-place, no extra memory | Kernel work per backend (CUDA / Metal) |

Backend A is for validating protocol adherence and accuracy. Backend B is the first candidate for measuring real speed-ups. C is only worth building if B's numbers justify it.

## Build

Same as upstream `llama.cpp`. Build the server, plus `llama-bench` if you want the depth benchmarks:

```bash
git clone https://github.com/edwardyoon/focus-llama.git
cd focus-llama

# CUDA (set the architecture of your GPU; "120" below is only an example)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DCMAKE_BUILD_TYPE=Release

# Metal (macOS) or CPU-only
# cmake -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j --target llama-server llama-cli llama-bench
```

## Smoke tests

The smoke tests are quick checks that the server-side mechanisms behave as intended on **one** prompt.
They are not benchmarks and say nothing about accuracy or speed.

All scripts need a server built from this fork. An older `llama-server` silently ignores the extra
request fields, which is why the scripts include checks that fail in that case.

### 1. KV-range removal (`da_rm`)

`da_rm` drops token ranges from the KV cache of a request. With `da_rm_at` the removal is applied
mid-prefill, so the tokens after the boundary (the question) are prefilled without the removed range.
Without `da_rm_at` it is applied after prefill.

```bash
./build/bin/llama-server -m <model>.gguf
python3 da-probe/da_server_smoke.py http://127.0.0.1:8080 [--hybrid]
```

Use `--hybrid` for models with recurrent / linear-attention layers (e.g. Gated DeltaNet). On those models
the text of the fact-removed runs is informational only.

Runs (all use `stop=["\n"]`, exact-match verdicts, and first-token logprobs):

| Run | Request | What it shows |
|---|---|---|
| baseline | no `da_rm` | answers exactly `ZEBRA-42` |
| strict | fact range, `da_rm_at` = question start | removal applied mid-prefill |
| paper | fact range, no `da_rm_at` | removal applied after prefill (decode-time restriction) |
| masked-all | whole document, `da_rm_at` | removal changes the computation |
| control | unrelated section, `da_rm_at` | answer stays `ZEBRA-42` (selectivity) |

Example summary (trimmed):

```
baseline         : PASS (exact 'ZEBRA-42')
mechanism (Δlp)  : baseline vs masked-all first-token Δlp=0.6217 - different prefill computation (mid-prefill removal ran)
selectivity      : PASS (exact 'ZEBRA-42')
path divergence  : strict vs paper first-token Δlp=0.9256 - different prefill paths
OVERALL          : PASS
```

- **mechanism** - first-token logprobs of baseline and masked-all differ, so the removal really changed
  the prefill. Identical values mean it was not applied mid-prefill (check the server log for `da_rm:` lines).
- **path divergence** - `strict` and `paper` must differ, otherwise both took the same path.
- **selectivity** - removing an unrelated section leaves the answer intact.

### 2. Two-stream backend B (`da_b`)

Backend B copies the kept prefix (scaffold + focus) into a reserved second sequence and decodes there.
The original sequence stays intact, so a later return to global attention would not need a re-prefill
(the return path itself is not implemented yet).

Requirements:

- `--kv-unified` is **required**. A partial-range `seq_cp` aborts on a non-unified pool, so without it
  the server falls back to logical removal (`da_rm`) and logs a WARN. The answers would still pass, which
  is why the server log is the real evidence.
- `--parallel 2` is only needed for the cache-integrity phase (phase 2 below).
- Add `-v` to see the per-step read-set lines.

```bash
./build/bin/llama-server -m <model>.gguf --kv-unified --parallel 2 -v
python3 da-probe/da_b_smoke.py http://127.0.0.1:8080 [--hybrid] [--runs k1,k2,...] [--skip-cache]
```

Options:

- `--hybrid` - same meaning as above.
- `--runs` - run a subset of phase-1 keys: `baseline,bkeep,bnofact,logical,bkeep2`.
- `--skip-cache` - skip phase 2.

**Phase 1** (one ZEBRA prompt, `cache_prompt=false`):

1. **mechanism** - `baseline` vs `bkeep` first-token logprobs must diverge. Identical values mean the B
   switch did not run (old binary, missing flags, or fallback).
2. **B equals A** - `bnofact` (B) and `logical` (A, `seq_rm`) apply the same removal by different means,
   so their first-token logprobs must agree within 5e-3 (0.0003 in one run on a local Bonsai-8B).
   A large difference points at a wrong keep set.
3. **answers** - `baseline`, `bkeep` and `bkeep2` must answer exactly `ZEBRA-42`. `bkeep2` is a second,
   consecutive `da_b` request and checks that the reserved sequence id can be reused after the first
   one finished.

**Phase 2** (cache integrity, needs `--parallel 2`): a request caches prompt P1 on one slot, a `da_b`
request runs on the other slot, then P1 is sent again. It must land back on the first slot with its
cache untouched, i.e. a B switch does not destroy another slot's cached prompt.

### 3. Tag parser (`da_chunks`) - the model decides the removal

Instead of the client choosing the ranges, the client sends the chunk *layout* and the model's own
`<focus magic_chunks="N">` tag decides what gets removed:

- `da_chunks` - the prompt token ranges of the document chunks, a list of `[lo, hi)` pairs. Index
  `N-1` is chunk `N` (the tag's number is 1-based).
- `da_filler` - the `[lo, hi)` range of a filler segment, removed together with the non-kept chunks.
  Omit it (or the server treats it as absent) when the layout has no filler.

When the first complete tag appears in the generated text, the server removes every chunk except the
named one, plus the filler, exactly once - via backend B if `da_b` is also set, else backend A. The
removal boundary is the current decode position, so in B mode the already-generated tokens are part of
the kept set. `da_chunks` supersedes a static `da_rm` in the same request. A model that never emits a
complete tag simply runs with full attention - no removal is applied.

The mechanism is **per-request opt-in**: a request without `da_*` fields runs exactly as before and
produces no `da_*` log lines. To use it, the client must build the prompt, tokenize it (e.g. with
`/tokenize`), and attach the resulting layout to the `/v1/completions` body:

```json
{
  "prompt": "<scaffold>\n[Chunk 1] ...\n[Chunk 2] ...\n...\n[Chunk 5] ...\n<filler>\n<instruction>",
  "da_chunks": [[8, 96], [96, 184], [184, 272], [272, 360], [360, 448]],
  "da_filler": [448, 520],
  "da_b": true,
  "max_tokens": 64
}
```

`da_chunks[i]` is the `[lo, hi)` prompt token range of chunk `i+1` (1-based, matching the tag's
`magic_chunks` number). Only the client that assembled the prompt knows these ranges - the server
cannot derive them from the text.

```bash
./build/bin/llama-server -m <model>.gguf --kv-unified --parallel 2 -v
python3 da-probe/da_tag_smoke.py http://127.0.0.1:8080 [--hybrid]
```

The smoke test builds a 5-chunk document (one codeword per chunk) plus filler and instruction. All
three runs (`baseline`, `tag_a`, `tag_b`) must emit `<focus magic_chunks="4">` before answering exactly
`ZEBRA-42`, and the server journal must show the tag close and the removal (`da_tag:` lines).

Note on tag emission: the script prompts raw (no chat template) and primes the model with an empty
closed thinking block (`
</think>

`), which makes small thinking models (e.g. Bonsai-8B) emit the
tag instead of answering straight. On a larger model with the proper chat template this priming should
not be needed.

### Reading the server log

```bash
grep 'da_b:' <server log>                 # or: journalctl -u <service> | grep 'da_b:'
grep 'da_tag:' <server log>               # tag parser: request start + tag close / removal
grep 'falling back to logical removal' <server log>    # must be empty for B runs
```

Per B request the server logs:

- `da_b: switched decode to seq N (...) - logical read set now X of Y prefix token(s) (Z removed, -P%)`
- one `da_b:   keep [lo, hi) n token(s)` line per kept range
- with `-v`, per-step `da_b: decode #p: logical read set X of Y ...` lines
- a final `da_b: finished on seq N - final logical read set ...` summary

**These numbers are logical, not measured reads.** In the unified KV pool the removed cells stay in place
and are masked, so the attention still scans the same cell range. Whether fully masked chunks are actually
skipped depends on the backend and the attention kernel (for example, single-token decode on CUDA does not
skip them), and no byte count is logged. Treat the percentage as the size of the attended token set, not as a
speed-up. Measuring real speed is still open.

## Relationship to FocusMemory

[FocusMemory](https://github.com/edwardyoon/FocusMemory) chunks and indexes long-term context. `focus-llama` is the inference-side counterpart: it lets the model read a compact index in `global` mode and then commit attention to specific chunks.

The two fit together because FocusMemory **assembles the prompt from its own chunks** - so its backend already knows the token range of every chunk it inserted, which is exactly what `da_chunks` asks for (section 3). The integration is a client-side change only: the FocusMemory backend tokenizes the assembled prompt (e.g. via `/tokenize`), maps each chunk to its `[lo, hi)` range (plus the filler, if any), and attaches `da_chunks` / `da_filler` (and `da_b`) to the `/v1/completions` request. From then on the model's own `<focus magic_chunks="N">` tag decides which chunk the next tokens attend to, and the server enforces it. Until that wiring exists, requests from a FocusMemory backend carry no `da_*` fields and run with full attention - the two projects remain independently usable.

## Production launch (recommended options)

The 123 node (192.168.219.123) runs the DA inference service - qwen3.8, multimodal - behind the
FocusMemory backend. The recommended `llama-server` launch line is:

```bash
llama-server \
  --parallel 1 \
  --metrics \
  --da-prompt-scan \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-ngl all
```

(`--parallel 1` keeps the single-tenant node on backend A - see Backend A vs B below.)

| Option | What it does | Why it is on |
|--------|--------------|--------------|
| `--metrics` | Exposes Prometheus metrics on the server port | Observability for a long-running service |
| `--da-prompt-scan` | **Prompt scanning** (P1): on each request the prompt is scanned for `<da:N>` chunk markers and their token ranges are pre-computed, so DA tags / static `da_rm` ranges resolve to real KV positions | The FocusMemory backend marks the chunks it assembled; scanning maps those markers to positions. A prompt with no markers runs with full attention (fail-open) |
| `--spec-type draft-mtp` | **Speculative decoding** with the model's MTP draft head | Speed-up on top of DA. Since P6, spec and DA **coexist**: while a slot is in DA mode (`da_seq` active) spec is paused automatically and resumes on the return to global attention - so spec stays ON without breaking DA |
| `--spec-draft-n-max 4` | Up to 4 draft tokens per step | Enough to overlap decode with drafting, without so many that rejections waste work |
| `--spec-draft-ngl all` | Puts the whole draft model on the GPU | The draft model is small; keeping it fully on-GPU avoids CPU round-trips that would erase the spec gain |

**Backend A vs B.** The focus/local restriction runs on backend A (`seq_rm` holes, monotonic - a removal
is irreversible within the request and a return to global attention needs a re-prefill) or backend B
(two streams, reversible, no re-prefill), selected by `--kv-unified`. Note that `--kv-unified` is
**implied by `--parallel`**: when `--parallel` is left at its default (auto), the server sets
`n_parallel=4` **and** `kv_unified=true`, so DA runs on backend B. To run backend A instead, launch with
an explicit `--parallel` (e.g. `--parallel 1`) and without `--kv-unified`. Confirm which backend a
request used from the scan log: `da_scan: ... - A path` vs `da_scan: ... - B path`.

**Verifying DA in the journal.** After a chat that carries DA markers, confirm the DA path ran:

```bash
journalctl -u qwen3.8-focus --since "10 min ago" | grep -E 'da_scan:|da_tag:'
```

- `da_scan:` - the prompt scanner found the `<da:N>` markers and built the chunk-to-range map
- `da_tag:` - a `<focus>`/`<local>` tag was parsed and the attention restriction applied

A plain chat (no DA markers) produces no `da_scan:` line and runs with full attention - that is the
intended fail-open behavior.

**Verifying DA end-to-end with curl.** Send a chat whose user message carries a well-formed marker
block (the same format the FocusMemory hook injects) and read the `timings` object of the response:

```bash
curl -s http://192.168.219.123:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen27b",
  "messages": [{"role": "user", "content": "Memory entries:\n<da:1>The capital of France is Paris.\n<da:2>The capital of Germany is Berlin.\n<da:filler>Instructions (Declarative Attention): The memory entries above are numbered magic chunks (1-2). First identify the chunk that contains the answer to the question, and output the tag <focus magic_chunks=\"N\"> on its own line, where N is the chunk number (1-2). Then answer the question.\n<da:layout:2>\nQuestion: What is the capital of France?"}],
  "max_tokens": 160, "temperature": 0, "stream": false
}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print({k:v for k,v in d['timings'].items() if k.startswith('da_')})"
```

How to read the result:

| Signal | Meaning |
|--------|---------|
| `timings.da_n_restricted_steps` > 0 | The model emitted a `<focus>`/`<local>` tag and the server ran the restricted-attention steps |
| `timings.da_n_attended_tokens` | Logical tokens attended per restricted step (sum over steps) - the reduction vs. the full prompt is the DA effect. **Logical, not physical** reads |
| `timings.da_path` = `A` / `B` | Which backend enforced the restriction (see Backend A vs B above) |
| `timings` has **no** `da_*` fields | Either the prompt had no valid marker block (fail-open vanilla) or the model never emitted a tag - check the journal `da_scan:`/`da_tag:` lines to tell which |
| `choices[0].message.content` | The model's answer. The `<focus ...>` tag is erased from the text; depending on how the tokens streamed it may or may not be visible to the client (v1-accepted) |

Expected outcomes per prompt shape:

- **Well-formed markers** (`<da:1>...<da:2>...<da:filler>...<da:layout:2>`) - journal shows
  `da_scan: 2 chunk(s) numbered 1..2 + filler ... - A path` followed by
  `da_tag: request start - 2 chunk(s)`, and (if the model commits to a chunk)
  `da_tag: <focus magic_chunks="N"> closed at generated token ... - A removal, ... range(s)`.
  A 500 / `substr` exception after a `da_tag:` line means the text-accounting broke - that is a bug.
- **No markers** - no `da_scan:` line at all, no `da_*` in `timings`, output identical to a vanilla
  server. This is the regression guard: unmarked traffic must be untouched.
- **Malformed markers** (e.g. footer says 3 but only 2 `<da:N>` markers) - journal WARN
  `da_scan: last block has 2 chunk marker(s) but the footer says 3 - failing open to vanilla`,
  no `da_*` in `timings`.

The `system_fingerprint` in the response carries the server's git short hash
(`b11090-88db36bc5` = commit `88db36bc5`) - use it to confirm which binary a node actually runs.

## Status and limits

Early work in progress.

- **Works:** `llama-server` accepts `da_rm` / `da_rm_at` to drop KV token ranges either mid-prefill or after prefill. Checked via first-token logprobs on a small smoke test.
- **Works:** backend B (`da_b`) - the kept ranges are copied to a reserved second sequence and decoded there, with the original sequence intact (section 2).
- **Works:** the server parses `<focus magic_chunks="N">` from the generated stream and removes the non-kept chunks at the tag close (`da_chunks`, section 3). The tag must be emitted by the model - on small thinking models without a chat template this may need the empty thinking-block priming from the smoke test.
- **Works:** prompt scanning (`--da-prompt-scan`) end-to-end with the FocusMemory marker block - verified on the 123 production node (qwen3.8, path A): the model emits the tag, the server applies the removal mid-decode alongside MTP speculation, and the response `timings` report `da_n_restricted_steps` / `da_n_attended_tokens` / `da_path`. Unmarked and malformed prompts fail open to vanilla.
- **Development aids only:** the standalone `da-probe/` probe binaries (multi-stream, per-step instrumentation) are not server features - the server mechanisms above are complete and smoke-tested.
- **Open:** physical (not just logical) read reduction on CUDA; measured end-to-end speed-ups (the 123 break-even analysis found no speed gain at context lengths up to 64K - see the production notes); the return to global attention on the hybrid model is lossy for the recurrent (GDN) state and needs a dedicated accuracy probe.
- **Hybrid models** (e.g. Gated DeltaNet): only the attention layers are affected, as in the paper.
- **Evidence so far is small:** one-prompt smoke tests and a 5-prompt tag-adherence check. For the paper's numbers and caveats, see arXiv:2609.02737.

## Reference

```bibtex
@article{ho2026da,
  title   = {Language Models Can Control Their Own Attention},
  author  = {Ho, Namgyu and Ahmad, Huzama and Koh, Woosung and Yun, Se-Young and Schuster, Tal and dos Santos, Cicero Nogueira},
  journal = {arXiv preprint arXiv:2609.02737},
  year    = {2026}
}
```

## License

MIT, following upstream `llama.cpp`.
