# focus-llama

> **A [`llama.cpp`](https://github.com/ggml-org/llama.cpp) fork with two production-verified engines: Declarative Attention (DA) - the model declares, in its own output, which parts of the KV cache the next tokens may attend to, and the engine enforces it at decode time - and kv-offload, a lossless evict/recall context store that makes long-horizon sessions viable: a 1-token re-prefill after eviction, and a ~1–2 s lossless chunk recall instead of a ~5.3 min lossy compaction.**

**Status: v2.0 - production-ready.** Running in production (123, qwen3.8-27B MROPE, RTX 5090) with `--da-auto --fm-offload --kv-offload-holes`. DA physical read reduction, DA survival across auto-compaction, the MROPE mid-hole gate (R1), and the kv-offload evict/hole/recall cycle are all verified end to end (below).

## Verified: physical KV read reduction (CUDA, 2026-09-22)

Declarative attention now reduces the **physical** KV read volume during decode on CUDA, not just the logical attention set. The flash-attention VEC kernel was ported to the `n_kv_max` sparse path - it gathers K/V rows by compact index, so the kernel reads only the attended rows instead of the whole cache - and the MMA f16 sparse gate was extended to square MHA head dims. A/B against the dense path (`FOCUS_DA_DENSE=1`) on an RTX 5090, 2655-token prompt with 97% of the KV ranges removed, 3 runs per arm:

| Path | Result |
|---|---|
| **VEC sparse - f16 KV** | ✅ 3/3 byte-identical output, max\|Δlogprob\| = 0.000e+00, journal `n_kv_max 0→512` |
| **VEC sparse - q4_0 KV** (production config) | ✅ 3/3 byte-identical output, max\|Δlogprob\| = 0.000e+00, journal `n_kv_max 0→512` |

The dense path stays bit-identical (constexpr folding), so sparse never changes results when no ranges are removed.

**Measured in production traffic (123, 09-23).** The sparse-gate journal (`da_sparse[VEC|MMA]:`, one line per ~512-token gather-bound step while sparse, plus the dense fallback with its reason) was captured from live `qwen3.8-focus` traffic, 15:48–21:10:

| Path | Sparse decisions | Mean read | Mean reduction | Range |
|---|---|---|---|---|
| VEC (q4_0 KV, production) | 91 | 34.0% | 66.0% | 8–50% |
| MMA (f16 KV) | 255 | 36.5% | 63.5% | 8–50% |
| **all** | **346** | **35.9%** | **64.1%** | **8–50%** |

The 50% top of the range is the gate's own bound: the sparse path is only active while it
reads at most half the KV (`K >= 2*n_kv_max`). Dense fallbacks (108) carry their reason:
`no sparse kernel variant` (68, MMA head-dim not yet covered) and the 50% decay
(`K < 2*n_kv_max`, 40). No multi-token-batch fallbacks were observed: spec decode is paused
while a slot is in DA mode, so DA decode is single-token.

**Status: v1.0** - physical read reduction verified in production traffic (above); tag-quote
hijack fixed (line-start-only entry tags, `632e31c20`).

---

## Verified: DA survives auto-compaction (GPU, 2026-09-23)

The production failure mode - an agent session that ran DA for several turns gets
auto-compacted (the conversation is replaced by a summary) and then behaves as if it lost
all memory - is fixed and verified end to end on the production model (qwen3.8-27B, RTX 5090).

Root cause: the summarizer reproduces the DA marker text verbatim inside the summary, and
the old "last footer wins" scanner then pinned attention to that dead block. The fix is
**tail anchoring**: the marker scanner only accepts a block that sits after the last
user-message boundary of the rendered prompt, so a dead block copied into a summary is never
a candidate and the request fails open to the live block instead.

`da-probe/da_e2e_compaction.py` reproduces the failure mode on the marker path
(`--da-prompt-scan`) with the real FocusMemory hook block shape and a simulated compaction:

| Phase | Scenario | Result |
|---|---|---|
| A | 3 turns, live `[[da:N]]` blocks (1-5 / 6-10 / 11-15) | ✅ 3/3 CORRECT |
| B | Summary carries a **dead** block (1-5) + a new live block (16-20) | ✅ CORRECT - dead block ignored |
| C | Post-compact turn, live block (21-25) | ✅ CORRECT |

Journal: 5/5 `da_scan:` lines with the expected chunk ranges (1..5 → 21..25), 0 fail-open
lines. Supporting engine work verified on the same box: P1 dead-marker smoke 3/3, P2a
multi-turn reversibility + instruction placement 3/3, P2b paper-aligned chunking 4/4
(packing / hard-cap / prose / 50K - all mean chunk sizes ≤ the 2560-token cap).

**Status: verified on the GPU box and running in production (123 `qwen3.8-focus`, 09-23).**

---

## Verified: lossless context without compaction (kv-offload, 2026-09-26)

The kv-offload cycle (*kv-offload* section below) replaces lossy auto-compaction with a
lossless evict/recall cycle. Every number below was measured on 2026-09-26, not modeled:

| Metric | Measured | Where |
|---|---|---|
| MROPE mid-hole gate (R1) | 4/4 checks PASS - kept chunk read through the mid-sequence hole, answer-token Δlogprob **+0.000 nats** vs the full-attention baseline | 123, qwen3.8-27B MROPE |
| Re-prefill after eviction | **1 token in 26 ms** vs 13.7 s for a full re-prefill (`n_past=3163` → 1-token prefill) | local smoke, Bonsai-8B + stub store |
| Hole application | 1412-token hole cut from the main sequence, prompt cache kept across `release()`, 0 aborts | local smoke (same run) |
| One chunk recall (GET + re-prefill) | **~1–2 s, lossless** (the original text, not a summary) | measured |
| One full compaction (baseline) | **~5.3 min, lossy** | measured |
| Store round-trip in a live session | 87 chunks (~400 KB) PUT/GET round-tripped; per-hole KV cuts applied turn after turn | 123 production |

A single recall is two to three orders of magnitude cheaper than a compaction and lossless:
context beyond `--kv-offload-threshold` is parked in the store and brought back verbatim on
demand, instead of being compressed into a lossy summary. The default Option C mode
(evict from the prompt) kept its behavior intact in the same smoke (prompt text reduction
12813 → 6484 tokens, no behavioral change).

**Status: verified (v2.0)** - running in production on 123 since 2026-09-26 (`-c 180000
--kv-offload-threshold 160000 --kv-offload-holes`). Intentionally out of scope for v2.0:
the sparse `n_kv_max` path with holes (R3) - v1 forces the dense path, the same physical
state the R1 gate and the DA A path already run in production.

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

### 4. DA survives auto-compaction (`da_e2e_compaction`)

The end-to-end check for the production failure mode: a DA session that gets
auto-compacted (the conversation replaced by a summary) must keep retrieving.
The driver sends the real FocusMemory hook block shape (a verbatim port of
`buildDaBlock`) and simulates a compaction by replacing the history with a
summary that carries a **dead** marker block, then a new turn with a **live**
block. The marker scanner must tail-anchor to the last user message, ignore
the dead block, and validate the live one.

```bash
./build/bin/llama-server -m <model>.gguf --da-prompt-scan --parallel 1 -c 8192 -v
python3 da-probe/da_e2e_compaction.py http://127.0.0.1:8086 --log /tmp/da_e2e.log
```

Three phases (A: 3 live-block turns, B: summary with a dead block + a new live
block, C: a post-compact turn). PASS requires all five answers CORRECT, 5/5
`da_scan:` journal lines with chunk ranges 1..5 → 21..25, and 0 fail-open
lines. The Phase B line is the money check: the compacted prompt was scanned
(not failed open) and the live block won over the dead summary block.

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

There are two ways to get a chunk layout onto the wire:

1. **Marker path (`--da-prompt-scan` + client markers).** FocusMemory assembles the
   prompt from its own chunks and wraps them in `[[da:N]]` ... `[[da:layout:N]]` markers (the
   legacy `<da:N>` form is accepted as well; the hook emits the bracket form because angle
   brackets get mangled by markdown/HTML escaping between the hook and the rendered prompt). The
   server scans the rendered prompt, maps the markers to token ranges, and the model's
   `<focus magic_chunks="N">` tag decides which chunk the next tokens attend to. The marker block
   sits at the prompt tail, so DA only touches the injected memory index - the rest of the
   conversation is untouched. The scanner **tail-anchors** to the last user-message boundary, so a
   marker block that a compaction summary reproduces in the history is ignored (fail-open) and a DA
   session keeps working after auto-compaction - see *Verified: DA survives auto-compaction*.
   Because it never touches the conversation (only the injected index), it cannot corrupt the
   session - but it also cannot reduce the conversation's KV, so it is an index-scoped
   complement to the auto path, not a replacement for it.
2. **Auto-chunking (`--da-auto`, the production path for physical reduction).** The server
   splits the rendered chat prompt itself into ~`--da-chunk-tokens` magic chunks (hard cap
   5/4 × target), packing across message boundaries and falling back paragraph → line → sentence
   → clause → word when a message has too few boundaries, so the client sends a plain request
   with nothing extra. Because it re-chunks the *entire* conversation (system + history + tool
   I/O) on every turn once the prompt passes `--da-min-ctx`, it is the path that produces
   conversation-level, physical KV read reduction - the marker path only ever scopes the injected
   index. It requires `--kv-unified` (backend B) since the A path is irreversible. Known
   limitations, observed in production and under stabilization (see
   `plans/focus-llama-da-stabilization.md`):
   - **Tag-quote hijack.** The tag state machine operates on the model's whole output stream; when
     the model *quotes* a DA tag in its reasoning (debugging DA, quoting a commit message), the
     parser can consume the quote as a control tag, causing an unintended mode switch and a gap in
     the output text. Fixed in v1.0 (line-start-only entry tags, `632e31c20`): a tag quoted
     mid-line is data, not a control tag.
   - **Bigger blast radius than the marker path.** A misfired tag restricts the session's own
     conversation, not just the injected index.

The two paths coexist (marker layout wins when present; auto takes over otherwise - see
Production launch). Without either, requests run with full attention - the two projects remain
independently usable.

## kv-offload (auto-compact replacement)

`--fm-offload` replaces the client's lossy auto-compaction with a **lossless** evict/recall
cycle backed by an external store (the [FocusMemory](https://github.com/edwardyoon/FocusMemory)
dumb KV store). When a rendered prompt exceeds `--kv-offload-threshold` tokens, the engine
evicts the oldest *middle* messages (never the system message or the last user message) before
`--da-auto` re-chunks: each evicted message's text is PUT to the store. Two modes control what
"evict" does to the KV:

- **`--kv-offload-holes` (production path)** - the evicted text *stays in the prompt*; the
  engine cuts its KV out of the main sequence (`seq_rm` holes, positions not re-based) once the
  prefix is known to exist. The client keeps sending the full prompt, so every re-request
  re-matches at `n_past = full` and re-prefills **1 token** instead of the evicted tail
  (~64 s → ~2 s). Holes are applied per range (disjoint, validated `hi <= n_past`),
  `release()` keeps the prompt cache, and the generic KQ -inf mask covers the holes - the same
  physical state the DA A path runs in production. The R1 gate (an MROPE mid-sequence hole
  behaves like -inf masking) passed on the production 27B with answer-token Δlogprob
  +0.000 nats. Mid-sequence holes are legal on non-MROPE models too - the batch position
  check only requires the next token at `seq_max + 1`, so a hole in the middle does not
  break contiguity (verified in the local smoke, *Verified* section above).
- **default** - the evicted range is removed from the prompt and the segment becomes a
  **virtual chunk** numbered `M+1..M+K` (after the `M` active chunks) listed in the DA
  instruction; when the model emits `<focus magic_chunks="N">` for an offloaded chunk, the
  engine GETs its text back and re-prefills it at the tail of the sequence (get-on-focus) so
  the model reads the original content - not a compaction summary.

Any store error fails open (the segment stays in the prompt), so a downed store degrades to a
normal DA session, never to data loss. The cycle is complementary to both DA layout paths: it
decides *which* middle messages leave the KV and *how* to bring one back on demand; `--da-auto`
still owns the re-chunking. It requires `--da-auto` + `--kv-unified` (the eviction runs inside
the auto-chunk gate) and a store reachable at `--focus-memory-host`.

**Status: verified (v2.0)** - see *Verified: lossless context without compaction* above.
Running in production (123, qwen3.8-27B MROPE, `-c 180000 --kv-offload-threshold 160000`,
since 09-26) with `--kv-offload-holes`: evictions PUT to the store and holes cut the KV on
every turn of a live session.

> **Flag naming.** The engine flag is `--fm-offload` (env `LLAMA_ARG_FM_OFFLOAD`), not
> `--kv-offload`: the stock llama.cpp `-kvo/--kv-offload` flag (KV-cache offloading) already
> owns that name and the `LLAMA_ARG_KV_OFFLOAD` env var, so a `--kv-offload` here would be
> silently consumed by the stock flag and eviction would never engage.

## Production launch (recommended options)

The production node runs the DA inference service - qwen3.8, multimodal - behind the
FocusMemory backend. The recommended `llama-server` launch line is:

```bash
llama-server \
  --parallel 1 \
  --metrics \
  --da-auto \
  --kv-unified \
  --da-min-ctx 2048 \
  --da-chunk-tokens 4096 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-ngl all \
  # kv-offload (auto-compact replacement) - needs a reachable store:
  --fm-offload \
  --kv-offload-holes \
  --kv-offload-threshold 160000 \
  --focus-memory-host http://<store-host>:3900 \
  --focus-memory-token <CONTEXT_API_TOKEN>
```

(`--da-auto` is the production path: it re-chunks the whole rendered prompt, so DA restricts
attention over the *entire* conversation - which is what makes the physical KV read reduction
real (see *Scope of the effect* below). It requires `--kv-unified` (backend B): the A path is
irreversible, so a wrong focus would permanently delete the answer chunk. The
post-compaction drift item (tag-quote hijack) was fixed in v1.0 (line-start-only entry
tags); residual items are tracked in `plans/focus-llama-da-stabilization.md`. A client that
also injects FocusMemory markers can add
`--da-prompt-scan`; the marker layout then wins per request when present.)

(`--parallel 1` keeps the node single-tenant; with `--kv-unified` present it runs backend B - see
Backend A vs B below.)

| Option | What it does | Why it is on |
|--------|--------------|--------------|
| `--metrics` | Exposes Prometheus metrics on the server port | Observability for a long-running service |
| `--da-auto` | **Auto-chunking**: when a rendered prompt has no client layout markers and is at least `--da-min-ctx` tokens, the server splits it into magic chunks by message boundaries so the model's `<focus magic_chunks="N">` tags can restrict attention over the whole conversation | Production path for **physical** KV read reduction - re-chunks system + history + tool I/O, not just an injected index |
| `--kv-unified` | Enables backend B (reversible two-stream KV) | Required by `--da-auto`: the A path (`seq_rm`) is irreversible, a wrong focus permanently deletes the answer chunk |
| `--da-min-ctx 2048` | Minimum prompt length (tokens) before auto-chunking engages | Short prompts stay full-attention (no DA overhead); default is 0 (always) |
| `--da-chunk-tokens 4096` | Target magic-chunk size (tokens); hard cap 5/4 × target | Production chunk size (default 2048); larger = fewer, coarser spans |
| `--spec-type draft-mtp` | **Speculative decoding** with the model's MTP draft head | Speed-up on top of DA. Since P6, spec and DA **coexist**: while a slot is in DA mode (`da_seq` active) spec is paused automatically and resumes on the return to global attention - so spec stays ON without breaking DA |
| `--spec-draft-n-max 4` | Up to 4 draft tokens per step | Enough to overlap decode with drafting, without so many that rejections waste work |
| `--spec-draft-ngl all` | Puts the whole draft model on the GPU | The draft model is small; keeping it fully on-GPU avoids CPU round-trips that would erase the spec gain |
| `--fm-offload` | **kv-offload**: evict the oldest middle messages to the FocusMemory store once the prompt exceeds `--kv-offload-threshold`, and re-prefill them on demand when the model focuses an offloaded chunk | Replaces lossy auto-compaction with a lossless evict/refill cycle (see *kv-offload* above). Optional - off by default |
| `--kv-offload-threshold 130000` | Token count at which kv-offload eviction engages | Below this the prompt is kept whole; the threshold should sit under the client's auto-compact point (e.g. 70% of a 200K window) |
| `--focus-memory-host` | Base URL of the FocusMemory KV store (`PUT`/`GET` `/v1/kv-offload/chunk`) | Empty = kv-offload disabled even with `--fm-offload` on (fail-open) |
| `--focus-memory-token` | Bearer token for the store API (`CONTEXT_API_TOKEN`) | Empty = no auth header; set it to match the store |

**Scope of the effect.** The two paths are not two settings of the same effect. The marker
path restricts attention only over the client-declared marker region (the FocusMemory index
at the prompt tail); the rest of the conversation - system, history, tool I/O - is scaffold
and always keeps full attention. Its read reduction is therefore bounded by the size of the
injected index, and a prompt with no markers gets no DA at all (fail-open, full attention).
`--da-auto` is the opposite: it re-chunks the *whole* rendered prompt (system + history + tool
I/O) and can attend to a single chunk of it - which is why the recommended line above runs it,
and why it is the path that produces conversation-level, physical KV read reduction.

**Two layout paths, one priority.** `--da-prompt-scan` (marker path) and `--da-auto` (server path)
are independent and can both be on. Per request the server first scans for a client-provided marker
block; if that yields a valid layout it wins, and auto-chunking is skipped for that request. If the
scan finds nothing (no markers, or malformed markers that fail open), auto-chunking takes over for
prompts at or above `--da-min-ctx`. So a node can serve both marker-aware clients and plain
OpenAI-compatible clients with the same launch line.

Both paths fail open to vanilla (full attention) on any mismatch - a wrong layout must never
restrict attention to the wrong ranges. The auto path tolerates bounded tokenizer drift (up to 1%
of the text length) when mapping the headers it inserted itself, because large rendered prompts
(50K+ tokens with repeated content) do not round-trip tokenization exactly; drift beyond that
still fails open.

**Backend A vs B.** The focus/local restriction runs on backend A (`seq_rm` holes, monotonic - a removal
is irreversible within the request and a return to global attention needs a re-prefill) or backend B
(two streams, reversible, no re-prefill), selected by `--kv-unified`. Note that `--kv-unified` is
**implied by `--parallel`**: when `--parallel` is left at its default (auto), the server sets
`n_parallel=4` **and** `kv_unified=true`, so DA runs on backend B. To run backend A instead, launch with
an explicit `--parallel` (e.g. `--parallel 1`) and without `--kv-unified`. Confirm which backend a
request used from the scan log: `da_scan: ... - A path` vs `da_scan: ... - B path`.

**Verifying DA in the journal.** After a chat that carries DA markers, confirm the DA path ran:

```bash
journalctl -u qwen3.8-focus --since "10 min ago" | grep -E 'da_scan:|da_auto:|da_tag:|kv_offload:'
```

- `da_scan:` - the prompt scanner found the `[[da:N]]`/`<da:N>` markers and built the chunk-to-range map (marker path)
- `da_auto:` - auto-chunking split the prompt (the recommended production line runs `--da-auto`)
- `da_tag:` - a `<focus>`/`<local>` tag was parsed and the attention restriction applied
- `kv_offload: gate` - kv-offload engaged for the request: logs the threshold, store host, current token count, and session
- `kv_offload: evict plan` / `PUT ok` / `evicted N segment(s)` - the eviction sequence: the plan, each successful store PUT, and the prompt shrink
- `kv_offload: get-on-focus` / `GET ok ... re-prefilling` - the model focused an offloaded chunk and it was fetched + re-prefilled at the tail
- `kv_offload: disabled` (one-time warning) - `--fm-offload` was not parsed or `--focus-memory-host` is empty, so eviction can never engage

A chat with no markers and a prompt below `--da-min-ctx` produces no `da_scan:`/`da_auto:` line
and runs with full attention - that is the intended fail-open behavior. On a node running
`--da-auto`, a markerless prompt at or above `--da-min-ctx` produces a `da_auto:` line instead.

**Verifying DA end-to-end with curl.** Send a chat whose user message carries a well-formed marker
block (the same format the FocusMemory hook injects) and read the `timings` object of the response:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen27b",
  "messages": [{"role": "user", "content": "Memory entries:\n[[da:1]]The capital of France is Paris.\n[[da:2]]The capital of Germany is Berlin.\n[[da:filler]]Instructions (Declarative Attention): The memory entries above are numbered magic chunks (1-2). First identify the chunk that contains the answer to the question, and output the tag <focus magic_chunks=\"N\"> on its own line, where N is the chunk number (1-2). Then answer the question.\n[[da:layout:2]]\nQuestion: What is the capital of France?"}],
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
