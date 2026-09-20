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
| **B. Two streams** | Stream 0 holds the full context permanently; stream 1 holds the kept (scaffold + focus) chunks + response, copied with `seq_cp` (a cell retag in the unified pool, no data copy) | Logical read-set reduction only - the attended tokens shrink, but the physical KV scan is unchanged (removed cells stay in place, masked with `-inf`); a physical reduction needs the FA kernel to skip masked tiles (C). B's unique value: the original stream is preserved, so a global return needs no re-prefill | ~2x KV metadata (unified pool) |
| **C. Kernel skipping** | Skip fully-masked tiles/blocks in the flash-attention kernels | Potentially in-place, no extra memory | Kernel work per backend (CUDA / Metal) |

Backend A is for validating protocol adherence and accuracy. Backend B is the first candidate for measuring real speed-ups. C is only worth building if B's numbers justify it.

## Build

Same as upstream `llama.cpp`:

```bash
git clone https://github.com/edwardyoon/focus-llama.git
cd focus-llama
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
     # or omit for CPU, or use Metal on macOS
cmake --build build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-bench
```

Once done, you can smoke test like below:

```bash
$ python3 ~/focus-llama/da-probe/da_server_smoke.py http://127.0.0.1:8080 --hybrid
base   : http://127.0.0.1:8080  (hybrid expectations)
prompt tokens            : 175
document start           : 8
fact range (S2)          : [69, 106)
other range (S3)         : [116, 145)
question start (da_rm_at): 146
------------------------------------------------------------------------
baseline: 'ZEBRA-42'  (ZEBRA-42: True)
          prompt_tokens=175 completion_tokens=7
          first_token='Z' lp=-0.1332 top=[('Z', -0.1332), ('The', -3.1368), ('**', -3.648)]
strict    (fact, at Q): 'The user is asking for the "emergency shutdown codeword for the facility" based on the provided technical document.'  (ZEBRA-42: False)
          prompt_tokens=175 completion_tokens=24
          first_token='The' lp=-1.0588 top=[('The', -1.0588), ('There', -2.0083), ('Answer', -2.0664)]
paper     (fact, end): 'Z'  (ZEBRA-42: False)
          prompt_tokens=175 completion_tokens=2
          first_token='Z' lp=-0.1332 top=[('Z', -0.1332), ('The', -3.1368), ('**', -3.648)]
masked-all (doc, at Q): 'The document does not contain an emergency shutdown codeword.'  (ZEBRA-42: False)
          prompt_tokens=175 completion_tokens=13
          first_token='The' lp=-0.7549 top=[('The', -0.7549), ('<think>', -2.0124), ('**', -2.1791)]
control   (S3, at Q)  : 'ZEBRA-42'  (ZEBRA-42: True)
          prompt_tokens=175 completion_tokens=7
          first_token='Z' lp=-0.1385 top=[('Z', -0.1385), ('The', -3.2115), ('**', -3.6142)]
------------------------------------------------------------------------
baseline         : PASS (exact 'ZEBRA-42')
mechanism (Δlp)  : baseline vs masked-all first-token Δlp=0.6217 — different prefill computation (mid-prefill removal ran)
mechanism (text) : clean (informational on hybrid — with a Δlp divergence above, a leading 'Z' means the fact survives via the recurrent state, not the full-attention KV; confirm with the codeword-swap control QUOKKA-17)
selectivity      : PASS (exact 'ZEBRA-42')
strict leak      : clean (informational on hybrid — paper semantics, surviving-KV leak; see da_probe masked-A)
path divergence  : strict vs paper first-token Δlp=0.9256 — different prefill paths
OVERALL          : PASS
paper run is informational on both architectures (decode-time restriction).
also check the server terminal for a WARN 'clamping end' line from the masked-all run.
```

### Two-stream backend B smoke test (`da-probe/da_b_smoke.py`)

Backend B copies the kept prefix (scaffold + focus) to a reserved second stream and decodes there; the original stream is preserved, so a return to global attention needs no re-prefill. The server **must** be started with `--kv-unified --parallel 2` (a partial-range `seq_cp` aborts on a non-unified pool, and B needs one extra reserved sequence id). Without those flags the server falls back to logical removal (`da_rm`) with a WARN — the smoke answers would still pass, so the journal is the real evidence.

```bash
$ ./build/bin/llama-server -m <model>.gguf --kv-unified --parallel 2 -v
$ python3 ~/focus-llama/da-probe/da_b_smoke.py http://127.0.0.1:8080 [--hybrid] [--runs k1,k2,...] [--skip-cache]
```

- `--hybrid` — model has recurrent/linear-attention layers (e.g. Qwen3-Next): the fact-removed runs' TEXT is informational (the fact can survive via the recurrent state); the logprob checks are architecture-independent.
- `--runs` — run a subset of phase-1 keys: `baseline,bkeep,bnofact,logical,bkeep2`.
- `--skip-cache` — skip phase 2 (cache integrity; needs `--parallel 2`).

Phase 1 (one ZEBRA prompt, `cache_prompt=false`) checks:

1. **mechanism** — `baseline` vs `bkeep` first-token logprobs diverge: the question was prefilled against the copied KV, so identical logprobs mean the B switch did not run (old binary, missing flags, or fallback).
2. **B vs A equal** — `bnofact` (B) and `logical` (A, `seq_rm`) apply the same removal by different means; the attended token sets are identical, so the first-token logprobs must agree within 5e-3 (observed 0.0003 on local Bonsai-8B). A large delta means B's keep-set computation is wrong.
3. **answers** — `baseline`/`bkeep`/`bkeep2` exactly `ZEBRA-42` (the fact must be readable from the copied ranges); `bkeep2` (a consecutive `da_b`) proves the reserved id is reusable after the previous B slot released.

Phase 2 (cache integrity, `--parallel 2`): C1 caches P1 on slot S1, C2 (ZEBRA + `da_b`) runs on S2, C3 (P1 again) must land back on S1 with its cache untouched — `C3 == C1` proves a B switch does not destroy another slot's prompt cache.

**Reading the read-reduction log** — the server logs the logical read set for every B request. In the terminal, or via journal on the service node:

```bash
$ grep 'da_b:' <server log>        # or: journalctl -u llama-server | grep 'da_b:'
```

Per B request: `da_b: switched decode to seq N (at da_rm_at) - logical read set now X of Y prefix token(s) (Z removed, -P%)`, one `da_b:   keep [lo, hi) n token(s)` line per kept range, per-step `da_b: decode #p: logical read set X of Y ...` lines (with `-v`), and a final `da_b: finished on seq N - final logical read set ...` summary. These are logical (attended) counts; the physical KV load is reduced too (the FA kernels skip fully-masked chunks) but is chunk-quantized and not logged as a byte count. Also verify `grep 'falling back to logical removal'` is empty.

## Relationship to FocusMemory

[FocusMemory](https://github.com/edwardyoon/FocusMemory) chunks and indexes long-term context. `focus-llama` is the inference-side counterpart: it lets the model read a compact index in `global` mode and then commit attention to specific chunks. The two are independent and can be used separately.

## Status and limits

Early work in progress.

- **Works:** `llama-server` accepts `da_rm` / `da_rm_at` to drop KV token ranges either mid-prefill or after prefill. Checked via first-token logprobs on a small smoke test.
- **Probe only:** parsing `<focus magic_chunks="N">` during generation and removing ranges at that point (`da-probe/`). Not in the server yet.
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
