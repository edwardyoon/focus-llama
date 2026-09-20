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
