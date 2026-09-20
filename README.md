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
│ tool-use     │   │   parse tag on closing '>'        │   │   (physical compaction)  │
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
| **B. Two streams** | Stream 0 holds the full context permanently; stream 1 holds scaffold + focus chunks + response; new tokens are copied across with `seq_cp` | Real reduction in KV read (attended region is physically small) | ~2x KV memory, small copy per switch |
| **C. Kernel skipping** | Skip fully-masked tiles/blocks in the flash-attention kernels | Potentially in-place, no extra memory | Kernel work per backend (CUDA / Metal) |

Backend A is for validating protocol adherence and accuracy. Backend B is the first candidate for measuring real speed-ups. C is only worth building if B's numbers justify it.

## Roadmap

- [ ] **P0. Prompt + parser, no masking (DA-nm).** Chunker, tool-use transcript prompt, DA state machine, attended-token logger. Measures protocol adherence and the accuracy of the prompt format alone.
- [ ] **P1. Backend A (`seq_rm`).** Exact-mask accuracy; compare Vanilla / DA-nm / DA on short-context tasks first.
- [ ] **P2. Backend B (two streams).** Verify that decoding a stream while the other stays idle works with the public API; measure per-step KV read and wall-clock.
- [ ] **P3. Model coverage.** Check behaviour on pure-attention vs. hybrid (recurrent) architectures.
- [ ] **P4. Backend C.** Only if P2 shows a clear win.
- [ ] **P5. FocusMemory integration.** Use chunk indexes/summaries as the global-mode view.

## Expectations and known limits

Numbers below are from the DA paper (zero-shot, vLLM, batched serving), **not** measurements of this fork.

- Attended KV tokens per response dropped **52.0%** (Gemma-4-31B) and **31.1%** (Qwen-3.6-27B), with accuracy drops of 1.27pp and 2.75pp on 15 long-context tasks.
- Per-step savings are large only in `focus`/`local` steps (76-99% fewer attended tokens). `global` steps save nothing and account for most of the remaining attended tokens, more so at longer contexts.
- DA generates **15-35% more decode steps** than vanilla. Whether that nets out depends on how much of decode time is spent on global-attention KV reads. The paper's wall-clock figures (0.71x / 0.77x) are roofline estimates for large-batch serving, not measurements. **Single-stream local inference is a different regime and may see little or no gain.**
- The mask only applies to global-attention layers; sliding-window and recurrent (e.g. Gated DeltaNet) layers are untouched. On hybrid models `seq_rm` reproduces the paper's semantics — the attention KV of the removed range is freed, but the recurrent/linear-attention state keeps the removed content, so a removed fact can still be answerable. **Isolation** (a removed fact becomes unanswerable) is a property of pure-attention models only and is not what DA promises on hybrids. Hybrids also save less per token (attention KV is a fraction of the layers), so the speed-up factor is smaller.
- Thinking mode must be disabled; models did not follow the protocol inside thinking traces.
- Zero-shot protocol adherence needs a capable model. Small models (around 4B) fail often to emit valid `focus` calls.
- Prefill cost and resident KV memory are unchanged.

## Build

Same as upstream `llama.cpp`:

```bash
git clone https://github.com/edwardyoon/focus-llama.git
cd focus-llama
cmake -B build -DGGML_CUDA=ON     # or omit for CPU, or use Metal on macOS
cmake --build build --config Release -j
```

## Usage (planned, not yet implemented)

A dedicated tool is planned rather than changes to `llama-server`:

```bash
# sketch only: flags and binary name are not final
./build/bin/llama-da \
  -m model.gguf \
  --da-context ./doc.txt \
  --da-chunk-tokens 2048 \
  --da-backend seq_rm \
  --da-log attended.jsonl \
  -p "Question about the document"
```

Each run is intended to log, per step: mode, attended token count, and total decode steps, so results can be compared against the paper's metrics.

## Relationship to FocusMemory

FocusMemory chunks and indexes long-term context. `focus-llama` is the inference-side counterpart: it lets the model read a compact index in `global` mode and then commit attention to specific chunks. The two are independent and can be used separately.

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
