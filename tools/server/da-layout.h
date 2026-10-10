#pragma once

// DA (Declarative Attention) prompt layout, extracted from server-context.cpp.
//
// Two paths share the lenient char->token mapping helpers:
//  - P1 marker scan (da_scan_prompt): the client hook (FocusMemory,
//    FOCUSMEMORY_DA) appends a [[da:N]]/... marker block to the rendered
//    prompt; the scan recovers the chunk token ranges from the prompt string +
//    tokenization and fills the task params (da_chunks, da_filler, da_b).
//  - P4 auto-chunking (da_auto_chunk): a marker-less prompt is split into
//    [Magic Chunk N] chunks and the DA instruction (+ optional Σ anchor) is
//    appended.
// Fail-open: no markers (a normal request) or any mismatch leaves the request
// vanilla.
//
// Free functions, no per-request state. Depends on llama.h (llama_vocab /
// llama_tokens) and server-task.h (task_params), both already pulled in by
// server-context.h; the heavy includes (server-common.h for SRV_INF/SRV_WRN,
// common.h for the tokenizer) live in da-layout.cpp only.

#include "llama.h"
#include "server-task.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Result of the P4 auto-chunk layout: the modified prompt (headers + DA
// instruction + optional Σ anchor inserted) and the char ranges the caller
// needs to drive the B/A KV paths.
struct da_auto_layout {
    bool ok = false;
    std::string modified;              // prompt with [Magic Chunk N] headers + DA instruction (+ Σ anchor) inserted
    std::vector<size_t> header_pos;    // final char offset of each [Magic Chunk N] header (ascending chunk order)
    size_t tail_pos = 0;               // start of the last user message after header insertion
    std::pair<size_t, size_t> filler = { 0, 0 };  // char range of the DA instruction (removed in FOCUS mode)
    std::pair<size_t, size_t> sigma  = { 0, 0 };  // char range of the Σ anchor (kept in FOCUS mode)
    size_t n_source_msgs = 0;
};

// kv-offload (Option B, --kv-offload-holes): an evicted piece is always its
// own chunk, so the KV hole (the evicted message) aligns with the chunk
// boundary. The auto-chunker treats each evicted message's char range in the
// original prompt text as a hard cut boundary (the message gets its own chunk,
// with its [Magic Chunk N] header just before it).
struct da_evict_bound {
    size_t lo = 0;
    size_t hi = 0;
    std::string hint;
};

// Lenient char->token round-trip (P4, da_auto; also used by da_scan):
// offsets[i] = char offset of token i in the detokenized concat (and
// offsets[n] = concat length). The strict round trip
// (detokenize(tokenize(text)) == text) is not guaranteed for large rendered
// prompts, so drift of up to 1% of the text length is tolerated and the
// (monotonic) offsets are returned anyway; larger drift returns {} and the
// caller fails open.
std::vector<size_t> da_token_offsets_lenient(const llama_vocab * vocab, const std::string & text, const llama_tokens & tokens);

// Nearest token boundary (tolerant of drift); -1 only when offsets is empty.
// da_auto uses this so a header char offset that lands a few chars off a true
// boundary (tokenizer drift) still maps to the adjacent token.
int32_t da_char_to_token_nearest(const std::vector<size_t> & offsets, size_t c);

// P1 marker scan: recover the chunk token ranges from the prompt string +
// tokenization and fill params.da_chunks/da_filler/da_chunk_base/da_b.
// Fail-open: returns false (the request stays vanilla) on any mismatch.
bool da_scan_prompt(const llama_vocab * vocab, const std::string & text, const llama_tokens & tokens, bool kv_unified, task_params & params);

// P4 auto-chunker: split a marker-less prompt into [Magic Chunk N] chunks and
// append the DA instruction (+ optional Σ anchor). The default arguments live
// here only; the definition in da-layout.cpp omits them.
da_auto_layout da_auto_chunk(const llama_vocab * vocab, const std::string & text, int32_t da_chunk_tokens,
        const std::vector<std::string> & offloaded_hints = {},
        const std::vector<da_evict_bound> & evict_bounds = {},
        const std::string & sigma_anchor = {});
