#pragma once

// Sigma-strip (direction B, plans/2026-11-03-compressfast-auto-trigger.md) —
// server-side prompt compression at the messages-JSON stage.
//
// The client re-sends the full original history on every request (stateless).
// Once a session's usage (prompt tokens / --sigma-strip-window-tokens) crosses
// --sigma-strip-threshold, its K boundary advances in one big jump so the
// stripped usage reaches --sigma-strip-target, and every COMPACTABLE tool
// result inside messages[0..K) is replaced by a deterministic stub (media
// parts are dropped). The stub is a pure function of (tool, args, original
// token count) — no randomness, no timestamps — so the same input always
// yields a byte-identical rewrite, and the frozen prefix [0..K) keeps the KV
// prefix cacheable.
//
// Modes: off (default, no-op), shadow (compute + log only, body untouched),
// on (rewrites body["messages"] in place; the refresh/Σ wiring is step 3).
//
// Fail-open: any anomaly (no messages, no session key, malformed JSON,
// exception) leaves the request untouched.
//
// Session state is in-memory (process-lifetime) for the shadow phase; the
// plan's persistence ("서버 재시작을 견디도록 영속화") lands with step 3,
// when the rewrite is actually applied and a lost K would cost real work.

#include "json.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct llama_vocab;
struct server_http_req;
struct common_params;

// Outcome of one sigma-strip pass over a chat request.
struct sigma_strip_result {
    bool        eligible           = false; // real session key + parseable messages
    bool        prefix_reset       = false; // K reset this request (client compact / /clear / resume)
    bool        k_advanced         = false; // K advanced this request
    bool        target_unreachable = false; // cannot reach target even stripping everything (phase-0 data)
    size_t      K                  = 0;     // K after this request
    size_t      n_messages         = 0;
    size_t      n_stripped         = 0;     // tool results stubbed within [0, K)
    int64_t     tokens_before      = 0;     // estimated prompt tokens of the original messages
    int64_t     tokens_after       = 0;     // estimated prompt tokens of the rewritten messages
    double      usage_before       = 0.0;   // resident usage before this request's decision
    double      usage_after        = 0.0;   // resident usage after this request
    std::string rewrite_hash;               // hash16 of the rewritten messages dump ("" when not eligible)
    std::string session;
};

// Compute the sigma-strip rewrite plan for one chat request.
//   vocab:  tokenizer (nullptr = deterministic byte/4 estimate, for unit tests)
//   params: sigma_strip_mode / sigma_strip_threshold / sigma_strip_target /
//           sigma_strip_window_tokens
//   req:    used only for session resolution (X-Session-Id / user)
//   body:   the parsed /chat/completions request; with apply=true and mode
//           "on", body["messages"] is rewritten in place
// Returns an empty result (and leaves body untouched) when not eligible.
sigma_strip_result sigma_strip_maybe_rewrite(
        const struct llama_vocab * vocab,
        const common_params & params,
        const server_http_req & req,
        common_json & body,
        bool apply);
