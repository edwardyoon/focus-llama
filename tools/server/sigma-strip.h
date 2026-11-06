#pragma once

// Sigma-strip (direction B, plans/2026-11-03-compressfast-auto-trigger.md) —
// server-side prompt compression at the messages-JSON stage.
//
// The client re-sends the full original history on every request (stateless).
// Once a session's usage (prompt tokens / --sigma-strip-window-tokens) crosses
// --sigma-strip-threshold, its K boundary advances in one big jump so the
// stripped usage reaches --sigma-strip-target, and every COMPACTABLE tool
// result inside messages[0..K) is replaced by a deterministic stub (media
// parts are dropped). Two floors limit the jump: K never crosses the start of
// the last --sigma-strip-keep-recent tool rounds (recent working context), and
// the stripped prompt never drops below --sigma-strip-min-keep-tokens (the
// effective goal is max(target x window, min_keep)). The stub is a pure
// function of (tool, args, original
// token count) — no randomness, no timestamps — so the same input always
// yields a byte-identical rewrite, and the frozen prefix [0..K) keeps the KV
// prefix cacheable.
//
// Modes: off (default, no-op), shadow (compute + log only, body untouched),
// on (rewrites body["messages"] in place). Stage 2 (--sigma-sum
// off|rule|llm) adds whole-message removal of [0,K_sum) + a frozen Σ block:
// rule = deterministic local block, llm = fetched from focus-memory
// /sigma/refresh (rule fallback after repeated failures).
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
    bool        floor_limited      = false; // advance stopped at the token floor, before the cap
    int         skip               = 0;     // futility guard skip: 0=none, 1=gain, 2=rearm
    size_t      K                  = 0;     // K after this request
    size_t      n_messages         = 0;
    size_t      n_stripped         = 0;     // tool results stubbed within [0, K)
    int64_t     tokens_before      = 0;     // estimated prompt tokens of the original messages
    int64_t     tokens_after       = 0;     // estimated prompt tokens of the rewritten messages
    double      usage_before       = 0.0;   // (resident + est_offset) / window before this request's decision
    double      usage_after        = 0.0;   // (tokens_after + est_offset) / window after this request
    int64_t     est_offset         = 0;     // reported-space offset applied to this request's gate/floors
    std::string rewrite_hash;               // hash16 of the rewritten messages dump ("" when not eligible)
    // stage 2 (Σ block, plans/sigma_strip_compaction.md): whole-message removal
    // of [0,K_sum) + a frozen Σ block prepended to the first surviving user
    // message. sum_applied means the block was committed this request (or was
    // already committed and re-applied); K_sum/sum_hash describe the committed
    // boundary/snapshot after this request.
    bool        sum_applied        = false; // stage 2 in effect for this request's rewrite
    size_t      K_sum              = 0;     // committed removal boundary after this request
    std::string sum_hash;                  // hash16 of the frozen Σ block ("" when none committed)
    int64_t     sum_fetch_ms       = 0;     // Σ block generation/fetch time (rule: local gen)
    bool        sum_partial        = false; // block was partial (llm mode)
    int         sum_fail           = 0;     // consecutive stage-2 fetch failures (0 = healthy)
    bool        bypass             = false; // futility hard-limit bypass fired (usage >= bypass usage)
    std::string session;
};

// Compute the sigma-strip rewrite plan for one chat request.
//   vocab:  tokenizer (nullptr = deterministic byte/4 estimate, for unit tests)
//   params: sigma_strip_mode / sigma_strip_threshold / sigma_strip_target /
//           sigma_strip_min_keep_tokens / sigma_strip_keep_recent /
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

// Capture the reported prompt-token count of a completed chat response and
// update the session's est offset: offset = clamp(reported - last_est, 0,
// window/2), where last_est is the previous request's post-rewrite
// estimate. The server's message-level estimate omits the chat template,
// tool schemas and system overhead that the client's full prompt carries,
// so the gate and floors run in reported space (estimate + offset). Also
// logs the per-response observation line. The window is read from the
// session state (stamped by sigma_strip_maybe_rewrite), so this takes no
// common_params — it can be called from the response path (server_context),
// which does not hold one. No-op when the session is unknown (non-chat
// endpoints, sigma off, no session key) or has no previous request.
void sigma_strip_note_response(
        const std::string & session,
        int64_t reported_prompt_tokens);

// Startup validation of the sigma-strip configuration (call once before
// serving). Returns a non-empty error when the floors make the K advance
// unusable: floor = max(target x window, min_keep_tokens) >= threshold x
// window means a full advance still leaves the usage at/above the
// threshold, so the session either stalls (no advance) or re-advances
// every request (chain re-prefill) until native compact. A thin margin
// (floor within 10% of the window below the threshold) works but risks
// the same chain, so it is warned, not rejected. Empty when the feature
// is off or has no window.
std::string sigma_strip_config_check(const common_params & params);

// Test-only fault injection (tests/test-sigma-strip.cpp): when on, the
// stage-2 fetch in [B] throws before the HTTP request is sent, verifying
// the fail-open path releases the per-session inflight flag. Compiled into
// test builds only (SIGMA_STRIP_TESTS) — not part of the production binary.
#ifdef SIGMA_STRIP_TESTS
void sigma_strip_test_set_fetch_fault(bool on);
#endif
