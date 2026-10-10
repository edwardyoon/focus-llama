#pragma once

// kv-offload (auto-compact replacement) — FocusMemory store client + eviction
// planning, extracted from server-context.cpp so the store I/O, the eviction
// planner and the in-memory per-session state live in one self-contained unit.
//
// ODR note: the HTTP helpers (kv_offload_put/get/session_get/get_sigma) build an
// httplib::Client via common_http_client (common/http.h). That layout depends on
// CPPHTTPLIB_SSL_ENABLED, so this file must only be compiled inside the
// server-context target, which links cpp-httplib PRIVATE and shares the compile
// definition (see CMakeLists.txt — 2026-09-28 kv_offload_session_get stack
// corruption incident).

#include "json.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

// full definitions pulled in by kv-offload.cpp
struct server_http_req;
struct llama_vocab;

// One evicted segment planned for upload to the FocusMemory store.
struct kv_offload_evict_segment {
    std::string key;     // stable content hash (the FocusMemory key)
    std::string hint;    // one-line hint shown in the DA instruction
    std::string text;    // the evicted message range text (for the PUT)
    int32_t     tokens = 0;
    size_t      range_lo = 0;  // original-text range [range_lo, range_hi)
    size_t      range_hi = 0;
};

// In-memory kv-offload state, one instance per server process, owned by
// server_routes. Bundles the per-session upload ledger, the logical (full
// prompt) token counts, the B4 pin-released flags and the mutex that guards
// all three.
struct kv_offload_state {
    std::mutex put_mutex;
    // session -> (content-hash key -> offloaded token count). Storing the token
    // count per key lets GET /kv_state report the total offloaded tokens.
    std::map<std::string, std::map<std::string, int64_t>> uploaded; // session -> keys -> tokens
    // session -> latest logical (full prompt) token count, recorded at the
    // eviction gate. GET /kv_state derives resident = logical - offloaded.
    std::map<std::string, int64_t> session_logical; // session -> logical tokens
    // sessions whose first-user-message pin has been released by the
    // FocusMemory state worker (the user revoked the original task).
    std::set<std::string> pin_released; // session ids
};

// --- pure utils ---
// FNV-1a 64-bit of the segment text -> 16 hex chars (stable content key).
std::string kv_offload_hash16(const std::string & s);
// Minimal JSON string escape (for the PUT body).
std::string kv_offload_json_escape(const std::string & s);
// Percent-encode a value for use in a URL query string.
std::string kv_offload_url_encode(const std::string & s);
// Resolve the FocusMemory session id for one request.
std::string kv_offload_resolve_session(const server_http_req & req, const common_json & data);

// --- FocusMemory store HTTP client (fail-open: false on any error) ---
bool kv_offload_put(const std::string & host, const std::string & token,
                    const std::string & session_id, const std::string & key,
                    const std::string & text, int32_t tokens);
bool kv_offload_get(const std::string & host, const std::string & token,
                    const std::string & session_id, const std::string & key,
                    std::string & out_text);
bool kv_offload_session_get(const std::string & host, const std::string & token,
                            const std::string & session_id, bool & out_pin_released);
bool kv_offload_get_sigma(const std::string & host, const std::string & token,
                          const std::string & session_id, std::string & out_anchor);

// --- eviction planning ---
// Scan the chat text, evict the oldest middle messages until under threshold +
// retain, and append the resulting segments to out_segments.
bool kv_offload_evict(const llama_vocab * vocab, const std::string & text,
                      int32_t total_tokens, int32_t threshold,
                      std::vector<kv_offload_evict_segment> & out_segments,
                      bool release_pin = false, int32_t retain = 0);

// --- /kv_state payload ---
// Build the per-session /kv_state payload. Assumes state.put_mutex is held by
// the caller. Returns null if the session is unknown (no logical recording and
// no ledger entry).
common_json kv_offload_build_state_json(const kv_offload_state & state, const std::string & session, int64_t buffer_size);
