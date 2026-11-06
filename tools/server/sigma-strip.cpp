// sigma-strip.cpp — server-side prompt compression at the messages-JSON stage
// (direction B, plans/2026-11-03-compressfast-auto-trigger.md).
//
// Stage 2 (shadow): when a session's usage (estimated prompt tokens /
// sigma_strip_window_tokens) crosses sigma_strip_threshold, the session's K
// boundary advances in one big jump so the stripped prompt reaches
// sigma_strip_target; tool results in messages[0..K) are replaced by
// deterministic stubs. Shadow mode computes and logs only — the request body
// is left untouched.
//
// Two floors limit the jump:
// - protection: K never crosses the start of the last keep_recent tool
//   rounds (tool_calls + their results) — recent working context. Stubbing
//   it leaves the model nothing to act on (empty responses).
// - token: the stripped prompt never drops below min_keep_tokens (effective
//   goal = max(target x window, min_keep)).
// If a floor blocks the target, the target_unreachable path still advances
// what it can; a session that stays above threshold is handed to the native
// compact safety net. A thin threshold-floor margin is warned once per
// session (chain re-prefill risk).
//
// Guarantees:
// - byte-identical rewrite for a fixed prefix + fixed K (deterministic stubs,
//   no randomness / timestamps)
// - message count and tool_call_id pairing are preserved
// - the last keep_recent tool rounds are never stubbed
// - prefix fingerprint (hash of messages[0..K)) mismatches reset K
// - fail-open: any error leaves the request untouched
#include "sigma-strip.h"
#include "kv-offload.h"
#include "server-common.h"
#include "http.h"          // common_http_client (cpp-httplib)
#include "common.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {

//
// token estimation
//

// Deterministic prompt-token estimate. With a vocab: real tokenization.
// Without (unit tests): ceil(bytes / 4) — stable across runs.
int64_t estimate_tokens(const struct llama_vocab * vocab, const std::string & text) {
    if (text.empty()) {
        return 0;
    }
    if (vocab) {
        const std::vector<llama_token> tokens = common_tokenize(vocab, text, /*add_special*/ false);
        return (int64_t) tokens.size();
    }
    return (int64_t) ((text.size() + 3) / 4);
}

// String value of an object key, "" if absent or not a string.
std::string json_string_or(const common_json & obj, const char * key) {
    if (obj.is_object() && obj.contains(key) && obj.at(key).is_string()) {
        return obj.at(key).get<std::string>();
    }
    return "";
}

// All text of a message: content (string or array of parts, text parts only)
// plus assistant tool_calls (id + function name + arguments).
std::string message_text(const common_json & msg) {
    std::string out;
    if (msg.contains("content")) {
        const common_json & c = msg.at("content");
        if (c.is_string()) {
            out = c.get<std::string>();
        } else if (c.is_array()) {
            for (auto it = c.begin(); it != c.end(); ++it) {
                if (json_string_or(*it, "type") == "text") {
                    out += json_string_or(*it, "text");
                }
            }
        }
    }
    if (msg.contains("tool_calls") && msg.at("tool_calls").is_array()) {
        for (auto it = msg.at("tool_calls").begin(); it != msg.at("tool_calls").end(); ++it) {
            const common_json & tc = *it;
            if (!tc.is_object()) {
                continue;
            }
            out += json_string_or(tc, "id");
            if (tc.contains("function") && tc.at("function").is_object()) {
                const common_json & fn = tc.at("function");
                out += json_string_or(fn, "name");
                if (fn.contains("arguments")) {
                    const common_json & a = fn.at("arguments");
                    if (a.is_string()) {
                        out += a.get<std::string>();
                    } else if (!a.is_null()) {
                        out += a.dump();
                    }
                }
            }
        }
    }
    return out;
}

int64_t message_tokens(const struct llama_vocab * vocab, const common_json & msg) {
    return estimate_tokens(vocab, message_text(msg));
}

//
// tool resolution
//

// COMPACTABLE_TOOLS (case-insensitive). MCP-prefixed variants (mcp__<server>__<tool>)
// match on the suffix after the last "__".
const std::set<std::string> & compactable_tools() {
    static const std::set<std::string> tools = {
        "read_file", "run_shell_command", "shell", "grep_search", "grep", "glob",
        "web_fetch", "web_search", "search_web", "read_mcp_resource",
        "edit", "write_file", "skill", "search_memory",
    };
    return tools;
}

bool is_compactable_tool(const std::string & name) {
    std::string n = name;
    for (char & c : n) {
        c = (char) tolower((unsigned char) c);
    }
    if (compactable_tools().count(n)) {
        return true;
    }
    const auto pos = n.rfind("__");
    if (pos != std::string::npos) {
        return compactable_tools().count(n.substr(pos + 2));
    }
    return false;
}

// tool_call_id -> (name, args) over all assistant tool_calls, in message
// order (the first non-empty name wins, matching the old per-id scan).
using tool_call_map_t = std::map<std::string, std::pair<std::string, std::string>>;

// Built once per request so each tool message resolves its call in O(1)
// instead of scanning every message (O(n^2) overall for long sessions).
static tool_call_map_t build_tool_call_map(const common_json & messages) {
    tool_call_map_t m;
    for (size_t i = 0; i < messages.size(); ++i) {
        const common_json & msg = messages.at(i);
        if (!msg.is_object() || json_string_or(msg, "role") != "assistant") {
            continue;
        }
        if (!msg.contains("tool_calls") || !msg.at("tool_calls").is_array()) {
            continue;
        }
        for (auto it = msg.at("tool_calls").begin(); it != msg.at("tool_calls").end(); ++it) {
            const common_json & tc = *it;
            if (!tc.is_object()) {
                continue;
            }
            const std::string id = json_string_or(tc, "id");
            if (id.empty() || m.count(id) != 0) {
                continue;
            }
            std::string name, args;
            if (tc.contains("function") && tc.at("function").is_object()) {
                name = json_string_or(tc.at("function"), "name");
                if (tc.at("function").contains("arguments")) {
                    const common_json & a = tc.at("function").at("arguments");
                    if (a.is_string()) {
                        args = a.get<std::string>();
                    } else if (!a.is_null()) {
                        args = a.dump();
                    }
                }
            }
            if (name.empty()) {
                name = json_string_or(tc, "name");
            }
            if (!name.empty()) {
                m[id] = std::make_pair(name, args);
            }
        }
    }
    return m;
}

// Resolve the tool name + arguments for a role=tool message: tool_call_id
// looked up in the prebuilt call map, explicit "name" field as fallback.
bool resolve_tool_call(const tool_call_map_t & calls, const common_json & msg,
        std::string & out_name, std::string & out_args) {
    const std::string explicit_name = json_string_or(msg, "name");
    const std::string id = json_string_or(msg, "tool_call_id");
    if (!id.empty()) {
        const auto it = calls.find(id);
        if (it != calls.end()) {
            out_name = it->second.first;
            out_args = it->second.second;
            return true;
        }
    }
    if (!explicit_name.empty()) {
        out_name = explicit_name;
        return true;
    }
    return false;
}

//
// stub construction — pure function of (tool, args, original tokens)
//

// "[<tool> result stripped — <hint>, <N> tokens. Re-read if needed]"
std::string make_stub(const std::string & tool, const std::string & args_json, int64_t orig_tokens) {
    std::string hint;
    if (!args_json.empty()) {
        try {
            const common_json args = common_json::parse(args_json);
            if (args.is_object()) {
                for (const char * key : { "path", "file_path", "filepath", "query", "pattern", "url" }) {
                    const std::string v = json_string_or(args, key);
                    if (!v.empty()) {
                        hint = std::string(key) + "=" + (v.size() > 80 ? v.substr(0, 80) + "..." : v);
                        break;
                    }
                }
            }
        } catch (const std::exception &) {
            // unparseable arguments: fall through to the hash hint
        }
    }
    if (hint.empty()) {
        hint = "args=" + kv_offload_hash16(args_json);
    }
    return "[" + tool + " result stripped — " + hint + ", " + std::to_string(orig_tokens) + " tokens. Re-read if needed]";
}

// Keep only the text parts of an array content; replace with "" if nothing left.
void strip_media(common_json & content) {
    common_json kept = common_json::array();
    for (auto it = content.begin(); it != content.end(); ++it) {
        if (json_string_or(*it, "type") == "text") {
            kept.push_back(*it);
        }
    }
    if (kept.empty()) {
        content = std::string("");
    } else {
        content = kept;
    }
}

//
// prefix fingerprint
//

// hash16 of the original messages[0..K) — the KV-prefix stability anchor.
std::string prefix_fingerprint(const common_json & messages, size_t K) {
    common_json prefix = common_json::array();
    for (size_t i = 0; i < K && i < messages.size(); ++i) {
        prefix.push_back(messages.at(i));
    }
    return kv_offload_hash16(prefix.dump());
}

//
// per-session state (in-memory, process lifetime — shadow phase)
//
struct sigma_strip_session_state {
    size_t      K = 0;             // K after the last request
    std::string prefix_fp;         // hash16 of the original messages[0..K)
    // window stamped from common_params at maybe_rewrite time, so
    // sigma_strip_note_response (response path, no common_params in scope)
    // can clamp and warn without one.
    int64_t     window = 0;
    // est calibration (E1): the server's message-level estimate omits the
    // chat template, tool schemas and system overhead that the client's
    // full prompt carries. est_offset = last (reported - last_est),
    // clamped to [0, window/2]; the gate and floors run in reported space.
    int64_t     est_offset = 0;
    int64_t     last_est   = 0;    // last request's sent estimate (tokens_after)
    int64_t     last_reported = 0; // last response's reported prompt tokens
    // decision snapshot for the per-response observation line (E2)
    size_t      last_K = 0;
    size_t      last_stripped = 0;
    bool        last_advanced = false;
    bool        last_floor_limited = false;
    // one-shot warnings (E3)
    bool        window_warned_big = false; // reported > server window
    bool        resets_warned = false;     // repeated prefix resets
    int         n_resets = 0;
    // stage 2 (Σ block, plans/sigma_strip_compaction.md): whole-message removal
    // boundary [0,K_sum) and the frozen snapshot. All indices are on the
    // ORIGINAL messages basis (invariant K >= K_sum). The block is frozen
    // between advances so the prefix stays stable (no re-prefill).
    size_t      K_sum = 0;                 // last committed removal boundary
    std::string sum_block;                 // frozen Σ block (rule/llm)
    std::string sum_hash;                  // hash16 of the frozen block
    uint64_t    sum_gen = 0;               // generation counter (revalidation)
    bool        sum_inflight = false;      // a refresh is in flight (llm mode)
    int         sum_fetch_fail_count = 0;  // consecutive refresh failures
    int64_t     sum_last_fail_ms = 0;      // steady-clock ms of last fetch failure (cooldown)
};

std::mutex & sigma_strip_mutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, sigma_strip_session_state> & sigma_strip_sessions() {
    static std::map<std::string, sigma_strip_session_state> m;
    return m;
}

// Test-only fault injection (tests/test-sigma-strip.cpp): when set, the
// stage-2 fetch in [B] throws before the HTTP request is sent, exercising
// the fail-open path that must release the per-session inflight flag.
// Never set in production.
std::atomic<bool> g_test_fetch_fault{false};

// RAII release of the per-session sum_inflight flag. [A] sets the flag before
// the (throwable) [B] fetch and [C] settles it; if [B] or [C] throws, the
// outer catch returns fail-open and a plain flag would stay true forever —
// stage 2 and the stage-1 async absorption (both gated on !sum_inflight)
// would stall for the session until a prefix reset. The guard releases the
// flag on scope exit, but only when this request still owns it (sum_gen
// unchanged): a prefix reset or a newer request has already taken over.
// gen == 0 means the guard is inactive (no fetch was set in [A]).
struct inflight_guard {
    const std::string & session;
    uint64_t gen = 0;
    inflight_guard(const std::string & s) : session(s) {}
    inflight_guard(const inflight_guard &) = delete;
    inflight_guard & operator=(const inflight_guard &) = delete;
    ~inflight_guard() {
        if (gen == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(sigma_strip_mutex());
        auto it = sigma_strip_sessions().find(session);
        if (it != sigma_strip_sessions().end()
                && it->second.sum_inflight
                && it->second.sum_gen == gen) {
            it->second.sum_inflight = false;
        }
    }
};

//
// step-1 diagnostic: classification of the post-stage-1 message composition.
// At a K advance, record where the remaining tokens sit so the stage-2
// decision (whole-message removal of [0,K_sum)) is grounded in the actual
// distribution. Deletable = the non-system messages in [0,K_cap) that stage 2
// could remove (stubs / assistant text+args / resident non-compactable tool
// results / user). Irreducible = system (stage 2 keeps it) + the protected
// recent rounds [K_cap,n) + the template/tool-schema/system overhead carried
// by est_offset. stage2_can_reach says whether removing the whole deletable
// region could still drop the prompt to the target — if the irreducible floor
// alone exceeds the target, only native compact can get there.
struct sigma_classify {
    // deletable region [0,K_cap) by role
    size_t  n_stubs = 0;      int64_t tok_stubs = 0;
    size_t  n_assistant = 0;  int64_t tok_assistant = 0;
    size_t  n_tool_res = 0;   int64_t tok_tool_res = 0;   // non-compactable tool results
    size_t  n_user = 0;       int64_t tok_user = 0;
    int64_t tok_deletable = 0;
    // irreducible
    size_t  n_system = 0;     int64_t tok_system = 0;    // kept by stage 2
    size_t  n_protected = 0;  int64_t tok_protected = 0; // [K_cap,n) recent rounds
    int64_t fixed_off = 0;                             // est_offset
    int64_t tok_floor = 0;      // tok_system + tok_protected + fixed_off
    bool    stage2_can_reach = false;
};

// Classify the post-stage-1 (stubbed) message distribution. tok[i] is the
// original message tokens minus the stub savings when the message is an
// active stub (i < K and stubbed), else the original tokens — the stage-1
// stubbed state, which is what the stage-2 decision (whole-message removal of
// [0,K_sum)) is grounded in. Media-stripped content is not accounted for (a
// distribution diagnostic, not a byte-exact measure).
sigma_classify classify_post_stage1(const common_json & messages,
        const std::vector<int64_t> & orig,
        const std::vector<std::string> & stubs,
        const std::vector<int64_t> & savings,
        size_t K, size_t K_cap, size_t n, int64_t off, double target_tokens) {
    sigma_classify c;
    c.fixed_off = off;
    for (size_t i = 0; i < n && i < messages.size(); ++i) {
        const common_json & msg = messages.at(i);
        if (!msg.is_object()) continue;
        const int64_t tok = (i < K && !stubs[i].empty())
            ? (orig[i] - savings[i]) : orig[i];
        const std::string role = json_string_or(msg, "role");
        if (role == "system") {
            c.n_system++; c.tok_system += tok;      // stage 2 keeps system
            continue;
        }
        if (i >= K_cap) {
            c.n_protected++; c.tok_protected += tok;
            continue;
        }
        // deletable region [0,K_cap), non-system
        c.tok_deletable += tok;
        if (role == "assistant") {
            c.n_assistant++; c.tok_assistant += tok;
        } else if (role == "tool") {
            if (i < K && !stubs[i].empty()) { c.n_stubs++; c.tok_stubs += tok; }
            else { c.n_tool_res++; c.tok_tool_res += tok; }
        } else {
            c.n_user++; c.tok_user += tok;
        }
    }
    c.tok_floor = c.tok_system + c.tok_protected + c.fixed_off;
    c.stage2_can_reach = (c.tok_floor <= target_tokens);
    return c;
}

//
// stage 2 (Σ block): K_sum computation + deterministic rule block
// (plans/sigma_strip_compaction.md §5.2/§5.4). The rule block is a pure,
// byte-deterministic function of the removed messages [0,K_sum) — no LLM, no
// timestamps — so the frozen snapshot is stable across requests (no
// re-prefill) and the rewrite is byte-identical for a fixed prefix.
//

// Lowercase + strip the MCP "mcp__<server>__" prefix (last "__" group).
std::string normalize_tool_name(const std::string & name) {
    std::string n = name;
    for (char & c : n) {
        c = (char) tolower((unsigned char) c);
    }
    const auto pos = n.rfind("__");
    if (pos != std::string::npos) {
        n = n.substr(pos + 2);
    }
    return n;
}

// Byte-clamp to `max` chars, backing off a split UTF-8 sequence, with an
// ellipsis. Deterministic (same input -> same output).
std::string clamp_chars(const std::string & s, size_t max) {
    if (s.size() <= max) {
        return s;
    }
    std::string out = s.substr(0, max);
    while (!out.empty() && ((unsigned char) out.back() & 0xC0) == 0x80) {
        out.pop_back();
    }
    return out + "…";
}

// Removes client-appended context blocks (user-prompt-submit context, system
// reminders) so the rule block keeps the user's own words — mirrors
// stripInjectedBlocks in focus-memory (kvoffload.js). Without this, a
// <system-reminder> from the removed region is re-injected into the active
// context through the frozen summary (the old contamination path). Only
// complete tag pairs are removed, matching the JS regex; the result is
// trimmed.
std::string strip_injected_blocks(const std::string & s) {
    static const struct { const char * open; const char * close; } kTags[] = {
        { "<qwen:user-prompt-submit-context>", "</qwen:user-prompt-submit-context>" },
        { "<system-reminder>", "</system-reminder>" },
    };
    std::string out = s;
    for (const auto & tag : kTags) {
        const std::string o(tag.open);
        const std::string c(tag.close);
        size_t pos = 0;
        while ((pos = out.find(o, pos)) != std::string::npos) {
            const size_t end = out.find(c, pos + o.size());
            if (end == std::string::npos) {
                break;   // unclosed tag: leave as-is (same as the JS regex)
            }
            out.erase(pos, (end + c.size()) - pos);
        }
    }
    const size_t b = out.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = out.find_last_not_of(" \t\r\n");
    return out.substr(b, e - b + 1);
}

// Best-effort ok/error for a shell command: scan the first 200 chars of the
// matching tool result for error markers. The body itself is never placed in
// the block (plan §4.5 "no tool bodies") — only a status flag is derived.
std::string shell_status(const common_json & messages, const std::string & tool_call_id) {
    if (tool_call_id.empty()) {
        return "ok";
    }
    for (size_t j = 0; j < messages.size(); ++j) {
        const common_json & m = messages.at(j);
        if (!m.is_object() || json_string_or(m, "role") != "tool") {
            continue;
        }
        if (json_string_or(m, "tool_call_id") != tool_call_id) {
            continue;
        }
        std::string head = json_string_or(m, "content").substr(0, 200);
        for (char & c : head) {
            c = (char) tolower((unsigned char) c);
        }
        if (head.find("error") != std::string::npos
                || head.find("exception") != std::string::npos
                || head.find("traceback") != std::string::npos
                || head.find("command not found") != std::string::npos) {
            return "error";
        }
        return "ok";
    }
    return "ok";
}

// The smallest user-message boundary b (1 <= b <= min(K_cap, n-1)) such that
// removing the whole messages [0,b) (system kept) and stubbing the surviving
// [b,K) drops the non-block prompt to the estimate-space floor
// max(target x window, min_keep) - block_budget - off. The first surviving
// message being a user keeps role alternation intact and anchors the block
// (plan §5.4). Returns 0 when no boundary reaches the floor (native-compact
// territory) — except when the floor itself is at or below the offset
// (floor2 == 0, target x window - budget < off): the estimate can never hit
// the target, so it returns the largest user boundary within K_cap as a
// best-effort shrink instead. O(n) via prefix/suffix token sums.
size_t compute_k_sum(const common_json & messages,
        const std::vector<int64_t> & orig,
        const std::vector<std::string> & stubs,
        const std::vector<int64_t> & savings,
        size_t K, size_t K_cap, size_t n,
        int64_t tokens_before, double floor_tokens,
        int64_t block_budget, int64_t off,
        double window, size_t K_sum_committed) {
    (void) tokens_before;
    const int64_t floor2 = std::max<int64_t>(0, (int64_t) floor_tokens - block_budget - off);
    // final non-block tokens at boundary b = (system tokens in [0,b))
    //   + sum of contrib[i] over [b,n)
    // contrib[i] = orig[i] - savings[i] for an active stub (i<K, stubbed),
    // else orig[i].
    std::vector<int64_t> prefix_system(n + 1, 0);
    std::vector<int64_t> suffix(n + 2, 0);
    for (size_t i = 0; i < n; ++i) {
        const bool sys = messages.at(i).is_object()
            && json_string_or(messages.at(i), "role") == "system";
        prefix_system[i + 1] = prefix_system[i] + (sys ? orig[i] : 0);
    }
    // suffix must be built backwards: suffix[i] = sum of contrib[i..n). A
    // forward loop reads suffix[i+1] before it is set (still 0), collapsing
    // the sum to a single contrib[i] and undercounting the post-removal total.
    for (size_t i = n; i > 0; --i) {
        const size_t j = i - 1;
        const int64_t contrib = (j < K && !stubs[j].empty())
            ? (orig[j] - savings[j]) : orig[j];
        suffix[j] = suffix[j + 1] + contrib;
    }
    const size_t limit = std::min(K_cap, n - 1); // a survivor must exist
    size_t last_user = 0; // largest user boundary within K_cap (best-effort fallback)
    size_t result = 0;
    for (size_t b = 1; b <= limit; ++b) {
        if (!messages.at(b).is_object() || json_string_or(messages.at(b), "role") != "user") {
            continue; // not a user boundary
        }
        last_user = b;
        if (prefix_system[b] + suffix[b] <= floor2) {
            result = b; // smallest user boundary that reaches the floor
            break;
        }
    }
    // Best-effort: no boundary reached the floor. This covers both the
    // floor2 == 0 case (the reported-space offset dominates — the estimate
    // can never hit the target, the target-0.17 / off-45k launch-line case)
    // and the case where floor2 > 0 but every user boundary still leaves the
    // prompt above it (coarse boundaries, e.g. floor2 7.5k but system + the
    // protected round alone weigh 15k). Remove up to the largest user
    // boundary within K_cap to shrink the prompt as much as safely possible
    // instead of never firing.
    //
    // Gate: the removal must save at least 10% of the window, measured over
    // [K_sum_committed, last_user) — the MARGINAL reduction this advance
    // adds, not the whole removed region. A trivially small save is not worth
    // the re-prefill; and because best-effort cannot reach the target (usage
    // lingers above the threshold), the marginal gate is what stops the chain
    // of re-advances that would otherwise fire on every new user boundary.
    if (result == 0 && last_user > 0 && window > 0) {
        int64_t reduction = 0;
        for (size_t i = K_sum_committed; i < last_user && i < n; ++i) {
            if (messages.at(i).is_object()
                    && json_string_or(messages.at(i), "role") == "system") {
                continue; // system is kept, not removed
            }
            reduction += (i < K && !stubs[i].empty())
                ? (orig[i] - savings[i]) : orig[i];
        }
        if (reduction >= (int64_t) (0.10 * window)) {
            result = last_user;
        }
    }
    return result;
}

// Deterministic rule Σ block: a compact history of the removed region
// [0,K_sum) — files read/edited, shell commands (with ok/error), earlier user
// requests. Past-tense, evidence-based, no instructions (plan §4.5). Capped to
// block_budget tokens (sections dropped from the least important first).
std::string make_rule_block(const common_json & messages, size_t K_sum,
        const struct llama_vocab * vocab, int64_t block_budget) {
    // path -> (count, last index) so the lists can be ordered most-recent
    // first (a std::map key order would be alphabetical).
    std::map<std::string, std::pair<int, size_t>> files_read, files_edited;
    std::vector<std::string> commands, requests;
    for (size_t i = 0; i < K_sum && i < messages.size(); ++i) {
        const common_json & msg = messages.at(i);
        if (!msg.is_object()) {
            continue;
        }
        const std::string role = json_string_or(msg, "role");
        if (role == "system") {
            continue;
        }
        if (role == "user") {
            const std::string t = strip_injected_blocks(message_text(msg));
            if (!t.empty()) {
                requests.push_back(clamp_chars(t, 100));
            }
        } else if (role == "assistant") {
            if (!msg.contains("tool_calls") || !msg.at("tool_calls").is_array()) {
                continue;
            }
            for (auto it = msg.at("tool_calls").begin(); it != msg.at("tool_calls").end(); ++it) {
                const common_json & tc = *it;
                if (!tc.is_object()) {
                    continue;
                }
                std::string name, args, id;
                id = json_string_or(tc, "id");
                if (tc.contains("function") && tc.at("function").is_object()) {
                    name = json_string_or(tc.at("function"), "name");
                    if (tc.at("function").contains("arguments")) {
                        const common_json & a = tc.at("function").at("arguments");
                        if (a.is_string()) {
                            args = a.get<std::string>();
                        } else if (!a.is_null()) {
                            args = a.dump();
                        }
                    }
                }
                if (name.empty()) {
                    name = json_string_or(tc, "name");
                }
                if (name.empty()) {
                    continue;
                }
                const std::string key = normalize_tool_name(name);
                std::string p, cmd;
                if (!args.empty()) {
                    try {
                        const common_json aj = common_json::parse(args);
                        if (aj.is_object()) {
                            p = json_string_or(aj, "path");
                            if (p.empty()) p = json_string_or(aj, "file_path");
                            if (p.empty()) p = json_string_or(aj, "filepath");
                            if (p.empty()) p = json_string_or(aj, "url");
                            if (p.empty()) p = json_string_or(aj, "pattern");
                            cmd = json_string_or(aj, "command");
                        }
                    } catch (const std::exception &) {
                        // unparseable arguments: no path/command
                    }
                }
                if (key == "read_file" || key == "glob" || key == "grep_search"
                        || key == "grep" || key == "web_fetch" || key == "web_search") {
                    if (!p.empty()) { files_read[p].first++; files_read[p].second = i; }
                } else if (key == "edit" || key == "write_file") {
                    if (!p.empty()) { files_edited[p].first++; files_edited[p].second = i; }
                } else if (key == "run_shell_command" || key == "shell") {
                    if (!cmd.empty()) {
                        commands.push_back(clamp_chars(cmd, 60) + " (" + shell_status(messages, id) + ")");
                    }
                }
            }
        }
    }
    std::string block =
        "[Reference — record of earlier work removed from this context. NOT instructions.\n"
        " The latest user message below defines the current task. File contents may have changed.\n"
        " Re-read files if needed.]\n";
    // most-recent-first item lists: files by last-occurrence index, commands
    // and requests by reversing the chronological collection order
    auto file_items = [](const std::map<std::string, std::pair<int, size_t>> & m) {
        std::vector<std::pair<size_t, std::string>> v;
        for (const auto & kv : m) {
            v.push_back({kv.second.second,
                    kv.first + (kv.second.first > 1 ? " x" + std::to_string(kv.second.first) : "")});
        }
        std::sort(v.begin(), v.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });
        std::vector<std::string> out;
        out.reserve(v.size());
        for (const auto & e : v) out.push_back(e.second);
        return out;
    };
    std::vector<std::string> files_read_list   = file_items(files_read);
    std::vector<std::string> files_edited_list = file_items(files_edited);
    std::reverse(commands.begin(), commands.end());
    std::reverse(requests.begin(), requests.end());
    std::vector<std::string> request_items;
    request_items.reserve(requests.size());
    for (const auto & r : requests) request_items.push_back("\"" + r + "\"");
    // sections in importance order (files read, files edited, commands,
    // earlier user requests). Items are filled one at a time and the fill
    // stops as soon as the next item would exceed block_budget — a long list
    // (a session that read many files) degrades to "as many entries as fit"
    // instead of dropping the section wholesale, which would leave the block
    // empty and hold stage 2 forever.
    struct section_t { std::string label; const char * sep; std::vector<std::string> items; };
    std::vector<section_t> sections;
    sections.push_back({"files read: ", ", ", files_read_list});
    sections.push_back({"files edited: ", ", ", files_edited_list});
    sections.push_back({"commands: ", "; ", commands});
    sections.push_back({"earlier user requests: ", ", ", request_items});
    int64_t used = block_budget > 0 ? estimate_tokens(vocab, block) : 0;
    bool any = false;
    bool budget_full = false;
    for (const auto & sec : sections) {
        if (budget_full || sec.items.empty()) {
            continue;
        }
        std::string line;
        bool first = true;
        for (const auto & item : sec.items) {
            const std::string piece = first ? sec.label + item : std::string(sec.sep) + item;
            const std::string new_line = line.empty() ? piece : line + piece;
            if (block_budget > 0) {
                const int64_t cost = estimate_tokens(vocab, new_line + "\n");
                if (used + cost > block_budget) {
                    budget_full = true;
                    break;
                }
            }
            line = new_line;
            first = false;
        }
        if (!line.empty()) {
            block += line + "\n";
            used += estimate_tokens(vocab, line + "\n");
            any = true;
        }
    }
    // A header-only block (nothing to summarize, or no item fit the budget)
    // carries no information — return "" so the caller does not commit an
    // empty shell (block_ok = !block.empty() stays false and stage 2 is held
    // for this request).
    return any ? block : "";
}

// Stage-2 llm mode: after this many consecutive /sigma/refresh failures the
// server stops holding stage 2 and advances with the deterministic rule block
// (plan §5.3 rule 2, N = 2~3 -> 3).
constexpr int SIGMA_SUM_RULE_FALLBACK_FAILS = 3;

// llm mode: fetch the rendered Σ block from focus-memory (plan §5.2 B1-B4).
// POST {host}/sigma/refresh with {"session", "cut", "wait"}; the server
// extracts the removed region's history into Σ (single-flight per session;
// wait:true blocks until this cut is absorbed), then answers
// {"block", "partial"}. Fail-open: false on any error (connection, timeout,
// non-2xx, malformed body, empty block) — the caller decides between holding
// stage 2 and the rule-block fallback. Runs OUTSIDE the sigma_strip mutex
// (the A/B/C split): a slow fetch must not stall every session's decision.
bool sigma_refresh_fetch(const std::string & host, const std::string & token,
        const std::string & session, size_t cut, bool wait, int timeout_ms,
        std::string & out_block, bool & out_partial, std::string & err) {
    if (host.empty() || session.empty()) {
        err = "no host/session";
        return false;
    }
    try {
        auto [cli, parts] = common_http_client(host);
        cli.set_connection_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);
        cli.set_read_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);
        cli.set_write_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);
        if (!token.empty()) {
            cli.set_default_headers({ { "Authorization", "Bearer " + token } });
        }
        std::string path = "/sigma/refresh";
        if (!parts.path.empty() && parts.path != "/") {
            path = parts.path + path;
        }
        common_json body;
        body["session"] = session;
        body["cut"] = (uint64_t) cut;
        body["wait"] = wait;
        auto res = cli.Post(path, body.dump(), "application/json");
        if (!res) {
            err = "no response (connection failed or timeout)";
            return false;
        }
        if (res->status != 200 && res->status != 202) {
            err = "status " + std::to_string(res->status);
            return false;
        }
        common_json j = common_json::parse(res->body);
        if (!j.is_object() || !j.contains("block") || !j["block"].is_string()) {
            err = "malformed body";
            return false;
        }
        out_block = j["block"].get<std::string>();
        out_partial = j.contains("partial") && j["partial"].is_boolean() && j["partial"].get<bool>();
        if (out_block.empty()) {
            err = "empty block";
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

// Option A (incidents/2026-11-06-sigma-partial-block-not-completed.md) drain
// caps: the background thread keeps firing wait:false /sigma/refresh passes
// (one 30k-char chunk each) until the removed region is fully absorbed
// (partial:false). A 2.6 MB region is ~87 passes; at 15-40 s per 27B
// extraction plus the inter-pass delay that is ~30-65 min, so a 1 h wall /
// 200-pass cap bounds the drain while still covering most regions. A cap hit
// leaves the partial block in place; the next K_sum advance re-fetches a
// fuller block.
constexpr int     SIGMA_SUM_DRAIN_MAX_PASSES  = 200;
constexpr int64_t SIGMA_SUM_DRAIN_MAX_WALL_MS = 3600 * 1000;  // 1 h

// Option A: after a PARTIAL Σ block is committed (llm mode), the server's
// hist_last_bytes is short of the removed region's end. This background
// thread keeps firing wait:false /sigma/refresh passes (one 30k-char chunk
// each; the server is single-flight per session, so this is safe alongside
// live chat's own stage-2 fetch and the 5a async fire) until the server
// answers partial:false (region fully absorbed). On completion the committed
// block is refreshed to the complete Σ (lock + K_sum revalidation).
// Superseded (a newer stage-2 commit moved K_sum, or a prefix reset) -> stop;
// the newer fetch owns the region. A pass failure stops the drain (the next
// stage-2 fetch retries). Runs OUTSIDE the sigma_strip mutex except for the
// short revalidation/commit sections. The supersession check is
// st.K_sum != cut (NOT sum_gen) — a 5a async fire bumps sum_gen but leaves
// K_sum unchanged and must not kill the drain (it absorbs the same region;
// the drain continues past it).
void sigma_strip_drain(std::string session, size_t cut,
        std::string host, std::string token, int timeout_ms, int64_t delay_ms) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int pass = 1; pass <= SIGMA_SUM_DRAIN_MAX_PASSES; ++pass) {
        if (pass > 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() > SIGMA_SUM_DRAIN_MAX_WALL_MS) {
            SRV_WRN("sigma-strip: session=%s Σ drain wall cap (%lld ms) at pass %d: block left partial\n",
                    session.c_str(), (long long) SIGMA_SUM_DRAIN_MAX_WALL_MS, pass);
            return;
        }
        // superseded? (a newer stage-2 commit moved K_sum, or a prefix reset)
        {
            std::lock_guard<std::mutex> lock(sigma_strip_mutex());
            auto it = sigma_strip_sessions().find(session);
            if (it == sigma_strip_sessions().end() || it->second.K_sum != cut) {
                SRV_INF("sigma-strip: session=%s Σ drain superseded before pass %d: stopping\n",
                        session.c_str(), pass);
                return;
            }
        }
        std::string block;
        bool partial = true;
        std::string err;
        const bool fetched = sigma_refresh_fetch(host, token, session, cut,
                /*wait*/ false, timeout_ms, block, partial, err);
        if (!fetched) {
            SRV_WRN("sigma-strip: session=%s Σ drain pass %d failed (%s): stopping (next stage-2 fetch retries)\n",
                    session.c_str(), pass, err.c_str());
            return;
        }
        if (partial) {
            continue;  // more of the region remains — next pass
        }
        // region fully absorbed — refresh the committed block to the complete Σ
        {
            std::lock_guard<std::mutex> lock(sigma_strip_mutex());
            auto it = sigma_strip_sessions().find(session);
            if (it != sigma_strip_sessions().end() && it->second.K_sum == cut) {
                it->second.sum_block = block;
                it->second.sum_hash = kv_offload_hash16(block);
                SRV_INF("sigma-strip: session=%s Σ drain complete: block refreshed to %zu chars after %d pass(es)\n",
                        session.c_str(), block.size(), pass);
            }
        }
        return;
    }
    SRV_WRN("sigma-strip: session=%s Σ drain pass cap (%d): block left partial\n",
            session.c_str(), SIGMA_SUM_DRAIN_MAX_PASSES);
}

} // namespace

//
// startup configuration validation
//
std::string sigma_strip_config_check(const common_params & params) {
    // Stage 2 "llm": the Σ block is fetched from focus-memory
    // /sigma/refresh (same host/token as the kv-offload store). Without a
    // host there is nothing to fetch from, so the deterministic server
    // "rule" block is used instead — warn once at startup.
    if (params.sigma_sum_mode == "llm") {
        if (params.focus_memory_host.empty()) {
            SRV_WRN("%s", "sigma-strip: --sigma-sum llm without --focus-memory-host: using the deterministic server 'rule' block instead\n");
        } else {
            SRV_INF("sigma-strip: --sigma-sum llm: fetching Σ blocks from %s/sigma/refresh (timeout %d ms, rule fallback after %d consecutive failures)\n",
                    params.focus_memory_host.c_str(), params.sigma_sum_timeout_ms, SIGMA_SUM_RULE_FALLBACK_FAILS);
        }
    }
    if (params.sigma_strip_mode == "off" || params.sigma_strip_window_tokens <= 0) {
        return "";
    }
    const double window = (double) params.sigma_strip_window_tokens;
    const double floor = std::max(params.sigma_strip_target * window,
                                  (double) params.sigma_strip_min_keep_tokens);
    const double margin = params.sigma_strip_threshold * window - floor;
    if (margin <= 0.0) {
        return "sigma-strip: floor " + std::to_string((long long) floor) +
               " tokens (max of --sigma-strip-target x window and --sigma-strip-min-keep-tokens) >= "
               "threshold " + std::to_string((long long) (params.sigma_strip_threshold * window)) +
               " tokens: the K advance cannot keep the usage below the threshold (stall or chain "
               "re-prefill until native compact). Lower --sigma-strip-target or "
               "--sigma-strip-min-keep-tokens, or raise --sigma-strip-threshold.";
    }
    if (margin < 0.10 * window) {
        SRV_WRN("sigma-strip: thin margin (threshold - floor = %.0f tokens < 10%% of window): "
                "post-advance usage lingers near the threshold -> repeated advances (chain re-prefill) "
                "until native compact; lower the floors or raise the threshold", margin);
    }
    return "";
}

//
// entry point
//
sigma_strip_result sigma_strip_maybe_rewrite(
        const struct llama_vocab * vocab,
        const common_params & params,
        const server_http_req & req,
        common_json & body,
        bool apply) {
    sigma_strip_result res;

    // gates: mode + window + parseable messages
    if (params.sigma_strip_mode == "off" || params.sigma_strip_window_tokens <= 0) {
        return res;
    }
    if (!body.is_object() || !body.contains("messages") || !body.at("messages").is_array()) {
        return res;
    }

    const std::string session = kv_offload_resolve_session(req, body);
    if (session.empty() || session == "kv-offload-default") {
        return res; // no real session key: excluded from rewriting
    }
    res.session = session;

    try {
        const common_json & messages = body.at("messages");
        const size_t n = messages.size();
        res.n_messages = n;
        if (n == 0) {
            return res;
        }

        // 1) token estimate of the original messages + per-message savings.
        // orig[i] (per-message original tokens) is kept for the stage-2
        // K_sum computation (plan §5.4).
        int64_t tokens_before = 0;
        std::vector<int64_t> orig(n, 0);
        std::vector<int64_t> savings(n, 0);
        std::vector<std::string> stubs(n);
        const tool_call_map_t tool_calls = build_tool_call_map(messages);
        for (size_t i =0; i < n; ++i) {
            const common_json & msg = messages.at(i);
            if (!msg.is_object()) {
                continue;
            }
            orig[i] = message_tokens(vocab, msg);
            tokens_before += orig[i];
            if (json_string_or(msg, "role") == "tool") {
                std::string tool;
                std::string args;
                if (resolve_tool_call(tool_calls, msg, tool, args) && is_compactable_tool(tool)) {
                    stubs[i] = make_stub(tool, args, orig[i]);
                    savings[i] = std::max<int64_t>(0, orig[i] - estimate_tokens(vocab, stubs[i]));
                }
            }
        }
        res.tokens_before = tokens_before;

        const double window = (double) params.sigma_strip_window_tokens;
        const double target_tokens = params.sigma_strip_target * window;
        const double floor_tokens = std::max(target_tokens, (double) params.sigma_strip_min_keep_tokens);
        // stage 2 (Σ block): on unless --sigma-sum off. The block budget is
        // --sigma-sum-max-tokens (default 1500 tokens).
        const bool sum_enabled = (params.sigma_sum_mode != "off");
        const int64_t sum_budget = (params.sigma_sum_max_tokens > 0)
            ? (int64_t) params.sigma_sum_max_tokens : 1500;

        // stage-2 decision locals: set in [A], consumed in [B]/[C]
        bool do_fetch = false;
        bool rule_fallback = false;  // threshold reached in [A]: rule block, no llm fetch
        size_t K_sum_cand = 0;
        uint64_t gen_at = 0;
        std::string fp_at;
        int fail_count_at = 0;   // consecutive failures when the fetch was set (llm fallback)
        size_t K_cap = n;
        size_t K_stage1 = 0;
        int64_t off = 0;
        // Releases sum_inflight on any scope exit (including the outer
        // catch's fail-open return) when this request still owns it.
        inflight_guard guard{session};

        // ===== [A] compute — under the lock, pure decision, no commit =====
        {
            std::lock_guard<std::mutex> lock(sigma_strip_mutex());
            sigma_strip_session_state & st = sigma_strip_sessions()[session];
            st.window = params.sigma_strip_window_tokens;

            // 2) prefix fingerprint check (reset on client compact / /clear / resume)
            if (st.K > 0) {
                if (st.K > n) {
                    st.K = 0;
                    st.prefix_fp.clear();
                    res.prefix_reset = true;
                } else {
                    const std::string fp = prefix_fingerprint(messages, st.K);
                    if (fp != st.prefix_fp) {
                        st.K = 0;
                        st.prefix_fp.clear();
                        res.prefix_reset = true;
                    }
                }
            }
            if (res.prefix_reset) {
                // E1: the client compacted/cleared below the server threshold —
                // the template/tool state may have changed with it, so the est
                // calibration is void until a fresh request re-anchors it.
                st.est_offset = 0;
                st.n_resets++;
                // stage 2: the prefix the Σ block was built from is gone — drop
                // the committed boundary/snapshot too (plan §5.4).
                st.K_sum = 0;
                st.sum_block.clear();
                st.sum_hash.clear();
                st.sum_gen++;
                st.sum_inflight = false;
                st.sum_fetch_fail_count = 0;
                // E3: repeated resets mean the client's own compaction is firing
                // below the server threshold — its window is likely smaller than
                // the configured server window. Warn once per session.
                if (st.n_resets >= 3 && !st.resets_warned) {
                    st.resets_warned = true;
                    SRV_WRN("sigma-strip: session=%s %d prefix resets (client compaction below the server threshold): the client context window is likely smaller than --sigma-strip-window-tokens %d — set them equal",
                            session.c_str(), st.n_resets, params.sigma_strip_window_tokens);
                }
            }

            // 3) K advance: one big jump when the reported-space usage crosses
            // the threshold. The estimate is the previous round's post-rewrite
            // value plus the est offset — the template/tool-schema/system
            // overhead the client's full prompt carries but this message-level
            // estimate omits. Starting from the raw tokens here would re-advance
            // K on every request. The resident already accounts for the stage-2
            // removed region [0,K_sum) (orig tokens gone, system kept) and the
            // frozen Σ block.
            off = st.est_offset;
            res.est_offset = off;
            double resident = (double) tokens_before;
            for (size_t i = 0; i < st.K; ++i) {
                if (i < st.K_sum) {
                    if (!(messages.at(i).is_object() && json_string_or(messages.at(i), "role") == "system")) {
                        resident -= (double) orig[i]; // removed
                    }
                } else {
                    resident -= (double) savings[i];  // stubbed
                }
            }
            if (st.K_sum > 0 && !st.sum_block.empty()) {
                resident += (double) estimate_tokens(vocab, st.sum_block);
            }
            res.usage_before = (resident + (double) off) / window;
            // Protection boundary (start of the last keep_recent tool rounds).
            // Refined inside the advance block; defaults to n (nothing protected)
            // when the gate is not crossed, where it is never used.
            if (res.usage_before >= params.sigma_strip_threshold) {
                // Protection floor: K never crosses the start of the last
                // keep_recent tool rounds (tool_calls + their results) — recent
                // working context ("what was I doing"). Fewer rounds than
                // requested: protect them all.
                const size_t keep_recent = (size_t) std::max<int32_t>(1, params.sigma_strip_keep_recent);
                {
                    size_t rounds = 0, first_round = n;
                    for (size_t i = n; i-- > 0;) {
                        const common_json & m = messages.at(i);
                        if (m.is_object() && json_string_or(m, "role") == "assistant"
                                && m.contains("tool_calls") && m.at("tool_calls").is_array()
                                && !m.at("tool_calls").empty()) {
                            ++rounds;
                            first_round = i;
                            if (rounds == keep_recent) {
                                K_cap = i;
                                break;
                            }
                        }
                    }
                    if (rounds < keep_recent) {
                        K_cap = first_round;
                    }
                }
                if (st.K > K_cap) {
                    st.K = K_cap;
                    st.prefix_fp = prefix_fingerprint(messages, st.K);
                }
                // Token floor: the stripped prompt never drops below
                // max(target x window, min_keep_tokens). In reported space the
                // off tokens are unstrippable overhead (chat template, tool
                // schemas, system), so the effective floor for the strippable
                // content is lowered by off — with off = 0 this is the original
                // floor.
                const double floor_adj = std::max(0.0, floor_tokens - (double) off);
                double stripped = resident;
                size_t K2 = st.K;
                while (K2 < K_cap && stripped > floor_adj) {
                    stripped -= (double) savings[K2];
                    K2++;
                }
                // K safety boundary: K must never split a parallel tool group
                // (assistant tool_calls | its tool results) — messages[K] has to
                // be a user/assistant message so a future insertion at K cannot
                // break tool_call_id pairing. When K2 lands inside a group, roll
                // it back to the group start (the assistant). The assistant
                // carries no savings, so the rollback cannot violate the floor
                // and the rewrite bytes are unchanged.
                if (K2 > 0 && K2 < n && json_string_or(messages.at(K2), "role") == "tool") {
                    size_t i = K2;
                    while (i > 0 && json_string_or(messages.at(i - 1), "role") == "tool") {
                        i--;
                    }
                    if (i > 0 && json_string_or(messages.at(i - 1), "role") == "assistant") {
                        // i is the first tool of the run; the group-start
                        // assistant is at i - 1.
                        K2 = i - 1;
                        // the rolled-back tool results stay resident: recompute
                        // stripped for the observability fields below
                        stripped = resident;
                        for (size_t j = st.K; j < K2; ++j) {
                            stripped -= (double) savings[j];
                        }
                    }
                }
                if (K2 > st.K) {
                    st.K = K2;
                    st.prefix_fp = prefix_fingerprint(messages, st.K);
                    res.k_advanced = true;
                }
                // floor_limited: the token floor (above the target) stopped the
                // advance before the cap. With min_keep = 0 the floor equals the
                // target, so a plain target stop is not floor-limited.
                res.floor_limited = (floor_tokens > target_tokens && K2 < K_cap && stripped <= floor_adj);
                // target_unreachable in reported space: even after stripping
                // everything strippable, the unstrippable overhead keeps the
                // client's reported usage above the target.
                if (stripped + (double) off > target_tokens) {
                    res.target_unreachable = true;
                }
            }
            K_stage1 = st.K;
            res.K = st.K;

            // 3b) stage-2 decision (plan §5.2 A): fire only when stage 1 could
            // not reach the target and the Σ feature is on. K_sum is the
            // smallest user-message boundary at or below K_cap that drops the
            // non-block prompt to floor - Σ budget - off. A new advance
            // (K_sum_cand > committed) sets the inflight flag + generation so
            // [B] can generate and [C] can revalidate; an already-committed
            // boundary reuses the frozen snapshot (no fetch), and a request
            // that finds another mid-fetch passes through on committed state.
            if (sum_enabled && res.target_unreachable) {
                K_sum_cand = compute_k_sum(messages, orig, stubs, savings,
                        st.K, K_cap, n, tokens_before, floor_tokens, sum_budget, off,
                        window, st.K_sum);
                if (K_sum_cand > st.K_sum && !st.sum_inflight) {
                    bool fire = true;
                    // Cooldown: after a failed llm fetch, skip the next
                    // attempt(s) for sigma_sum_fetch_cooldown_ms so a down
                    // focus-memory does not cost up to sigma_sum_timeout_ms
                    // on every request. Only in the 1..(N-1) failure range —
                    // at N consecutive failures the rule fallback (no HTTP)
                    // takes over.
                    if (params.sigma_sum_mode == "llm"
                            && !params.focus_memory_host.empty()
                            && st.sum_fetch_fail_count > 0
                            && st.sum_fetch_fail_count < SIGMA_SUM_RULE_FALLBACK_FAILS
                            && params.sigma_sum_fetch_cooldown_ms > 0) {
                        const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
                        if (now_ms - st.sum_last_fail_ms < params.sigma_sum_fetch_cooldown_ms) {
                            fire = false;
                            SRV_INF("sigma-strip: session=%s /sigma/refresh in cooldown (%lld ms since last fail < %d ms): stage 1 only this request\n",
                                    session.c_str(), (long long) (now_ms - st.sum_last_fail_ms),
                                    params.sigma_sum_fetch_cooldown_ms);
                        }
                    }
                    if (fire) {
                        do_fetch = true;
                        st.sum_inflight = true;
                        st.sum_gen++;
                        gen_at = st.sum_gen;
                        guard.gen = gen_at;
                        fp_at = prefix_fingerprint(messages, st.K);
                        fail_count_at = st.sum_fetch_fail_count;
                        rule_fallback = (fail_count_at >= SIGMA_SUM_RULE_FALLBACK_FAILS);
                    }
                }
            }
            // (lock released here)
        }

        // ===== [B] fetch — no lock =====
        // rule mode: the Σ block is a deterministic local function of the
        // removed messages (no HTTP). llm mode: POST /sigma/refresh to
        // focus-memory (wait:true — its Σ extraction must have absorbed this
        // cut before the block goes on the wire). The A/B/C split keeps the
        // fetch out of the lock so a slow one cannot stall every session's
        // strip decision. Failure rule (plan §5.3 rule 2): hold stage 2
        // (stage-1 only) until SIGMA_SUM_RULE_FALLBACK_FAILS consecutive
        // failures, then advance with the deterministic rule block — from
        // that point on the llm fetch is skipped entirely (an attempt would
        // only cost up to sigma_sum_timeout_ms per request for the same
        // rule outcome).
        std::string block;
        bool block_ok = false;
        bool block_partial = false;
        int64_t fetch_ms = 0;
        if (do_fetch) {
            if (g_test_fetch_fault.load(std::memory_order_relaxed)) {
                throw std::runtime_error("injected fetch fault (unit test)");
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (rule_fallback) {
                // The consecutive-failure threshold was already reached when
                // [A] decided: skip the llm fetch and commit the
                // deterministic rule block (plan §5.3 rule 2).
                SRV_WRN("sigma-strip: session=%s /sigma/refresh failed %d times in a row: rule block, llm fetch skipped\n",
                        session.c_str(), fail_count_at);
                block = make_rule_block(messages, K_sum_cand, vocab, sum_budget);
                block_ok = !block.empty();
            } else if (params.sigma_sum_mode == "llm" && !params.focus_memory_host.empty()) {
                std::string err;
                const bool fetched = sigma_refresh_fetch(params.focus_memory_host, params.focus_memory_token,
                            session, K_sum_cand, /*wait*/ true, params.sigma_sum_timeout_ms,
                            block, block_partial, err);
                if (fetched) {
                    block_ok = true;
                    if (block_partial) {
                        // A partial block (JS-side claim skip, cut-mapping
                        // failure, or the chunk cap) is a valid cumulative Σ of
                        // what the server has absorbed so far — apply it. The
                        // server's hist_last_bytes advances only to the consumed
                        // bytes, so the next refresh (stage-1 async absorption or
                        // a later stage-2 fetch) absorbs the remainder from the
                        // transcript and the next committed block completes the
                        // Σ. Applying a partial block beats holding stage 2 and
                        // removing the region with no summary at all.
                        SRV_INF("sigma-strip: session=%s /sigma/refresh returned a partial block: applying partial Σ, the server absorbs the remainder on subsequent refreshes\n",
                                session.c_str());
                    }
                } else {
                    // fail_count_at < SIGMA_SUM_RULE_FALLBACK_FAILS here: a
                    // request at or above the threshold takes the
                    // rule_fallback path above instead of paying the fetch.
                    SRV_WRN("sigma-strip: session=%s /sigma/refresh failed (%s), failure %d of %d: stage 1 only this request\n",
                            session.c_str(), err.c_str(), fail_count_at + 1, SIGMA_SUM_RULE_FALLBACK_FAILS);
                }
            } else {
                block = make_rule_block(messages, K_sum_cand, vocab, sum_budget);
                block_ok = !block.empty();
            }
            fetch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0).count();
        }

        // Whether this request ran the stage-2 wait:true fetch — the ATTEMPT,
        // not the commit: [C] resets do_fetch when the fetch is discarded.
        // 5a) skips the async refresh whenever the attempt happened: on
        // success its extraction already covers [hist_last, K_sum); on
        // failure the 3-strike rule owns the session's stage-2 flow and a
        // concurrent refresh would only steal the single-flight flag and
        // stall the failure count.
        const bool did_stage2 = do_fetch;

        // ===== [C] commit + apply — under the lock =====
        {
            std::lock_guard<std::mutex> lock(sigma_strip_mutex());
            sigma_strip_session_state & st = sigma_strip_sessions()[session];
            // C1: revalidate — if another request changed the state while we
            // were generating, discard our result and pass through on the
            // committed state (plan §5.3 rule 4).
            if (do_fetch) {
                const bool commit_ok = (st.sum_inflight
                        && st.sum_gen == gen_at
                        && st.prefix_fp == fp_at
                        && block_ok);
                if (st.sum_gen == gen_at) {
                    // still own the inflight slot: settle the failure count
                    // (the guard releases the flag on scope exit —
                    // an unconditional clear here could drop a newer
                    // request's flag after a prefix reset).
                    st.sum_fetch_fail_count = block_ok ? 0 : st.sum_fetch_fail_count + 1;
                    if (!block_ok) {
                        st.sum_last_fail_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
                    }
                }
                if (!commit_ok) {
                    do_fetch = false; // discard; use committed state below
                }
                // C2: commit K_sum + K=max(K,K_sum) + snapshot atomically.
                if (do_fetch) {
                    st.K_sum = K_sum_cand;
                    if (st.K < st.K_sum) {
                        st.K = st.K_sum;
                        st.prefix_fp = prefix_fingerprint(messages, st.K);
                    }
                    st.sum_block = block;
                    st.sum_hash = kv_offload_hash16(block);
                }
            }
            res.sum_applied = (st.K_sum > 0 && !st.sum_block.empty());
            res.K_sum = st.K_sum;
            res.sum_hash = res.sum_applied ? st.sum_hash : "";
            res.sum_fetch_ms = do_fetch ? fetch_ms : 0;
            res.sum_partial = (do_fetch && block_ok) ? block_partial : false;
            res.sum_fail = st.sum_fetch_fail_count;
            res.K = st.K;

            // 4) build the rewrite: remove [0,K_sum) (system kept) + the frozen
            // Σ block prepended to the first surviving user message + stub the
            // surviving [K_sum,K). The message count is NOT preserved by stage
            // 2 (plan §5.4); tool_call_id pairing and role alternation are.
            int64_t tokens_after = tokens_before;
            size_t n_stripped = 0;
            bool have_rewrite = false;
            common_json rewritten;
            if (st.K > 0 || st.K_sum > 0) {
                rewritten = common_json::array();
                bool block_placed = false;
                for (size_t i = 0; i < n; ++i) {
                    const common_json & m = messages.at(i);
                    if (i < st.K_sum) {
                        // removed region: keep system, drop the rest
                        if (m.is_object() && json_string_or(m, "role") == "system") {
                            rewritten.push_back(m);
                        } else {
                            tokens_after -= orig[i];
                        }
                        continue;
                    }
                    common_json msg = m; // copy
                    if (i < st.K && msg.is_object()) {
                        if (!stubs[i].empty()) {
                            msg["content"] = stubs[i];
                            n_stripped++;
                            tokens_after -= savings[i];
                        } else if (msg.contains("content") && msg.at("content").is_array()) {
                            strip_media(msg.at("content"));
                        }
                    }
                    // the block: prepend to the first surviving user message
                    if (res.sum_applied && !block_placed
                            && msg.is_object() && json_string_or(msg, "role") == "user") {
                        if (msg.contains("content") && msg.at("content").is_array()) {
                            // parts array: prepend a text part so the original
                            // parts survive the rewrite (a string overwrite would
                            // drop them)
                            common_json part = common_json::object();
                            part["type"] = std::string("text");
                            part["text"] = st.sum_block;
                            common_json new_content = common_json::array();
                            new_content.push_back(part);
                            for (const auto & e : msg.at("content")) {
                                new_content.push_back(e);
                            }
                            msg["content"] = std::move(new_content);
                        } else {
                            std::string content;
                            if (msg.contains("content") && msg.at("content").is_string()) {
                                content = msg.at("content").get<std::string>();
                            }
                            msg["content"] = st.sum_block + "\n\n" + content;
                        }
                        block_placed = true;
                        tokens_after += estimate_tokens(vocab, st.sum_block);
                    }
                    rewritten.push_back(msg);
                }
                have_rewrite = true;
                res.rewrite_hash = kv_offload_hash16(rewritten.dump());
            }
            res.tokens_after = tokens_after;
            res.n_stripped = n_stripped;
            res.usage_after = ((double) tokens_after + (double) off) / window;
            res.eligible = true;

            // E1/E2: decision snapshot for the next response's observation line
            st.last_est = tokens_after;
            st.last_K = st.K;
            st.last_stripped = n_stripped;
            st.last_advanced = res.k_advanced;
            st.last_floor_limited = res.floor_limited;

            // 5) apply in "on" mode
            if (params.sigma_strip_mode == "on" && apply && have_rewrite) {
                body["messages"] = rewritten;
            }
            // (lock released here)
        }

        // Σ drain (option A, incidents/2026-11-06-sigma-partial-block-not-completed.md):
        // the committed block is partial — the removed region is larger than one
        // 30k-char pass, so the server absorbed only a prefix. A detached thread
        // keeps firing wait:false refreshes (one 30k-char chunk each, spaced by
        // sigma_sum_drain_delay_ms so live chat can interleave on the shared
        // slot) until the server answers partial:false, then refreshes the
        // committed block to the complete Σ. A newer commit moving K_sum (or a
        // prefix reset) or a pass failure stops it — the next stage-2 fetch
        // retries. Caps: 200 passes / 1 h.
        if (res.sum_partial && params.sigma_sum_drain) {
            std::thread(sigma_strip_drain, session, K_sum_cand,
                    params.focus_memory_host, params.focus_memory_token,
                    params.sigma_sum_timeout_ms,
                    (int64_t) params.sigma_sum_drain_delay_ms).detach();
        }

        // 5a) stage-1 absorption (llm mode only): a stage-1 advance stubs
        // [old_K, K) — fire a best-effort wait:false refresh so focus-memory
        // absorbs that region's history in the background. Shares the
        // per-session inflight flag (single-flight); skipped when this
        // request already ran the stage-2 wait:true fetch — its extraction
        // covers [hist_last, K_sum) and the remainder [K_sum, K) rides the
        // next refresh (hist_last_bytes advances only on absorbed chunks).
        if (params.sigma_sum_mode == "llm" && !params.focus_memory_host.empty()
                && res.k_advanced && !did_stage2) {
            bool fire = false;
            size_t cut = 0;
            uint64_t gen = 0;
            {
                std::lock_guard<std::mutex> lock(sigma_strip_mutex());
                sigma_strip_session_state & st = sigma_strip_sessions()[session];
                if (!st.sum_inflight) {
                    st.sum_inflight = true;
                    st.sum_gen++;
                    gen = st.sum_gen;
                    fire = true;
                    cut = st.K;
                }
            }
            if (fire) {
                const std::string host = params.focus_memory_host;
                const std::string token = params.focus_memory_token;
                const int timeout_ms = params.sigma_sum_timeout_ms;
                std::thread([host, token, session, cut, timeout_ms, gen]() {
                    std::string block;
                    bool partial = false;
                    std::string err;
                    if (!sigma_refresh_fetch(host, token, session, cut, /*wait*/ false,
                                timeout_ms, block, partial, err)) {
                        SRV_WRN("sigma-strip: session=%s async /sigma/refresh (wait:false) failed: %s\n",
                                session.c_str(), err.c_str());
                    }
                    std::lock_guard<std::mutex> lock(sigma_strip_mutex());
                    auto it = sigma_strip_sessions().find(session);
                    // Only release the flag if this generation still owns it —
                    // a prefix reset or a newer fire bumped sum_gen, and an
                    // unconditional clear would drop the newer owner's flag.
                    if (it != sigma_strip_sessions().end()
                            && it->second.sum_inflight
                            && it->second.sum_gen == gen) {
                        it->second.sum_inflight = false;
                    }
                }).detach();
            }
        }

        // 5b) step-1 diagnostic (no lock): at an advance, classify the
        // post-stage-1 message composition (deletable vs. irreducible tokens)
        // so the stage-2 decision is grounded in the real distribution.
        if (res.k_advanced) {
            const sigma_classify c = classify_post_stage1(messages, orig, stubs,
                    savings, K_stage1, K_cap, n, off, target_tokens);
            SRV_INF("sigma-strip-classify: session=%s K=%zu n=%zu deletable=%lld (stubs=%zu/%lld asst=%zu/%lld toolres=%zu/%lld user=%zu/%lld) system=%lld protected=%zu/%lld fixed=%lld floor=%lld target=%lld stage2_reach=%d",
                    session.c_str(), K_stage1, n,
                    (long long) c.tok_deletable,
                    c.n_stubs, (long long) c.tok_stubs,
                    c.n_assistant, (long long) c.tok_assistant,
                    c.n_tool_res, (long long) c.tok_tool_res,
                    c.n_user, (long long) c.tok_user,
                    (long long) c.tok_system,
                    c.n_protected, (long long) c.tok_protected,
                    (long long) c.fixed_off, (long long) c.tok_floor,
                    (long long) target_tokens, (int) c.stage2_can_reach);
        }

        // 6) shadow / on log line (with the stage-2 Σ fields)
        SRV_INF("sigma-strip: session=%s mode=%s K=%zu/%zu K_sum=%zu reset=%d advanced=%d stripped=%zu tokens %lld->%lld usage %.3f->%.3f off=%lld rewrite=%s sum=%d block=%s fetch_ms=%lld partial=%d fail=%d%s%s",
                session.c_str(), params.sigma_strip_mode.c_str(), res.K, n, res.K_sum,
                (int) res.prefix_reset, (int) res.k_advanced, res.n_stripped,
                (long long) res.tokens_before, (long long) res.tokens_after,
                res.usage_before, res.usage_after, (long long) res.est_offset,
                res.rewrite_hash.c_str(), (int) res.sum_applied, res.sum_hash.c_str(),
                (long long) res.sum_fetch_ms, (int) res.sum_partial, res.sum_fail,
                res.target_unreachable ? " target-unreachable" : "",
                res.floor_limited ? " floor-limited" : "");
        return res;
    } catch (const std::exception & e) {
        SRV_WRN("sigma-strip: fail-open (%s), request untouched", e.what());
        return sigma_strip_result();
    }
}

//
// per-response est calibration + observation line
//
void sigma_strip_note_response(
        const std::string & session,
        int64_t reported_prompt_tokens) {
    if (session.empty() || session == "kv-offload-default") {
        return;
    }
    std::lock_guard<std::mutex> lock(sigma_strip_mutex());
    auto it = sigma_strip_sessions().find(session);
    if (it == sigma_strip_sessions().end()) {
        return; // no request was ever seen for this session
    }
    sigma_strip_session_state & st = it->second;
    if (st.last_est == 0 || st.window <= 0) {
        return; // no estimate yet to calibrate against
    }
    const int64_t window = st.window;
    // E1: calibrate the est offset from the response's reported count.
    // Clamp to [0, window/2]: a negative gap means the estimate overshot
    // (nothing to correct — the gate stays conservative); a huge positive
    // gap means the client's window is far larger than configured.
    const int64_t delta = reported_prompt_tokens - st.last_est;
    st.est_offset = std::min<int64_t>(std::max<int64_t>(0, delta), window / 2);
    st.last_reported = reported_prompt_tokens;

    // E3: the response reports more than the configured server window — the
    // client's real window is larger than --sigma-strip-window-tokens.
    // Warn once per session.
    if (reported_prompt_tokens > window && !st.window_warned_big) {
        st.window_warned_big = true;
        SRV_WRN("sigma-strip: session=%s response reports %lld prompt tokens > --sigma-strip-window-tokens %lld: the client's real window is larger than configured — raise the flag to match the client",
                session.c_str(), (long long) reported_prompt_tokens, (long long) window);
    }

    // E2: per-response observation line (every completed chat response of an
    // eligible session, including the K=0 pass-throughs).
    SRV_INF("sigma-strip: session=%s complete reported=%lld est=%lld off=%lld K=%zu stripped=%zu adv=%d floor=%d",
            session.c_str(), (long long) reported_prompt_tokens, (long long) st.last_est,
            (long long) st.est_offset, st.last_K, st.last_stripped,
            (int) st.last_advanced, (int) st.last_floor_limited);
}

// Test-only fault injection (tests/test-sigma-strip.cpp) — see
// g_test_fetch_fault above.
void sigma_strip_test_set_fetch_fault(bool on) {
    g_test_fetch_fault.store(on, std::memory_order_relaxed);
}
