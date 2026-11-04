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
#include "common.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <mutex>
#include <set>
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

// Resolve the tool name + arguments for a role=tool message: explicit "name"
// field first, then tool_call_id matched against earlier assistant tool_calls.
bool resolve_tool_call(const common_json & messages, const common_json & msg,
        std::string & out_name, std::string & out_args) {
    const std::string explicit_name = json_string_or(msg, "name");
    const std::string id = json_string_or(msg, "tool_call_id");
    if (!id.empty()) {
        for (size_t i = 0; i < messages.size(); ++i) {
            const common_json & m = messages.at(i);
            if (!m.is_object() || json_string_or(m, "role") != "assistant") {
                continue;
            }
            if (!m.contains("tool_calls") || !m.at("tool_calls").is_array()) {
                continue;
            }
            for (auto it = m.at("tool_calls").begin(); it != m.at("tool_calls").end(); ++it) {
                const common_json & tc = *it;
                if (!tc.is_object() || json_string_or(tc, "id") != id) {
                    continue;
                }
                std::string name;
                if (tc.contains("function") && tc.at("function").is_object()) {
                    name = json_string_or(tc.at("function"), "name");
                    if (tc.at("function").contains("arguments")) {
                        const common_json & a = tc.at("function").at("arguments");
                        if (a.is_string()) {
                            out_args = a.get<std::string>();
                        } else if (!a.is_null()) {
                            out_args = a.dump();
                        }
                    }
                }
                if (name.empty()) {
                    name = json_string_or(tc, "name");
                }
                if (!name.empty()) {
                    out_name = name;
                    return true;
                }
            }
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
};

std::mutex & sigma_strip_mutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, sigma_strip_session_state> & sigma_strip_sessions() {
    static std::map<std::string, sigma_strip_session_state> m;
    return m;
}

} // namespace

//
// startup configuration validation
//
std::string sigma_strip_config_check(const common_params & params) {
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

        // 1) token estimate of the original messages + per-message savings
        int64_t tokens_before = 0;
        std::vector<int64_t> savings(n, 0);
        std::vector<std::string> stubs(n);
        for (size_t i = 0; i < n; ++i) {
            const common_json & msg = messages.at(i);
            if (!msg.is_object()) {
                continue;
            }
            const int64_t orig = message_tokens(vocab, msg);
            tokens_before += orig;
            if (json_string_or(msg, "role") == "tool") {
                std::string tool;
                std::string args;
                if (resolve_tool_call(messages, msg, tool, args) && is_compactable_tool(tool)) {
                    stubs[i] = make_stub(tool, args, orig);
                    savings[i] = std::max<int64_t>(0, orig - estimate_tokens(vocab, stubs[i]));
                }
            }
        }
        res.tokens_before = tokens_before;

        const double window = (double) params.sigma_strip_window_tokens;

        // 2) prefix fingerprint check (reset on client compact / /clear / resume)
        std::lock_guard<std::mutex> lock(sigma_strip_mutex());
        sigma_strip_session_state & st = sigma_strip_sessions()[session];
        st.window = params.sigma_strip_window_tokens;
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
        // value (raw tokens minus the savings of the already-stripped
        // prefix) plus the est offset — the template/tool-schema/system
        // overhead the client's full prompt carries but this message-level
        // estimate omits. Starting from the raw tokens here would re-advance
        // K on every request.
        const int64_t off = st.est_offset;
        res.est_offset = off;
        double resident = (double) tokens_before;
        for (size_t i = 0; i < st.K; ++i) {
            resident -= (double) savings[i];
        }
        res.usage_before = (resident + (double) off) / window;
        const double target_tokens = params.sigma_strip_target * window;
        if (res.usage_before >= params.sigma_strip_threshold) {
            // Protection floor: K never crosses the start of the last
            // keep_recent tool rounds (tool_calls + their results) — recent
            // working context ("what was I doing"). Fewer rounds than
            // requested: protect them all.
            const size_t keep_recent = (size_t) std::max<int32_t>(1, params.sigma_strip_keep_recent);
            size_t K_cap = n;
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
            const double floor_tokens = std::max(target_tokens, (double) params.sigma_strip_min_keep_tokens);
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
        res.K = st.K;

        // 4) build the rewrite (K > 0 only)
        int64_t tokens_after = tokens_before;
        size_t n_stripped = 0;
        bool have_rewrite = false;
        common_json rewritten;
        if (st.K > 0) {
            rewritten = common_json::array();
            for (size_t i = 0; i < n; ++i) {
                common_json msg = messages.at(i); // copy
                if (i < st.K && msg.is_object()) {
                    if (!stubs[i].empty()) {
                        msg["content"] = stubs[i];
                        n_stripped++;
                        tokens_after -= savings[i];
                    } else if (msg.contains("content") && msg.at("content").is_array()) {
                        strip_media(msg.at("content"));
                    }
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

        // 6) shadow / on log line
        SRV_INF("sigma-strip: session=%s mode=%s K=%zu/%zu reset=%d advanced=%d stripped=%zu tokens %lld->%lld usage %.3f->%.3f off=%lld rewrite=%s%s%s",
                session.c_str(), params.sigma_strip_mode.c_str(), res.K, n,
                (int) res.prefix_reset, (int) res.k_advanced, res.n_stripped,
                (long long) res.tokens_before, (long long) res.tokens_after,
                res.usage_before, res.usage_after, (long long) res.est_offset,
                res.rewrite_hash.c_str(),
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
