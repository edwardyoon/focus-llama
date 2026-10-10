#include "kv-offload.h"

#include "http.h"            // common_http_client (cpp-httplib)
#include "server-common.h"   // SRV_WRN; transitively common.h (common_tokenize), llama.h, json.h
#include "server-http.h"     // server_http_req (full definition)

#include <cctype>            // tolower
#include <regex>             // std::regex, std::sregex_iterator
#include <utility>           // std::move

// FNV-1a 64-bit of the segment text -> 16 hex chars (stable content key).
std::string kv_offload_hash16(const std::string & s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= (uint64_t) c;
        h *= 1099511628211ULL;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return std::string(buf, 16);
}

// Minimal JSON string escape (for the PUT body).
std::string kv_offload_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

// Resolve the FocusMemory session id for one request. Priority:
// 1. X-Session-Id header (the Qwen Code client carries the real session
//    UUID here; matched case-insensitively, the wire case is not guaranteed),
// 2. the OpenAI `user` field, 3. the shared "kv-offload-default" constant
// (clients that send neither). Every store call of the request must use the
// same value (evict PUT, refill GET via task.params.kv_offload_session)
// or the per-session store files would fragment.
std::string kv_offload_resolve_session(const server_http_req & req, const common_json & data) {
    auto header_ci = [&req](const std::string & want) -> std::string {
        std::string w = want;
        for (auto & c : w) c = (char) tolower((unsigned char) c);
        for (const auto & kv : req.headers) {
            std::string k = kv.first;
            for (auto & c : k) c = (char) tolower((unsigned char) c);
            if (k == w) return kv.second;
        }
        return "";
    };
    const std::string h = header_ci("X-Session-Id");
    if (!h.empty()) return h;
    if (data.contains("user") && data["user"].is_string() && !data["user"].get<std::string>().empty()) {
        return data["user"].get<std::string>();
    }
    return "kv-offload-default";
}

// PUT one evicted segment's text to the FocusMemory store. Returns true on
// success; false on any error (the caller then keeps the segment in the prompt).
bool kv_offload_put(
        const std::string & host, const std::string & token,
        const std::string & session_id, const std::string & key,
        const std::string & text, int32_t tokens) {
    if (host.empty() || key.empty() || text.empty()) return false;
    try {
        auto [cli, parts] = common_http_client(host);
        cli.set_read_timeout(5, 0);
        if (!token.empty()) {
            cli.set_default_headers({ { "Authorization", "Bearer " + token } });
        }
        std::string body = "{\"session_id\":\"" + kv_offload_json_escape(session_id)
                         + "\",\"key\":\"" + kv_offload_json_escape(key)
                         + "\",\"text\":\"" + kv_offload_json_escape(text)
                         + "\",\"tokens\":" + std::to_string(tokens) + "}";
        std::string path = "/v1/kv-offload/chunk";
        if (!parts.path.empty() && parts.path != "/") path = parts.path + path;
        auto res = cli.Put(path, body, "application/json");
        if (!res || res->status != 200) {
            SRV_WRN("kv_offload: PUT failed (status %d) key=%s\n", res ? res->status : -1, key.c_str());
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        SRV_WRN("kv_offload: PUT exception: %s\n", e.what());
        return false;
    }
}

// Percent-encode a value for use in a URL query string (RFC 3986 unreserved
// set: A-Z a-z 0-9 - _ . ~). Everything else becomes %XX so that a session id
// or key containing '&' '?' '=' etc. cannot break the query structure.
std::string kv_offload_url_encode(const std::string & s) {
    static const char * hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '~') {
            out += (char) c;
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 0xF];
        }
    }
    return out;
}

// GET one segment's text back from the FocusMemory store (get-on-focus).
// Returns true and fills out_text on success; false on any error (fail-open:
// the caller proceeds without the chunk).
bool kv_offload_get(
        const std::string & host, const std::string & token,
        const std::string & session_id, const std::string & key,
        std::string & out_text) {
    if (host.empty() || key.empty()) return false;
    try {
        auto [cli, parts] = common_http_client(host);
        cli.set_read_timeout(5, 0);
        if (!token.empty()) {
            cli.set_default_headers({ { "Authorization", "Bearer " + token } });
        }
        std::string path = "/v1/kv-offload/chunk?session_id=" + kv_offload_url_encode(session_id) + "&key=" + kv_offload_url_encode(key);
        if (!parts.path.empty() && parts.path != "/") path = parts.path + path;
        auto res = cli.Get(path);
        if (!res || res->status != 200) return false;
        common_json j = common_json::parse(res->body);
        if (!j.contains("text")) return false;
        out_text = j["text"].get<std::string>();
        return !out_text.empty();
    } catch (const std::exception &) {
        return false;
    }
}

// Query the FocusMemory store for a session's pin-released flag (B4). The
// state worker sets it when the user has cancelled or superseded the
// session's ORIGINAL first task: a pinned cancelled request keeps steering
// the model while the cancelling instructions sit in KV holes (2026-09-28
// incident). While false (the default) the first user message stays pinned
// (task anchor - 2026-09-26 fix); once true it becomes a normal evictable
// middle message (re-surfaced per turn by the store's instruction ledger).
// Returns true and fills out_pin_released on success; false on any error
// (fail-open: the caller keeps the pin - current behavior).
bool kv_offload_session_get(
        const std::string & host, const std::string & token,
        const std::string & session_id, bool & out_pin_released) {
    if (host.empty() || session_id.empty()) return false;
    try {
        auto [cli, parts] = common_http_client(host);
        cli.set_connection_timeout(1, 0);
        cli.set_read_timeout(5, 0);
        if (!token.empty()) {
            cli.set_default_headers({ { "Authorization", "Bearer " + token } });
        }
        std::string path = "/v1/kv-offload/session?session_id=" + kv_offload_url_encode(session_id);
        if (!parts.path.empty() && parts.path != "/") path = parts.path + path;
        auto res = cli.Get(path);
        if (!res || res->status != 200) return false;
        common_json j = common_json::parse(res->body);
        out_pin_released = j.contains("pin_released") && j["pin_released"].is_boolean() && j["pin_released"].get<bool>();
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

// kv-offload (mid-turn Σ, 2026-10-09): fetch the session's state anchor from
// FocusMemory. The Stop/UserPromptSubmit hooks only refresh Σ at TURN
// boundaries, so a long autonomous turn (an hour of reason/read without a
// stop) runs on a frozen state record while the eviction PUTs keep distilling
// new state server-side. The caller appends the anchor to the DA scaffold's
// offloaded-chunks note so the model re-grounds on every request. This runs
// on the request path: 200ms cap total, fail-open (empty anchor = nothing to
// inject) — it must never delay a generation.
bool kv_offload_get_sigma(
        const std::string & host, const std::string & token,
        const std::string & session_id, std::string & out_anchor) {
    if (host.empty() || session_id.empty()) return false;
    try {
        auto [cli, parts] = common_http_client(host);
        cli.set_connection_timeout(0, 200000); // 200ms — request-path budget
        cli.set_read_timeout(0, 200000);
        if (!token.empty()) {
            cli.set_default_headers({ { "Authorization", "Bearer " + token } });
        }
        std::string path = "/v1/kv-offload/session/sigma?session_id=" + kv_offload_url_encode(session_id);
        if (!parts.path.empty() && parts.path != "/") path = parts.path + path;
        auto res = cli.Get(path);
        if (!res || res->status != 200) return false;
        common_json j = common_json::parse(res->body);
        if (j.contains("anchor") && j["anchor"].is_string()) {
            out_anchor = j["anchor"].get<std::string>();
            return true;
        }
        return false;
    } catch (const std::exception &) {
        return false;
    }
}

bool kv_offload_evict(
        const llama_vocab * vocab,
        const std::string & text,
        int32_t total_tokens,
        int32_t threshold,
        std::vector<kv_offload_evict_segment> & out_segments,
        bool release_pin,
        int32_t retain) {
    if (threshold <= 0 || total_tokens <= threshold + retain) return false;

    struct msg_t { size_t start; size_t role_end; std::string role; };
    std::vector<msg_t> msgs;
    {
        static const std::regex msg_re(R"(<\|im_start\|>([a-zA-Z_]+)\n)");
        for (std::sregex_iterator it(text.cbegin(), text.cend(), msg_re), end; it != end; ++it) {
            msg_t m;
            m.start    = (size_t) it->position(0);
            m.role_end = m.start + (size_t) it->length(0);
            m.role     = (*it)[1].str();
            msgs.push_back(m);
        }
    }
    if (msgs.size() < 3 || msgs[0].role != "system") return false;

    // first user message = the task anchor (original request); last user = the
    // current question. Both are protected from eviction (see evict_start below).
    size_t first_user = 0, last_user = 0;
    for (size_t i = 0; i < msgs.size(); i++) {
        if (msgs[i].role != "user") continue;
        if (first_user == 0) first_user = i;
        last_user = i;
    }
    if (first_user == 0 || last_user < 2) return false;

    auto n_tok = [&](const std::string & s) -> int32_t {
        return (int32_t) common_tokenize(vocab, s, true, true).size();
    };
    auto msg_range = [&](size_t i) -> std::pair<size_t, size_t> {
        size_t lo = msgs[i].start;
        size_t hi = (i + 1 < msgs.size()) ? msgs[i + 1].start : text.size();
        return { lo, hi };
    };

    // Evict from the oldest middle message forward until under the threshold
    // + retain margin (retain = minimum recent tokens kept in the KV, 0 = current).
    // The system message (index 0) is never evicted. The first user message
    // (the task anchor) is pinned by default - eviction starts after it;
    // with release_pin it becomes a normal candidate (eviction starts at it).
    const size_t evict_start = release_pin ? first_user : first_user + 1;
    if (evict_start >= last_user) return false;  // nothing between start and last user
    int32_t remaining = total_tokens;
    std::vector<size_t> evict_idx;
    for (size_t i = evict_start; i < last_user; i++) {
        if (remaining <= threshold + retain) break;
        auto [lo, hi] = msg_range(i);
        if (hi <= lo) continue;
        evict_idx.push_back(i);
        remaining -= n_tok(text.substr(lo, hi - lo));
    }
    if (evict_idx.empty()) return false;

    for (size_t i : evict_idx) {
        auto [lo, hi] = msg_range(i);
        const std::string seg_text = text.substr(lo, hi - lo);
        kv_offload_evict_segment seg;
        seg.text     = seg_text;
        seg.tokens   = n_tok(seg_text);
        seg.key      = kv_offload_hash16(seg_text);
        seg.range_lo = lo;
        seg.range_hi = hi;
        std::string first = text.substr(msgs[i].role_end, hi - msgs[i].role_end);
        if (first.size() > 60) first.resize(60);
        for (auto & ch : first) if (ch == '\n' || ch == '\r') ch = ' ';
        seg.hint = msgs[i].role + ": " + first;
        out_segments.push_back(std::move(seg));
    }
    return true;
}

common_json kv_offload_build_state_json(const kv_offload_state & state, const std::string & session, int64_t buffer_size) {
    // Assumes state.put_mutex is held by the caller.
    const auto it_logical = state.session_logical.find(session);
    const auto it_ledger  = state.uploaded.find(session);
    if (it_logical == state.session_logical.end() && it_ledger == state.uploaded.end()) {
        return nullptr; // unknown session
    }
    int64_t logical   = (it_logical != state.session_logical.end()) ? it_logical->second : 0;
    int64_t offloaded = 0;
    int64_t evictions = 0;
    if (it_ledger != state.uploaded.end()) {
        evictions = (int64_t) it_ledger->second.size();
        for (const auto & kv : it_ledger->second) {
            offloaded += kv.second;
        }
    }
    // In both offload modes (holes keeps evicted text in the prompt and cuts the
    // KV cells; the default removes the text) the KV-resident count is the full
    // logical size minus the offloaded tokens.
    int64_t resident = logical - offloaded;
    if (resident < 0) resident = 0;
    return {
        {"session",   session},
        {"logical",   logical},
        {"resident",  resident},
        {"offloaded", offloaded},
        {"evictions", evictions},
        {"buffer",    (int64_t) buffer_size},
    };
}
