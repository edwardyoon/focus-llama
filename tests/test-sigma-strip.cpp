// Unit tests for sigma-strip (tools/server/sigma-strip.cpp) — the stage-2
// shadow behavior: deterministic stubs, K state, prefix-fingerprint reset,
// fail-open. Plain-assert style, matching the project's test convention.
//
// vocab is nullptr throughout: token estimates use the deterministic
// byte/4 fallback, so the assertions are stable across builds.
//
// Session state is a process-global map keyed by session id, so every test
// case uses its own session id to stay isolated.

#include "sigma-strip.h"
#include "server-http.h"
#include "common.h"

#include <cstdio>
#include <functional>
#include <string>

static int n_failed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_failed++; \
    } \
} while (0)

// server_http_req carries a const reference member (should_stop) that must
// outlive every request instance.
static std::function<bool()> g_stop = [] { return false; };

static server_http_req make_req(const std::string & session_id) {
    // server_http_req has a const-reference member (should_stop), so it is
    // aggregate-initialized: params, headers, path, query_string, body,
    // files, should_stop.
    server_http_req req{ {}, {}, std::string(), std::string(), std::string(), {}, g_stop };
    if (!session_id.empty()) {
        req.headers["X-Session-Id"] = session_id;
    }
    return req;
}

static common_params make_params(const std::string & mode, int32_t window) {
    common_params p;
    p.sigma_strip_mode = mode;
    p.sigma_strip_window_tokens = window;
    return p;
}

// system + user + n_rounds x (assistant tool_call + tool result).
// Each result is result_chars of 'x' -> byte/4 estimate = result_chars/4.
// Layout: [0]=system [1]=user [2]=asst [3]=tool(call_0) [4]=asst [5]=tool(call_1) ...
static common_json make_messages(size_t n_rounds, int result_chars,
                                 const std::string & sys = "You are a coding agent.") {
    common_json msgs = common_json::array();
    {
        common_json m = common_json::object();
        m["role"] = std::string("system");
        m["content"] = sys;
        msgs.push_back(m);
    }
    {
        common_json m = common_json::object();
        m["role"] = std::string("user");
        m["content"] = std::string("Do the task.");
        msgs.push_back(m);
    }
    for (size_t i = 0; i < n_rounds; ++i) {
        common_json fn = common_json::object();
        fn["name"] = std::string("read_file");
        fn["arguments"] = std::string("{\"path\":\"/tmp/file") + std::to_string(i) + ".txt\"}";
        common_json call = common_json::object();
        call["id"] = std::string("call_") + std::to_string(i);
        call["type"] = std::string("function");
        call["function"] = fn;
        common_json tcs = common_json::array();
        tcs.push_back(call);
        common_json asst = common_json::object();
        asst["role"] = std::string("assistant");
        asst["content"] = std::string("");
        asst["tool_calls"] = tcs;
        msgs.push_back(asst);

        common_json tool = common_json::object();
        tool["role"] = std::string("tool");
        tool["tool_call_id"] = std::string("call_") + std::to_string(i);
        tool["name"] = std::string("read_file");
        tool["content"] = std::string((size_t) result_chars, 'x');
        msgs.push_back(tool);
    }
    return msgs;
}

static common_json make_body(const common_json & messages) {
    common_json body = common_json::object();
    body["messages"] = messages;
    return body;
}

int main() {
    // --- mode off: no-op, body untouched ----------------------------------
    {
        common_json body = make_body(make_messages(5, 2000));
        const std::string before = body.dump();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("off", 100000), make_req("t-off"), body, true);
        CHECK(!res.eligible);
        CHECK(body.dump() == before);
    }

    // --- window 0: no-op ----------------------------------------------------
    {
        common_json body = make_body(make_messages(5, 2000));
        const std::string before = body.dump();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 0), make_req("t-window0"), body, true);
        CHECK(!res.eligible);
        CHECK(body.dump() == before);
    }

    // --- no session key (kv-offload-default): excluded, body untouched ------
    {
        common_json body = make_body(make_messages(5, 2000));
        const std::string before = body.dump();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 1000), make_req(""), body, true);
        CHECK(!res.eligible);
        CHECK(body.dump() == before);
    }

    // --- malformed messages (not an array): fail-open, no crash -------------
    {
        common_json body = common_json::object();
        body["messages"] = common_json::object();
        const std::string before = body.dump();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 1000), make_req("t-malformed"), body, true);
        CHECK(!res.eligible);
        CHECK(body.dump() == before);
    }

    // --- shadow below threshold: eligible, K stays 0, no rewrite ------------
    {
        common_json body = make_body(make_messages(5, 2000));
        const std::string before = body.dump();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 100000), make_req("t-below"), body, true);
        CHECK(res.eligible);
        CHECK(res.session == "t-below");
        CHECK(res.K == 0);
        CHECK(res.n_stripped == 0);
        CHECK(res.tokens_after == res.tokens_before);
        CHECK(res.rewrite_hash.empty());
        CHECK(body.dump() == before);
    }

    // --- shadow above threshold: one big K jump, body untouched (shadow) ----
    {
        common_json body = make_body(make_messages(5, 2000));
        const std::string before = body.dump();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 3000), make_req("t-above"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(res.K > 0);
        CHECK(res.K <= res.n_messages);
        CHECK(res.n_stripped >= 1);
        CHECK(res.tokens_after < res.tokens_before);
        CHECK(res.usage_after < res.usage_before);
        CHECK(!res.target_unreachable);
        CHECK(!res.rewrite_hash.empty());
        CHECK(body.dump() == before);  // shadow: compute + log only
    }

    // --- byte-identical across N iterations; K fixed -> prefix invariant ----
    {
        const common_json source = make_body(make_messages(5, 2000));
        const std::string src = source.dump();

        sigma_strip_result first;
        {
            common_json body = common_json::parse(src);
            first = sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 3000), make_req("t-byte"), body, true);
        }
        CHECK(first.k_advanced);
        CHECK(!first.rewrite_hash.empty());
        for (int i = 0; i < 4; ++i) {
            common_json body = common_json::parse(src);
            const sigma_strip_result res =
                sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 3000), make_req("t-byte"), body, true);
            CHECK(res.K == first.K);
            CHECK(res.rewrite_hash == first.rewrite_hash);
            CHECK(!res.prefix_reset);
            CHECK(!res.k_advanced);
            CHECK(body.dump() == src);
        }
    }

    // --- prefix change (client compact) -> K reset ---------------------------
    {
        common_json a = make_body(make_messages(5, 2000, "System prompt A."));
        common_json b = make_body(make_messages(5, 2000, "System prompt B."));
        sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 3000), make_req("t-reset"), a, true);
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("shadow", 3000), make_req("t-reset"), b, true);
        CHECK(res.prefix_reset);
    }

    // --- on + apply: body rewritten, message count and pairing preserved ----
    {
        const common_json messages = make_messages(5, 2000);
        const size_t n = messages.size();
        common_json body = make_body(messages);
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("on", 3000), make_req("t-on"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        const common_json & msgs = body.at("messages");
        CHECK(msgs.size() == n);  // message count preserved
        for (size_t i = 0; i < res.K; ++i) {
            const common_json & m = msgs.at(i);
            if (m.at("role").get<std::string>() == "tool") {
                const std::string content = m.at("content").get<std::string>();
                CHECK(content.rfind("[", 0) == 0);
                CHECK(content.find("result stripped") != std::string::npos);
                // tool_call_id pairing intact
                CHECK(m.at("tool_call_id").get<std::string>() ==
                      std::string("call_") + std::to_string((i - 3) / 2));
            }
        }
        // assistant tool_calls untouched (first assistant is index 2)
        CHECK(msgs.at(2).at("tool_calls").size() == 1);
        // messages beyond K untouched
        if (res.K < n) {
            const std::string tail = msgs.at(n - 1).at("content").get<std::string>();
            CHECK(tail.size() == 2000);
        }
    }

    // --- on + apply=false: compute only, body untouched ----------------------
    {
        common_json body = make_body(make_messages(5, 2000));
        const std::string before = body.dump();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("on", 3000), make_req("t-on-noapply"), body, false);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(body.dump() == before);
    }

    // --- media removal within [0,K): non-text parts dropped ------------------
    {
        common_json msgs = common_json::array();
        {
            common_json m = common_json::object();
            m["role"] = std::string("system");
            m["content"] = std::string("You are a coding agent.");
            msgs.push_back(m);
        }
        {
            common_json m = common_json::object();
            m["role"] = std::string("user");
            common_json parts = common_json::array();
            common_json t = common_json::object();
            t["type"] = std::string("text");
            t["text"] = std::string("look at this");
            parts.push_back(t);
            common_json img = common_json::object();
            img["type"] = std::string("image_url");
            common_json iu = common_json::object();
            iu["url"] = std::string("data:image/png;base64,AAAA");
            img["image_url"] = iu;
            parts.push_back(img);
            m["content"] = parts;
            msgs.push_back(m);
        }
        // append 5 tool rounds
        for (size_t i = 0; i < 5; ++i) {
            common_json fn = common_json::object();
            fn["name"] = std::string("read_file");
            fn["arguments"] = std::string("{\"path\":\"/tmp/file") + std::to_string(i) + ".txt\"}";
            common_json call = common_json::object();
            call["id"] = std::string("call_") + std::to_string(i);
            call["type"] = std::string("function");
            call["function"] = fn;
            common_json tcs = common_json::array();
            tcs.push_back(call);
            common_json asst = common_json::object();
            asst["role"] = std::string("assistant");
            asst["content"] = std::string("");
            asst["tool_calls"] = tcs;
            msgs.push_back(asst);

            common_json tool = common_json::object();
            tool["role"] = std::string("tool");
            tool["tool_call_id"] = std::string("call_") + std::to_string(i);
            tool["name"] = std::string("read_file");
            tool["content"] = std::string(2000, 'x');
            msgs.push_back(tool);
        }
        common_json body = make_body(msgs);
        const size_t n = msgs.size();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("on", 3000), make_req("t-media"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        const common_json & out = body.at("messages");
        CHECK(out.size() == n);
        // the user message (index 1) is inside [0,K): media part dropped, text kept
        const common_json & user = out.at(1);
        CHECK(user.at("content").is_array());
        CHECK(user.at("content").size() == 1);
        CHECK(user.at("content").at(0).at("type").get<std::string>() == "text");
    }

    // --- K cap: the last assistant tool_calls round is never stubbed --------
    // Two-turn layout: [0]=system [1]=user1 [2]=asst(tc call_0) [3]=tool
    // [4]=asst(text) [5]=user2 [6]=asst(tc call_1) [7]=tool. K must stop at 6
    // even though the target is unreachable from there (target_unreachable),
    // and the current turn's tool result must survive verbatim.
    {
        common_json msgs = common_json::array();
        auto add = [&msgs](const std::string & role, const std::string & content,
                           const std::string & tcid = "", const std::string & tcargs = "") {
            common_json m = common_json::object();
            m["role"] = role;
            m["content"] = content;
            if (!tcid.empty()) {
                common_json fn = common_json::object();
                fn["name"] = std::string("read_file");
                fn["arguments"] = tcargs;
                common_json call = common_json::object();
                call["id"] = tcid;
                call["type"] = std::string("function");
                call["function"] = fn;
                common_json tcs = common_json::array();
                tcs.push_back(call);
                m["tool_calls"] = tcs;
            }
            if (!tcid.empty() && role == "tool") {
                m["tool_call_id"] = tcid;
                m["name"] = std::string("read_file");
            }
            msgs.push_back(m);
        };
        const std::string big(8000, 'x');
        add("system", "You are a coding agent.");
        add("user", "First request.");
        add("assistant", "", "call_0", "{\"path\":\"/tmp/a.txt\"}");
        add("tool", big, "call_0", "");
        add("assistant", "First answer.");
        add("user", "Second request.");
        add("assistant", "", "call_1", "{\"path\":\"/tmp/b.txt\"}");
        add("tool", big, "call_1", "");

        common_json body = make_body(msgs);
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, make_params("on", 3000), make_req("t-cap"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(res.K == 6);                 // capped at the last assistant tool_calls
        CHECK(res.n_stripped == 1);        // only the first round's tool result
        CHECK(res.target_unreachable);     // protected round keeps the prompt above target
        const common_json & out = body.at("messages");
        CHECK(out.size() == 8);
        const std::string stubbed = out.at(3).at("content").get<std::string>();
        CHECK(stubbed.rfind("[", 0) == 0);
        CHECK(stubbed.find("result stripped") != std::string::npos);
        // the current turn's tool result survives verbatim
        CHECK(out.at(7).at("content").get<std::string>() == big);
        CHECK(out.at(7).at("tool_call_id").get<std::string>() == "call_1");
    }

    if (n_failed) {
        std::printf("test-sigma-strip: %d check(s) FAILED\n", n_failed);
        return 1;
    }
    std::printf("test-sigma-strip: all checks passed\n");
    return 0;
}
