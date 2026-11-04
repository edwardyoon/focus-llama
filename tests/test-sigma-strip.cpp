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
#include <vector>

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

    // --- token floor: min_keep stops the advance before the cap ------------
    // 5 rounds (K_cap=10 with keep_recent=1). Target-only would strip all 5
    // results (K=10); min_keep=4500 stops it after 3 (tokens_after ~4100
    // <= 4500, K=8 < 10) -> floor_limited.
    {
        common_params p = make_params("on", 3000);
        p.sigma_strip_min_keep_tokens = 4500;
        common_json body = make_body(make_messages(5, 8000));
        const size_t n = 12;
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-floor"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(res.K == 8);
        CHECK(res.n_stripped == 3);
        CHECK(res.floor_limited);
        CHECK(res.target_unreachable);
        const std::string big(8000, 'x');
        const common_json & out = body.at("messages");
        CHECK(out.size() == n);
        CHECK(out.at(3).at("content").get<std::string>() != big);
        CHECK(out.at(9).at("content").get<std::string>() == big);
        CHECK(out.at(11).at("content").get<std::string>() == big);
    }

    // --- protection floor: keep_recent=2 keeps the last two rounds ---------
    // 4 rounds (tool_calls at 2,4,6,8). K_cap=6: the last two rounds
    // (results 7,9) survive verbatim; the older two (results 3,5) are stubbed.
    {
        common_params p = make_params("on", 3000);
        p.sigma_strip_keep_recent = 2;
        common_json body = make_body(make_messages(4, 8000));
        const size_t n = 10;
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-keeprecent"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(res.K == 6);
        CHECK(res.n_stripped == 2);
        CHECK(!res.floor_limited);
        CHECK(res.target_unreachable);
        const std::string big(8000, 'x');
        const common_json & out = body.at("messages");
        CHECK(out.size() == n);
        CHECK(out.at(3).at("content").get<std::string>() != big);
        CHECK(out.at(5).at("content").get<std::string>() != big);
        CHECK(out.at(7).at("content").get<std::string>() == big);
        CHECK(out.at(9).at("content").get<std::string>() == big);
    }

    // --- token floor off: a plain target stop is not floor_limited ---------
    // min_keep = 0 -> floor == target. The advance stops at the target with
    // K below the cap (5 rounds, K_cap = 10): the old condition
    // (K2 < K_cap && stripped <= floor) flagged this as floor-limited.
    {
        common_params p = make_params("on", 10000);
        p.sigma_strip_target = 0.55;   // floor = target = 5500
        common_json body = make_body(make_messages(5, 8000));
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-flooroff"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(res.K == 8);
        CHECK(res.n_stripped == 3);
        CHECK(!res.floor_limited);
        CHECK(!res.target_unreachable);
    }

    // --- config check: unusable floors are rejected at startup -------------
    // threshold 0.70 x window 10000 = 7000; 10% of window = 1000.
    {
        common_params p = make_params("on", 10000);
        p.sigma_strip_min_keep_tokens = 7000;  // floor 7000 >= threshold 7000
        CHECK(!sigma_strip_config_check(p).empty());
        p.sigma_strip_min_keep_tokens = 0;
        p.sigma_strip_target = 0.8;            // floor 8000 > threshold 7000
        CHECK(!sigma_strip_config_check(p).empty());
        p.sigma_strip_target = 0.68;           // floor 6800, margin 200 < 1000
        CHECK(sigma_strip_config_check(p).empty());   // thin margin: warn only
        p.sigma_strip_target = 0.35;           // floor 3500, margin 3500 >= 1000
        CHECK(sigma_strip_config_check(p).empty());
        p.sigma_strip_min_keep_tokens = 6500;  // floor 6500, margin 500 < 1000
        CHECK(sigma_strip_config_check(p).empty());
        common_params q = make_params("off", 10000);
        q.sigma_strip_min_keep_tokens = 7000;
        CHECK(sigma_strip_config_check(q).empty());  // feature off: no check
        common_params r = make_params("on", 0);
        r.sigma_strip_min_keep_tokens = 7000;
        CHECK(sigma_strip_config_check(r).empty());  // no window: no check
    }

    // --- E1-1: est offset calibrates the gate into reported space ---------
    // 5 rounds x 3200 chars -> raw estimate 4059 < 0.70 x 10000: no advance.
    // The response reports 3000 more (template/tool-schema/system overhead
    // the message-level estimate omits) -> est_offset = 3000. The identical
    // re-send now measures (4059+3000)/10000 = 0.706 >= 0.70: the offset
    // alone triggers the advance (to the K_cap at 10).
    {
        common_params p = make_params("shadow", 10000);
        common_json body = make_body(make_messages(5, 3200));
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-e1-1"), body, true);
        CHECK(r1.eligible);
        CHECK(r1.tokens_before == 4059);
        CHECK(r1.usage_before < 0.70);
        CHECK(r1.K == 0);
        CHECK(!r1.k_advanced);
        sigma_strip_note_response("t-e1-1", r1.tokens_before + 3000);
        common_json body2 = make_body(make_messages(5, 3200));
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-e1-1"), body2, true);
        CHECK(r2.est_offset == 3000);
        CHECK(r2.usage_before >= 0.70);
        CHECK(r2.k_advanced);
        CHECK(r2.K == 10);
        CHECK(r2.n_stripped == 4);
    }

    // --- E1-2: prefix reset voids the est calibration ----------------------
    // After the advance (K=10, off=3000) the client compacts (system prompt
    // changed inside [0,K)) -> K resets and est_offset drops to 0; the
    // request is judged on the raw estimate (0.406 < 0.70) until a fresh
    // response re-anchors the offset.
    {
        common_params p = make_params("shadow", 10000);
        common_json b1 = make_body(make_messages(5, 3200));
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-e1-2"), b1, true);
        sigma_strip_note_response("t-e1-2", r1.tokens_before + 3000);
        common_json b2 = make_body(make_messages(5, 3200));
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-e1-2"), b2, true);
        CHECK(r2.k_advanced);
        CHECK(r2.K == 10);
        common_json b3 = make_body(make_messages(5, 3200, "A different system prompt."));
        const sigma_strip_result r3 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-e1-2"), b3, true);
        CHECK(r3.prefix_reset);
        CHECK(r3.est_offset == 0);
        CHECK(r3.K == 0);
        CHECK(!r3.k_advanced);
        CHECK(r3.usage_before < 0.70);
    }

    // --- E1-3: K safety boundary (parallel tool group) ---------------------
    // Layout: [0]=system [1]=user [2]=asst (2 parallel calls) [3]=tool
    // [4]=tool [5]=asst [6]=tool. Raw estimate 7238 >= 0.70 x 10000, target
    // 0.65 -> floor 6500 stops the raw advance at K2=4, inside the parallel
    // group. K must roll back to the group-start assistant at 2 — never to a
    // tool message — and the rollback is a no-op rewrite (nothing compactable
    // below 2).
    {
        common_params p = make_params("shadow", 10000);
        p.sigma_strip_target = 0.65;
        common_json msgs = common_json::array();
        auto add_asst_calls = [&msgs](const std::vector<std::string> & ids) {
            common_json tcs = common_json::array();
            for (const std::string & id : ids) {
                common_json fn = common_json::object();
                fn["name"] = std::string("read_file");
                fn["arguments"] = std::string("{\"path\":\"/tmp/\"") + id.substr(5) + ".txt\"}";
                common_json call = common_json::object();
                call["id"] = id;
                call["type"] = std::string("function");
                call["function"] = fn;
                tcs.push_back(call);
            }
            common_json m = common_json::object();
            m["role"] = std::string("assistant");
            m["content"] = std::string("");
            m["tool_calls"] = tcs;
            msgs.push_back(m);
        };
        auto add_tool = [&msgs](const std::string & tcid, const std::string & content) {
            common_json m = common_json::object();
            m["role"] = std::string("tool");
            m["content"] = content;
            m["tool_call_id"] = tcid;
            m["name"] = std::string("read_file");
            msgs.push_back(m);
        };
        {
            common_json m = common_json::object();
            m["role"] = std::string("system");
            m["content"] = std::string("You are a coding agent.");
            msgs.push_back(m);
        }
        {
            common_json m = common_json::object();
            m["role"] = std::string("user");
            m["content"] = std::string("Do the task.");
            msgs.push_back(m);
        }
        const std::string big(9600, 'x');
        add_asst_calls({ "call_0", "call_1" });
        add_tool("call_0", big);
        add_tool("call_1", big);
        add_asst_calls({ "call_2" });
        add_tool("call_2", big);
        common_json body = make_body(msgs);
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-e1-3"), body, true);
        CHECK(res.eligible);
        CHECK(res.tokens_before == 7238);
        CHECK(res.usage_before >= 0.70);
        CHECK(res.k_advanced);
        CHECK(res.K == 2);
        // the invariant: K is 0 or points at a user/assistant message, so a
        // future insertion at K cannot break tool_call_id pairing
        const std::string roleK = msgs.at(res.K).at("role").get<std::string>();
        CHECK(res.K == 0 || roleK == "user" || roleK == "assistant");
        CHECK(res.n_stripped == 0);
        CHECK(res.tokens_after == res.tokens_before);
    }

    // --- E1-4: the token floor runs in reported space ----------------------
    // min_keep 6000 > target x window 3500. With est_offset 3000 the
    // effective floor is floor_adj = 6000 - 3000 = 3000, so the advance goes
    // further (K=8) than the raw floor 6000 would allow (K=4) and stops
    // floor-limited.
    {
        common_params p = make_params("shadow", 10000);
        p.sigma_strip_min_keep_tokens = 6000;
        common_json b1 = make_body(make_messages(5, 4800));
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-e1-4"), b1, true);
        CHECK(r1.tokens_before == 6059);
        CHECK(r1.K == 0);
        CHECK(!r1.k_advanced);
        sigma_strip_note_response("t-e1-4", r1.tokens_before + 3000);
        common_json b2 = make_body(make_messages(5, 4800));
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-e1-4"), b2, true);
        CHECK(r2.est_offset == 3000);
        CHECK(r2.k_advanced);
        CHECK(r2.K == 8);
        CHECK(r2.n_stripped == 3);
        CHECK(r2.floor_limited);
        CHECK(r2.target_unreachable);
    }

    if (n_failed) {
        std::printf("test-sigma-strip: %d check(s) FAILED\n", n_failed);
        return 1;
    }
    std::printf("test-sigma-strip: all checks passed\n");
    return 0;
}
