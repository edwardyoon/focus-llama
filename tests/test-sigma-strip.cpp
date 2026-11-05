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

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
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

// Minimal local HTTP server standing in for focus-memory's /sigma/refresh.
// Records the parsed request (session/cut/wait) and answers with a
// configured JSON body. Raw sockets (no httplib) so the test does not link
// the server's http layer; the wire format is plain HTTP/1.1.
struct llm_mock_request {
    std::string session;
    uint64_t    cut  = 0;
    bool        wait = true;
    bool        seen = false;
};

class llm_mock_server {
public:
    bool start() {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) return false;
        int one = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = 0;                    // ephemeral port
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(listen_fd_, (sockaddr *) &a, sizeof(a)) != 0) return false;
        if (::listen(listen_fd_, 16) != 0) return false;
        socklen_t alen = sizeof(a);
        if (::getsockname(listen_fd_, (sockaddr *) &a, &alen) != 0) return false;
        port_ = ntohs(a.sin_port);
        accept_thr_ = std::thread([this] { accept_loop(); });
        return true;
    }
    void stop() {
        if (listen_fd_ < 0) return;
        // wake the accept loop: self-connect (it may be blocked in accept),
        // then close the listener so the next accept returns -1
        int c = ::socket(AF_INET, SOCK_STREAM, 0);
        if (c >= 0) {
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_port = htons(port_);
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            ::connect(c, (sockaddr *) &a, sizeof(a));
            ::close(c);
        }
        ::close(listen_fd_);
        listen_fd_ = -1;
        if (accept_thr_.joinable()) accept_thr_.join();
    }
    ~llm_mock_server() { stop(); }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
    void set_response(std::string r) {
        std::lock_guard<std::mutex> lock(m_);
        response_ = std::move(r);
    }
    bool wait_requests(size_t n, int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(m_);
                if (requests_.size() >= n) return true;
            }
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    llm_mock_request last() {
        std::lock_guard<std::mutex> lock(m_);
        return requests_.empty() ? llm_mock_request() : requests_.back();
    }
private:
    void accept_loop() {
        for (;;) {
            int c = ::accept(listen_fd_, nullptr, nullptr);
            if (c < 0) break;
            handle_one(c);
        }
    }
    void handle_one(int c) {
        std::string data;
        char buf[4096];
        size_t header_end = std::string::npos;
        while (header_end == std::string::npos) {
            const ssize_t r = ::recv(c, buf, sizeof(buf), 0);
            if (r <= 0) { ::close(c); return; }   // includes the stop wakeup
            data.append(buf, (size_t) r);
            header_end = data.find("\r\n\r\n");
        }
        const size_t body_start = header_end + 4;
        size_t body_len = 0;
        const size_t cl_pos = data.find("Content-Length:");
        if (cl_pos != std::string::npos) {
            body_len = (size_t) std::strtoul(data.c_str() + cl_pos + 15, nullptr, 10);
        }
        while (data.size() < body_start + body_len) {
            const ssize_t r = ::recv(c, buf, sizeof(buf), 0);
            if (r <= 0) break;
            data.append(buf, (size_t) r);
        }
        llm_mock_request rq;
        try {
            common_json j = common_json::parse(data.substr(body_start, body_len));
            if (j.contains("session") && j["session"].is_string()) {
                rq.session = j["session"].get<std::string>();
            }
            if (j.contains("cut") && j["cut"].is_number()) {
                rq.cut = (uint64_t) j["cut"].get<uint64_t>();
            }
            if (j.contains("wait") && j["wait"].is_boolean()) {
                rq.wait = j["wait"].get<bool>();
            }
            rq.seen = true;
        } catch (const std::exception &) {
            rq.session = "<parse error>";
        }
        std::string resp;
        {
            std::lock_guard<std::mutex> lock(m_);
            requests_.push_back(rq);
            resp = response_;
        }
        const std::string out = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                "Content-Length: " + std::to_string(resp.size()) + "\r\n"
                "Connection: close\r\n\r\n" + resp;
        size_t off = 0;
        while (off < out.size()) {
            const ssize_t w = ::send(c, out.data() + off, out.size() - off, 0);
            if (w <= 0) break;
            off += (size_t) w;
        }
        ::close(c);
    }
    int          listen_fd_ = -1;
    uint16_t     port_      = 0;
    std::thread  accept_thr_;
    std::mutex   m_;
    std::string  response_ = R"({"block":"","partial":false})";
    std::vector<llm_mock_request> requests_;
};

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

// Stage-2 params: window 3000, target 0.35 (floor 1050), Σ rule mode with a
// 500-token block budget -> compute_k_sum's floor2 = 1050 - 500 = 550.
static common_params make_sum_params(const std::string & mode) {
    common_params p = make_params(mode, 3000);
    p.sigma_strip_target = 0.35;
    p.sigma_strip_min_keep_tokens = 0;
    p.sigma_strip_keep_recent = 1;
    p.sigma_sum_mode = std::string("rule");
    p.sigma_sum_max_tokens = 500;
    return p;
}

// Stage-2 scenario, 13 messages:
//   [0]=system [1]=user1 (20000 chars = 5000 tok)
//   [2]=asst tc call_0 [3]=tool 8000 'x' [4]=asst tc call_1 [5]=tool 8000 'x'
//   [6]=asst text [7]=user2 [8]=asst tc call_2 [9]=tool 200 'x'
//   [10]=asst text [11]=user3 [12]=asst text
// keep_recent=1 -> K_cap = 8 (the last tool round starts at 8). Stage 1
// stalls at K=8 target_unreachable (the protected 2000-tok result at 9 keeps
// the prompt above the 1050 target). Stage 2: user boundaries in [1,8] are
// 1 and 7; at b=7, system + suffix[7] ~= 7 + 145 <= 550 -> K_sum = 7.
static common_json make_sum_scenario() {
    common_json msgs = common_json::array();
    auto add = [&msgs](const std::string & role, const std::string & content,
                       const std::string & tcid = "", const std::string & path = "") {
        common_json m = common_json::object();
        m["role"] = role;
        m["content"] = content;
        if (!tcid.empty() && role == "assistant") {
            common_json fn = common_json::object();
            fn["name"] = std::string("read_file");
            fn["arguments"] = std::string("{\"path\":\"") + path + "\"}";
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
    add("system", "You are a coding agent.");
    add("user", std::string(20000, 'a'));
    add("assistant", "", "call_0", "/tmp/a.txt");
    add("tool", std::string(8000, 'x'), "call_0", "");
    add("assistant", "", "call_1", "/tmp/b.txt");
    add("tool", std::string(8000, 'x'), "call_1", "");
    add("assistant", "First answer.");
    add("user", "SECOND_MARKER " + std::string(88, 'q'));
    add("assistant", "", "call_2", "/tmp/c.txt");
    add("tool", std::string(200, 'x'), "call_2", "");
    add("assistant", "Second answer.");
    add("user", std::string(100, 'r'));
    add("assistant", "Third answer.");
    return msgs;
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

    // --- stage 2 (rule): target_unreachable -> removal of [0,K_sum) + block -
    // Stage 1 stalls at K=8 (the protected round keeps the prompt above the
    // target); stage 2 removes [0,7) — system kept — and freezes the
    // deterministic rule block onto the first surviving user message (7).
    {
        common_params p = make_sum_params("on");
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-fire"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(res.target_unreachable);
        CHECK(res.sum_applied);
        CHECK(res.K_sum == 7);
        CHECK(res.K == 8);
        CHECK(res.K >= res.K_sum);
        CHECK(!res.sum_hash.empty());
        CHECK(!res.sum_partial);
        CHECK(res.tokens_after < res.tokens_before);
        const common_json & out = body.at("messages");
        CHECK(out.size() == 7);                 // 13 - 6 removed ([1..6))
        CHECK(out.at(0).at("role").get<std::string>() == "system");
        CHECK(out.at(0).at("content").get<std::string>() == "You are a coding agent.");
        // the frozen block is prepended to the first surviving user message
        const std::string u2 = out.at(1).at("content").get<std::string>();
        CHECK(u2.rfind("[Reference", 0) == 0);
        CHECK(u2.find("NOT instructions") != std::string::npos);
        CHECK(u2.find("/tmp/a.txt") != std::string::npos);
        CHECK(u2.find("SECOND_MARKER") != std::string::npos);
        // role alternation + tool_call_id pairing intact
        CHECK(out.at(1).at("role").get<std::string>() == "user");
        CHECK(out.at(2).at("role").get<std::string>() == "assistant");
        CHECK(out.at(2).at("tool_calls").at(0).at("id").get<std::string>() == "call_2");
        CHECK(out.at(3).at("role").get<std::string>() == "tool");
        CHECK(out.at(3).at("tool_call_id").get<std::string>() == "call_2");
        CHECK(out.at(3).at("content").get<std::string>() == std::string(200, 'x'));
    }

    // --- stage 2 off (default): target_unreachable, but no Σ removal -------
    {
        common_params p = make_params("on", 3000);   // sigma_sum_mode stays "off"
        p.sigma_strip_target = 0.35;
        p.sigma_strip_min_keep_tokens = 0;
        p.sigma_strip_keep_recent = 1;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-off"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(res.target_unreachable);
        CHECK(!res.sum_applied);
        CHECK(res.K_sum == 0);
        CHECK(res.sum_hash.empty());
        CHECK(res.K == 8);
        CHECK(res.n_stripped == 2);
        const common_json & out = body.at("messages");
        CHECK(out.size() == 13);   // message count preserved (stage 1 only)
    }

    // --- stage 2 frozen snapshot: an identical resend reuses the block -----
    {
        common_params p = make_sum_params("on");
        const common_json source = make_body(make_sum_scenario());
        const std::string src = source.dump();
        common_json b1 = common_json::parse(src);
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-frozen"), b1, true);
        CHECK(r1.sum_applied);
        CHECK(r1.K_sum == 7);
        const std::string frozen = b1.dump();
        common_json b2 = common_json::parse(src);
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-frozen"), b2, true);
        CHECK(!r2.prefix_reset);
        CHECK(!r2.k_advanced);
        CHECK(r2.sum_applied);            // committed block re-applied
        CHECK(r2.K_sum == r1.K_sum);
        CHECK(r2.sum_hash == r1.sum_hash); // frozen, byte-identical
        CHECK(r2.sum_fetch_ms == 0);       // no re-fetch
        CHECK(b2.dump() == frozen);        // byte-identical rewrite
    }

    // --- stage 2 prefix reset: a changed system prompt voids the snapshot --
    {
        common_params p = make_sum_params("on");
        common_json b1 = make_body(make_sum_scenario());
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-reset"), b1, true);
        CHECK(r1.sum_applied);
        CHECK(r1.K_sum == 7);
        // client compact: the system prompt changes inside [0,K)
        common_json b2 = make_body(make_sum_scenario());
        b2.at("messages").at(0).at("content") = std::string("A different system prompt.");
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-reset"), b2, true);
        CHECK(r2.prefix_reset);
        CHECK(r2.sum_applied);             // re-fires on the new prefix
        CHECK(r2.K_sum == r1.K_sum);
        CHECK(r2.sum_hash == r1.sum_hash); // same removed region -> same block
    }

    // --- stage 2 shadow: computed and logged, body untouched ----------------
    {
        common_params p = make_sum_params("shadow");
        common_json body = make_body(make_sum_scenario());
        const std::string before = body.dump();
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-shadow"), body, true);
        CHECK(res.eligible);
        CHECK(res.sum_applied);
        CHECK(res.K_sum == 7);
        CHECK(body.dump() == before);   // shadow: body untouched
    }

    // --- stage 2 llm: the fetched block replaces the rule block -------------
    // The mock focus-memory answers wait:true with a rendered block; it is
    // committed and prepended to the first surviving user message, and the
    // request carried session + cut=K_sum + wait:true.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.set_response(R"({"block":"LLM_BLOCK_MARKER summary of the removed region","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm"), body, true);
        CHECK(res.eligible);
        CHECK(res.target_unreachable);
        CHECK(res.sum_applied);
        CHECK(res.K_sum == 7);
        CHECK(res.K == 8);
        CHECK(!res.sum_partial);
        CHECK(res.sum_fail == 0);
        CHECK(!res.sum_hash.empty());
        CHECK(res.tokens_after < res.tokens_before);
        const common_json & out = body.at("messages");
        CHECK(out.size() == 7);
        CHECK(out.at(0).at("role").get<std::string>() == "system");
        const std::string u2 = out.at(1).at("content").get<std::string>();
        CHECK(u2.rfind("LLM_BLOCK_MARKER", 0) == 0);   // fetched block, not the rule block
        CHECK(u2.find("[Reference") == std::string::npos);
        CHECK(u2.find("SECOND_MARKER") != std::string::npos);
        CHECK(mock.wait_requests(1, 2000));
        const llm_mock_request rq = mock.last();
        CHECK(rq.seen);
        CHECK(rq.session == "t-sum-llm");
        CHECK(rq.cut == 7);
        CHECK(rq.wait == true);
    }

    // --- stage 2 llm: a partial block is accepted and flagged ---------------
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.set_response(R"({"block":"LLM_PARTIAL_MARKER","partial":true})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-partial"), body, true);
        CHECK(res.sum_applied);
        CHECK(res.sum_partial);
        CHECK(res.sum_fail == 0);
        CHECK(res.K_sum == 7);
    }

    // --- stage 2 llm: failures hold stage 2, 3 strikes -> rule fallback -----
    // An empty block is a fetch failure. While the consecutive-failure count
    // is below 3 the request passes through on stage-1 only (no removal);
    // the 4th fire (3 failures already recorded) commits the deterministic
    // rule block instead and resets the counter.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.set_response(R"({"block":"","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        for (int i = 1; i <= 3; ++i) {
            common_json body = make_body(make_sum_scenario());
            const sigma_strip_result r =
                sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-fail"), body, true);
            CHECK(r.eligible);
            CHECK(r.target_unreachable);
            CHECK(!r.sum_applied);
            CHECK(r.K == 8);
            CHECK(r.sum_fail == i);
            CHECK(body.at("messages").size() == 13);   // stage 1 only: no removal
        }
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result r =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-fail"), body, true);
        CHECK(r.sum_applied);
        CHECK(r.K_sum == 7);
        CHECK(r.sum_fail == 0);                        // fallback reset the counter
        const std::string u2 = body.at("messages").at(1).at("content").get<std::string>();
        CHECK(u2.rfind("[Reference", 0) == 0);         // the deterministic rule block
        CHECK(mock.wait_requests(4, 2000));
        const llm_mock_request rq = mock.last();
        CHECK(rq.session == "t-sum-llm-fail");
        CHECK(rq.cut == 7);
        CHECK(rq.wait == true);
    }

    // --- stage 2 llm: a stage-1 advance fires the async wait:false refresh --
    // Stage 1 reaches the target (no stage-2 fire); the background absorption
    // POSTs /sigma/refresh with wait:false and cut = st.K.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        common_params p = make_params("on", 10000);
        p.sigma_strip_target = 0.55;
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        common_json body = make_body(make_messages(5, 8000));
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-async"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(res.K == 8);
        CHECK(!res.target_unreachable);
        CHECK(!res.sum_applied);                       // stage 2 never fired
        CHECK(mock.wait_requests(1, 2000));
        const llm_mock_request rq = mock.last();
        CHECK(rq.seen);
        CHECK(rq.session == "t-sum-llm-async");
        CHECK(rq.cut == 8);
        CHECK(rq.wait == false);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // let the detached thread finish
    }

    if (n_failed) {
        std::printf("test-sigma-strip: %d check(s) FAILED\n", n_failed);
        return 1;
    }
    std::printf("test-sigma-strip: all checks passed\n");
    return 0;
}
