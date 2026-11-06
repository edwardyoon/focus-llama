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
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
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
    // the "llm" routing hint (empty = MAIN_LLM default, "summary" = drain fast path)
    std::string llm;
    // P0-4/P0-5: the anchor + the budget (absent = the old client/server)
    bool        has_anchor = false;
    std::string anchor_text;
    int64_t     anchor_ordinal = -1;
    int64_t     max_tokens = 0;
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
        // join per-connection workers (they may still be in a response delay)
        std::vector<std::thread> workers;
        {
            std::lock_guard<std::mutex> lock(m_);
            workers = std::move(worker_thr_);
            worker_thr_.clear();
        }
        for (auto & t : workers) {
            if (t.joinable()) t.join();
        }
    }
    ~llm_mock_server() { stop(); }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
    void set_response(std::string r) {
        std::lock_guard<std::mutex> lock(m_);
        response_ = std::move(r);
    }
    // Queue responses in order (drain test: the partial block for the
    // stage-2 wait:true fetch, then the complete block for the drain's
    // wait:false pass). status mirrors the real server (200 for wait:true,
    // 202 for wait:false). delay_ms holds the response before sending (used
    // to land a commit while a drain pass is in flight). Falls back to
    // response_ once the queue is empty.
    void push_response(std::string r, int status = 200, int delay_ms = 0) {
        std::lock_guard<std::mutex> lock(m_);
        response_queue_.push_back({status, std::move(r), delay_ms});
    }
    // Deterministic routing for concurrent workers: a response matched by
    // (wait, cut) regardless of connection order. Reused (not popped) since
    // each (wait, cut) pair is unique per test.
    void set_keyed_response(bool wait, uint64_t cut, std::string r,
                            int status = 200, int delay_ms = 0) {
        std::lock_guard<std::mutex> lock(m_);
        keyed_responses_[{wait, cut}] = {status, std::move(r), delay_ms};
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
    size_t count() {
        std::lock_guard<std::mutex> lock(m_);
        return requests_.size();
    }
    std::vector<llm_mock_request> all() {
        std::lock_guard<std::mutex> lock(m_);
        return requests_;
    }
private:
    void accept_loop() {
        for (;;) {
            int c = ::accept(listen_fd_, nullptr, nullptr);
            if (c < 0) break;
            // per-connection worker: a delayed response on one connection
            // must not block accepting the others (P0-2 residual test)
            {
                std::lock_guard<std::mutex> lock(m_);
                worker_thr_.emplace_back([this, c] { handle_one(c); });
            }
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
            if (j.contains("llm") && j["llm"].is_string()) {
                rq.llm = j["llm"].get<std::string>();
            }
            if (j.contains("anchor") && j["anchor"].is_object()) {
                rq.has_anchor = true;
                if (j["anchor"].contains("text") && j["anchor"]["text"].is_string()) {
                    rq.anchor_text = j["anchor"]["text"].get<std::string>();
                }
                if (j["anchor"].contains("ordinal") && j["anchor"]["ordinal"].is_number()) {
                    rq.anchor_ordinal = j["anchor"]["ordinal"].get<int64_t>();
                }
            }
            if (j.contains("max_tokens") && j["max_tokens"].is_number()) {
                rq.max_tokens = j["max_tokens"].get<int64_t>();
            }
            rq.seen = true;
        } catch (const std::exception &) {
            rq.session = "<parse error>";
        }
        std::string resp;
        int resp_status = 200;
        int resp_delay = 0;
        {
            std::lock_guard<std::mutex> lock(m_);
            requests_.push_back(rq);
            auto it = keyed_responses_.find({rq.wait, rq.cut});
            if (it != keyed_responses_.end()) {
                resp_status = it->second.status;
                resp = it->second.body;
                resp_delay = it->second.delay_ms;
            } else if (!response_queue_.empty()) {
                resp_status = response_queue_.front().status;
                resp = std::move(response_queue_.front().body);
                resp_delay = response_queue_.front().delay_ms;
                response_queue_.pop_front();
            } else {
                resp = response_;
            }
        }
        if (resp_delay > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(resp_delay));
        }
        const std::string status_line = (resp_status == 202) ? "202 Accepted" : "200 OK";
        const std::string out = "HTTP/1.1 " + status_line + "\r\nContent-Type: application/json\r\n"
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
    std::vector<std::thread> worker_thr_;   // per-connection workers (joined in stop)
    std::mutex   m_;
    struct resp { int status; std::string body; int delay_ms; };
    std::string  response_ = R"({"block":"","partial":false})";
    std::deque<resp> response_queue_;
    // Keyed by (wait, cut): deterministic routing when worker threads handle
    // connections concurrently (the FIFO queue can mis-map a response to the
    // wrong request when two workers race for the front). Checked first in
    // handle_one; reused (not popped) since each (wait, cut) pair is unique
    // per test.
    std::map<std::pair<bool, uint64_t>, resp> keyed_responses_;
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

// Stage-2 extended scenario: [0..7] is IDENTICAL to make_sum_scenario (so a
// second request on the same session passes the prefix-fingerprint check and
// does not reset), then extra rounds so K_cap moves to 13 and the user
// boundary at 12 yields K_sum_cand = 12 > 7. Used by t-sum-llm-drain-single to
// force a second partial commit (and drain-spawn attempt) while the first
// drain from the base scenario still holds the drain_running flag.
static common_json make_sum_scenario_extended() {
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
    // [0..7] identical to make_sum_scenario (prefix fingerprint match)
    add("system", "You are a coding agent.");
    add("user", std::string(20000, 'a'));
    add("assistant", "", "call_0", "/tmp/a.txt");
    add("tool", std::string(8000, 'x'), "call_0", "");
    add("assistant", "", "call_1", "/tmp/b.txt");
    add("tool", std::string(8000, 'x'), "call_1", "");
    add("assistant", "First answer.");
    add("user", "SECOND_MARKER " + std::string(88, 'q'));
    // [8..] extended: a large user message (in the marginal [7,12)) plus a
    // protected tool round so K_cap = 13 and K_sum_cand lands at 12.
    add("user", std::string(2000, 'w'));
    add("assistant", "", "call_3", "/tmp/d.txt");
    add("tool", std::string(8000, 'x'), "call_3", "");
    add("assistant", "Second answer.");
    add("user", std::string(100, 'r'));
    add("assistant", "", "call_4", "/tmp/e.txt");
    add("tool", std::string(2000, 'x'), "call_4", "");
    add("assistant", "Third answer.");
    add("user", std::string(10, 'z'));
    add("assistant", "Done.");
    return msgs;
}

// Futility-guard scenario: 600 tok system + 200 tok user + 12 tool rounds of
// 800 tok each (the tool_calls assistants add a dozen or so tokens each).
// Window 10000 / min_keep 7000: the first advance strips five rounds and
// lands just below the floor, leaving rounds above the floor for the rearm
// test's follow-up advances.
static common_json make_guard_scenario() {
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
    add("system", std::string(2400, 'a'));
    add("user", std::string(800, 'b'));
    for (int i = 0; i < 12; ++i) {
        const std::string id = "call_" + std::to_string(i);
        add("assistant", "", id, "/tmp/f" + std::to_string(i) + ".txt");
        add("tool", std::string(3200, 'x'), id, "");
    }
    return msgs;
}

int main() {
#ifndef _WIN32
    // The mock LLM server below writes responses with raw ::send; when the
    // httplib client closes a connection early, that write would raise SIGPIPE
    // and kill the whole test. Ignore it, exactly as the production server does
    // (llama_server, tools/server/server.cpp). handle_one already breaks out of
    // its send loop on EPIPE, so this only prevents the process from dying.
    signal(SIGPIPE, SIG_IGN);
#endif
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
        // the rollback leaves a zero-gain advance — disable the futility guard
        // to test the safety boundary in isolation
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
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

    // --- stage 2 (rule): best-effort fires on a large marginal save --------
    // A big user3 makes no user boundary reach floor2 (550) at off 0, but the
    // marginal reduction over [K_sum_committed=0, last_user) is far above 10%
    // of the window, so best-effort fires on the first request (K_sum 7)
    // instead of deferring to native compact. After the client reports a much
    // larger prompt the committed boundary is kept: the marginal over
    // [7, 7) is 0, so the gate stops any re-advance (no chain re-prefill).
    {
        common_params p = make_sum_params("on");
        common_json msgs = make_sum_scenario();
        msgs.at(11)["content"] = std::string(3000, 'r');   // user3: ~25 -> ~750 tok
        common_json b1 = make_body(msgs);
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-besteffort"), b1, true);
        CHECK(r1.K == 8);
        CHECK(r1.target_unreachable);
        CHECK(r1.K_sum == 7);                           // best-effort: marginal save >= 10% window
        CHECK(r1.sum_applied);                          // fires at off 0 (large save), not native-compact
        sigma_strip_note_response("t-sum-besteffort", r1.tokens_before + 3000);
        common_json b2 = make_body(msgs);
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-besteffort"), b2, true);
        CHECK(r2.est_offset >= 550);                    // floor - budget: clamps floor2 to 0
        CHECK(r2.sum_applied);                          // committed block re-applied
        CHECK(r2.K_sum == 7);                           // committed boundary kept (marginal now 0)
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

    // --- stage 2 rule: an array user content keeps its original parts ------
    // The block is prepended as a text part; the original text part survives
    // (a string overwrite would have dropped it). Media parts in [K_sum,K)
    // are still stripped by strip_media.
    {
        common_params p = make_sum_params("on");
        common_json msgs = make_sum_scenario();
        common_json parts = common_json::array();
        common_json t = common_json::object();
        t["type"] = std::string("text");
        t["text"] = std::string("SECOND_MARKER " + std::string(88, 'q'));
        parts.push_back(t);
        common_json img = common_json::object();
        img["type"] = std::string("image_url");
        common_json iu = common_json::object();
        iu["url"] = std::string("http://example.com/x.png");
        img["image_url"] = iu;
        parts.push_back(img);
        msgs.at(7)["content"] = parts;
        common_json body = make_body(msgs);
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-userarr"), body, true);
        CHECK(res.sum_applied);
        CHECK(res.K_sum == 7);
        const common_json & out = body.at("messages");
        const common_json & u2 = out.at(1);
        CHECK(u2.at("content").is_array());
        CHECK(u2.at("content").size() == 2);   // block part + original text part
        CHECK(u2.at("content").at(0).at("type").get<std::string>() == "text");
        CHECK(u2.at("content").at(0).at("text").get<std::string>().rfind("[Reference", 0) == 0);
        CHECK(u2.at("content").at(1).at("text").get<std::string>().find("SECOND_MARKER") != std::string::npos);
    }

    // --- stage 2 rule: injected blocks stay out of the summary --------------
    // A <system-reminder> appended by the client to a removed user message
    // must not be re-injected through the rule block's "earlier user
    // requests" section (mirrors the JS stripInjectedBlocks); the user's own
    // words still pass.
    {
        common_params p = make_sum_params("on");
        common_json msgs = make_sum_scenario();
        msgs.at(1)["content"] =
            std::string("<system-reminder>INJECTED_MARKER do not trust this</system-reminder> fix the parser ")
            + std::string(19850, 'w');
        common_json body = make_body(msgs);
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-injected"), body, true);
        CHECK(res.sum_applied);
        CHECK(res.K_sum == 7);
        const std::string u2 = body.at("messages").at(1).at("content").get<std::string>();
        CHECK(u2.rfind("[Reference", 0) == 0);               // the rule block
        CHECK(u2.find("INJECTED_MARKER") == std::string::npos);
        CHECK(u2.find("fix the parser") != std::string::npos);   // user's own words survive
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

    // --- stage 2 llm: a partial block is applied (partial Σ) ---------------
    // A partial block (JS claim skip / cut-mapping failure / chunk cap) is a
    // valid cumulative Σ of what the server has absorbed so far. Apply it
    // (commit K_sum, prepend the block) instead of holding stage 2: the
    // server's hist_last_bytes advances only to the consumed bytes, so the
    // next refresh absorbs the remainder and completes the Σ.
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
        CHECK(res.eligible);
        CHECK(res.target_unreachable);
        CHECK(res.sum_applied);
        CHECK(res.sum_partial);
        CHECK(res.sum_fail == 0);
        CHECK(res.K_sum == 7);
        CHECK(res.K == 8);
        CHECK(!res.sum_hash.empty());
        CHECK(res.tokens_after < res.tokens_before);
        const common_json & out = body.at("messages");
        CHECK(out.size() == 7);
        CHECK(out.at(0).at("role").get<std::string>() == "system");
        const std::string u2 = out.at(1).at("content").get<std::string>();
        CHECK(u2.rfind("LLM_PARTIAL_MARKER", 0) == 0);   // partial block applied
        CHECK(mock.wait_requests(1, 2000));
        const llm_mock_request rq = mock.last();
        CHECK(rq.seen);
        CHECK(rq.session == "t-sum-llm-partial");
        CHECK(rq.cut == 7);
        CHECK(rq.wait == true);
    }

    // --- stage 2 llm: drain complete -> pending, applied only at advance (P0-1) ---
    // Request 1 commits a PARTIAL block (llm mode). The background drain absorbs
    // the remainder and must hold the complete block PENDING — not swap
    // sum_block out-of-band (it is prepended to the front of the prefix: an
    // out-of-band swap would invalidate the whole KV from there, an un-planned
    // re-prefill). A non-advance resend keeps the partial block and the same
    // rewrite_hash. The next K/K_sum advance applies the pending complete
    // block: a new tool round lifts K_cap to 13 (no new user boundary ->
    // K_sum_cand stays 7, no new fetch), and the lowered threshold gives the
    // gate reported headroom after the removal.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.push_response(R"({"block":"LLM_PARTIAL_MARKER","partial":true})");
        mock.push_response(R"({"block":"LLM_COMPLETE_MARKER","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 0;   // no inter-pass spacing (fast test)
        p.sigma_strip_threshold = 0.50;   // gate headroom after the removal
        p.sigma_strip_min_gain = 0.0;     // isolate P0-1 (the advance gain is small)
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-drain"), body, true);
        CHECK(res.eligible);
        CHECK(res.target_unreachable);
        CHECK(res.sum_applied);
        CHECK(res.sum_partial);
        CHECK(res.K_sum == 7);
        CHECK(res.K == 8);
        const std::string u2 = body.at("messages").at(1).at("content").get<std::string>();
        CHECK(u2.rfind("LLM_PARTIAL_MARKER", 0) == 0);   // partial block applied
        // request 1 = stage-2 wait:true fetch (default MAIN_LLM, no llm hint);
        // request 2 = the drain's pass (routes to the lightweight SUMMARY_LLM).
        CHECK(mock.wait_requests(2, 5000));
        const std::vector<llm_mock_request> reqs = mock.all();
        CHECK(reqs.size() >= 2);
        CHECK(reqs[0].llm.empty());
        const llm_mock_request rq = reqs[1];
        CHECK(rq.session == "t-sum-llm-drain");
        CHECK(rq.cut == 7);
        CHECK(rq.wait == false);   // the drain's pass
        CHECK(rq.llm == "summary");
        // let the drain thread store the pending block (it returns right after)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // non-advance resend: the complete Σ is PENDING — the partial block
        // stays and the rewrite is byte-identical to request 1's (same hash).
        common_json body2 = make_body(make_sum_scenario());
        const sigma_strip_result res2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-drain"), body2, true);
        CHECK(!res2.k_advanced);
        CHECK(res2.rewrite_hash == res.rewrite_hash);   // P0-1: pending not applied
        const std::string u2b = body2.at("messages").at(1).at("content").get<std::string>();
        CHECK(u2b.rfind("LLM_PARTIAL_MARKER", 0) == 0);
        // advance request: a new tool round (K_cap 8 -> 13, no new user
        // boundary -> K_sum_cand stays 7, no fetch)
        sigma_strip_note_response("t-sum-llm-drain", res.tokens_after + 1500);
        common_json msgs15 = make_sum_scenario();
        {
            common_json fn = common_json::object();
            fn["name"] = std::string("read_file");
            fn["arguments"] = std::string("{\"path\":\"/tmp/f.txt\"}");
            common_json call = common_json::object();
            call["id"] = std::string("call_5");
            call["type"] = std::string("function");
            call["function"] = fn;
            common_json tcs = common_json::array();
            tcs.push_back(call);
            common_json asst = common_json::object();
            asst["role"] = std::string("assistant");
            asst["content"] = std::string("");
            asst["tool_calls"] = tcs;
            msgs15.push_back(asst);
            common_json tool = common_json::object();
            tool["role"] = std::string("tool");
            tool["tool_call_id"] = std::string("call_5");
            tool["name"] = std::string("read_file");
            tool["content"] = std::string(100, 'x');
            msgs15.push_back(tool);
        }
        common_json body3 = make_body(msgs15);
        const sigma_strip_result res3 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-drain"), body3, true);
        CHECK(res3.k_advanced);
        CHECK(res3.K == 13);
        CHECK(res3.K_sum == 7);
        CHECK(res3.sum_applied);
        CHECK(res3.rewrite_hash != res.rewrite_hash);  // the pending complete Σ applied
        const std::string u2c = body3.at("messages").at(1).at("content").get<std::string>();
        CHECK(u2c.rfind("LLM_COMPLETE_MARKER", 0) == 0);
    }

    // --- stage 2 llm: chain advance mid-drain — one thread picks the new cut (P0-2) ---
    // Request 1 (base) commits K_sum=7 partial -> drain A spawns. Request 2
    // (extended; same [0..7] prefix, no reset) advances K_sum to 12 and
    // commits a second partial; its own spawn attempt is rejected
    // (drain_running held). The running drain A must NOT stop on the K_sum
    // move: it re-queries the cut on its next pass and absorbs the new region
    // (cut 12) — one thread, no partial left behind.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        // queue: req1 stage-2 fetch (partial), drain A pass 1 cut 7 (partial),
        // req2 stage-2 fetch (partial), drain A pass 2 cut 12 (complete).
        mock.push_response(R"({"block":"LLM_D1","partial":true})");
        mock.push_response(R"({"block":"LLM_D2","partial":true})");
        mock.push_response(R"({"block":"LLM_D3","partial":true})");
        mock.push_response(R"({"block":"LLM_D4","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 100;   // space drain passes so the flag
                                            // is held across request 2's check
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-drain-single"), body, true);
        CHECK(res.sum_partial);
        CHECK(res.K_sum == 7);
        // request 1 fetch + drain A pass 1 (the drain is now in its pass-2 sleep)
        CHECK(mock.wait_requests(2, 5000));
        // Request 2: extended scenario -> K_sum advances to 12, second partial
        // commit, drain-spawn attempt (must be blocked by the held flag).
        common_json body2 = make_body(make_sum_scenario_extended());
        const sigma_strip_result res2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-llm-drain-single"), body2, true);
        CHECK(res2.sum_partial);
        CHECK(res2.K_sum > 7);
        // drain A wakes, re-queries the cut (now 12) and absorbs the new region
        CHECK(mock.wait_requests(4, 5000));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        int drain_passes = 0, new_cut_passes = 0;
        for (const auto & r : mock.all()) {
            if (r.wait == false) {
                ++drain_passes;
                if (r.cut == res2.K_sum) ++new_cut_passes;
            }
        }
        CHECK(drain_passes == 2);        // one thread: pass 1 (cut 7) + pass 2 (cut 12)
        CHECK(new_cut_passes == 1);      // the running thread picked up the new cut
    }

    // --- P0-2 residual: K_sum moves while a COMPLETE (partial:false) pass is in flight ---
    // The drain's pass 1 fetches cut 7; the mock answers partial:false (the
    // old cut's region is "complete"). But before that response reaches the
    // drain, request 2 commits K_sum=12 (a newer region; its own drain spawn
    // is rejected because the running drain holds the flag). The drain must
    // NOT stop on the stale complete answer — the completion branch sees
    // K_sum (12) != cut (7) and continues; the next pass re-queries the new
    // cut and finishes it. The keyed mock response + a 600 ms delay on the
    // stale pass makes the race deterministic (the delay holds the drain's
    // response while request 2 lands).
    {
        llm_mock_server mock;
        CHECK(mock.start());
        // FIFO: req1 stage-2 (7, partial), req2 stage-2 (new cut, partial),
        // drain pass 2 (new cut, complete). The stale drain pass 1 (cut 7,
        // complete) is keyed + delayed so it cannot mis-map the FIFO while it
        // is in flight with request 2.
        mock.push_response(R"({"block":"P2R_D1","partial":true})", 200);
        mock.push_response(R"({"block":"P2R_D3","partial":true})", 200);
        mock.push_response(R"({"block":"P2R_D4","partial":false})", 202);
        mock.set_keyed_response(false, 7,
                R"({"block":"P2R_D2_STALE","partial":false})", 202, 600);
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 100;
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p02r"), body, true);
        CHECK(res.sum_partial);
        CHECK(res.K_sum == 7);
        // req1 stage-2 + drain pass 1 (the stale response is held 600 ms)
        CHECK(mock.wait_requests(2, 5000));
        // while the stale pass-1 response is held: request 2 advances K_sum
        // (second partial commit; its drain spawn must be rejected)
        common_json body2 = make_body(make_sum_scenario_extended());
        const sigma_strip_result res2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p02r"), body2, true);
        CHECK(res2.sum_partial);
        CHECK(res2.K_sum > 7);
        // drain pass 1 wakes: K_sum != cut (7) -> continue, not stop. Pass 2
        // re-queries the new cut and completes -> 2 drain passes, one on the
        // new cut.
        CHECK(mock.wait_requests(4, 8000));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        int drain_passes = 0, new_cut_passes = 0;
        for (const auto & r : mock.all()) {
            if (r.wait == false) {
                ++drain_passes;
                if (r.cut == res2.K_sum) ++new_cut_passes;
            }
        }
        CHECK(drain_passes == 2);        // stale pass 1 (cut 7) + pass 2 (new cut)
        CHECK(new_cut_passes == 1);      // the running thread finished the new cut
    }

    // --- P0-1 (reset): a prefix reset discards the pending Σ ----------------
    // The drain completes (pending stored). A system change resets the session
    // (K, K_sum, pending). The reset request itself re-fetches (K_sum 0 -> 7):
    // it must commit the fresh mock block, not resurrect the discarded
    // pending one.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.push_response(R"({"block":"P01R_PARTIAL","partial":true})");
        mock.push_response(R"({"block":"P01R_COMPLETE","partial":false})");
        mock.push_response(R"({"block":"P01R_POSTRESET","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 0;
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p01r"), body, true);
        CHECK(res.sum_partial);
        CHECK(res.K_sum == 7);
        CHECK(mock.wait_requests(2, 5000));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // drain stores pending
        // client compact: the system message (inside the fingerprinted
        // prefix) changes, everything else identical
        common_json msgs2 = make_sum_scenario();
        msgs2.at(0)["content"] = std::string(2400, 'c');
        common_json body2 = make_body(msgs2);
        const sigma_strip_result res2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p01r"), body2, true);
        CHECK(res2.prefix_reset);
        CHECK(res2.sum_applied);
        CHECK(res2.K_sum == 7);
        const std::string u2 = body2.at("messages").at(1).at("content").get<std::string>();
        CHECK(u2.rfind("P01R_POSTRESET", 0) == 0);
        CHECK(u2.find("P01R_COMPLETE") == std::string::npos);  // pending discarded
    }

    // --- P0-5: wait:false without explicit partial:false = in progress ------
    // The real server answers the drain's wait:false passes with 202. A
    // response that omits the boolean partial field must read as "in
    // progress" (not complete), and an empty block on the wait:false path is
    // "not rendered yet", not a failure — the drain keeps passing instead of
    // saving an unfinished block as the Σ.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.push_response(R"({"block":"P05_PARTIAL","partial":true})", 200);
        mock.push_response(R"({"block":"P05_P1","partial":true})", 202);
        // wait:false, 202, empty block, no partial field -> in progress
        mock.push_response(R"({"block":""})", 202);
        mock.push_response(R"({"block":"P05_DONE","partial":false})", 202);
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 0;
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p05"), body, true);
        CHECK(res.sum_partial);
        CHECK(res.K_sum == 7);
        // drain: pass 1 (202, partial:true) -> pass 2 (202, empty, no partial
        // field -> in progress, not a failure) -> pass 3 (complete)
        CHECK(mock.wait_requests(4, 5000));
        int drain_passes = 0;
        for (const auto & r : mock.all()) if (r.wait == false) ++drain_passes;
        CHECK(drain_passes == 3);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // the drain completed on pass 3 -> the complete block is PENDING
        // (applied at the next advance, P0-1): the partial block stays
        common_json body2 = make_body(make_sum_scenario());
        const sigma_strip_result res2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p05"), body2, true);
        CHECK(!res2.k_advanced);
        const std::string u2 = body2.at("messages").at(1).at("content").get<std::string>();
        CHECK(u2.rfind("P05_PARTIAL", 0) == 0);
    }

    // --- P0-4: the anchor fingerprint (unit test C1) -------------------------
    // The byte contract with the JS anchorNorm: role must be user, injected
    // blocks stripped, every whitespace byte dropped, the first 60 code
    // points kept (UTF-8 boundary safe).
    {
        common_json user = common_json::object();
        user["role"] = std::string("user");
        user["content"] = std::string("hello world");
        CHECK(sigma_strip_test_anchor(user) == "helloworld");
        common_json tool = common_json::object();
        tool["role"] = std::string("tool");
        tool["content"] = std::string("hello world");
        CHECK(sigma_strip_test_anchor(tool).empty());   // non-user: no anchor
        common_json tagged = common_json::object();
        tagged["role"] = std::string("user");
        tagged["content"] = std::string("<system-reminder>noise</system-reminder>abc");
        CHECK(sigma_strip_test_anchor(tagged) == "abc");   // the block is stripped
        std::string ko;
        for (int i = 0; i < 70; ++i) ko += "\xEC\xA0\x9C";   // 70 Korean cp, 3 bytes each
        common_json longko = common_json::object();
        longko["role"] = std::string("user");
        longko["content"] = ko;
        CHECK(sigma_strip_test_anchor(longko).size() == 180);   // 60 cp x 3 bytes
    }

    // --- P0-4/P1-2: the stage-2 fetch carries the anchor + the budget (C2) ---
    // The base scenario's boundary K_sum=7 is the "SECOND_MARKER" user: the
    // fingerprint is its 60 code points, the ordinal counts the user
    // messages in [0,7) (one), the budget is p.sigma_sum_max_tokens.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.set_response(R"({"block":"C2_BLOCK","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-anchor-c2"), body, true);
        CHECK(res.sum_applied);
        CHECK(res.K_sum == 7);
        CHECK(mock.wait_requests(1, 2000));
        const llm_mock_request rq = mock.last();
        CHECK(rq.wait == true);
        CHECK(rq.has_anchor);
        CHECK(rq.anchor_text == "SECOND_MARKER" + std::string(47, 'q'));   // 13 + 47 = 60 cp
        CHECK(rq.anchor_ordinal == 1);
        CHECK(rq.max_tokens == 500);   // make_sum_params: sigma_sum_max_tokens = 500
    }

    // --- P0-4: the 5a async fire sends no anchor but the budget (C3) --------
    // A stage-1 advance (no stage-2 fire): the background absorption POSTs
    // wait:false with cut = st.K. The 5a region [K_sum, K) has no boundary
    // fingerprint worth sending — the count path maps it.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        common_params p = make_params("on", 10000);
        p.sigma_strip_target = 0.55;
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        common_json body = make_body(make_messages(5, 8000));
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-anchor-c3"), body, true);
        CHECK(res.eligible);
        CHECK(res.k_advanced);
        CHECK(!res.target_unreachable);
        CHECK(!res.sum_applied);                       // stage 2 never fired
        CHECK(mock.wait_requests(1, 2000));
        const llm_mock_request rq = mock.last();
        CHECK(rq.wait == false);
        CHECK(!rq.has_anchor);                         // no anchor key at all
        CHECK(rq.max_tokens == 1500);                  // the default budget fallback
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // --- P0-5: the drain stops after 3 no-progress passes (C4) --------------
    // The server's hist_last_bytes is stuck at 100 (e.g. the anchor fails on
    // every pass and the count path is stuck): three partial passes without
    // progress stop the drain — exactly 3 passes, no 4th, the flag is
    // released (a stuck flag would reject every later drain).
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.push_response(R"({"block":"C4_PARTIAL","partial":true,"hist_last_bytes":100})");
        mock.push_response(R"({"block":"C4_P1","partial":true,"hist_last_bytes":100})");
        mock.push_response(R"({"block":"C4_P2","partial":true,"hist_last_bytes":100})");
        mock.push_response(R"({"block":"C4_P3","partial":true,"hist_last_bytes":100})");
        mock.push_response(R"({"block":"C4_P4","partial":true,"hist_last_bytes":100})");   // must go unused
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 0;
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-stall-c4"), body, true);
        CHECK(res.sum_partial);
        CHECK(res.K_sum == 7);
        // stage-2 + exactly 3 drain passes
        CHECK(mock.wait_requests(4, 5000));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        int drain_passes = 0;
        for (const auto & r : mock.all()) {
            if (r.wait == false) ++drain_passes;
        }
        CHECK(drain_passes == 3);
        CHECK(!sigma_strip_test_drain_running("t-stall-c4"));   // the flag is released
    }

    // --- P0-5: advancing hist_last_bytes resets the stall counter (C5) -----
    // 100 -> 200 -> 300 between passes: the stall counter never reaches 3
    // and the drain completes normally.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.push_response(R"({"block":"C5_PARTIAL","partial":true,"hist_last_bytes":100})");
        mock.push_response(R"({"block":"C5_P1","partial":true,"hist_last_bytes":200})");
        mock.push_response(R"({"block":"C5_P2","partial":true,"hist_last_bytes":300})");
        mock.push_response(R"({"block":"C5_DONE","partial":false,"hist_last_bytes":400})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 0;
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-prog-c5"), body, true);
        CHECK(res.sum_partial);
        CHECK(mock.wait_requests(4, 5000));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        int drain_passes = 0;
        for (const auto & r : mock.all()) {
            if (r.wait == false) ++drain_passes;
        }
        CHECK(drain_passes == 3);   // no stall stop: the drain completed
        CHECK(!sigma_strip_test_drain_running("t-prog-c5"));
    }

    // --- P0-5: an old server (no progress fields) must not stall-stop (C6) --
    // Six partial passes without hist_last_bytes, then complete: the stall
    // detector stays off (missing = -1) and the drain finishes normally.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.push_response(R"({"block":"C6_PARTIAL","partial":true})");
        for (int i = 1; i <= 5; ++i) {
            mock.push_response(std::string(R"X({"block":"C6_P)X") + std::to_string(i)
                    + R"X(","partial":true})X");
        }
        mock.push_response(R"({"block":"C6_DONE","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 0;
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-old-c6"), body, true);
        CHECK(res.sum_partial);
        // stage-2 + 6 drain passes
        CHECK(mock.wait_requests(7, 8000));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        int drain_passes = 0;
        for (const auto & r : mock.all()) {
            if (r.wait == false) ++drain_passes;
        }
        CHECK(drain_passes == 6);
        CHECK(!sigma_strip_test_drain_running("t-old-c6"));
    }

    // --- P0-4: the drain re-reads cut AND anchor when K_sum moves (C7) -----
    // The chain-advance scenario: pass 1 is cut 7 (ordinal 1, the
    // "SECOND_MARKER" user), pass 2 is cut 12 (ordinal 3, the 100-r user).
    // The anchor must move with the boundary.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.push_response(R"({"block":"C7_D1","partial":true})");
        mock.push_response(R"({"block":"C7_D2","partial":true})");
        mock.push_response(R"({"block":"C7_D3","partial":true})");
        mock.push_response(R"({"block":"C7_D4","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        p.sigma_sum_drain = true;
        p.sigma_sum_drain_delay_ms = 100;
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json b1 = make_body(make_sum_scenario());
        const sigma_strip_result res1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-anchor-c7"), b1, true);
        CHECK(res1.sum_partial);
        CHECK(res1.K_sum == 7);
        CHECK(mock.wait_requests(2, 5000));
        common_json b2 = make_body(make_sum_scenario_extended());
        const sigma_strip_result res2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-anchor-c7"), b2, true);
        CHECK(res2.sum_partial);
        CHECK(res2.K_sum == 12);
        CHECK(mock.wait_requests(4, 5000));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // all() returns a copy — take values, not addresses (the temporary
        // dies at the end of the loop).
        bool have_p1 = false;
        bool have_p2 = false;
        llm_mock_request p1;
        llm_mock_request p2;
        for (const auto & r : mock.all()) {
            if (r.wait == false) {
                if (r.cut == 7) {
                    p1 = r;
                    have_p1 = true;
                } else if (r.cut == 12) {
                    p2 = r;
                    have_p2 = true;
                }
            }
        }
        CHECK(have_p1);
        CHECK(have_p2);
        CHECK(p1.anchor_ordinal == 1);
        CHECK(p1.anchor_text == "SECOND_MARKER" + std::string(47, 'q'));
        CHECK(p2.anchor_ordinal == 3);
        CHECK(p2.anchor_text == std::string(60, 'r'));
    }

    // --- stage 2 llm: failures hold stage 2, 3 strikes -> rule fallback -----
    // An empty block is a fetch failure. While the consecutive-failure count
    // is below 3 the request passes through on stage-1 only (no removal);
    // the 4th fire (3 failures already recorded) skips the llm fetch and
    // commits the deterministic rule block directly, resetting the counter.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.set_response(R"({"block":"","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        // This test isolates the 3-strike rule: disable the 5000 ms failure
        // cooldown (now the default) so the 3 strikes land back to back. The
        // cooldown itself has its own test below.
        p.sigma_sum_fetch_cooldown_ms = 0;
        // and the futility guard (its rearm deadband would hold the back-to-
        // back retries — a flat-usage scenario the guard exists to suppress)
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
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
        CHECK(mock.wait_requests(3, 2000));
        CHECK(mock.count() == 3);                      // the 4th fire skipped the llm fetch
        const llm_mock_request rq = mock.last();
        CHECK(rq.session == "t-sum-llm-fail");
        CHECK(rq.cut == 7);
        CHECK(rq.wait == true);
    }

    // --- stage 2 llm: a failed fetch starts the failure cooldown ------------
    // The default 5000 ms cooldown holds the next attempt: the second request
    // (a few ms after the first failure) is served stage-1 only and does not
    // hit the mock again; the failure count is preserved (not reset).
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.set_response(R"({"block":"","partial":false})");   // failure
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();
        CHECK(p.sigma_sum_fetch_cooldown_ms == 5000);   // the new default
        // isolate the cooldown mechanism: disable the futility guard (its
        // rearm deadband would hold the retry before the cooldown is reached)
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        common_json b1 = make_body(make_sum_scenario());
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-cooldown"), b1, true);
        CHECK(r1.sum_fail == 1);
        CHECK(mock.count() == 1);
        common_json b2 = make_body(make_sum_scenario());
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-cooldown"), b2, true);
        CHECK(r2.sum_fail == 1);        // cooldown keeps the count (no new attempt)
        CHECK(!r2.sum_applied);
        CHECK(mock.count() == 1);       // the second request did NOT hit the mock
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

    // --- stage 2 llm: an exception in [B] releases the inflight flag -------
    // [A] sets sum_inflight before the (throwable) fetch; without the RAII
    // guard the fail-open return would leave the flag true and stage 2
    // (gated on !sum_inflight) would stall for the session. With the guard,
    // the next request fires stage 2 again and commits the fetched block.
    {
        llm_mock_server mock;
        CHECK(mock.start());
        mock.set_response(R"({"block":"FAULT_RELEASE_BLOCK","partial":false})");
        common_params p = make_sum_params("on");
        p.sigma_sum_mode = "llm";
        p.focus_memory_host = mock.url();

        sigma_strip_test_set_fetch_fault(true);
        {
            common_json body = make_body(make_sum_scenario());
            const sigma_strip_result r =
                sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-fault"), body, true);
            CHECK(!r.eligible);                         // fail-open: untouched result
            CHECK(body.at("messages").size() == 13);    // body untouched
            CHECK(mock.count() == 0);                   // the throw precedes the HTTP send
        }
        sigma_strip_test_set_fetch_fault(false);

        // the flag was released: stage 2 fires on the next request
        common_json body = make_body(make_sum_scenario());
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-sum-fault"), body, true);
        CHECK(r2.eligible);
        CHECK(r2.sum_applied);
        CHECK(r2.K_sum == 7);
        CHECK(r2.sum_fail == 0);
        CHECK(mock.wait_requests(1, 2000));
        CHECK(mock.count() == 1);                       // only the 2nd request fetched
        const llm_mock_request rq = mock.last();
        CHECK(rq.seen);
        CHECK(rq.session == "t-sum-fault");
        CHECK(rq.cut == 7);
        CHECK(rq.wait == true);
    }

    // --- futility guard: min-gain skip (guard on) vs advance (guard off) ---
    // The advance's expected gain (raw - floor) is far below min_gain 0.99 x
    // window: skip=1, K stays 0, body untouched (fail-open). The same request
    // with the gain gate disabled advances to the floor.
    {
        common_params p = make_params("shadow", 10000);
        p.sigma_strip_min_keep_tokens = 7000;
        p.sigma_strip_keep_recent = 1;
        p.sigma_strip_min_gain = 0.99;
        p.sigma_strip_futility_bypass_usage = 0.0;   // isolate the gain gate
        common_json body = make_body(make_guard_scenario());
        const sigma_strip_result res =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-gain"), body, true);
        CHECK(res.eligible);
        CHECK(res.skip == 1);
        CHECK(!res.k_advanced);
        CHECK(res.K == 0);
        CHECK(body.at("messages").size() == 26);

        common_params p2 = make_params("shadow", 10000);
        p2.sigma_strip_min_keep_tokens = 7000;
        p2.sigma_strip_keep_recent = 1;
        p2.sigma_strip_min_gain = 0.0;
        common_json body2 = make_body(make_guard_scenario());
        const sigma_strip_result res2 =
            sigma_strip_maybe_rewrite(nullptr, p2, make_req("t-gain2"), body2, true);
        CHECK(res2.k_advanced);
        CHECK(res2.skip == 0);
        CHECK(res2.tokens_after <= 7000);
    }

    // --- futility guard: rearm deadband, growth re-advances ---------------
    // r1 advances to the floor (anchor = usage_after). r2 re-crosses the
    // threshold (reported +400 over the original estimate -> est offset 400;
    // shadow: last_est is tokens_before, P0-4) but stays inside the rearm
    // deadband (usage_before < anchor + 0.05): skip=2, K held. A larger
    // reported growth (offset 560) leaves the deadband and the advance lands.
    // (560, not 500: at the exact boundary the double rounding of
    // anchor + 0.05 can still read as larger, and the skip side is the
    // fail-safe one.)
    {
        common_params p = make_params("shadow", 10000);
        p.sigma_strip_min_keep_tokens = 7000;
        p.sigma_strip_keep_recent = 1;
        common_json body = make_body(make_guard_scenario());
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-rearm"), body, true);
        CHECK(r1.k_advanced);
        CHECK(r1.tokens_after >= 6600);   // precondition for the offsets below
        CHECK(r1.tokens_after <= 7000);

        sigma_strip_note_response("t-rearm", r1.tokens_before + 400);
        common_json body2 = make_body(make_guard_scenario());
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-rearm"), body2, true);
        CHECK(r2.skip == 2);
        CHECK(!r2.k_advanced);
        CHECK(r2.K == r1.K);
        CHECK(r2.tokens_after == r1.tokens_after);
        CHECK(r2.usage_before >= 0.70);   // the gate fired; the deadband held
        CHECK(r2.usage_before < r1.usage_after + 0.05);

        sigma_strip_note_response("t-rearm", r2.tokens_before + 560);
        common_json body3 = make_body(make_guard_scenario());
        const sigma_strip_result r3 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-rearm"), body3, true);
        CHECK(r3.k_advanced);
        CHECK(r3.skip == 0);
        CHECK(r3.K == r1.K + 2);          // one asst+tool round past the floor
        CHECK(r3.tokens_after < r2.tokens_after);
    }

    // --- futility guard: a prefix reset voids the rearm anchor ------------
    // After a rearm skip, a changed prefix (client compact) resets K and the
    // anchor: the next request advances again despite the flat usage.
    {
        common_params p = make_params("shadow", 10000);
        p.sigma_strip_min_keep_tokens = 7000;
        p.sigma_strip_keep_recent = 1;
        common_json body = make_body(make_guard_scenario());
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-rearm-reset"), body, true);
        CHECK(r1.k_advanced);
        CHECK(r1.tokens_after >= 6600);

        sigma_strip_note_response("t-rearm-reset", r1.tokens_before + 400);
        common_json body2 = make_body(make_guard_scenario());
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-rearm-reset"), body2, true);
        CHECK(r2.skip == 2);
        CHECK(!r2.k_advanced);

        // client compact: the system message (inside the fingerprinted
        // prefix) changes, everything else identical
        common_json msgs3 = make_guard_scenario();
        msgs3.at(0)["content"] = std::string(2400, 'c');
        common_json body3 = make_body(msgs3);
        const sigma_strip_result r3 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-rearm-reset"), body3, true);
        CHECK(r3.prefix_reset);
        CHECK(r3.k_advanced);
        CHECK(r3.K == r1.K);
        CHECK(r3.tokens_after == r1.tokens_after);
    }

    // --- P0-3: futility guard hard-limit bypass ------------------------------
    // min-gain would skip a ~3.8k gain (far below 0.99 x window). Below the
    // bypass usage (0.90) the skip stands; once the reported usage reaches
    // the hard limit the bypass voids the guard — a small re-prefill beats
    // riding usage to 1.0 and the multi-minute native compact.
    {
        common_params p = make_params("shadow", 14000);
        p.sigma_strip_min_keep_tokens = 7000;
        p.sigma_strip_keep_recent = 1;
        p.sigma_strip_min_gain = 0.99;
        common_json b1 = make_body(make_guard_scenario());
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p03"), b1, true);
        CHECK(r1.usage_before >= 0.70);
        CHECK(r1.usage_before < 0.90);
        CHECK(r1.skip == 1);
        CHECK(!r1.k_advanced);
        CHECK(r1.K == 0);
        CHECK(!r1.bypass);
        // the reported usage pushed over the hard limit (same session:
        // note_response only calibrates sessions that went through the gate)
        sigma_strip_note_response("t-p03", r1.tokens_before + 2500);
        common_json b2 = make_body(make_guard_scenario());
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p03"), b2, true);
        CHECK(r2.usage_before >= 0.90);
        CHECK(r2.k_advanced);
        CHECK(r2.skip == 0);
        CHECK(r2.bypass);
        CHECK(r2.tokens_after <= 7000);
    }

    // --- P0-4: shadow last_est is the original (tokens_before) --------------
    // Shadow mode never rewrites the body, so the reported prompt count is
    // original-based: the offset must not absorb the strip savings (the old
    // code anchored on tokens_after and over-counted by the full savings).
    {
        common_params p = make_params("shadow", 10000);
        p.sigma_strip_min_keep_tokens = 7000;
        p.sigma_strip_keep_recent = 1;
        p.sigma_strip_min_gain = 0.0;
        p.sigma_strip_rearm_delta = 0.0;
        p.sigma_strip_futility_bypass_usage = 0.0;
        common_json b1 = make_body(make_guard_scenario());
        const sigma_strip_result r1 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p04"), b1, true);
        CHECK(r1.k_advanced);
        CHECK(r1.K > 0);
        CHECK(r1.tokens_after < r1.tokens_before);
        sigma_strip_note_response("t-p04", r1.tokens_before + 3000);
        common_json b2 = make_body(make_guard_scenario());
        const sigma_strip_result r2 =
            sigma_strip_maybe_rewrite(nullptr, p, make_req("t-p04"), b2, true);
        CHECK(r2.est_offset == 3000);   // not 3000 + savings (clamped to window/2)
    }

    if (n_failed) {
        std::printf("test-sigma-strip: %d check(s) FAILED\n", n_failed);
        return 1;
    }
    std::printf("test-sigma-strip: all checks passed\n");
    return 0;
}
