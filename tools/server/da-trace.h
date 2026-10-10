#pragma once

// Standalone trace file for the kv-offload / DA refill process.
// (2026-10-06 14:21 incident: a <focus magic_chunks="N"> tag closed
// mid-tool-call, the tail re-prefill split the model's own tag across the
// recalled block, and the continuation leaked a headless tool-call tail).
// Everything about the refill is logged here - NOT to the journal - so the
// timeline can be read from the file alone, without the session record or
// the offloaded chunk contents. One line per event:
//   <ISO8601>.<us> tid=<hex> | <message>
// Path: env FOCUS_DA_TRACE_LOG, default /tmp/da-trace.log. Append + flush
// per line; the volume is a few lines per request. Tracing failures are
// silent - they must never break the request path.
//
// Split out of server-context.cpp (2026-10-18): the functions were static
// there; they are inline now. The ofstream is a persistent static (one open
// for the process lifetime) guarded by a mutex so concurrent callers from
// different decode threads cannot interleave mid-line.

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <mutex>
#include <string>
#include <thread>

inline const std::string & da_trace_path() {
    static const std::string path = []() {
        const char * env = getenv("FOCUS_DA_TRACE_LOG");
        if (env && *env) return std::string(env);
        return std::string("/tmp/da-trace.log");
    }();
    return path;
}

// fmt must carry a trailing newline.
inline void da_trace(const char * fmt, ...) {
    char msg[16384];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    const auto now = std::chrono::system_clock::now();
    const auto us  = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", &tm_buf);

    static std::mutex mtx;
    static std::ofstream f(da_trace_path(), std::ios::app);
    if (!f) return;
    std::lock_guard<std::mutex> lk(mtx);
    f << tbuf << '.' << std::setw(6) << std::setfill('0') << us
      << " tid=" << std::hex << std::hash<std::thread::id>{}(std::this_thread::get_id()) << std::dec
      << " | " << msg;
    f.flush();
}

// escape for one-line da_trace output (newlines made visible, control
// chars replaced, truncated with "...")
inline std::string dbg_esc(const std::string & in, size_t max = 400) {
    std::string o;
    o.reserve(std::min(in.size(), max) + 8);
    for (unsigned char c : in) {
        if (c == '\n')      o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c < 0x20)  o += '?';
        else                o += (char) c;
        if (o.size() >= max) { o += "..."; break; }
    }
    return o;
}

// number of non-overlapping occurrences of needle in s[0, end)
inline int dbg_count(const std::string & s, const std::string & needle, size_t end) {
    int n = 0;
    for (size_t p = s.find(needle); p != std::string::npos && p < end; p = s.find(needle, p + needle.size())) n++;
    return n;
}
