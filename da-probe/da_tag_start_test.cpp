// Standalone test for the DA tag-start rule (S1 + S1-CLOSE: line-start
// only, all six tags).
//
// The logic under test is da_tag_start_allowed() in
// tools/server/server-context.cpp - the shared start rule used by
// scan_da_tag / da_tag_hold_len / da_tag_inflight. S1 tightened the two
// ENTRY tags (the focus and the local opener) to "beginning of text or
// right after a newline"; S1-CLOSE extends that to the three RETURN
// (close) tags and the global tag, dropping the legacy "after any
// whitespace" rule and the P3 glued-return exception. Every tag now
// starts only at the beginning of the text or right after a newline,
// and the verdict no longer depends on the DA mode.
//
// The tags are assembled from parts at runtime: full DA tag tokens must
// not appear inline in this file (a DA session's scanner consumes them
// from the model's output stream, and editor tooling splits them across
// physical lines).
//
// Build & run (seconds, no server needed):
//   g++ -O2 -std=c++17 -o /tmp/da_tag_start_test da_tag_start_test.cpp \
//       && /tmp/da_tag_start_test
//
// Exit code 0 = all cases pass.

#include <cctype>
#include <cstdio>
#include <string>

enum da_mode_t { DA_MODE_GLOBAL, DA_MODE_FOCUS, DA_MODE_LOCAL };

static const std::string T_LOCAL  = std::string("<") + "local" + ">";
static const std::string T_GLOBAL = std::string("<") + "global" + ">";
static const std::string C_FOCUS  = std::string("<") + "/focus" + ">";
static const std::string C_LOCAL  = std::string("<") + "/local" + ">";
static const std::string C_GLOBAL = std::string("<") + "/global" + ">";
static const std::string T_FOCUS  = std::string("<") + "focus magic_chunks=\"1\"";

// The start rule (must stay in sync with da_tag_start_allowed in
// server-context.cpp).
static bool da_tag_start_allowed(const std::string & text, size_t lt, da_mode_t mode) {
    (void) mode;  // the start rule no longer depends on the mode (S1-CLOSE)
    if (lt == 0) {
        return true;
    }
    return (unsigned char) text[lt - 1] == '\n';
}

static int failures = 0;

/**
 * Check one (text, tag-position, mode) case against the expected verdict.
 * @param {const char *} label
 * @param {const std::string &} text - text containing the tag at lt
 * @param {const std::string &} tag - the tag placed at lt (for locating)
 * @param {da_mode_t} mode
 * @param {bool} expect - expected da_tag_start_allowed() result
 * @returns {void}
 */
static void check(const char * label, const std::string & text, const std::string & tag,
                  da_mode_t mode, bool expect) {
    const size_t lt = text.find(tag);
    if (lt == std::string::npos) {
        std::printf("FAIL %-46s tag not found in text\n", label);
        failures++;
        return;
    }
    const bool got = da_tag_start_allowed(text, lt, mode);
    if (got != expect) {
        std::printf("FAIL %-46s expected %d got %d\n", label, (int) expect, (int) got);
        failures++;
    } else {
        std::printf("ok   %s\n", label);
    }
}

int main() {
    // ── ENTRY tags: line-start only (S1) ────────────────────────────
    check("local @ text start", T_LOCAL, T_LOCAL, DA_MODE_GLOBAL, true);
    check("local after newline", std::string("abc\n") + T_LOCAL, T_LOCAL, DA_MODE_GLOBAL, true);
    check("local after space (mid-line)", std::string("abc ") + T_LOCAL, T_LOCAL, DA_MODE_GLOBAL, false);
    check("local after tab", std::string("abc\t") + T_LOCAL, T_LOCAL, DA_MODE_GLOBAL, false);
    check("local backtick-quoted", std::string("use `") + T_LOCAL + "` in the", T_LOCAL, DA_MODE_GLOBAL, false);
    check("local in JSON string", std::string("{\"tag\": \"") + T_LOCAL + "\"}", T_LOCAL, DA_MODE_GLOBAL, false);
    check("focus @ text start", T_FOCUS + ">", T_FOCUS, DA_MODE_GLOBAL, true);
    check("focus after newline", std::string("abc\n") + T_FOCUS + ">", T_FOCUS, DA_MODE_GLOBAL, true);
    check("focus after space (mid-line)", std::string("abc ") + T_FOCUS + ">", T_FOCUS, DA_MODE_GLOBAL, false);
    check("focus partial after space", std::string("abc <foc"), std::string("<foc"), DA_MODE_GLOBAL, false);
    check("local partial after newline", std::string("abc\n<loc"), std::string("<loc"), DA_MODE_GLOBAL, true);
    check("local after newline in LOCAL mode", std::string("abc\n") + T_LOCAL, T_LOCAL, DA_MODE_LOCAL, true);

    // ── RETURN (close) tags: line-start only (S1-CLOSE) ─────────────
    // The three close tags now follow the same rule as the entry tags:
    // beginning of text or right after a newline. Mid-line (after a
    // space/tab), glued to the answer, and backtick-quoted instances are
    // data, not control tags - regardless of the DA mode.
    check("close-focus @ text start", C_FOCUS, C_FOCUS, DA_MODE_GLOBAL, true);
    check("close-focus after newline", std::string("abc\n") + C_FOCUS, C_FOCUS, DA_MODE_GLOBAL, true);
    check("close-focus after space (mid-line)", std::string("abc ") + C_FOCUS, C_FOCUS, DA_MODE_GLOBAL, false);
    check("close-focus after tab (mid-line)", std::string("abc\t") + C_FOCUS, C_FOCUS, DA_MODE_GLOBAL, false);
    check("close-focus glued to answer", std::string("CODE") + C_FOCUS, C_FOCUS, DA_MODE_LOCAL, false);
    check("close-focus backtick-quoted", std::string("close `") + C_FOCUS + "` now", C_FOCUS, DA_MODE_FOCUS, false);
    check("close-local @ text start", C_LOCAL, C_LOCAL, DA_MODE_GLOBAL, true);
    check("close-local after newline", std::string("abc\n") + C_LOCAL, C_LOCAL, DA_MODE_GLOBAL, true);
    check("close-local after space (mid-line)", std::string("abc ") + C_LOCAL, C_LOCAL, DA_MODE_GLOBAL, false);
    check("close-local glued to answer", std::string("CODE") + C_LOCAL, C_LOCAL, DA_MODE_FOCUS, false);
    check("close-global @ text start", C_GLOBAL, C_GLOBAL, DA_MODE_GLOBAL, true);
    check("close-global after newline", std::string("abc\n") + C_GLOBAL, C_GLOBAL, DA_MODE_GLOBAL, true);
    check("close-global after space (mid-line)", std::string("abc ") + C_GLOBAL, C_GLOBAL, DA_MODE_GLOBAL, false);
    check("close-global glued to answer", std::string("CODE") + C_GLOBAL, C_GLOBAL, DA_MODE_LOCAL, false);

    // ── global tag: line-start only (S1-CLOSE) ───────────────────────
    check("global @ text start", T_GLOBAL, T_GLOBAL, DA_MODE_GLOBAL, true);
    check("global after newline", std::string("abc\n") + T_GLOBAL, T_GLOBAL, DA_MODE_GLOBAL, true);
    check("global after space (mid-line)", std::string("abc ") + T_GLOBAL, T_GLOBAL, DA_MODE_GLOBAL, false);
    check("global glued to answer", std::string("CODE") + T_GLOBAL, T_GLOBAL, DA_MODE_LOCAL, false);
    check("global backtick-quoted", std::string("use `") + T_GLOBAL + "` mode", T_GLOBAL, DA_MODE_GLOBAL, false);

    // ── mode independence (S1-CLOSE dropped the mode dependence) ─────
    check("close-focus after newline in FOCUS", std::string("abc\n") + C_FOCUS, C_FOCUS, DA_MODE_FOCUS, true);
    check("close-local glued in LOCAL (no P3)", std::string("CODE") + C_LOCAL, C_LOCAL, DA_MODE_LOCAL, false);
    check("global after space in LOCAL", std::string("abc ") + T_GLOBAL, T_GLOBAL, DA_MODE_LOCAL, false);

    if (failures) {
        std::printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("\nALL PASS (%d cases)\n", 34);
    return 0;
}
