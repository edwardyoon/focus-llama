// Standalone test for the DA tag-start rule (S1: line-start only).
//
// The logic under test is da_tag_start_allowed() in
// tools/server/server-context.cpp - the shared start rule used by
// scan_da_tag / da_tag_hold_len / da_tag_inflight. S1 tightens the two
// ENTRY tags (the focus and the local opener) to "beginning of text or
// right after a newline"; the three RETURN (close) tags and the global
// tag keep the legacy rule (after any whitespace, plus the glued-return
// exception in a restricted mode - P3).
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
    if (lt == 0) {
        return true;
    }
    const unsigned char prev = (unsigned char) text[lt - 1];
    // entry tags: line-start only (S1)
    if (text.compare(lt, 4, "<loc") == 0 || text.compare(lt, 4, "<foc") == 0) {
        return prev == '\n';
    }
    if (std::isspace(prev)) {
        return true;
    }
    // glued return, restricted mode only (P3)
    return (mode != DA_MODE_GLOBAL) &&
           (text.compare(lt, 5, "</foc") == 0 ||
            text.compare(lt, 5, "</loc") == 0 ||
            text.compare(lt, 5, "</glo") == 0);
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

    // ── RETURN tags: legacy rule (any whitespace + glued in restricted) ──
    check("close-focus after space GLOBAL", std::string("abc ") + C_FOCUS, C_FOCUS, DA_MODE_GLOBAL, true);
    check("close-focus after space LOCAL", std::string("abc ") + C_FOCUS, C_FOCUS, DA_MODE_LOCAL, true);
    check("close-focus glued LOCAL (P3)", std::string("CODE") + C_FOCUS, C_FOCUS, DA_MODE_LOCAL, true);
    check("close-focus glued GLOBAL", std::string("CODE") + C_FOCUS, C_FOCUS, DA_MODE_GLOBAL, false);
    check("close-local glued FOCUS (P3)", std::string("CODE") + C_LOCAL, C_LOCAL, DA_MODE_FOCUS, true);
    check("close-global glued LOCAL (P3)", std::string("CODE") + C_GLOBAL, C_GLOBAL, DA_MODE_LOCAL, true);
    check("close-global glued GLOBAL", std::string("CODE") + C_GLOBAL, C_GLOBAL, DA_MODE_GLOBAL, false);
    check("close-focus @ text start", C_FOCUS, C_FOCUS, DA_MODE_GLOBAL, true);

    // ── global tag: legacy rule (any whitespace, no glued exception) ──
    check("global after space", std::string("abc ") + T_GLOBAL, T_GLOBAL, DA_MODE_GLOBAL, true);
    check("global after newline", std::string("abc\n") + T_GLOBAL, T_GLOBAL, DA_MODE_GLOBAL, true);
    check("global glued (no exception)", std::string("CODE") + T_GLOBAL, T_GLOBAL, DA_MODE_LOCAL, false);

    if (failures) {
        std::printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("\nALL PASS (%d cases)\n", 23);
    return 0;
}
