// Unit tests for the DA (decoupled-KV) control-tag parser (da-tag.h).
// Plain-assert style, matching the project's test convention.
//
// Regression focus: the hold logic (da_tag_prefix / da_tag_hold_len) must
// never release a partially emitted tag - the 09-25 19:29 (task 5952) bug
// released the hold at the comma, streaming the tag head to the client
// before the complete tag could be erased.

#include "da-tag.h"

#include <cassert>
#include <cstdio>
#include <string>

static int n_failed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_failed++; \
    } \
} while (0)

int main() {
    const da_mode_t G = DA_MODE_GLOBAL;

    // --- scan_da_tag: the six tag types at a line start -----------------
    {
        const std::string s = "<focus magic_chunks=\"1\">";
        da_tag_t t = scan_da_tag(s, 0, G);
        CHECK(t.type == 0);
        CHECK(t.keep_nums.size() == 1 && t.keep_nums[0] == 1);
        CHECK(t.start == 0);
        CHECK(t.end == s.size());
    }
    {
        da_tag_t t = scan_da_tag("<local>", 0, G);
        CHECK(t.type == 1);
        CHECK(t.end == 7);
    }
    {
        da_tag_t t = scan_da_tag("</focus>", 0, DA_MODE_FOCUS);
        CHECK(t.type == 2);
        CHECK(t.end == 8);
    }
    {
        da_tag_t t = scan_da_tag("</local>", 0, DA_MODE_LOCAL);
        CHECK(t.type == 3);
        CHECK(t.end == 8);
    }
    {
        da_tag_t t = scan_da_tag("<global>", 0, G);
        CHECK(t.type == 4);
        CHECK(t.end == 8);
    }
    {
        da_tag_t t = scan_da_tag("</global>", 0, G);
        CHECK(t.type == 5);
        CHECK(t.end == 9);
    }

    // --- scan_da_tag: multi-chunk keep list ------------------------------
    {
        da_tag_t t = scan_da_tag("<focus magic_chunks=\"12,13\">", 0, G);
        CHECK(t.type == 0);
        CHECK(t.keep_nums.size() == 2 && t.keep_nums[0] == 12 && t.keep_nums[1] == 13);
    }

    // --- scan_da_tag: quoted/embedded mid-line tags are data, not tags ---
    {
        da_tag_t t = scan_da_tag("answer: use <local> here\n", 0, G);
        CHECK(t.type < 0);
    }
    {
        // backtick-quoted close mid-line - the 345642 (09-24) case
        da_tag_t t = scan_da_tag("done `</focus>` now\n", 0, DA_MODE_FOCUS);
        CHECK(t.type < 0);
    }
    {
        // a tag on its own line after text IS a tag
        const std::string s = "text\n</focus>";
        da_tag_t t = scan_da_tag(s, 0, DA_MODE_FOCUS);
        CHECK(t.type == 2);
        CHECK(t.start == 5);
    }

    // --- scan_da_tag: from-offset skips earlier text ---------------------
    {
        da_tag_t t = scan_da_tag("<local>\nmore</focus>", 10, G);
        CHECK(t.type < 0);  // the </focus> is mid-line
    }
    {
        da_tag_t t = scan_da_tag("<local>\nmore\n</focus>", 10, DA_MODE_FOCUS);
        CHECK(t.type == 2);
        CHECK(t.start == 13);
    }

    // --- token-at-a-time feeding: the hold never releases early ----------
    {
        // feed <focus magic_chunks="12,13"> one char at a time; the hold
        // must cover the whole fragment while the tag is incomplete and
        // drop to 0 only once it is complete (and thus consumable).
        const std::string full = "<focus magic_chunks=\"12,13\">";
        for (size_t n = 1; n < full.size(); n++) {
            const std::string frag = full.substr(0, n);
            const size_t hold = da_tag_hold_len(frag, G);
            CHECK(hold == frag.size());
        }
        CHECK(da_tag_hold_len(full, G) == 0);  // complete: consumed, not held
    }

    // --- no-quote variant: the quote is optional in the hold logic -------
    {
        CHECK(da_tag_prefix("<focus magic_chunks=12") == true);
        CHECK(da_tag_prefix("<focus magic_chunks=12>") == false);  // complete
    }

    // --- whitespace / indentation: not a tag start ------------------------
    {
        CHECK(da_tag_hold_len("  <local>", G) == 0);        // indented
        CHECK(da_tag_hold_len("text <local>", G) == 0);     // mid-line
        CHECK(da_tag_inflight("text <local>", G) == false);
        CHECK(da_tag_hold_len("text\n  <local>", G) == 0);  // indented line
        CHECK(da_tag_hold_len("text\n<local>", G) == 0);    // complete: consumed
        CHECK(da_tag_hold_len("text\n<loc", G) > 0);        // partial at line start
    }

    // --- da_tag_inflight mirrors the hold ---------------------------------
    {
        CHECK(da_tag_inflight("thinking\n<focus magic_chunks=\"1\"", G) == true);
        CHECK(da_tag_inflight("thinking\n<focus magic_chunks=\"1\">", G) == false);
        CHECK(da_tag_inflight("no tags here", G) == false);
    }

    // --- da_tag_start_allowed ----------------------------------------------
    {
        CHECK(da_tag_start_allowed("abc", 0, G) == true);
        CHECK(da_tag_start_allowed("ab\nc", 3, G) == true);
        CHECK(da_tag_start_allowed("ab c", 3, G) == false);
        CHECK(da_tag_start_allowed("ab\tc", 3, G) == false);  // tab is not a line start
    }

    if (n_failed) {
        std::printf("test-da-tag: %d check(s) FAILED\n", n_failed);
        return 1;
    }
    std::printf("test-da-tag: all checks passed\n");
    return 0;
}
