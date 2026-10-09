// Standalone test for the DA tail floor (plans/continous_work2.md).
//
// The logic under test is da_apply_keep_floor() + da_ranges_total() in
// tools/server/server-context.cpp: next to the Sigma anchor subtraction,
// the most recent K tokens before the removal boundary [bound-K, bound)
// are always kept (subtracted from the removal ranges via
// da_subtract_range). The keep set is the union of the tag selection, the
// Sigma anchor and the tail, so the work region the model was just in
// (not covered by any magic_chunks tag) survives a B switch.
//
// The pure range logic is duplicated below (same as da_placement_test.cpp
// and da_tag_start_test.cpp) - keep in sync with server-context.cpp.
//
// Build & run (seconds, no server needed):
//   g++ -O2 -std=c++17 -o /tmp/da_tail_floor_test da_tail_floor_test.cpp \
//       && /tmp/da_tail_floor_test
//
// Exit code 0 = all cases pass.

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

using ranges_t = std::vector<std::pair<int32_t, int32_t>>;

// ── duplicated logic (must stay in sync with server-context.cpp) ────────

static void da_subtract_range(ranges_t & ranges, int32_t sub_lo, int32_t sub_hi) {
    if (sub_hi <= sub_lo) {
        return;
    }
    ranges_t out;
    for (const auto & r : ranges) {
        const int32_t lo = r.first;
        const int32_t hi = r.second;
        if (hi <= sub_lo || lo >= sub_hi) {
            out.emplace_back(lo, hi);  // no overlap
            continue;
        }
        if (lo < sub_lo) {
            out.emplace_back(lo, sub_lo);  // left part
        }
        if (hi > sub_hi) {
            out.emplace_back(sub_hi, hi);  // right part
        }
    }
    ranges = std::move(out);
}

static size_t da_ranges_total(const ranges_t & ranges) {
    size_t total = 0;
    for (const auto & r : ranges) {
        total += (size_t) (r.second - r.first);
    }
    return total;
}

// Core of da_apply_keep_floor (the pure part): returns the newly-kept
// token count. K <= 0 or bound <= 0: no-op (legacy behavior).
static size_t da_apply_keep_floor(ranges_t & rm_ranges, int32_t bound, int32_t K) {
    if (K <= 0 || bound <= 0) {
        return 0;
    }
    const int32_t lo = std::max(bound - K, 0);
    const int32_t hi = bound;
    const size_t before = da_ranges_total(rm_ranges);
    da_subtract_range(rm_ranges, lo, hi);
    return before - da_ranges_total(rm_ranges);
}

// Bucketed remainder (distance of the range end from the tail), as logged
// by da_apply_keep_floor.
static std::vector<size_t> da_removed_buckets(const ranges_t & rm_ranges, int32_t bound) {
    const int32_t edges[] = { 1024, 4096, 8192, 16384, 32768 };
    std::vector<size_t> bkt(6, 0);
    for (const auto & r : rm_ranges) {
        int32_t dist = bound - r.second;
        if (dist < 0) {
            dist = 0;
        }
        int b = 0;
        while (b < 5 && dist >= edges[b]) {
            ++b;
        }
        bkt[b] += (size_t) (r.second - r.first);
    }
    return bkt;
}

// ── checks ───────────────────────────────────────────────────────────────

static int failures = 0;

static void check(const char * name, bool ok, const char * detail = nullptr) {
    if (!ok) {
        failures++;
    }
    printf("%-32s %s%s\n", name, ok ? "PASS" : "FAIL", (ok || !detail) ? "" : detail);
}

static bool same(const ranges_t & a, const ranges_t & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].first != b[i].first || a[i].second != b[i].second) {
            return false;
        }
    }
    return true;
}

int main() {
    // 1. plan case: tail and Sigma overlap.
    // rm [1000,9500), Sigma [5000,6000) subtracted first, bound=10000, K=4000.
    // after Sigma: [1000,5000) [6000,9500); floor [6000,10000) -> [1000,5000).
    {
        ranges_t rm = { {1000, 9500} };
        da_subtract_range(rm, 5000, 6000);  // Sigma, as apply_da_rm/apply_da_b do
        const size_t kept = da_apply_keep_floor(rm, 10000, 4000);
        check("1. tail+sigma overlap", kept == 3500 && same(rm, ranges_t{ {1000, 5000} }),
              "expect kept=3500 rm=[1000,5000)");
    }

    // 2. plan case: bound < K. bound=3000, K=8192 -> floor [0,3000) keeps
    // everything.
    {
        ranges_t rm = { {0, 3000} };
        const size_t kept = da_apply_keep_floor(rm, 3000, 8192);
        check("2. bound < K", kept == 3000 && rm.empty(), "expect kept=3000 rm=empty");
    }

    // 3. plan case: range already kept (disjoint from the floor).
    // rm [100,200), bound=10000, K=8192 -> floor [1808,10000), no overlap.
    {
        ranges_t rm = { {100, 200} };
        const size_t kept = da_apply_keep_floor(rm, 10000, 8192);
        check("3. range already kept", kept == 0 && same(rm, ranges_t{ {100, 200} }),
              "expect kept=0 rm unchanged");
    }

    // 4. K=0 -> legacy no-op.
    {
        ranges_t rm = { {0, 9000} };
        const size_t kept = da_apply_keep_floor(rm, 10000, 0);
        check("4. K=0 legacy no-op", kept == 0 && same(rm, ranges_t{ {0, 9000} }),
              "expect kept=0 rm unchanged");
    }

    // 5. range straddling the floor start. rm [5000,9000), bound=10000,
    // K=4000 -> floor [6000,10000) -> [5000,6000).
    {
        ranges_t rm = { {5000, 9000} };
        const size_t kept = da_apply_keep_floor(rm, 10000, 4000);
        check("5. straddling floor start", kept == 3000 && same(rm, ranges_t{ {5000, 6000} }),
              "expect kept=3000 rm=[5000,6000)");
    }

    // 6. range fully inside the floor. rm [7000,8000), bound=10000, K=4000
    // -> floor [6000,10000) -> empty.
    {
        ranges_t rm = { {7000, 8000} };
        const size_t kept = da_apply_keep_floor(rm, 10000, 4000);
        check("6. fully inside floor", kept == 1000 && rm.empty(), "expect kept=1000 rm=empty");
    }

    // 7. multiple ranges: one removed, one partially, one untouched.
    // rm [100,200) [5000,9000) [7000,8000) (note: the last two overlap;
    // call sites see near-merged ranges, the total is per-range)
    // bound=10000, K=4000 -> floor [6000,10000):
    //   [100,200) untouched; [5000,9000) -> [5000,6000); [7000,8000) -> gone.
    {
        ranges_t rm = { {100, 200}, {5000, 9000}, {7000, 8000} };
        const size_t kept = da_apply_keep_floor(rm, 10000, 4000);
        check("7. multi-range", kept == 4000 && same(rm, ranges_t{ {100, 200}, {5000, 6000} }),
              "expect kept=4000 rm=[100,200) [5000,6000)");
    }

    // 8. buckets: remainder [100,200) [5000,6000) at bound=10000.
    //   [100,200): dist=9800 -> 8-16k; [5000,6000): dist=4000 -> 1-4k
    //   (4000 < 4096, the first bucket edge).
    {
        ranges_t rm = { {100, 200}, {5000, 6000} };
        const auto bkt = da_removed_buckets(rm, 10000);
        const bool ok = bkt[0] == 0 && bkt[1] == 1000 && bkt[2] == 0 && bkt[3] == 100
                        && bkt[4] == 0 && bkt[5] == 0;
        char detail[128];
        snprintf(detail, sizeof detail, "got <1k=%zu 1-4k=%zu 4-8k=%zu 8-16k=%zu 16-32k=%zu 32k+=%zu",
                 bkt[0], bkt[1], bkt[2], bkt[3], bkt[4], bkt[5]);
        check("8. distance buckets", ok, detail);
    }

    // 9. buckets: range touching the tail (dist=0) and one beyond 32k.
    //   [9000,10000): dist=0 -> <1k (1000); [0,1000): dist=9000 -> 8-16k.
    {
        ranges_t rm = { {9000, 10000}, {0, 1000} };
        const auto bkt = da_removed_buckets(rm, 10000);
        const bool ok = bkt[0] == 1000 && bkt[1] == 0 && bkt[2] == 0 && bkt[3] == 1000
                        && bkt[4] == 0 && bkt[5] == 0;
        char detail[128];
        snprintf(detail, sizeof detail, "got <1k=%zu 1-4k=%zu 4-8k=%zu 8-16k=%zu 16-32k=%zu 32k+=%zu",
                 bkt[0], bkt[1], bkt[2], bkt[3], bkt[4], bkt[5]);
        check("9. buckets edge distances", ok, detail);
    }

    // 10. the 10-09 12:43 amnesia shape: bound=63370, K=8192, the masked
    // region [42060,63370) as the removal. The floor [55178,63370) must keep
    // the last 8192 tokens of it.
    {
        ranges_t rm = { {42060, 63370} };
        const size_t kept = da_apply_keep_floor(rm, 63370, 8192);
        check("10. 12:43 amnesia shape", kept == 8192 && same(rm, ranges_t{ {42060, 55178} }),
              "expect kept=8192 rm=[42060,55178)");
    }

    printf("\n%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
