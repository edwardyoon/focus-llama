// Standalone test for the da_auto instruction-placement logic.
//
// The logic under test is the block in tools/server/server-context.cpp
// da_auto_chunk() that decides where the DA instruction text goes inside
// the rendered qwen chat prompt. It must be placed at the END OF THE LAST
// USER MESSAGE (right before its terminator) so that the rendered prompt
// still ends with the template's final generation tail and generation
// starts from the model's normal turn position. Appending after the
// opener instead makes the model treat its own turn as already started
// and emit EOS immediately (A/B verified on Bonsai-8B: n_gen=1 vs 64).
//
// Rendered qwen prompt tails (all ASCII; verified against the GGUF
// tokenizer.chat_template hexdump):
//   non-thinking : ...content + USERTERM + LF + ASSTOPEN
//   thinking     : ...content + USERTERM + LF + ASSTOPEN + THINKOPEN
// where USERTERM is the im_end user terminator (10 bytes),
// ASSTOPEN is the im_start assistant opener (21 bytes), and
// THINKOPEN is the think open + close openers with blank lines (25 bytes).
//
// The tags are assembled from char codes at runtime because typing the
// full token inline gets split across physical lines by the editor
// tooling and breaks C string literals.
//
// Build & run (seconds, no server needed):
//   g++ -O2 -std=c++17 -o /tmp/da_placement_test da_placement_test.cpp \
//       && /tmp/da_placement_test
//
// Exit code 0 = all cases pass.

#include <cstdio>
#include <string>

static const std::string USER_TERM =
    std::string("<") + "|im_end|" + ">";                                     // 10 bytes
static const std::string ASST_OPEN =
    std::string("<") + "|im_start|>assistant\n";                             // 21 bytes
static const std::string THINK_OPEN =
    std::string("<") + "/think" + ">\n\n" + std::string("<") + "/think" + ">\n\n";  // 25 bytes

// The placement logic (must stay in sync with da_auto_chunk in
// server-context.cpp). Returns the insertion offset of the instruction
// text; legacy == true means "no safe spot found" (caller keeps the
// legacy tail append / fails open).
static size_t da_instr_pos(const std::string & modified, bool & legacy) {
    legacy = true;
    // the final assistant opener is always the LAST occurrence: the
    // prompt ends with it (plus the thinking openers, which cannot
    // contain the opener string)
    const size_t opener = modified.rfind(ASST_OPEN);
    if (opener == std::string::npos) {
        return modified.size();  // non-qwen shape: legacy append
    }
    // the last user message's terminator = the last USERTERM that ends
    // at or before the opener start. rfind(str, pos) matches
    // occurrences starting at or before pos, so bound the start to
    // opener - USERTERM.size().
    const size_t limit = opener >= USER_TERM.size() ? opener - USER_TERM.size() : 0;
    const size_t term = modified.rfind(USER_TERM, limit);
    if (term == std::string::npos) {
        return modified.size();  // degenerate: no terminator, fail open
    }
    legacy = false;
    return term;
}

static int failures = 0;

static void check(const char * name, const std::string & prompt, size_t expect) {
    bool legacy = false;
    const size_t got = da_instr_pos(prompt, legacy);
    const bool ok = (got == expect) && !legacy;
    if (!ok) { failures++; }
    printf("%-28s %s  N=%zu expect=%zu got=%zu legacy=%d\n",
           name, ok ? "PASS" : "FAIL", prompt.size(), expect, got, (int) legacy);
    // context around the insertion point (LFs made visible)
    const size_t a = got > 24 ? got - 24 : 0;
    std::string pre  = prompt.substr(a, got - a);
    std::string post = prompt.substr(got,
                       prompt.size() - got > 24 ? 24 : prompt.size() - got);
    for (auto & c : pre)  { if (c == '\n') { c = '?'; } }
    for (auto & c : post) { if (c == '\n') { c = '?'; } }
    printf("    ctx: [%s]INS[%s]\n", pre.c_str(), post.c_str());
}

int main() {
    const std::string SYS =
        std::string("<") + "|im_start|>system\n"
        "You are a precise retrieval assistant."
        + std::string("<") + "|im_end|>\n";
    const std::string Q  = "Question: What number is 2+2? Answer with the number only.";

    // offset right after the question content (inside the last user
    // message, before its terminator). Computed, not hand-arithmetic.
    const std::string U =
        std::string("<") + "|im_start|>user\n";
    const size_t after_q = SYS.size() + U.size() + Q.size();

    // T1: non-thinking. tail = Q + USERTERM + LF + ASSTOPEN
    {
        const std::string p = SYS + U + Q + USER_TERM + "\n" + ASST_OPEN;
        check("T1 non-thinking", p, after_q);
    }
    // T2: thinking (Bonsai-8B actual shape).
    //     tail = Q + USERTERM + LF + ASSTOPEN + THINKOPEN
    {
        const std::string p = SYS + U + Q + USER_TERM + "\n" + ASST_OPEN + THINK_OPEN;
        check("T2 thinking", p, after_q);
    }
    // T3: thinking, content ends with an LF (instruction starts on its
    //     own line)
    {
        const std::string p = SYS + U + Q + "\n" + USER_TERM + "\n" + ASST_OPEN + THINK_OPEN;
        check("T3 thinking + content LF", p, after_q + 1);
    }
    // T4: content itself ends with a literal terminator tag (pathological,
    //     must land after the content tag, before the real terminator)
    {
        const std::string p = SYS + U + Q + USER_TERM + USER_TERM + "\n" + ASST_OPEN;
        check("T4 content ends with term", p, after_q + USER_TERM.size());
    }
    // T5: multi-turn - an earlier assistant turn precedes the last user
    //     message; the LAST opener must win
    {
        const std::string asst_turn =
            std::string("<") + "|im_start|>assistant\n"
            "earlier answer"
            + std::string("<") + "|im_end|>\n";
        const std::string p = SYS + asst_turn + U + Q + USER_TERM + "\n" + ASST_OPEN + THINK_OPEN;
        check("T5 multi-turn", p, SYS.size() + asst_turn.size() + U.size() + Q.size());
    }
    // T6: non-qwen shape (no opener at the tail) -> legacy append
    {
        const std::string p = "plain prompt, no template tags at all";
        bool legacy = false;
        const size_t got = da_instr_pos(p, legacy);
        const bool ok = legacy && got == p.size();
        if (!ok) { failures++; }
        printf("%-28s %s  N=%zu got=%zu legacy=%d\n",
               "T6 non-qwen (legacy)", ok ? "PASS" : "FAIL", p.size(), got, (int) legacy);
    }

    // T7: invariant - the bytes from the insertion point to the end of
    //     the prompt must be EXACTLY the template tail (terminator + LF
    //     + opener + thinking openers), unchanged by the insertion
    {
        const std::string tail = USER_TERM + "\n" + ASST_OPEN + THINK_OPEN;
        const std::string p = SYS + U + Q + tail;
        bool legacy = false;
        const size_t pos = da_instr_pos(p, legacy);
        std::string m = p;
        m.insert(pos, "INSTRUCTION-TEXT");
        const bool ok = !legacy &&
            m.size() >= tail.size() &&
            m.compare(m.size() - tail.size(), tail.size(), tail) == 0;
        if (!ok) { failures++; }
        printf("%-28s %s\n", "T7 tail intact", ok ? "PASS" : "FAIL");
    }

    printf("%s (%d failure(s))\n", failures ? "OVERALL: FAIL" : "OVERALL: PASS", failures);
    return failures ? 1 : 0;
}
