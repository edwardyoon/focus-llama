// da_probe_dynamic.cpp — DA dynamic tag parsing probe (Phase 1b) — paper semantics
//
// Paper semantics: the scaffold is prefilled with FULL attention. While
// decoding, the probe watches the generated text for the magic-chunk tag
// <focus magic_chunks="N">. When the tag closes, the probe removes every
// document chunk EXCEPT the tagged one (plus the neutral filler) with
// llama_memory_seq_rm — mid-decode — and decoding continues under the
// restricted attention. The first generated token and the tag itself are
// produced with full attention: scaffold leakage is the paper's accepted
// premise, not a failure. The answer comes AFTER the removal (the model
// wraps it in the tag), so it is generated under restricted attention.
//
// Runs:
//   baseline : no removal at all                          -> expect tag + code
//   static   : all chunks except the GROUND-TRUTH one are removed right
//              after prefill, before the first token      -> "ideal DA":
//              restricted attention from the very first token (reference for
//              the decode tok/s of the reduced KV)
//   dynamic  : parse the tag mid-decode, remove at tag close -> the paper's
//              mechanism
//
// Metrics (this phase is a MEASUREMENT, not an isolation verdict):
//   - tag parse: did the model emit the tag, which chunk(s), at which
//     generated token index
//   - LOGICAL attended tokens: per decode step k, before n_prompt + k,
//     after n_prompt - R + k (R = removed tokens). This is what the paper's
//     mask achieves at the logit level. It is NOT a read reduction: seq_rm
//     removes cells logically (the kernel still reads up to n_kv, holes
//     are -inf-masked), so the before/after tok/s are expected to be
//     nearly equal — reported as a demonstration of the limitation. A real
//     speed gain needs the 2-stream/defrag path (plan: Phase 2 / backend B).
//   - answer: does the code still come out (expected: yes — the focus chunk
//     survives; informational, the paper allows scaffold leakage)
//
// Equivalence note (source-reading ONLY, per the 2026-09-20 scope decision):
// the public API has no per-position custom attention mask (llama_memory_*
// is range operations only), so seq_rm cannot be compared against a real
// -inf mask in this probe. From the source: seq_rm deletes the KV cells of
// the removed positions, so the attention layers cannot read them — the
// same effect the paper's block-table mask has on the global-attention
// layers. Record this as "source-reading only", never as "equivalence
// verified".
//
// One-stream-only decode batch (default on, --no-onestream to skip):
// separate micro-verification — with n_seq_max=2, decode steps that touch
// only seq 0 (seq 1 absent from the batch) must succeed and must not
// disturb seq 1's state, and seq 1 must resume afterwards. A prerequisite
// for the Phase 2 2-stream design (a pattern similar to speculative
// decoding's n_seq=2).
//
// Verdict:
//   PASS = baseline answered the code AND dynamic parsed the tag, applied
//          the removal (seq_rm true) and continued decoding. The static run
//          and the answers under restriction are reported, not gated.
//   No tag in dynamic => FAIL (compliance issue, see Phase 0 — the engine
//          path is untested on this model).
//
// Model-agnostic (same discipline as da_probe.cpp): the prompt framing is
// rendered from the model's own chat template; seams are newline-terminated
// and verified by detokenization. The document/instruction text is identical
// to da_phase0.py so Phase 0 and Phase 1b results are comparable.
//
// Usage: da_probe_dynamic <model.gguf> [max_tokens] [--ctx N] [--quiet]
//        [--filler N] [--question N] [--runs baseline,static,dynamic]
//        [--no-onestream] | --render
// Exit:  0 = pass, 1 = fail, 2 = usage error

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---- protocol text (identical to da_phase0.py) ----
const char * const SYSTEM_TEXT =
    "You are a precise reading engine. The document below contains several chunks. "
    "Some chunks contain a codeword in the form WORD-NUMBER.";

const char * const DOC_HEADER =
    "The following is a technical document about the facility's control systems.\n\n";

// newline-terminated: the Qwen pre-tokenizer never merges a \n into the
// following word, so cumulative-prefix tokenization gives exact boundaries
const char * const CHUNKS[5] = {
    "[Chunk 1]\nThe main power grid is monitored by a redundant array of relay units. "
    "The power grid reset phrase used by operators is ECHO-9. It is stored in the backup "
    "terminal and must be confirmed by a second operator before use.\n\n",
    "[Chunk 2]\nThe ventilation system recirculates air through a bank of axial fans. "
    "The ventilation restart code for the control room is TANGO-7. The fans return to "
    "nominal speed within ninety seconds after the code is accepted.\n\n",
    "[Chunk 3]\nFire suppression is handled by a distributed network of halon valves. "
    "The fire suppression override code is KILO-31. A full discharge locks the valves "
    "for four hours until a manual reset.\n\n",
    "[Chunk 4]\nThe emergency shutdown sequence is triggered from the control room "
    "console. The emergency shutdown codeword for the facility is ZEBRA-42. Operators "
    "must memorize it. It is printed on the wall of the control room in red letters.\n\n",
    "[Chunk 5]\nElevator movement during an alarm is governed by a dedicated controller. "
    "The elevator lockdown token is SIERRA-17. The token disables all car calls until "
    "it is cleared from the controller.\n\n",
};

const char * const INSTRUCTION =
    "Instructions:\n"
    "1. First identify the chunk that contains the answer to the question, and output "
    "exactly the tag <focus magic_chunks=\"N\"> where N is the chunk number (1-5).\n"
    "2. Then answer the question with the code only, nothing else.\n\n";

// (question, ground-truth chunk 1-based, expected code) — same table as da_phase0.py
struct Question {
    const char * text;
    int chunk;
    const char * code;
};
const Question QUESTIONS[5] = {
    {"What is the power grid reset phrase used by operators?", 1, "ECHO-9"},
    {"What is the ventilation restart code for the control room?", 2, "TANGO-7"},
    {"What is the fire suppression override code?", 3, "KILO-31"},
    {"What is the emergency shutdown codeword for the facility?", 4, "ZEBRA-42"},
    {"What is the elevator lockdown token?", 5, "SIERRA-17"},
};

const char * const FILLER_SENTENCE =
    "The mountain range stretches across the northern border of the valley. "
    "Hikers often begin their trails at dawn, when the light is soft and the air is cold. "
    "Small streams cross the path near the base camp, and the pines grow thicker above the ridge. "
    "The old stone bridge over the river has been repaired three times in the last century. ";

const char * const SENTINEL = "ZZSENTINEL42";

// ---- model-agnostic prompt framing (same technique as da_probe.cpp) ----

struct PromptParts {
    std::string scaffold;
    std::string question_suffix;
};

llama_token find_thinking_tag(const llama_vocab * vocab, bool want_start);  // fwd

PromptParts render_prompt(const llama_model * model) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        fprintf(stderr, "fatal: no chat template in model\n");
        exit(1);
    }
    const llama_chat_message msgs[2] = {
        {"system", SYSTEM_TEXT},
        {"user", SENTINEL},
    };
    std::vector<char> buf(std::strlen(tmpl) + 1024);
    int32_t n = -1;
    for (int attempt = 0; attempt < 8; attempt++) {
        n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int32_t)buf.size());
        if (n > 0 && n <= (int32_t)buf.size()) break;
        buf.resize(buf.size() * 2);
    }
    if (n <= 0 || n > (int32_t)buf.size()) {
        fprintf(stderr, "fatal: chat template render failed (n=%d)\n", n);
        exit(1);
    }
    const std::string rendered(buf.data(), (size_t)n);
    const size_t pos = rendered.find(SENTINEL);
    if (pos == std::string::npos || rendered.find(SENTINEL, pos + 1) != std::string::npos) {
        fprintf(stderr, "fatal: sentinel not found exactly once in rendered template\n");
        exit(1);
    }
    PromptParts p;
    p.scaffold = rendered.substr(0, pos);
    p.question_suffix = rendered.substr(pos + std::strlen(SENTINEL));
    return p;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, false, true);
    if (n < 0) n = -n;
    if (n == 0) {
        fprintf(stderr, "fatal: tokenize returned 0 tokens\n");
        exit(1);
    }
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), tokens.data(), n, false, true) < 0) {
        fprintf(stderr, "fatal: tokenize failed (fill)\n");
        exit(1);
    }
    return tokens;
}

// Segment layout (all seams newline-terminated):
//   seg 0 : scaffold (template framing, user open)
//   seg 1 : doc header
//   seg 2..6 : chunks 1..5            (chunk N = seg N+1)
//   seg 7 : filler (optional, --filler 0 disables it)
//   seg 8/7: instruction + question + question_suffix + empty thinking block
struct Layout {
    std::vector<std::string> segs;
    std::vector<int32_t> bounds;   // bounds[i] = token count of concat(segs[0..i])
    int filler_seg = -1;
    int32_t n_prompt = 0;
    std::vector<llama_token> all;

    // token range of chunk N (1-based). chunk N = seg (N+1); segment j
    // occupies [bounds[j-1], bounds[j]), so chunk N = [bounds[N], bounds[N+1])
    std::pair<int32_t, int32_t> chunk_range(int n) const {
        return {bounds[n], bounds[n + 1]};
    }
};

Layout build_layout(const llama_model * model, int filler_reps, const Question & q) {
    const PromptParts parts = render_prompt(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    Layout L;
    L.segs.push_back(parts.scaffold);
    L.segs.push_back(DOC_HEADER);
    for (int i = 0; i < 5; i++) L.segs.push_back(CHUNKS[i]);
    if (filler_reps > 0) {
        std::string filler;
        for (int i = 0; i < filler_reps; i++) filler += FILLER_SENTENCE;
        filler.back() = '\n';
        L.filler_seg = (int)L.segs.size();
        L.segs.push_back(filler);
    }
    // empty thinking block prefill (skip reasoning, answer directly)
    std::string think_block;
    const llama_token t_start = find_thinking_tag(vocab, true);
    const llama_token t_end = find_thinking_tag(vocab, false);
    if (t_start >= 0 && t_end >= 0) {
        char buf[128];
        int32_t n1 = llama_token_to_piece(vocab, t_start, buf, (int32_t)sizeof(buf), 0, true);
        if (n1 > 0) think_block.append(buf, (size_t)n1);
        think_block += "\n";
        int32_t n2 = llama_token_to_piece(vocab, t_end, buf, (int32_t)sizeof(buf), 0, true);
        if (n2 > 0) think_block.append(buf, (size_t)n2);
        think_block += "\n";
    }
    L.segs.push_back(std::string(INSTRUCTION) + "Question: " + q.text + "\n" +
                     parts.question_suffix + think_block);
    // cumulative-prefix boundaries + full token list
    std::string cum;
    for (const auto & s : L.segs) {
        cum += s;
        L.bounds.push_back((int32_t)tokenize(vocab, cum).size());
    }
    L.all = tokenize(vocab, cum);
    L.n_prompt = (int32_t)L.all.size();
    return L;
}

// ---- tag scanner ----

// Incremental scan of the accumulated generated text. Returns the chunk
// numbers of the FIRST complete <focus ... magic_chunks="N" ...> tag, or
// empty while no complete tag exists yet. Generation is short, so a full
// re-scan per step is fine.
std::vector<int32_t> scan_magic_tag(const std::string & text) {
    for (size_t p = 0; (p = text.find("<focus", p)) != std::string::npos; ) {
        const size_t close = text.find('>', p);
        if (close == std::string::npos) return {};  // tag not closed yet
        const std::string tag = text.substr(p, close - p + 1);
        const size_t q0 = tag.find("magic_chunks");
        if (q0 != std::string::npos) {
            std::vector<int32_t> nums;
            for (size_t q = q0 + std::strlen("magic_chunks"); q < tag.size(); q++) {
                if (std::isdigit((unsigned char)tag[q])) {
                    int v = 0;
                    while (q < tag.size() && std::isdigit((unsigned char)tag[q])) {
                        v = v * 10 + (tag[q] - '0');
                        q++;
                    }
                    nums.push_back(v);
                }
            }
            if (!nums.empty()) return nums;  // first complete tag wins
        }
        p = close + 1;  // this tag had no magic_chunks: keep looking
    }
    return {};
}

// ---- helpers (same discipline as da_probe.cpp) ----

llama_token argmax_token(const float * logits, int32_t n_vocab) {
    llama_token best = 0;
    float best_v = -1e30f;
    for (int32_t j = 0; j < n_vocab; j++) {
        if (logits[j] > best_v) {
            best_v = logits[j];
            best = j;
        }
    }
    return best;
}

llama_token find_thinking_tag(const llama_vocab * vocab, bool want_start) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        char piece[128];
        const int32_t n = llama_token_to_piece(vocab, i, piece, (int32_t)sizeof(piece), 0, true);
        if (n < 3 || n > 64) continue;
        const std::string s(piece, (size_t)n);
        if (s[0] != '<') continue;
        std::string l;
        l.reserve((size_t)n);
        for (int32_t k = 0; k < n; k++) l.push_back((char)std::tolower((unsigned char)piece[k]));
        if (l.find("think") == std::string::npos) continue;
        const bool is_close = (n >= 2 && s[1] == '/');
        if (want_start) { if (!is_close) return i; }
        else            { if (is_close) return i; }
    }
    return -1;
}

void quiet_log_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

bool contains_ci(const std::string & haystack, const char * needle) {
    std::string h = haystack;
    std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return (char)std::toupper(c); });
    std::string n = needle;
    std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return (char)std::toupper(c); });
    return h.find(n) != std::string::npos;
}

struct ArchInfo {
    std::string name;
    bool hybrid = false;
};

ArchInfo get_arch_info(const llama_model * model) {
    ArchInfo a;
    char buf[128] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) <= 0) {
        a.name = "(unknown)";
        return a;
    }
    a.name = buf;
    static const char * const HYBRID_ARCHS[] = {
        "qwen35", "qwen35moe", "qwen3next", "qwen4exp",
        "mamba", "rwkv6", "rwkv7",
        "gpt-oss", "minimax-01", "bailingmoe3", "plamo2",
        "kimi-linear", "kimi-k3",
    };
    for (const char * h : HYBRID_ARCHS) {
        if (a.name == h) { a.hybrid = true; break; }
    }
    return a;
}

// Prefill token range [from, to) of `all` with explicit positions (own
// counter rule — never derived from seq_pos_max). Single sequence. If
// capture_last, logits are enabled for the LAST position only and one
// n_vocab row is copied out (the first generated token is sampled from it).
void prefill(llama_context * ctx,
             const std::vector<llama_token> & all,
             int32_t from, int32_t to,
             const llama_vocab * vocab,
             std::vector<float> * capture_last) {
    const int32_t n = to - from;
    if (n <= 0) return;
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        batch.token[i] = all[from + i];
        batch.pos[i] = from + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (capture_last && i == n - 1) ? 1 : 0;
    }
    const int64_t t0 = llama_time_us();
    const int32_t rc = llama_decode(ctx, batch);
    const int64_t t1 = llama_time_us();
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "fatal: llama_decode rc=%d for pos [%d,%d)\n", rc, from, to);
        exit(1);
    }
    if (capture_last) {
        const float * logits = llama_get_logits(ctx);
        if (!logits) {
            fprintf(stderr, "fatal: no logits after prefill\n");
            exit(1);
        }
        capture_last->assign(logits, logits + llama_vocab_n_tokens(vocab));
    }
    std::printf("  prefill pos [%5d, %5d)  %8.2f tok/s\n", from, to, 1e6 * n / (t1 - t0));
}

// ---- generation ----

struct GenResult {
    std::string out;      // full generated text
    std::string answer;   // thinking block skipped, first line
    int32_t n_gen = 0;
    int32_t first_token_ms = -1;
    double t_decode_us = 0;
    // dynamic run
    bool tag_found = false;
    std::vector<int32_t> tag_chunks;
    int32_t tag_at_gen = -1;  // 1-based generated token index where the tag closed
    bool removal_ok = false;
    int32_t removed_tokens = 0;
    std::vector<std::pair<int32_t, int32_t>> removed_ranges;
    double t_before_us = 0; int32_t n_before = 0;  // steps decoded with full attention
    double t_after_us = 0;  int32_t n_after = 0;   // steps decoded under restriction
};

// Greedy generation (no mid-decode removal — that is inline in main, where
// the removal ranges are computed from the parsed tag). `first_logits` =
// logits of the last prompt position (the first token is sampled from it —
// no dummy step).
GenResult generate(llama_context * ctx, const llama_vocab * vocab,
                   const float * first_logits, int32_t start_pos, int32_t max_tokens) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token think_end = find_thinking_tag(vocab, false);
    GenResult r;
    llama_token next = argmax_token(first_logits, n_vocab);
    int32_t pos = start_pos;
    const int64_t t_start = llama_time_us();
    for (int32_t i = 0; i < max_tokens; i++) {
        if (llama_vocab_is_eog(vocab, next)) break;
        char piece[256];
        const int32_t n = llama_token_to_piece(vocab, next, piece, (int32_t)sizeof(piece), 0, false);
        if (n > 0) r.out.append(piece, n);
        // decode `next` at `pos` to obtain the logits of the following token
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        const int64_t t0 = llama_time_us();
        const int32_t rc = llama_decode(ctx, batch);
        const int64_t t1 = llama_time_us();
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "fatal: llama_decode step rc=%d\n", rc);
            exit(1);
        }
        const int64_t dt = t1 - t0;
        r.t_decode_us += dt;
        if (r.first_token_ms < 0) r.first_token_ms = (int32_t)(dt / 1000.0);
        const float * logits = llama_get_logits(ctx);
        if (!logits) {
            fprintf(stderr, "fatal: no logits\n");
            exit(1);
        }
        next = argmax_token(logits, n_vocab);
        pos++;
        r.n_gen = i + 1;
    }
    const int64_t t_end = llama_time_us();
    // extract the answer: skip a thinking block if the model emitted one
    r.answer = r.out;
    if (think_end >= 0) {
        char tag[128];
        const int32_t tn = llama_token_to_piece(vocab, think_end, tag, (int32_t)sizeof(tag), 0, true);
        if (tn > 0) {
            const std::string tagstr(tag, (size_t)tn);
            const size_t p = r.out.rfind(tagstr);
            if (p != std::string::npos) {
                r.answer = r.out.substr(p + tagstr.size());
                std::printf("  thinking block skipped (end tag tok %d)\n", (int)think_end);
            }
        }
    }
    const size_t nl = r.answer.find('\n');
    if (nl != std::string::npos) r.answer.erase(nl);
    std::printf("  decode  %d tokens in %.3f s (raw %zu chars), first token %d ms\n",
                r.n_gen, (t_end - t_start) / 1e6, r.out.size(), r.first_token_ms);
    return r;
}

// Removal ranges: every chunk except `keep` (1-based, 0 = none) + the filler.
std::vector<std::pair<int32_t, int32_t>> removal_ranges(const Layout & L, int keep) {
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int i = 1; i <= 5; i++) {
        if (i == keep) continue;
        r.push_back(L.chunk_range(i));
    }
    // segment j occupies [bounds[j-1], bounds[j]); the filler is seg filler_seg
    // (NEVER the tail — the instr+Q segment after it must survive so the
    // decode position stays contiguous)
    if (L.filler_seg >= 0) {
        r.push_back({L.bounds[L.filler_seg - 1], L.bounds[L.filler_seg]});
    }
    return r;
}

int32_t range_tokens(const std::vector<std::pair<int32_t, int32_t>> & r) {
    int32_t n = 0;
    for (const auto & p : r) n += p.second - p.first;
    return n;
}

// ---- one-stream-only decode batch micro-verification ----

// With n_seq_max=2: prefill both sequences, then decode steps that touch
// ONLY seq 0 (seq 1 absent from the batch). They must succeed, must not
// disturb seq 1's state, and seq 1 must resume afterwards. Prerequisite for
// the Phase 2 2-stream design.
int run_onestream_test(llama_model * model, const std::vector<llama_token> & prompt, int32_t n_ctx) {
    std::printf("\n=== one-stream-only decode batch (micro-verification) ===\n");
    // n_seq_max=2 splits the context EQUALLY (n_ctx_seq = n_ctx/2), so size it
    // so seq 0 can hold the full document plus the decode steps.
    const int32_t ctx_needed = 2 * ((int32_t)prompt.size() + 16);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (n_ctx > ctx_needed) ? n_ctx : ctx_needed;
    cp.n_batch = 2048;
    cp.n_seq_max = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "fatal: ctx init failed (n_seq_max=2)\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_memory_t mem = llama_get_memory(ctx);
    const int32_t n0 = (int32_t)prompt.size();
    prefill(ctx, prompt, 0, n0, vocab, nullptr);  // seq 0 (the shared helper uses seq_id 0)
    const std::vector<llama_token> s1 = tokenize(vocab, "The weather is fine today.\n");
    {
        // seq 1 prefill: the shared helper hardcodes seq_id 0, so build the
        // batch inline with seq_id 1 and its OWN position space (starts at 0)
        llama_batch b = llama_batch_init((int32_t)s1.size(), 0, 1);
        b.n_tokens = (int32_t)s1.size();
        for (int32_t i = 0; i < (int32_t)s1.size(); i++) {
            b.token[i] = s1[i];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 1;
            b.logits[i] = 0;
        }
        const int32_t rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            fprintf(stderr, "fatal: seq 1 prefill rc=%d\n", rc);
            llama_free(ctx);
            return 1;
        }
    }
    const int32_t s1_max = llama_memory_seq_pos_max(mem, 1);
    std::printf("  seq 0: %d tokens | seq 1: %d tokens (pos_max=%d)\n",
                n0, (int)s1.size(), s1_max);
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1;
        b.token[0] = prompt.back();
        b.pos[0] = n0 + i;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;  // seq 1 ABSENT from this batch
        b.logits[0] = 0;
        const int32_t rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            ok = false;
            std::printf("  decode step %d: rc=%d (FAIL)\n", i, rc);
            break;
        }
        std::printf("  decode step %d: seq 0 only (seq 1 absent) rc=0\n", i);
    }
    const int32_t s1_max2 = llama_memory_seq_pos_max(mem, 1);
    const bool idle_untouched = (s1_max2 == s1_max);
    std::printf("  seq 1 pos_max after idle steps: %d (unchanged: %s)\n",
                s1_max2, idle_untouched ? "yes" : "NO");
    if (!idle_untouched) ok = false;
    {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1;
        b.token[0] = s1.back();
        b.pos[0] = s1_max2 + 1;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 1;  // resume seq 1 at its own next position
        b.logits[0] = 0;
        const int32_t rc = llama_decode(ctx, b);
        llama_batch_free(b);
        std::printf("  resume seq 1 at pos %d: rc=%d (%s)\n", s1_max2 + 1, rc, rc == 0 ? "ok" : "FAIL");
        if (rc != 0) ok = false;
    }
    std::printf("  one-stream-only decode batch: %s\n", ok ? "PASS" : "FAIL");
    llama_free(ctx);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [max_tokens] [--ctx N] [--quiet] [--filler N]\n"
                        "       [--question N] [--runs baseline,static,dynamic] [--no-onestream] | --render\n",
                argv[0]);
        return 2;
    }
    int32_t max_tokens = 64;   // tag + answer is short; headroom for a stray thinking block
    int32_t n_ctx = 16384;
    int filler_reps = 12;
    int question_idx = 4;      // 1-based, default: the ZEBRA-42 question (continuity with 1a)
    bool want_render = false, want_quiet = false, want_onestream = true;
    bool run_baseline = true, run_static = true, run_dynamic = true;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--render") == 0) want_render = true;
        else if (strcmp(argv[i], "--quiet") == 0) want_quiet = true;
        else if (strcmp(argv[i], "--no-onestream") == 0) want_onestream = false;
        else if (strcmp(argv[i], "--ctx") == 0) { if (i + 1 < argc) n_ctx = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--filler") == 0) { if (i + 1 < argc) filler_reps = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--question") == 0) { if (i + 1 < argc) question_idx = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--runs") == 0) {
            if (i + 1 < argc) {
                const std::string sel = argv[++i];
                run_baseline = sel.find("baseline") != std::string::npos;
                run_static = sel.find("static") != std::string::npos;
                run_dynamic = sel.find("dynamic") != std::string::npos;
            }
        }
        else if (atoi(argv[i]) > 0) max_tokens = atoi(argv[i]);
    }
    if (want_quiet) llama_log_set(quiet_log_callback, nullptr);
    if (question_idx < 1 || question_idx > 5) {
        fprintf(stderr, "fatal: --question must be 1-5\n");
        return 2;
    }
    const Question q = QUESTIONS[question_idx - 1];

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "fatal: model load failed: %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const ArchInfo arch = get_arch_info(model);
    std::printf("architecture: %s (%s)\n", arch.name.c_str(),
                arch.hybrid ? "hybrid: has recurrent/linear-attention layers"
                            : "no recurrent layers in this build's model set");

    if (want_render) {
        const PromptParts p = render_prompt(model);
        std::printf("--- scaffold ---\n%s--- doc header ---\n%s--- chunk 1 ---\n%s"
                    "--- instruction+question ---\n%s%s%s\n",
                    p.scaffold.c_str(), DOC_HEADER, CHUNKS[0],
                    INSTRUCTION, "Question: ", q.text);
        std::printf("--- question_suffix ---\n%s", p.question_suffix.c_str());
        llama_model_free(model);
        return 0;
    }

    const Layout L = build_layout(model, filler_reps, q);
    std::printf("question   : %s (ground-truth chunk %d, code %s)\n", q.text, q.chunk, q.code);
    std::printf("prompt     : %d tokens | filler %d rep(s)\n", L.n_prompt, filler_reps);
    // segment j occupies [bounds[j-1], bounds[j]); chunk i (1-based) = seg i+1
    std::printf("layout     : scaffold [0,%d) header [%d,%d)", L.bounds[0], L.bounds[0], L.bounds[1]);
    for (int i = 1; i <= 5; i++) {
        std::printf(" chunk%d [%d,%d)", i, L.bounds[i], L.bounds[i + 1]);
    }
    if (L.filler_seg >= 0) {
        std::printf(" filler [%d,%d)", L.bounds[L.filler_seg - 1], L.bounds[L.filler_seg]);
    }
    std::printf(" instr+Q [%d,%d)\n", L.bounds[(int)L.bounds.size() - 2], L.n_prompt);

    // seam sanity: detokenizing at each boundary must start with the segment text
    {
        bool seams_ok = true;
        for (size_t i = 1; i < L.segs.size(); i++) {
            std::string piece;
            for (int32_t j = L.bounds[i - 1]; j < std::min<int32_t>(L.bounds[i - 1] + 6, L.n_prompt); j++) {
                char buf[256];
                const int32_t n = llama_token_to_piece(vocab, L.all[j], buf, (int32_t)sizeof(buf), 0, false);
                if (n > 0) piece.append(buf, n);
            }
            const size_t k = std::min(std::min<size_t>(8, piece.size()), L.segs[i].size());
            const bool ok = k > 0 && piece.compare(0, k, L.segs[i], 0, k) == 0;
            if (!ok) seams_ok = false;
            std::printf("  seam[%2zu] pos %5d: %s\n", i, L.bounds[i - 1], ok ? "OK" : "MISMATCH");
        }
        if (!seams_ok) {
            fprintf(stderr, "fatal: segment seam mismatch — boundary positions are wrong\n");
            llama_model_free(model);
            return 1;
        }
    }

    auto make_ctx = [&]() -> llama_context * {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;
        cparams.n_batch = 2048;
        cparams.n_seq_max = 1;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "fatal: ctx init failed\n");
            exit(1);
        }
        return ctx;
    };

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    GenResult res_baseline, res_static, res_dynamic;

    if (run_baseline) {
        std::printf("\n=== baseline (no removal) ===\n");
        llama_context * ctx = make_ctx();
        std::vector<float> first;
        prefill(ctx, L.all, 0, L.n_prompt, vocab, &first);
        res_baseline = generate(ctx, vocab, first.data(), L.n_prompt, max_tokens);
        std::printf("  output : %s\n", res_baseline.out.substr(0, 200).c_str());
        llama_free(ctx);
    }

    if (run_static) {
        std::printf("\n=== static (ground-truth chunk kept, removed after prefill, before first token) ===\n");
        llama_context * ctx = make_ctx();
        std::vector<float> first;
        prefill(ctx, L.all, 0, L.n_prompt, vocab, &first);
        const auto ranges = removal_ranges(L, q.chunk);
        for (const auto & rg : ranges) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, rg.first, rg.second)) {
                fprintf(stderr, "fatal: llama_memory_seq_rm([%d,%d)) rejected\n", rg.first, rg.second);
                llama_model_free(model);
                return 1;
            }
        }
        const int32_t n_removed = range_tokens(ranges);
        std::printf("  seq_rm: %d ranges, %d tokens removed (chunks kept: %d)\n",
                    (int)ranges.size(), n_removed, q.chunk);
        res_static = generate(ctx, vocab, first.data(), L.n_prompt, max_tokens);
        res_static.removed_tokens = n_removed;
        res_static.removed_ranges = ranges;
        std::printf("  output : %s\n", res_static.out.substr(0, 200).c_str());
        llama_free(ctx);
    }

    if (run_dynamic) {
        std::printf("\n=== dynamic (parse tag mid-decode, remove at tag close — paper semantics) ===\n");
        llama_context * ctx = make_ctx();
        std::vector<float> first;
        prefill(ctx, L.all, 0, L.n_prompt, vocab, &first);
        // Inline loop (not generate()): the removal ranges depend on the
        // PARSED tag, which is only known mid-decode — so the tag check and
        // the seq_rm at tag close live here. Same structure as generate(),
        // plus the tag scan. The tag token itself is decoded with full
        // attention; the NEXT step is the first under restriction.
        llama_token next = argmax_token(first.data(), n_vocab);
        int32_t pos = L.n_prompt;
        bool done = false;
        for (int32_t i = 0; i < max_tokens; i++) {
            if (llama_vocab_is_eog(vocab, next)) break;
            char piece[256];
            const int32_t n = llama_token_to_piece(vocab, next, piece, (int32_t)sizeof(piece), 0, false);
            if (n > 0) res_dynamic.out.append(piece, n);
            if (!done) {
                const std::vector<int32_t> nums = scan_magic_tag(res_dynamic.out);
                if (!nums.empty()) {
                    res_dynamic.tag_found = true;
                    res_dynamic.tag_chunks = nums;
                    res_dynamic.tag_at_gen = i + 1;
                }
            }
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.n_tokens = 1;
            batch.token[0] = next;
            batch.pos[0] = pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            const int64_t t0 = llama_time_us();
            const int32_t rc = llama_decode(ctx, batch);
            const int64_t t1 = llama_time_us();
            llama_batch_free(batch);
            if (rc != 0) {
                fprintf(stderr, "fatal: dynamic decode step rc=%d\n", rc);
                llama_model_free(model);
                return 1;
            }
            const int64_t dt = t1 - t0;
            res_dynamic.t_decode_us += dt;
            if (res_dynamic.first_token_ms < 0) res_dynamic.first_token_ms = (int32_t)(dt / 1000.0);
            const bool bucket_before = !done;  // this decode ran with full attention
            if (res_dynamic.tag_found && !res_dynamic.removal_ok) {
                const auto ranges = removal_ranges(L, res_dynamic.tag_chunks[0]);
                int rem = 0;
                for (const auto & rg : ranges) {
                    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, rg.first, rg.second)) {
                        fprintf(stderr, "fatal: llama_memory_seq_rm([%d,%d)) rejected mid-decode\n",
                                rg.first, rg.second);
                        llama_model_free(model);
                        return 1;
                    }
                    rem += rg.second - rg.first;
                }
                res_dynamic.removal_ok = true;
                res_dynamic.removed_tokens = rem;
                res_dynamic.removed_ranges = ranges;
                done = true;
                std::printf("  tag closed at generated token %d (pos %d) — %d tokens removed, decode continues\n",
                            res_dynamic.tag_at_gen, pos, rem);
            }
            if (bucket_before) { res_dynamic.t_before_us += dt; res_dynamic.n_before++; }
            else               { res_dynamic.t_after_us += dt; res_dynamic.n_after++; }
            const float * logits = llama_get_logits(ctx);
            if (!logits) {
                fprintf(stderr, "fatal: no logits\n");
                llama_model_free(model);
                return 1;
            }
            next = argmax_token(logits, n_vocab);
            pos++;
            res_dynamic.n_gen = i + 1;
        }
        // answer extraction (same as generate())
        const llama_token think_end = find_thinking_tag(vocab, false);
        res_dynamic.answer = res_dynamic.out;
        if (think_end >= 0) {
            char tag[128];
            const int32_t tn = llama_token_to_piece(vocab, think_end, tag, (int32_t)sizeof(tag), 0, true);
            if (tn > 0) {
                const std::string tagstr(tag, (size_t)tn);
                const size_t p = res_dynamic.out.rfind(tagstr);
                if (p != std::string::npos) {
                    res_dynamic.answer = res_dynamic.out.substr(p + tagstr.size());
                    std::printf("  thinking block skipped (end tag tok %d)\n", (int)think_end);
                }
            }
        }
        const size_t nl = res_dynamic.answer.find('\n');
        if (nl != std::string::npos) res_dynamic.answer.erase(nl);
        std::printf("  decode  %d tokens in %.3f s (raw %zu chars), first token %d ms\n",
                    res_dynamic.n_gen, res_dynamic.t_decode_us / 1e6, res_dynamic.out.size(),
                    res_dynamic.first_token_ms);
        std::printf("  output : %s\n", res_dynamic.out.substr(0, 200).c_str());
        llama_free(ctx);
    }

    // ---- results ----
    std::printf("\nRESULTS (question %d: %s)\n", q.chunk, q.code);
    if (run_baseline) {
        const bool ok = contains_ci(res_baseline.answer, q.code);
        std::printf("  baseline : answer %s | tag %s | output: %s\n",
                    ok ? "PASS" : "FAIL",
                    scan_magic_tag(res_baseline.out).empty() ? "absent" : "present",
                    res_baseline.answer.c_str());
    }
    if (run_static) {
        const bool ok = contains_ci(res_static.answer, q.code);
        std::printf("  static   : answer %s | removed %d tokens | output: %s\n",
                    ok ? "PASS" : "FAIL", res_static.removed_tokens, res_static.answer.c_str());
    }
    if (run_dynamic) {
        std::printf("  dynamic  : tag %s", res_dynamic.tag_found ? "parsed" : "NOT FOUND");
        if (res_dynamic.tag_found) {
            std::printf(" (chunk%s", res_dynamic.tag_chunks.size() > 1 ? "s" : "");
            for (size_t i = 0; i < res_dynamic.tag_chunks.size(); i++) {
                if (i) std::printf(",");
                std::printf(" %d", res_dynamic.tag_chunks[i]);
            }
            std::printf(" at generated token %d)", res_dynamic.tag_at_gen);
        }
        std::printf("\n");
        if (res_dynamic.tag_found) {
            std::printf("             removal %s (%d tokens) | answer %s | output: %s\n",
                        res_dynamic.removal_ok ? "applied" : "FAILED",
                        res_dynamic.removed_tokens,
                        contains_ci(res_dynamic.answer, q.code) ? "PASS" : "FAIL",
                        res_dynamic.answer.c_str());
        }
    }

    // ---- metrics ----
    // attended tokens: LOGICAL reduction (what the paper's mask achieves at
    // the logit level). seq_rm is a logical removal — the attention kernel
    // still reads up to n_kv (the max index of live cells); holes are
    // -inf-masked. So the before/after tok/s below are EXPECTED TO BE
    // NEARLY EQUAL: they demonstrate the limitation, they are not a speed
    // gain. A real read reduction needs the 2-stream (defrag) path — see
    // the plan (Phase 2 / backend B).
    std::printf("\nMETRICS\n");
    const int32_t R = res_dynamic.tag_found ? res_dynamic.removed_tokens
                    : (run_static ? res_static.removed_tokens : 0);
    if (arch.hybrid) {
        std::printf("  (hybrid architecture: the attended-token arithmetic below is NOT valid —\n"
                    "   recurrent layers do not read the KV. Use the tok/s figures.)\n");
    }
    std::printf("  LOGICAL attended per decode step k (pure attention): before = n_prompt + k, after = n_prompt - R + k\n");
    std::printf("  n_prompt = %d, R = %d removed (logical)\n", L.n_prompt, R);
    if (run_dynamic && res_dynamic.tag_found && res_dynamic.n_gen > 0) {
        const int32_t k = res_dynamic.n_gen;
        const int32_t att_before = L.n_prompt + k;
        const int32_t att_after = L.n_prompt - R + k;
        const double save = 100.0 * (att_before - att_after) / att_before;
        std::printf("  at final step (k=%d): attended before=%d after=%d  (logical reduction %.1f%%)\n",
                    k, att_before, att_after, save);
        if (res_dynamic.n_before > 0 && res_dynamic.n_after > 0) {
            std::printf("  decode tok/s: before removal %7.1f (%d steps) | after removal %7.1f (%d steps)\n",
                        1e6 * res_dynamic.n_before / res_dynamic.t_before_us, res_dynamic.n_before,
                        1e6 * res_dynamic.n_after / res_dynamic.t_after_us, res_dynamic.n_after);
            std::printf("  (before ~= after EXPECTED: seq_rm is logical, the kernel still reads up to n_kv;\n"
                        "   a real read reduction needs the 2-stream/defrag path)\n");
        }
        std::printf("  decode tok/s: overall %7.1f (%d steps)\n",
                    1e6 * res_dynamic.n_gen / res_dynamic.t_decode_us, res_dynamic.n_gen);
    }
    if (run_static && res_static.n_gen > 0) {
        std::printf("  static (restriction from step 1) decode tok/s: %7.1f (%d steps)\n",
                    1e6 * res_static.n_gen / res_static.t_decode_us, res_static.n_gen);
    }
    if (run_baseline && res_baseline.n_gen > 0 && res_baseline.t_decode_us > 0) {
        std::printf("  baseline (full KV throughout) decode tok/s: %7.1f (%d steps)\n",
                    1e6 * res_baseline.n_gen / res_baseline.t_decode_us, res_baseline.n_gen);
    }

    // ---- verdict ----
    bool ok = true;
    std::string verdict_parts;
    if (run_dynamic && !res_dynamic.tag_found) {
        ok = false;
        verdict_parts = "FAIL (dynamic: no magic tag in the output — compliance issue, see "
                        "Phase 0; the engine path is untested on this model)";
    } else if (run_dynamic && !res_dynamic.removal_ok) {
        ok = false;
        verdict_parts = "FAIL (dynamic: seq_rm rejected mid-decode)";
    } else if (run_baseline && !contains_ci(res_baseline.answer, q.code)) {
        ok = false;
        verdict_parts = "FAIL (baseline: the model cannot answer — probe invalid)";
    }
    if (ok) {
        std::string m = "PASS (";
        bool first = true;
        if (run_baseline) { m += "baseline answered"; first = false; }
        if (run_static)   { if (!first) m += "; "; m += "static applied the removal after prefill"; first = false; }
        if (run_dynamic)  { if (!first) m += "; "; m += "dynamic parsed the tag, applied the mid-decode removal and continued decoding"; }
        m += ")";
        verdict_parts = m;
    }
    std::printf("\nOVERALL: %s\n", verdict_parts.c_str());

    int rc = ok ? 0 : 1;
    if (want_onestream) {
        const int orc = run_onestream_test(model, L.all, n_ctx);
        if (orc != 0) rc = 1;
    }
    llama_model_free(model);
    return rc;
}
