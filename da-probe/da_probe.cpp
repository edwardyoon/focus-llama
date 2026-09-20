// da_probe.cpp — DA static mask probe (Phase 1a) — isolation verification
//
// Single-sequence design: the whole prompt AND the response live in seq 0.
// llama.cpp attention only permits cells of the same seq_id, so per-chunk
// seq_ids would make chunks invisible to each other (and the question
// unable to see any context) — removals are done by POSITION RANGE
// (b[i] = token count of concat(segments[0..i]), so segment i sits at
// [b[i-1], b[i]) and segment 0 at [0, b[0])):
//   seq_rm(mem, 0, b[0], b[1])  -> chunk A
//   seq_rm(mem, 0, b[2], b[3])  -> chunk C
//
// Segment order: scaffold, A, filler, C, D, question. D is a short neutral
// paragraph after C so that a removed chunk is NEVER the tail of the
// prefilled prefix: this version of llama_decode rejects a batch whose
// first position is not seq_pos_max + 1, and removing the tail would shrink
// seq_pos_max and break question prefill at its original position.
//
// Leak-free: context (scaffold+A+filler+C+D) is prefilled first, the masked
// chunk's KV is removed, and the QUESTION is prefilled AFTER the removal
// (explicit pos, own next_pos counter). The question's KV and the first
// generated token's logits are computed with the masked chunk absent.
//
// Generation starts from the LAST prompt token's logits (enabled in the
// question prefill) — no dummy-token step.
//
// Runs:
//   baseline : no removal                              -> expect ZEBRA-42
//   masked-A : chunk A removed before question prefill -> expect no ZEBRA-42
//   masked-C : chunk C removed before question prefill -> expect ZEBRA-42
//   keep-A   : question logits, chunk A present        -> control (differs)
//   rm-A     : question logits, chunk A removed        -> control (differs)
//
// NOTE on "gap-A" (chunk never prefilled): not constructible (the same
// continuity check rejects it) AND conceptually wrong for a single
// sequence — filler/C are prefilled while A is alive, so A's information
// is already mixed into their higher-layer KV; seq_rm deletes A's cells
// only. That matches the DA paper semantics (prefill = full attention,
// masking at decode time). Exact equivalence needs the KQ mask-injection
// path (to be built for the GDN hybrid) and is out of scope here.
//
// If masked-A still outputs ZEBRA-42, that is NOT necessarily a mask
// failure: A's content may have seeped into filler/C KV during the
// full-attention prefill. Distinguish before concluding.
//
// Pass = baseline AND masked-A AND masked-C
//        AND keep-A != rm-A (control: last-pos argmax must differ)
// A baseline failure invalidates every other result.
//
// Model-agnostic: the prompt framing is rendered from the model's own chat
// template at runtime (--render shows the result). Works on any model with
// a supported template; no per-model files, no hardcoded marker tokens.
//
// Usage: da_probe <model.gguf> [max_tokens] [--ctx N] [--quiet] | --render | --tokens | --tmpl
// Exit:  0 = pass, 1 = fail, 2 = usage error,
//        3 = seq_rm rejected by the memory backend (architecture gate:
//            hybrid/SSM memory cannot erase a middle range — a valid result)

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Segment {
    std::string text;
};

// Model-agnostic prompt construction. The template framing is rendered by
// the MODEL'S OWN chat template (llama_model_chat_template +
// llama_chat_apply_template) — no per-model prompt files, no hardcoded
// marker tokens (the Bonsai model uses LLaMA-style `|im_start|` ASCII at
// Qwen3 ID slots; other models differ, so hand-written markers are wrong
// by construction). The user message content is a sentinel string:
//   scaffold        = rendered text BEFORE the sentinel
//   question_suffix = rendered text AFTER it (user close + assistant open,
//                     exactly what this model's add_generation_prompt says)
// tokenize() uses parse_special=true so the rendered markers become real
// special tokens.
//
// Every content segment is newline-terminated: the Qwen pre-tokenizer never
// merges a \n into the following word, so cumulative-prefix tokenization
// gives exact boundary positions (a trailing space would let BPE merge
// " Chunk" into one token in the full text only — the seam bug).
const char * const SYSTEM_TEXT =
    "You are a precise reading engine. The document below contains several chunks. "
    "Some chunks contain a codeword in the form WORD-NUMBER.";
const char * const CHUNK_A = "Chunk 1: The codeword is ZEBRA-42. It is a test string.\n";
const char * const CHUNK_C = "Chunk 3: The codeword is TIGER-99. It is a test string.\n";
const char * const QUESTION_TEXT =
    "Question: What is the codeword in Chunk 1? Answer with only the codeword.\n";
const char * const SENTINEL = "ZZSENTINEL42";

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

std::vector<Segment> build_segments(const llama_model * model) {
    const PromptParts parts = render_prompt(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string filler;
    const char * sentence =
        "The mountain range stretches across the northern border of the valley. "
        "Hikers often begin their trails at dawn, when the light is soft and the air is cold. "
        "Small streams cross the path near the base camp, and the pines grow thicker above the ridge. "
        "The old stone bridge over the river has been repaired three times in the last century. ";
    for (int i = 0; i < 18; i++) filler += sentence;
    filler.back() = '\n';  // newline-terminate the seam (was a trailing space)
    // D: short neutral tail so a removed chunk is never the prefilled tail
    const std::string d = "The document ends here.\n";
    // Pre-fill an EMPTY thinking block (start tag + \n + end tag + \n) so a
    // thinking model skips reasoning and answers directly. The model's real
    // template does not pre-fill it; without this a thinking model emits a
    // long reasoning block before the answer. Non-thinking models: no tag
    // found, think_block stays empty.
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
    return {
        {parts.scaffold},
        {CHUNK_A},
        {filler},
        {CHUNK_C},
        {d},
        {std::string(QUESTION_TEXT) + parts.question_suffix + think_block},
    };
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    // this version returns -required_size when the buffer is too small
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

// Boundary i = token count of tokenize(concat(segments[0..i])). With all
// seams newline-terminated this equals the true position in tokenize(full).
std::vector<int32_t> segment_bounds(const llama_vocab * vocab, const std::vector<Segment> & segments) {
    std::vector<int32_t> bounds;
    std::string cum;
    for (const auto & s : segments) {
        cum += s.text;
        bounds.push_back((int32_t)tokenize(vocab, cum).size());
    }
    return bounds;
}

// Prefill token range [from, to) of `all` with explicit positions
// (pos = token index in `all`; the own-counter rule — never derived from
// seq_pos_max, which shrinks after tail removals). Single sequence (0).
// If capture_logits != nullptr, logits are enabled for every token in the
// range and copied out (n_rows = to-from, each n_vocab floats).
void prefill(llama_context * ctx,
             const std::vector<llama_token> & all,
             int32_t from, int32_t to,
             const llama_vocab * vocab,
             std::vector<float> * capture_logits) {
    const int32_t n = to - from;
    if (n <= 0) return;
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        batch.token[i] = all[from + i];
        batch.pos[i] = from + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;  // single sequence: everything attends to everything before it
        batch.logits[i] = capture_logits ? 1 : 0;
    }
    const int64_t t0 = llama_time_us();
    const int32_t rc = llama_decode(ctx, batch);
    const int64_t t1 = llama_time_us();
    llama_batch_free(batch);
    if (rc != 0) {
        // check rc BEFORE touching logits: a failed decode leaves a stale
        // pointer in llama_get_logits (the run-10 segfault)
        fprintf(stderr, "fatal: llama_decode rc=%d for pos [%d,%d)\n", rc, from, to);
        exit(1);
    }
    if (capture_logits) {
        const float * logits = llama_get_logits(ctx);
        if (!logits) {
            fprintf(stderr, "fatal: no logits after prefill\n");
            exit(1);
        }
        capture_logits->assign(logits, logits + (int64_t)n * llama_vocab_n_tokens(vocab));
    }
    std::printf("  prefill pos [%5d, %5d)  %7.2f tok/s\n", from, to, 1e6 * n / (t1 - t0));
}

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

// Find the thinking START / END marker tokens in the vocab (model-agnostic).
// Thinking models (Qwen3 family) wrap reasoning in a start/end tag pair. Match
// a token whose text contains "think" plus "start"/"begin" (start tag) or
// "end"/"close" (end tag). Return -1 when the model has no such token. The
// model's real template does NOT pre-fill an empty thinking block, so a
// thinking model would otherwise emit a (possibly long) reasoning block before
// the answer. Pre-filling an empty block makes it skip straight to the answer.
llama_token find_thinking_tag(const llama_vocab * vocab, bool want_start) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        char piece[128];
        const int32_t n = llama_token_to_piece(vocab, i, piece, (int32_t)sizeof(piece), 0, true);
        if (n < 3 || n > 64) continue;
        const std::string s(piece, (size_t)n);
        if (s[0] != '<') continue;  // XML-style tag only
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
llama_token find_thinking_end_token(const llama_vocab * vocab) {
    return find_thinking_tag(vocab, false);
}

// Greedy generation. The FIRST token is already sampled from the last
// prompt position's logits (first_logits) — no dummy decode step. Collects
// the FULL output (a thinking model emits a reasoning block first), then
// extracts the answer: text after the LAST thinking-end tag (if the model has
// one), truncated at the first newline. Returns the extracted answer.
std::string generate(llama_context * ctx, const llama_vocab * vocab,
                     const float * first_logits, int32_t start_pos, int32_t max_tokens) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token think_end = find_thinking_end_token(vocab);
    std::string out;
    llama_token next = argmax_token(first_logits, n_vocab);
    int32_t pos = start_pos;  // own next_pos counter
    int64_t t_start = llama_time_us();
    for (int32_t i = 0; i < max_tokens; i++) {
        if (llama_vocab_is_eog(vocab, next)) break;
        char piece[256];
        const int32_t n = llama_token_to_piece(vocab, next, piece, (int32_t)sizeof(piece), 0, false);
        if (n > 0) out.append(piece, n);
        // decode `next` at `pos` to obtain the logits of the following token
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;  // single sequence
        batch.logits[0] = 1;
        const int64_t t0 = llama_time_us();
        const int32_t rc = llama_decode(ctx, batch);
        const int64_t t1 = llama_time_us();
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "fatal: llama_decode step rc=%d\n", rc);
            exit(1);
        }
        const float * logits = llama_get_logits(ctx);
        if (!logits) {
            fprintf(stderr, "fatal: no logits\n");
            exit(1);
        }
        next = argmax_token(logits, n_vocab);
        pos++;
        if (i == 0) {
            std::printf("  decode  first token in %6.2f ms\n", (t1 - t0) / 1000.0);
        }
    }
    const int64_t t_end = llama_time_us();
    // Extract the answer: skip a thinking block if the model emitted one.
    std::string answer = out;
    if (think_end >= 0) {
        char tag[128];
        const int32_t tn = llama_token_to_piece(vocab, think_end, tag, (int32_t)sizeof(tag), 0, true);
        if (tn > 0) {
            const std::string tagstr(tag, (size_t)tn);
            const size_t p = out.rfind(tagstr);
            if (p != std::string::npos) {
                answer = out.substr(p + tagstr.size());
                std::printf("  decode  thinking block skipped (end tag tok %d)\n", (int)think_end);
            }
        }
    }
    const size_t nl = answer.find('\n');
    if (nl != std::string::npos) answer.erase(nl);
    std::printf("  decode  %d tokens in %.2f s (raw %zu chars)\n",
                pos - start_pos, (t_end - t_start) / 1e6, out.size());
    return answer;
}

// --quiet: suppress llama.cpp's INFO/WARN log spam (model load, graph
// reservation) and keep only real errors. The probe's own stdout is
// unaffected.
void quiet_log_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

void print_token(const llama_vocab * vocab, int32_t id) {
    char piece[256];
    const int32_t n = llama_token_to_piece(vocab, id, piece, (int32_t)sizeof(piece), 0, true);
    if (n > 0) std::printf("'%s'", piece);
    else std::printf("<tok %d>", id);
}

bool contains_ci(const std::string & haystack, const char * needle) {
    std::string h = haystack;
    std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return (char)std::toupper(c); });
    std::string n = needle;
    std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return (char)std::toupper(c); });
    return h.find(n) != std::string::npos;
}

struct LogitDiff {
    double max_abs = 0.0;
    int argmax_match = 0;
    int n_rows = 0;
    int32_t last_a = -1;
    int32_t last_b = -1;
    bool last_match = false;
};

LogitDiff compare_logits(const std::vector<float> & a, const std::vector<float> & b, const llama_vocab * vocab) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    LogitDiff d;
    d.n_rows = (int)(a.size() / n_vocab);
    for (int r = 0; r < d.n_rows; r++) {
        int32_t am = 0, bm = 0;
        double row_max = 0.0;
        float av = -1e30f, bv = -1e30f;
        for (int32_t j = 0; j < n_vocab; j++) {
            const double diff = std::fabs((double)a[r * n_vocab + j] - (double)b[r * n_vocab + j]);
            if (diff > row_max) row_max = diff;
            if (a[r * n_vocab + j] > av) { av = a[r * n_vocab + j]; am = j; }
            if (b[r * n_vocab + j] > bv) { bv = b[r * n_vocab + j]; bm = j; }
        }
        d.max_abs = std::max(d.max_abs, row_max);
        if (am == bm) d.argmax_match++;
        if (r == d.n_rows - 1) {
            d.last_a = am;
            d.last_b = bm;
            d.last_match = (am == bm);
        }
    }
    return d;
}

void print_diff(const char * label, const LogitDiff & d, const llama_vocab * vocab) {
    std::printf("%s\n", label);
    std::printf("  max |delta logit| = %.6f\n", d.max_abs);
    std::printf("  argmax match      = %d/%d\n", d.argmax_match, d.n_rows);
    std::printf("  last pos argmax   : ");
    print_token(vocab, d.last_a);
    std::printf("  |  ");
    print_token(vocab, d.last_b);
    std::printf("  -> %s\n", d.last_match ? "MATCH" : "MISMATCH");
}

}  // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: a crash must not swallow progress
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [max_tokens] [--ctx N] [--quiet] | --render | --tokens | --tmpl\n", argv[0]);
        return 2;
    }
    int32_t max_tokens = 256;  // headroom for a thinking block before the answer
    int32_t n_ctx = 16384;
    bool want_render = false, want_tokens = false, want_tmpl = false, want_thinkscan = false;
    bool want_quiet = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--render") == 0) want_render = true;
        else if (strcmp(argv[i], "--tokens") == 0) want_tokens = true;
        else if (strcmp(argv[i], "--tmpl") == 0) want_tmpl = true;
        else if (strcmp(argv[i], "--thinkscan") == 0) want_thinkscan = true;
        else if (strcmp(argv[i], "--quiet") == 0) want_quiet = true;
        else if (strcmp(argv[i], "--ctx") == 0) { if (i + 1 < argc) n_ctx = atoi(argv[++i]); }
        else if (atoi(argv[i]) > 0) max_tokens = atoi(argv[i]);
    }
    if (want_quiet) llama_log_set(quiet_log_callback, nullptr);

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "fatal: model load failed: %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (want_tmpl) {
        // full chat template stored in the GGUF (hex output: transport-safe)
        const char * tmpl = llama_model_chat_template(model, nullptr);
        if (!tmpl) {
            fprintf(stderr, "no chat template in model\n");
            return 1;
        }
        std::printf("len=%zu hex=", std::strlen(tmpl));
        for (const unsigned char * p = (const unsigned char *)tmpl; *p; p++) std::printf("%02x", *p);
        std::printf("\n");
        llama_model_free(model);
        return 0;
    }

    if (want_render) {
        // rendered scaffold / question suffix from the model's own template
        // (hex output: transport-safe; compare across models)
        const PromptParts p = render_prompt(model);
        std::printf("scaffold len=%zu hex=", p.scaffold.size());
        for (unsigned char c : p.scaffold) std::printf("%02x", c);
        std::printf("\nquestion_suffix len=%zu hex=", p.question_suffix.size());
        for (unsigned char c : p.question_suffix) std::printf("%02x", c);
        std::printf("\n");
        llama_model_free(model);
        return 0;
    }

    if (want_thinkscan) {
        // Dump every vocab token whose text contains "think" (hex, transport-safe)
        // to identify the real thinking start/end tags for this model.
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::printf("found=%d\n", (int)find_thinking_tag(vocab, true) >= 0 ? 1 : 0);
        for (int32_t i = 0; i < n_vocab; i++) {
            char piece[256];
            const int32_t n = llama_token_to_piece(vocab, i, piece, (int32_t)sizeof(piece), 0, true);
            if (n < 3 || n > 80) continue;
            std::string l;
            for (int32_t k = 0; k < n; k++) l.push_back((char)std::tolower((unsigned char)piece[k]));
            if (l.find("think") != std::string::npos) {
                std::printf("id=%d n=%d hex=", i, n);
                for (int32_t k = 0; k < n; k++) std::printf("%02x", (unsigned char)piece[k]);
                std::printf("\n");
            }
        }
        llama_model_free(model);
        return 0;
    }

    if (want_tokens) {
        const std::vector<Segment> segs = build_segments(model);
        std::string full;
        for (const auto & s : segs) full += s.text;
        const std::vector<llama_token> toks = tokenize(vocab, full);
        std::printf("n_tokens=%zu\n", toks.size());
        for (size_t i = 0; i < toks.size() && i < 12; i++) {
            char piece[256];
            const int32_t n = llama_token_to_piece(vocab, toks[i], piece, (int32_t)sizeof(piece), 0, true);
            std::printf("[%2zu] id=%6d %.*s\n", i, (int)toks[i], n, piece);
        }
        for (size_t i = toks.size() - 8; i < toks.size(); i++) {
            char piece[256];
            const int32_t n = llama_token_to_piece(vocab, toks[i], piece, (int32_t)sizeof(piece), 0, true);
            std::printf("[%zu] id=%6d %.*s\n", i, (int)toks[i], n, piece);
        }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        for (int32_t i = 0; i < n_vocab; i++) {
            char piece[256];
            const int32_t n = llama_token_to_piece(vocab, i, piece, (int32_t)sizeof(piece), 0, true);
            if (n > 6 && (strstr(piece, "im_start") != nullptr || strstr(piece, "im_end") != nullptr ||
                          strstr(piece, "user") != nullptr || strstr(piece, "assistant") != nullptr ||
                          strstr(piece, "system") != nullptr)) {
                std::printf("vocab: id=%d n=%d ", i, n);
                for (int32_t k = 0; k < n; k++) std::printf("%02x", (unsigned char)piece[k]);
                std::printf("\n");
            }
        }
        llama_model_free(model);
        return 0;
    }

    const std::vector<Segment> segments = build_segments(model);
    std::string full;
    for (const auto & s : segments) full += s.text;
    const std::vector<llama_token> all = tokenize(vocab, full);
    const std::vector<int32_t> b = segment_bounds(vocab, segments);
    const int32_t n_prompt = (int32_t)all.size();

    std::printf("prompt: %d tokens | A [%d,%d) | filler [%d,%d) | C [%d,%d) | D [%d,%d) | Q [%d,%d)\n",
                n_prompt, b[0], b[1], b[1], b[2], b[2], b[3], b[3], b[4], b[4], n_prompt);

    // Seam sanity: segment i (i>=1) starts at b[i-1]; detokenizing
    // all[b[i-1]..] must start with segment i's text.
    {
        bool seams_ok = true;
        for (int i = 1; i < (int)segments.size(); i++) {
            std::string piece;
            for (int32_t j = b[i - 1]; j < std::min<int32_t>(b[i - 1] + 6, n_prompt); j++) {
                char buf[256];
                const int32_t n = llama_token_to_piece(vocab, all[j], buf, (int32_t)sizeof(buf), 0, false);
                if (n > 0) piece.append(buf, n);
            }
            const size_t k = std::min(std::min<size_t>(8, piece.size()), segments[i].text.size());
            const bool ok = k > 0 && piece.compare(0, k, segments[i].text, 0, k) == 0;
            if (!ok) seams_ok = false;
            std::printf("  seam[%d] pos %4d: %s  (expect '%s...')\n",
                        i, b[i - 1], ok ? "OK" : "MISMATCH", segments[i].text.substr(0, 12).c_str());
        }
        if (!seams_ok) {
            fprintf(stderr, "fatal: segment seam mismatch — boundary positions are wrong\n");
            llama_model_free(model);
            return 1;
        }
    }

    auto make_ctx = [&]() -> llama_context * {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;   // n_seq_max = 1 -> n_ctx_seq = n_ctx
        cparams.n_batch = 2048;  // a single llama_decode call may not exceed n_batch
        cparams.n_seq_max = 1;   // single sequence: whole n_ctx usable
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "fatal: ctx init failed\n");
            exit(1);
        }
        return ctx;
    };

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // A seq_rm rejection is a MEANINGFUL architecture-gate result, not a bug:
    // hybrid/SSM memory (e.g. qwen35, 3/4 linear-attention layers) cannot
    // partially erase the recurrent state (only full reset or tail rollback),
    // so middle-range removal fails by design. Report it as a structured
    // verdict (exit 3) — "does partial-range removal work on a hybrid?" is
    // then answered by the probe itself, in minutes.
    std::string baseline_answer;
    auto seq_rm_rejected = [&](int32_t lo, int32_t hi) -> int {
        std::printf("\n=== SEQ_RM REJECTED (architecture gate) ===\n");
        std::printf("  llama_memory_seq_rm(seq 0, [%d,%d)) returned false\n", lo, hi);
        std::printf("  The memory backend does not support middle-range removal\n");
        std::printf("  (hybrid/SSM: the recurrent state cannot be partially erased —\n");
        std::printf("   only full reset or tail rollback are supported).\n");
        std::printf("  => seq_rm cannot express -inf masking on this model;\n");
        std::printf("     the KQ mask-injection path is required instead.\n");
        if (!baseline_answer.empty()) {
            std::printf("  baseline (no seq_rm) answered: %s\n", baseline_answer.c_str());
            std::printf("  (the baseline is valid: it needs no removal)\n");
        }
        std::printf("\nVERDICT: SEQ_RM_REJECTED (middle-range removal unsupported)\n");
        return 3;
    };

    // ---- behavioral runs (leak-free: question prefilled AFTER removal) ----
    bool beh_baseline = false, beh_maskedA = false, beh_maskedC = false;
    const struct { const char * name; int rm_lo; int rm_hi; } behs[] = {
        {"baseline", -1, -1},
        {"masked-A", b[0], b[1]},
        {"masked-C", b[2], b[3]},
    };
    for (const auto & beh : behs) {
        std::printf("\n=== %s (behavioral) ===\n", beh.name);
        llama_context * ctx = make_ctx();
        llama_memory_t mem = llama_get_memory(ctx);
        prefill(ctx, all, 0, b[4], vocab, nullptr);  // scaffold+A+filler+C+D, no question
        if (beh.rm_lo >= 0) {
            if (!llama_memory_seq_rm(mem, 0, beh.rm_lo, beh.rm_hi)) {
                return seq_rm_rejected(beh.rm_lo, beh.rm_hi);
            }
            std::printf("  seq_rm: seq 0 pos [%d,%d) removed\n", beh.rm_lo, beh.rm_hi);
        }
        std::vector<float> q_logits;
        prefill(ctx, all, b[4], n_prompt, vocab, &q_logits);  // question, AFTER removal
        const float * first_logits = q_logits.data() + (int64_t)(q_logits.size() / n_vocab - 1) * n_vocab;
        const std::string answer = generate(ctx, vocab, first_logits, n_prompt, max_tokens);
        std::printf("  answer: %s\n", answer.c_str());
        if (beh.rm_lo == -1) baseline_answer = answer;
        llama_free(ctx);
        const bool has_zebra = contains_ci(answer, "ZEBRA-42");
        if (beh.rm_lo == -1) beh_baseline = has_zebra;
        if (beh.rm_lo == b[0]) beh_maskedA = !has_zebra;
        if (beh.rm_lo == b[2]) beh_maskedC = has_zebra;
    }

    // ---- logits: keep-A (control) vs rm-A — must differ ----
    std::vector<float> keep_q, rm_q;
    {
        std::printf("\n=== keep-A (chunk A present, control) ===\n");
        llama_context * ctx = make_ctx();
        prefill(ctx, all, 0, b[4], vocab, nullptr);
        prefill(ctx, all, b[4], n_prompt, vocab, &keep_q);
        llama_free(ctx);
    }
    {
        std::printf("\n=== rm-A (chunk A removed, question prefilled after) ===\n");
        llama_context * ctx = make_ctx();
        llama_memory_t mem = llama_get_memory(ctx);
        prefill(ctx, all, 0, b[4], vocab, nullptr);
        if (!llama_memory_seq_rm(mem, 0, b[0], b[1])) {
            return seq_rm_rejected(b[0], b[1]);
        }
        std::printf("  seq_rm: seq 0 pos [%d,%d) removed\n", b[0], b[1]);
        prefill(ctx, all, b[4], n_prompt, vocab, &rm_q);
        llama_free(ctx);
    }

    const LogitDiff d_keep_rm = compare_logits(keep_q, rm_q, vocab);
    std::printf("\nLOGITS (question %d positions)\n", d_keep_rm.n_rows);
    print_diff("  [1] keep-A vs rm-A (expect DIFFERENT — control)", d_keep_rm, vocab);

    const bool logits_control_ok = !d_keep_rm.last_match;

    std::printf("\nRESULTS\n");
    std::printf("  baseline  : %s (expect ZEBRA-42)\n", beh_baseline ? "PASS" : "FAIL");
    if (!beh_baseline) {
        std::printf("  ** baseline failed — all other results are INVALID **\n");
    }
    std::printf("  masked-A  : %s (expect no ZEBRA-42)\n", beh_maskedA ? "PASS" : "FAIL");
    std::printf("  masked-C  : %s (expect ZEBRA-42)\n", beh_maskedC ? "PASS" : "FAIL");
    std::printf("  control keep-A != rm-A: %s (last-pos argmax must differ)\n", logits_control_ok ? "PASS" : "FAIL");

    const bool all_ok = beh_baseline && beh_maskedA && beh_maskedC && logits_control_ok;
    std::printf("\nOVERALL: %s\n", all_ok ? "PASS" : "FAIL");
    llama_model_free(model);
    return all_ok ? 0 : 1;
}
