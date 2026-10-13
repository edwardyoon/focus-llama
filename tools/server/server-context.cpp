#include "da-trace.h"
#include "da-tag.h"
#include "da-layout.h"
#include "recall-sanitize.h"
#include "server-context.h"
#include <atomic>
#include <thread>
#include "base64.hpp"
#include "server-chat.h"
#include "server-common.h"
#include "server-decision.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-realtime.h"
#include "server-schema.h"
#include "server-slot-sched.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "http.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "base64.hpp"

#include <string>
#include <vector>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstddef>
#include <cinttypes>
#include <ctime>
#include <deque>
#include <exception>
#include <functional>
#include <iomanip>
#include <memory>
#include <numeric>
#include <filesystem>
#include <random>
#include <regex>
#include <thread>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <chrono>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

// Upper bound on the generation tail (in KV cells) that the decoupled-KV
// reject gate and eviction clamp reserve. Reserving the full request n_predict
// (often 32768) makes resident + recall + gen exceed the buffer for most real
// prompts and causes spurious 400s; actual generation is far shorter. If a
// generation does outgrow this reservation, the buffer-cap graceful stop
// (process_token) ends it with finish_reason=length instead of a hard decode
// failure, so the full n_predict need not be reserved.
static const int64_t KV_GEN_TAIL_CAP = 8192;

// kv-offload (auto-compact replacement): kv_offload_refill is defined below
// (before da_auto_chunk) but apply_da_tag (mid-decode) calls it earlier, so it
// must be forward-declared. kv_offload_get is a free function from
// kv-offload.h, so it needs no local declaration.
struct server_slot;
// out_last_idx (optional): on success, receives the batch index (within the
// final sub-batch) of the refilled block's last token - the only output
// token of the final llama_decode, i.e. the logits row to sample the next
// generated token from (B1: the deferred refill must be the last decode
// before the slot's next sample).
// on_retry_idle (optional): called on a decode/draft failure - free an idle
// slot's KV cells (mirrors decode()'s try_clear_idle_slots); return true if a
// slot was cleared (the same sub-batch is retried).
// out_safe_failopen (optional): set true while no decode has run yet - a
// failure at that point (e.g. the free-cell hard gate) leaves the ctx output
// buffer holding the MAIN batch's rows, so the caller can fail-open (drop the
// recall, sample the closing token's row, keep the slot and its cache). Flipped
// to false once the decode loop starts (the buffer is overwritten; a failure
// then must abort the slot).
static bool kv_offload_refill(llama_context * ctx, server_slot & slot, const llama_tokens & toks, int32_t * out_last_idx = nullptr,
                              const std::function<bool()> & on_retry_idle = nullptr, bool * out_safe_failopen = nullptr);

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (!params.mmproj.path.empty()) {
        const auto mcaps = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        if (mcaps.gen_audio) {
            // some TTS models decode with embeddings output for the whole batch
            return { params.n_batch, params.n_batch };
        }
    }

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<llama_token> server_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < draft.size(); ++i) {
        // the token is discarded - the call is what advances the sampler over this position
        common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, draft[i], true);
        result.push_back(draft[i]);
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    common_batch view; // the rendered sub-batch [off, off + n_tokens), see render()

    struct token {
        int32_t id_slot;
        llama_token token;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        bool output;
        bool is_prompt; // for stats tracking
        int32_t decision_order = 0;
        int32_t i_embd = -1; // row in embd, -1 if this is a token
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // embedding entries, they can be mixed with tokens if the context supports it
    std::vector<float> embd; // n_embd per row

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    void init(llama_context * ctx, int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        view = common_batch(ctx);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, { pos, 0, 0, 0 }, output, is_prompt });
        return true;
    }

    // embd_in has n_embd values, pos has GGML_MROPE_SECTIONS values
    bool add_embd(int32_t id_slot, const float * embd_in, const llama_pos * pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, { pos[0], pos[1], pos[2], pos[3] }, output, is_prompt });
        tokens.back().i_embd = (int32_t) (embd.size() / n_embd);
        embd.insert(embd.end(), embd_in, embd_in + n_embd);
        return true;
    }

    bool has_embd() const {
        return !embd.empty();
    }

    void clear() {
        tokens.clear();
        embd.clear();
        view.clear();
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // remove the entries after the first n
    void truncate(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= size());
        for (int32_t i = n; i < size(); i++) {
            if (tokens[i].i_embd >= 0) {
                embd.resize((size_t) tokens[i].i_embd * n_embd);
                break;
            }
        }
        tokens.resize(n);
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void set_decision_order(int32_t idx, int32_t order) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].decision_order = order;
    }

    // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
    void render(int32_t off, int32_t n_tokens) {
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        view.clear();
        for (int32_t i = off; i < off + n_tokens; i++) {
            const auto & t = tokens[i];
            if (t.i_embd >= 0) {
                view.add_embd({ embd.data() + (size_t) t.i_embd * n_embd, 1, (size_t) n_embd }, t.pos.data(), t.id_slot, t.output);
            } else {
                view.add(t.token, t.pos[0], t.id_slot, t.output);
            }
            view.tokens.back().decision_order = t.decision_order;
        }
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    struct tts_ctx {
        mtmd_helper::gen_audio ctx;
        std::vector<float> h_state_prompt; // the first h_state after step_prompt done
        const float * h_state;
        llama_token   sampled;
        int32_t       n_decoded;
        bool          stop;
        bool is_supported() const {
            return ctx.valid();
        }
        void reset() {
            ctx.reset();
            h_state_prompt.clear();
            h_state   = nullptr;
            sampled   = LLAMA_TOKEN_NULL;
            n_decoded = 0;
            stop      = false;
        }
    };
    tts_ctx tts;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    common_chat_input generated;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    // Declarative Attention: da_rm ranges not yet applied to the KV cache
    bool da_rm_pending = false;

    // one-shot diagnostic emitted when the da_rm_at boundary is crossed
    // without applying the removals (e.g. media present)
    bool da_rm_gate_diag_done = false;

    // true when the prompt contains no media chunks (text tokens == all
    // tokens). has_mtmd is true whenever an mmproj is loaded, even for
    // text-only prompts, so the mid-prefill guards test this instead.
    // Computed once at request start: get_text_tokens() is O(n) and the
    // batch-fill loop runs per token.
    bool da_rm_text_only = false;

    // Declarative Attention 2-stream (B): >= 0 once the slot's decode has been
    // switched to a second sequence (the keep ranges were copied there via
    // seq_cp, see apply_da_b). All KV operations and batch tokens of the slot
    // then use kv_seq() instead of `id`; the original sequence `id` is left
    // intact. Reset in reset() and its KV cleared in prompt_clear().
    llama_seq_id da_seq = -1;
    // set by apply_da_b: the removal boundary position and the number of
    // prefix tokens copied into da_seq (the logical read set of the second
    // stream - the attended tokens, not the physical KV scan). Used by the
    // read-reduction logs (per-step DBG, final INF in release()).
    int32_t da_bound = -1;
    size_t  da_keep_count = 0;
    // DA mode state machine (P3) - see da_mode_t above.
    da_mode_t da_mode = DA_MODE_GLOBAL;
    // chunk indices (0-based, into task->params.da_chunks) currently kept:
    // B path = present on da_seq, A path = not removed. Only meaningful when
    // da_mode != DA_MODE_GLOBAL.
    std::vector<int32_t> da_keep_chunks;
    // DA tag scanner: char offset in generated_text up to which the text has
    // already been scanned for complete tags (each tag is consumed exactly
    // once; a tag may span several generated tokens).
    size_t da_tag_scan_pos = 0;
    // DA tag gate (P6): set by apply_da_tag() when a control tag closed -
    // the next draft step is skipped so the first token after the mode
    // transition follows the single-token trajectory (a batch-verified token
    // at the transition can diverge numerically and drop the answer).
    // Consumed by the draft decision.
    bool da_skip_draft = false;
    // B1 fix (2026-10-06 14:21): the tag-closing token is sampled in
    // post_decode() but committed to the KV only in the NEXT predict()'s
    // handle_last_sampled_token(). Refilling at tag close inserted the
    // recalled chunk BEFORE that token - the model's own tag got split
    // across the recalled block and the continuation derailed (a
    // mid-tool-call <focus> leaked a headless tool-call tail). The refill
    // tokens are now queued here and applied in the next post_decode() -
    // after the closing token is committed - and the next token is sampled
    // from the refill's last logits so the model continues after the
    // recalled content. Request-scoped (cleared in reset()).
    std::vector<llama_token> da_pending_refill;
    // batch index (within the refill's final sub-batch) of the last
    // refilled token's logits row; the next sample is taken from it
    // (-1 = none this step). Request-scoped.
    int32_t da_refill_out_idx = -1;
    // da_trace: number of post-tag tokens still to trace in detail (set at
    // tag close, decremented per generated token). Request-scoped.
    int32_t da_trace_left = 0;
    // P2: true once a DA removal has actually been applied to this slot's KV
    // (A path: holes cut into the original sequence; B path: switched to
    // da_seq). Drives the release() cache policy - a tag request that never
    // emitted a tag keeps its prompt cache like a vanilla request.
    bool da_applied = false;
    // P2: true once an A-path removal (seq_rm holes) was actually applied.
    // B path may fall back to A (apply_da_b preconditions), so the release()
    // cache policy keys on the path actually taken, not on task params:
    // A holes are baked into the cached sequence (clear), a B slot that
    // returned to the original sequence mid-decode keeps its cache.
    bool da_removed_a = false;
    // kv-offload Option B (--kv-offload-holes): holes cut into this slot's MAIN
    // sequence at eviction time (the evicted segments' KV is removed via
    // llama_memory_seq_rm, positions are NOT re-based). They persist across
    // release() between requests: release() keeps the prompt cache (the kv_
    // holes_active branch) so the next request re-matches at n_past=full and
    // re-prefills 1 token instead of the evicted tail.
    //   kv_hole_ranges : applied holes (the ledger) - drives the release()
    //                    cache policy and the n_kv_max dense override.
    // The pending holes (not yet applied) are request-scoped: they live in
    // task_params::kv_hole_pending (set by the eviction gate) and are read
    // directly at the n_past check - NOT stored in the slot (a slot takeover
    // would otherwise apply the previous session's stale holes to the new
    // sequence). Cleared only when the whole sequence is wiped (prompt_clear
    // or the n_past==0 takeover) - reset() must NOT clear them (the holes live
    // in the shared cache, not in the per-request slot state).
    std::vector<std::pair<llama_pos, llama_pos>> kv_hole_ranges;
    bool kv_holes_active = false;
    // kv-offload-holes: the FocusMemory session this slot's KV belongs to
    // (learned from task.params.kv_offload_session at launch, persists across
    // release() while the prompt cache is kept). Key of the context's
    // kv_session_cut entry - lets prompt_clear refresh it even when the slot
    // is idle (task already gone).
    std::string kv_session;
    // kv-offload-holes progressive apply (prefill-time hole cutting): per-hole
    // progress while the prompt is being (re-)prefilled. applied_until: [lo,
    // applied_until) is already cut from the KV. Request-scoped: cleared in
    // reset() and at the STARTED block - the gate re-plans the same holes
    // every request, so progress never carries over. The applied ledger
    // (kv_hole_ranges) records only fully cut holes; a partial cut that
    // survives to the next request is completed by the n_past batch path
    // (re-cutting the full range is idempotent).
    struct kv_hole_progress_entry {
        llama_pos lo = 0;
        llama_pos hi = 0;
        llama_pos applied_until = 0; // [lo, applied_until) already cut
        bool fail_logged = false;    // one-shot warn per rejected cut
    };
    std::vector<kv_hole_progress_entry> kv_hole_progress;
    // peak KV-resident tokens (processed - cut) observed during the
    // progressive prefill cut; 0 = not measured (no progressive cut this
    // request). Logged once at the DONE_PROMPT -> GENERATING transition.
    int32_t kv_hole_peak_resident = 0;
    // set by the n_cache_reuse shift block: the shift moved the KV, so the
    // pending hole coordinates (final token space) are stale - the n_past
    // batch path and the progressive path both defer the holes.
    bool kv_hole_shift_blocked = false;
    // kv-offload refill failure: a partial refill was written to the KV and
    // its attention-KV rollback failed (a recurrent-state hybrid cannot
    // partially roll back), so the slot's KV is inconsistent with
    // prompt.tokens. release() must prompt_clear() this slot instead of
    // keeping the prompt cache. Request-scoped: reset() clears it.
    bool refill_poisoned = false;
    // Phase 0 (measure-only): server-level flag copied at slot init. When set,
    // apply_da_rm/apply_da_b skip the KV mutation (output stays vanilla) so the
    // tag state machine + logging can measure g (tag position) and emission
    // rate on real traffic without touching the KV cache.
    bool da_measure_only = false;
    // DA tail floor (plans/continous_work2.md): the most recent da_tail_keep
    // tokens before the removal boundary are always kept - subtracted from the
    // removal ranges in apply_da_rm/apply_da_b next to the Sigma anchor. The
    // keep set is the union of the tag selection, the Sigma anchor and the
    // tail: without the floor, the work region the model was just in (not
    // covered by any magic_chunks tag) is masked out at a B switch and the
    // model forgets what it just did. Server-level, copied at slot init
    // (0 = legacy behavior).
    int32_t da_tail_keep = 8192;
    // DA 2-stream (B) no-benefit skip threshold (plans/continous_work2.md,
    // optional item): when > 0, apply_da_b skips the switch if fewer than
    // this many tokens would be removed - the switch cost outweighs a tiny
    // removal and the output stays vanilla. 0 = always switch (legacy).
    // Server-level, copied at slot init.
    int32_t da_b_min_remove = 0;
    // Phase 0: generated-token index at which the first <focus>/<local> tag
    // closed (-1 = no restriction tag emitted yet). release() logs it together
    // with the total n_gen to give g = first_tag / total (the global-step
    // share before the first attention restriction).
    int32_t da_first_tag_gen = -1;
    // P5 instrumentation (logical attended tokens, not measured bytes):
    // n_da_removed is the A-path removed-token count (read set = n_full -
    // n_da_removed). The per-step counters live in stats (server_slot_stats)
    // so they reach the response timings via stats.to_json().
    uint32_t n_da_removed = 0;
    llama_seq_id kv_seq() const {
        return da_seq >= 0 ? da_seq : (llama_seq_id) id;
    }

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        const size_t n_tgt = llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t n_dft = ctx_dft ? llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        if (n_tgt != cur_size_tgt || n_dft != cur_size_dft) {
            SLT_ERR(*this, "%s", "failed to save complete prompt state\n");
            prompt_cache.states.pop_back();
            return false;
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_INF(*this, "clearing prompt with %zu tokens (seq %d)\n", prompt.tokens.size(), id);

        mem.seq_rm(id, -1, -1);
        // a 2-stream (B) slot also holds the keep ranges on its second
        // sequence; without this the cells would leak into the unified pool
        // and the next request could pick up the same seq id with stale KV
        if (da_seq >= 0) {
            mem.seq_rm(da_seq, -1, -1);
        }

        prompt.clear();

        // Option B: the whole main sequence is gone, so the applied-hole
        // ledger is stale and must not survive into the next request
        kv_hole_ranges.clear();
        kv_holes_active = false;
        kv_hole_progress.clear();
        // the session's actual-cut entry must drop with the ledger, or the
        // watermark gate keeps subtracting holes that no longer exist
        // (2026-10-10: idle-slot clear lost the ledger, the gate stayed idle
        // and the KV refilled to the buffer cap)
        if (callback_on_kv_holes_change) {
            callback_on_kv_holes_change(*this);
        }
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    // for realtime transcription, these are the rows from the last audio stream step (size = n_rows * n_embd)
    std::vector<float> inp_embd;

    // realtime transcription, see SERVER_TASK_TYPE_TRANSCRIBE_RT
    mtmd::audio_stream_ptr audio_stream;
    bool audio_wait = false; // the stream needs more audio before it can produce rows
    std::deque<int64_t> audio_commit_rows; // end row of each non-final commit not yet completed

    // realtime conversation, see SERVER_TASK_TYPE_REALTIME
    std::unique_ptr<server_realtime_slot> rt;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()
    // kv-offload-holes: refresh the context's kv_session_cut entry after the
    // applied-hole ledger changes (set by the context; no-op if unset)
    std::function<void(const server_slot &)> callback_on_kv_holes_change;

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated      = {};
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        // 2-stream (B) switch + DA state machine are request-scoped
        da_seq           = -1;
        da_bound         = -1;
        da_keep_count    = 0;
        da_mode          = DA_MODE_GLOBAL;
        da_keep_chunks.clear();
        da_tag_scan_pos  = 0;
        da_skip_draft    = false;
        da_pending_refill.clear();
        da_refill_out_idx = -1;
        da_trace_left     = 0;
        da_applied       = false;
        da_removed_a     = false;
        n_da_removed     = 0;
        da_first_tag_gen = -1;

        // kv-offload refill poison is request-scoped: the poisoned KV dies with the
        // prompt cache at release(), so the flag must not leak into the next request
        // (unlike kv_holes_active, which persists in the shared cache).
        refill_poisoned = false;

        // kv-offload-holes progressive apply: request-scoped progress (the
        // applied ledger kv_hole_ranges persists in the shared cache)
        kv_hole_progress.clear();
        kv_hole_peak_resident = 0;
        kv_hole_shift_blocked = false;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();

        tts.reset();

        audio_stream.reset();
        audio_wait = false;
        audio_commit_rows.clear();
        inp_embd.clear();
        rt.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool can_split() const {
        GGML_ASSERT(task);
        // a decision over a non-causal context reads the whole prompt in one batch
        if (task->type == SERVER_TASK_TYPE_DECISION && !llama_get_causal_attn(ctx_tgt)) {
            return false;
        }
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!task->need_embd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (!llama_get_memory(ctx_tgt)) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const auto pooling = llama_pooling_type(ctx_tgt);
        if (pooling == LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == LLAMA_POOLING_TYPE_RANK && llama_get_causal_attn(ctx_tgt)) {
            return true;
        }
        // a decision task reads its outputs from the last batch
        if (task->type == SERVER_TASK_TYPE_DECISION) {
            return true;
        }
        return false;
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        // a joint decision head reads the whole batch
        if (!task->decision.order.empty() || !other_slot.task->decision.order.empty()) {
            return false;
        }

        // lfm3 d1 has no memory, its prompt is read as one flat ubatch and the short convs would mix the sequences
        // TODO: batch them along the sequence dimension, this needs ubatches of sequences with unequal lengths
        if (task->decision.no_batching || other_slot.task->decision.no_batching) {
            return false;
        }

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // DA 2-pass overhead accounting: count a mode transition only when the
    // mode actually changes, and attribute each generated token to the mode
    // active at sample time (a closing tag is processed after its last token
    // is counted, so the tag tokens are attributed to the mode they were
    // generated in - the global "search" phase, not the focus "re-reason").
    void da_set_mode(da_mode_t m) {
        if (da_mode != m) {
            stats.da_n_transitions += 1;
            da_mode = m;
        }
    }
    void da_count_generated_token() {
        switch (da_mode) {
            case DA_MODE_GLOBAL: stats.da_tokens_global += 1; break;
            case DA_MODE_FOCUS:  stats.da_tokens_focus  += 1; break;
            case DA_MODE_LOCAL:  stats.da_tokens_local  += 1; break;
        }
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        // the audio stream has no token before the first step, track the position with a dummy token
        const llama_token tok_track = sampled == LLAMA_TOKEN_NULL ? 0 : sampled;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                const int32_t n_rows = (int32_t) inp_embd.size() / batch.n_embd;
                GGML_ASSERT((int32_t) inp_embd.size() == n_rows * batch.n_embd);
                i_batch = batch.size() + n_rows - 1;
                for (int32_t i = 0; i < n_rows; i++) {
                    // text embeddings broadcast the same position across the M-RoPE sections
                    const llama_pos p = prompt.tokens.pos_next();
                    const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
                    add_ok &= batch.add_embd(kv_seq(), inp_embd.data() + (size_t) i * batch.n_embd, pos, i == n_rows - 1, false);
                    if (i < n_rows - 1) {
                        prompt.tokens.push_back(tok_track);
                    }
                }
                if (audio_stream) {
                    inp_embd.clear(); // rows are consumed, the next step produces new ones
                }
            } else {
                add_ok &= batch.add(kv_seq(), sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(kv_seq(), sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->kv_seq(), token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        // L4: the tokens already in the prompt just before this commit -
        // if a refill was (wrongly) inserted earlier, its tail shows up here
        if (da_trace_left > 0) {
            std::string s;
            const size_t n = prompt.tokens.size();
            for (size_t i = n > 12 ? n - 12 : 0; i < n; i++) {
                const llama_token t = prompt.tokens[i];
                if (t != LLAMA_TOKEN_NULL) {
                    s += string_format("%d:'%s' ", (int) t, dbg_esc(common_token_to_piece(ctx_tgt, t, true), 24).c_str());
                }
            }
            da_trace("slot=%d task=%d L4 commit sampled=%d pos=%d seq=%d prev12=[%s]\n",
                     (int) id, (int) (task ? task->id : -1),
                     (int) sampled, (int) prompt.tokens.pos_next(), (int) kv_seq(), s.c_str());
        }

        prompt.tokens.push_back(tok_track);
        prompt.tokens.insert(spec_draft);

        // 2-stream (B) read-reduction log: the logical read set (attended
        // tokens) is the kept prefix plus the tokens added after the boundary
        // ("read 30 of 100"). The FA kernels still scan the full [0, n_kv)
        // range but skip fully-masked chunks (see apply_da_b), so the
        // physical KV load shrinks at chunk granularity; this number is the
        // logical (attended) count, not a measured byte count.
        if (da_seq >= 0) {
            const int32_t n_full = (int32_t) prompt.n_tokens();
            const int32_t n_read = (int32_t) da_keep_count + (n_full - da_bound);
            stats.n_da_restricted_steps += 1;
            stats.n_da_attended_tokens  += (uint64_t) n_read;
            stats.da_removed_a          = false;
            SLT_DBG(*this, "da_b: decode #%d: logical read set %d of %d token(s) on seq %d (%zu kept + %d after boundary %d; seq %d holds all %d)\n",
                    n_full - 1, n_read, n_full, da_seq, da_keep_count, n_full - da_bound, da_bound, id, n_full);
        } else if (da_applied && n_da_removed > 0) {
            // A path: the holes stay in the original sequence; the logical
            // read set is the full length minus the removed tokens
            const int32_t n_full = (int32_t) prompt.n_tokens();
            const int32_t n_read = std::max(0, n_full - (int32_t) n_da_removed);
            stats.n_da_restricted_steps += 1;
            stats.n_da_attended_tokens  += (uint64_t) n_read;
            stats.da_removed_a          = true;
        }
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            da_trace("slot=%d task=%d L7 release stop=%d prompt=%d holes=%zu holes_active=%d da_seq=%d da_applied=%d mode=%d poisoned=%d pending_refill=%zu\n",
                     (int) id, (int) (task ? task->id : -1), (int) stop, (int) prompt.n_tokens(),
                     kv_hole_ranges.size(), (int) kv_holes_active, (int) da_seq, (int) da_applied, (int) da_mode, (int) refill_poisoned,
                     da_pending_refill.size());

            // 2-stream (B) final read-reduction summary (logical read set, see
            // handle_last_sampled_token)
            if (da_seq >= 0) {
                const int32_t n_full = (int32_t) prompt.n_tokens();
                const int32_t n_read = (int32_t) da_keep_count + (n_full - da_bound);
                SLT_INF_DBG(*this, "da_b: finished on seq %d - final logical read set %d of %d token(s) (%zu kept prefix + %d after boundary %d); seq %d kept intact\n",
                        da_seq, n_read, n_full, da_keep_count, n_full - da_bound, da_bound, id);
            }

            // P5: DA instrumentation summary (logical attended tokens, not
            // measured bytes) - also exposed in the response timings
            if (stats.n_da_restricted_steps > 0) {
                const uint64_t n_full = prompt.n_tokens();
                const uint64_t n_full_steps = n_full * stats.n_da_restricted_steps;
                const double reduction = n_full_steps > 0
                        ? 100.0 * (1.0 - (double) stats.n_da_attended_tokens / (double) n_full_steps) : 0.0;
                SLT_INF_DBG(*this, "da: %s path - %lu restricted step(s), logical attended %lu of %lu token-step(s) (-%.1f%%)\n",
                        stats.da_removed_a ? "A" : "B",
                        (unsigned long) stats.n_da_restricted_steps,
                        (unsigned long) stats.n_da_attended_tokens,
                        (unsigned long) n_full_steps, reduction);
            }

            // 2-pass overhead summary: mode-transition count and the
            // generated-token share per mode (global = search phase, focus =
            // re-reason phase). The transition sequence itself is in the
            // da_tag: lines above.
            if (stats.da_n_transitions > 0) {
                SLT_INF_DBG(*this, "da_overhead: %lu transition(s), generated tokens by mode: global %lu / focus %lu / local %lu (of %lu total)\n",
                        (unsigned long) stats.da_n_transitions,
                        (unsigned long) stats.da_tokens_global,
                        (unsigned long) stats.da_tokens_focus,
                        (unsigned long) stats.da_tokens_local,
                        (unsigned long) stats.n_gen);
            }

            // Phase 0 (measure-only): report the tag-position distribution
            // (g) on real traffic. g = first_tag / total is the global-step
            // share before the first attention restriction; the break-even
            // gate uses (1+d)*(g + (1-g)*r). Only logged in measure-only mode
            // (the production path keeps its current summary).
            if (da_measure_only && da_first_tag_gen >= 0) {
                const double total = (double) stats.n_gen;
                const double g     = total > 0 ? (double) da_first_tag_gen / total : 0.0;
                SLT_INF_DBG(*this, "da_measure: first <focus>/<local> tag at generated token %d of %lu - g=%.4f (global-step share before restriction)\n",
                        da_first_tag_gen, (unsigned long) stats.n_gen, g);
            }

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // DA cache policy (P2):
            // - child / stateless / audio / realtime slot: always clear (no reuse)
            // - static da_rm request: KV holes in the original sequence -> clear
            // - DA tag request that applied a removal (da_applied):
            //     B path: the original sequence is intact; copy the tail
            //             generated on da_seq back so it holds the full
            //             conversation, free da_seq, and KEEP the prompt cache
            //             (the next turn reuses the complete prefix)
            //     A path: holes persist in the original sequence -> clear
            // - DA tag request that never emitted a tag: cache kept (vanilla)
            // - kv-offload get-on-focus refill: the refilled chunk is appended
            //   at the tail with KV and prompt.tokens in sync, so it is just part
            //   of the generated tail - the B path copies it back (or the default
            //   keeps it) and the next request's prefix match (get_common_prefix
            //   + keep_first + seq_rm) trims it. Keeping the cache (not the old
            //   prompt_clear) avoids the full re-prefill stall.
            if (refill_poisoned) {
                // kv-offload refill failed mid-way and its attention-KV rollback was a
                // no-op (recurrent-state hybrid cannot partially roll back): the KV is
                // inconsistent with prompt.tokens, so the prompt cache cannot be kept.
                SLT_INF(*this, "clearing slot after poisoned kv-offload refill (KV rollback no-op): %zu prompt tokens\n",
                        prompt.tokens.size());
                prompt_clear();
            } else if (task->is_child() || task->is_stateless() || audio_stream || task->type == SERVER_TASK_TYPE_REALTIME) {
                prompt_clear();
            } else if (!task->params.da_rm.empty()) {
                SLT_INF(*this, "clearing slot after da_rm request: %zu prompt tokens, %zu range(s)\n",
                        prompt.tokens.size(), task->params.da_rm.size());
                prompt_clear();
            } else if (da_applied && da_seq >= 0) {
                const int32_t n_full = (int32_t) prompt.n_tokens();
                if (n_full > da_bound) {
                    mem.seq_cp(da_seq, id, da_bound, n_full);
                }
                mem.seq_rm(da_seq, -1, -1);
                SLT_INF_DBG(*this, "da_b: returned %d generated token(s) to seq %d - prompt cache kept\n",
                        n_full - da_bound, id);
                da_seq = -1;
            } else if (da_removed_a) {
                // A path (or a B request that fell back to A): holes persist
                // in the original sequence -> the cached prompt is not
                // reusable
                SLT_INF(*this, "clearing slot after DA removal (A path, holes persist): %zu prompt tokens\n",
                        prompt.tokens.size());
                prompt_clear();
            } else if (kv_holes_active) {
                // Option B (--kv-offload-holes): the evicted segments' KV was
                // cut out of the main sequence (positions not re-based) but
                // the prompt text is unchanged, so the next request re-matches
                // at n_past=full and re-prefills 1 token instead of the
                // evicted tail -> KEEP the prompt cache. (Placed after the DA
                // branches on purpose: da_applied must still get its tail
                // copy-back and da_removed_a must still clear - those win.)
                SLT_INF_DBG(*this, "kv-offload-holes: prompt cache kept (%zu prompt tokens, %zu hole range(s), %d token span)\n",
                        prompt.tokens.size(), kv_hole_ranges.size(),
                        std::accumulate(kv_hole_ranges.begin(), kv_hole_ranges.end(), 0,
                                        [](int s, const auto & r) { return s + (int) (r.second - r.first); }));
                // kv-probe (release): physical occupancy at cache-keep time, before
                // any checkpoint restore of the next turn. used (buffer - free)
                // already far above the ledger-based resident here means the cut
                // never happened (H1); a jump only after restore means H2
                // (2026-10-10 2nd incident).
                {
                    const int32_t n_free = llama_memory_n_free_cells(llama_get_memory(mem.ctx_tgt));
                    int64_t probe_cut = 0;
                    for (const auto & r : kv_hole_ranges) {
                        probe_cut += (int64_t) (r.second - r.first);
                    }
                    SLT_INF_DBG(*this, "kv-probe (release): free=%d ledger_cut=%d prompt=%d\n",
                            n_free, (int) probe_cut, (int) prompt.n_tokens());
                }
            }
            // a B slot that returned to the original sequence mid-decode
            // (</focus> before the end) has a plain sequence again: the
            // prompt cache is kept, like a vanilla request

            // mtmd_helper always re-eval the whole prompt
            if (task->type == SERVER_TASK_TYPE_TTS) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        SLT_INF(*this,
                "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
                llama_perf_context(ctx_tgt).n_reallocs_same, llama_perf_context(ctx_tgt).n_reallocs_diff);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated.empty() ? debug_generated_text : generated.text;
            }
        }

        return res;
    }

    // the other slot continues from the tokens processed so far
    void copy_prompt_to(server_slot & other) const {
        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.prompt = prompt.clone();
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// encode the mtmd chunk at idx, batched with as many of the next chunks as possible
// returns 0 on success
static int encode_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx) {
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);

    mbatch.reset(mtmd_batch_init(mctx));
    int32_t res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return 0;
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                struct cb_data_t {
                    common_speculative * spec;
                    llama_context * ctx;
                } cb_data = { slot.spec, slot.ctx_tgt };

                static auto cb = [](const mtmd_helper_embd_batch * b, void * user_data) {
                    const auto * data = static_cast<cb_data_t *>(user_data);

                    common_batch batch(data->ctx);
                    for (int32_t i = 0; i < b->n_tokens; ++i) {
                        llama_pos pos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < b->n_pos; ++j) {
                            pos[j] = b->pos[j * b->n_tokens + i];
                        }
                        batch.add_embd({ b->embd + (size_t) i * b->n_embd, 1, (size_t) b->n_embd }, pos, b->seq_id, false);
                    }

                    return common_speculative_process(data->spec, batch) ? 0 : 1;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    &cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    if (encode_mtmd_chunk(slot, mbatch, idx) != 0) {
        return -1;
    }

    return try_decode();
}

// add an encoded mtmd chunk to the batch of the text tokens, instead of decoding it on its own like process_mtmd_chunk()
// embd is the output of the encoder for this chunk
// returns false if the batch is full
static bool add_mtmd_chunk(const server_slot & slot, const mtmd_input_chunk * chunk, const float * embd, server_batch & batch) {
    // positions are the ones of mtmd_helper_decode_image_chunk()
    const auto * image    = mtmd_input_chunk_get_tokens_image(chunk);
    const bool   is_mrope = mtmd_decode_use_mrope(slot.mctx);
    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_embd   = batch.n_embd;
    const llama_pos pos_0 = slot.prompt.tokens.pos_next();

    for (size_t i = 0; i < n_tokens; i++) {
        const llama_pos p = pos_0 + (llama_pos) i;
        llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, p };
        if (is_mrope && image) {
            const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
            pos[0] = rel.t;
            pos[1] = rel.y;
            pos[2] = rel.x;
            pos[3] = rel.z;
        }
        if (!batch.add_embd(slot.id, embd + i * n_embd, pos, slot.need_embd(), /* is_prompt */ true)) {
            return false;
        }
    }
    return true;
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    mtmd_helper_cap mtmd_cap = {};
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    // prompt of the "asr" chat template, for live ASR
    llama_tokens asr_prompt;

    bool has_cap_tts() const {
        return mtmd_cap.audio_gen;
    }

    bool has_cap_chat() const {
        return mtmd_cap.chat;
    }

    server_decision_context decision;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

    // kv-offload-holes: refresh the session's actual-cut entry from the
    // slot's applied-hole ledger (kv_hole_ranges). Called at every ledger
    // mutation (n_past apply/expire, progressive-cut completion, prompt_clear)
    // so the watermark gate sees the holes that are physically cut in the KV
    // right now, not the cumulative PUT history (see the kv_session_cut
    // member for why). No-op for slots without a known session (plain
    // traffic, or before the first da_auto turn). Public: the watermark gate
    // in server_routes reads the entry via kv_session_cut_get().
    void kv_session_cut_update(const server_slot & slot) {
        if (slot.kv_session.empty()) {
            return;
        }
        int64_t cut = 0;
        for (const auto & r : slot.kv_hole_ranges) {
            cut += (int64_t) (r.second - r.first);
        }
        std::lock_guard<std::mutex> lk(kv_session_cut_mutex);
        kv_session_cut[slot.kv_session] = cut;
        // 2026-10-12: mirror the ranges too - the gate needs per-range
        // coordinates to apply the expire rule (the total alone cannot say
        // which holes survive the new n_past).
        kv_session_holes[slot.kv_session] = slot.kv_hole_ranges;
    }

    // kv-offload-holes: the actual-cut count for the watermark gate. 0 when
    // the session has no (known) cut holes - e.g. a fresh session, or after
    // its ledger was dropped by a prompt_clear.
    int64_t kv_session_cut_get(const std::string & session) {
        std::lock_guard<std::mutex> lk(kv_session_cut_mutex);
        const auto it = kv_session_cut.find(session);
        return it != kv_session_cut.end() ? it->second : 0;
    }

    // kv-offload-holes (2026-10-12): the shared hole-expire rule (formerly
    // inline at task start): a hole survives a re-prefill at n_past iff its
    // KV range is entirely inside the matched prefix (hi <= n_past);
    // otherwise its cells were freshly rewritten and the cut no longer
    // applies. Returns the surviving ranges. Static so the task-start path
    // (queue thread, the real n_past) and the gate's kv_session_valid_cut
    // (HTTP thread, the simulated n_past) use the exact same rule.
    static std::vector<std::pair<int32_t,int32_t>> kv_holes_survive(
            const std::vector<std::pair<int32_t,int32_t>> & ranges, int64_t n_past) {
        std::vector<std::pair<int32_t,int32_t>> out;
        for (const auto & r : ranges) {
            if (r.second <= n_past) {
                out.push_back(r);
            }
        }
        return out;
    }

    // kv-offload-holes (2026-10-12): the gate-side valid-cut estimate (HTTP
    // thread). The cut-ledger total (kv_session_cut_get) overcounts the holes
    // that survive this request: the client rewrites/compacts the history, so
    // the new prompt's LCP with the previous one (the task-start n_past) is
    // far below the previous prompt size, and every hole with hi > n_past
    // expires (its KV is re-prefilled). Projecting resident = prompt -
    // ledger_cut then pins the gate "idle" while the physical KV re-fills to
    // the buffer cap (2026-10-12 13:35-13:54: 11x full re-prefill, ~55s
    // each, then the find_slot cascade).
    //
    // The LCP is computed against the session's last accepted prompt (the
    // kv_last_prompt copy - the slot's residual prompt is queue-thread owned
    // and must not be read from the HTTP thread). The effective n_past is the
    // LCP minus the given margin: a checkpoint restore can land below the
    // LCP by up to the checkpoint spacing, and the da_auto re-tokenization
    // can shift the chunk headers by up to one chunk. When uncertain, bias to
    // the evict-plan side - over-planning is a few idempotent PUTs,
    // under-planning is a 55s prefill + failure.
    int64_t kv_session_valid_cut(const std::string & session, const llama_tokens & cur,
                                 int64_t margin, size_t * lcp_out = nullptr) {
        std::lock_guard<std::mutex> lk(kv_session_cut_mutex);
        const auto pit = kv_last_prompt.find(session);
        if (pit == kv_last_prompt.end()) {
            return 0; // no history - nothing can be assumed cut
        }
        const size_t max_idx = std::min(cur.size(), pit->second.size());
        size_t lcp = 0;
        while (lcp < max_idx && cur[lcp] == pit->second[lcp]) {
            lcp++;
        }
        if (lcp_out) {
            *lcp_out = lcp;
        }
        const int64_t n_eff = (int64_t) lcp - margin;
        if (n_eff < 0) {
            return 0;
        }
        const auto hit = kv_session_holes.find(session);
        if (hit == kv_session_holes.end()) {
            return 0;
        }
        int64_t cut = 0;
        for (const auto & r : kv_holes_survive(hit->second, n_eff)) {
            cut += (int64_t) (r.second - r.first);
        }
        return cut;
    }

    // kv-offload-holes (2026-10-12): record the session's last accepted
    // prompt (the gate's LCP reference for the next request).
    void kv_last_prompt_set(const std::string & session, const llama_tokens & tokens) {
        std::lock_guard<std::mutex> lk(kv_session_cut_mutex);
        kv_last_prompt[session] = tokens;
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    // DA (W2): the n_kv_max value last pushed to ctx_tgt, for the change log
    int64_t da_n_kv_max_last = 0;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    // kv-offload-holes: session -> tokens currently cut from the KV (the sum
    // of the owning slot's kv_hole_ranges ledger). The watermark gate reads
    // this (kv_session_cut_get) instead of the cumulative PUT volume: holes
    // expire when the KV is re-prefilled and the ledger is dropped by
    // prompt_clear, so the PUT history overstates the offloaded amount and
    // the gate stays "idle" while the physical KV refills to the buffer cap
    // (2026-10-10 incident: resident 19919 reported, ~101k actually resident).
    // Written on the queue thread (ledger mutations), read on the HTTP thread
    // (the gate) - mutex-guarded.
    std::mutex kv_session_cut_mutex;
    std::map<std::string, int64_t> kv_session_cut;
    // kv-offload-holes (2026-10-12): session -> the owning slot's applied-hole
    // ranges (kv_hole_ranges), mirrored for the HTTP-thread gate. The gate
    // must not touch the slot's ledger (queue-thread owned - the 2026-10-12
    // 13:35 gate ran 3ms before the prior task's release). Written by
    // kv_session_cut_update on every ledger mutation, same mutex.
    std::map<std::string, std::vector<std::pair<int32_t,int32_t>>> kv_session_holes;
    // kv-offload-holes (2026-10-12): session -> the last accepted request's
    // original prompt tokens (the gate's LCP reference). The gate (HTTP
    // thread) compares the current prompt against this stored copy - NOT the
    // slot's residual prompt (queue-thread owned). Updated when a request
    // passes the gate; a rejected request leaves it untouched (the KV state
    // did not change, so the reference must not move either).
    std::map<std::string, llama_tokens> kv_last_prompt;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    server_slot_sched slot_sched;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end() ||
                              std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        mtmd_memory_usage mmproj_usage = {{}, -1, false};
        int64_t mmproj_usage_t_us = 0;
        if (has_mmproj) {
            int64_t t_start = ggml_time_us();
            mmproj_usage = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            mmproj_usage_t_us = ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        // on resume params_base already holds the margin from the first load
        if (has_mmproj && params_base.fit_params && !is_resume) {
            const auto & mmproj_mem = mmproj_usage.backend_mem_usage;
            const int64_t t_elapsed = mmproj_usage_t_us;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        // 2-stream (B) reserves sequence ids beyond the slot ids (da_seq is
        // always >= n_slots, so it can never collide with a slot's own sequence
        // or an idle slot's cached prompt). apply_da_b scans [n_slots,
        // n_seq_max) for a free id and gives each B slot its own, so with
        // n_parallel slots all able to sit on the B path at once, n_seq_max
        // must reach 2*n_slots or the extra slots fall back to the
        // irreversible A path. In --kv-unified mode n_ctx_seq is the full pool
        // regardless of n_seq_max, so the extra ids cost only sequence
        // metadata, not KV capacity. n_parallel is the sole driver of
        // cparams.n_seq_max (common.cpp), so bump it only around context
        // creation and restore it before the slot loop reads it.
        //
        // n_outputs_max must move with it: llama_context::output_reserve sizes
        // the output buffer for max(n_outputs, n_seq_max) rows and asserts
        // n_outputs_max >= n_seq_max, so the reserved ids need that many rows.
        const int n_da_b_slots   = params_base.n_parallel;
        const int n_da_b_seq_max = 2 * n_da_b_slots;
        if (params_base.kv_unified) {
            params_base.n_parallel = n_da_b_seq_max;
            params_base.n_outputs_max = std::max(params_base.n_outputs_max, n_da_b_seq_max);
        }

        llama_init = common_init_from_params(params_base);

        params_base.n_parallel = n_da_b_slots;

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        try {
            decision.init(model_tgt);
        } catch (const std::exception & e) {
            SRV_ERR("failed to init decision model: %s\n", e.what());
            return false;
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // the draft context must accept the same da_seq ids as the
                // target context: common_memory::seq_cp operates on both, and
                // a B slot reserves an id in [n_slots, n_seq_max).
                // common_base_params_to_speculative fixed n_outputs_max to the
                // original slot count, so widen n_seq_max AND n_outputs_max
                // together (llama_context::output_reserve asserts
                // n_outputs_max >= n_seq_max).
                if (params_base.kv_unified) {
                    params_dft.n_parallel = n_da_b_seq_max;
                    params_dft.n_outputs_max = std::max(params_dft.n_outputs_max, n_da_b_seq_max);
                }

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            // the image must fit in one ubatch if the model needs non-causal attention on it
            // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
            {
                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                const bool is_mixed = use_mixed_batch();
                if (is_mixed || mmproj_usage.use_non_causal) {
                    const int n_max = is_mixed ? n_ubatch / 2 : n_ubatch;
                    if (mmproj_usage.image_max_tokens > n_max) {
                        SRV_WRN("cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", mmproj_usage.image_max_tokens, n_max, n_ubatch,
                                is_mixed ? "the model processes the prompt in one ubatch" : "model needs non-causal attention on image");
                        SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
                        mparams.image_max_tokens = n_max;
                        mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_max);
                    }
                }
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        mtmd_cap = mtmd_helper_get_capabilities(model_tgt, mctx);
        // the helper only knows about the built-in video decoder
        mtmd_cap.video = mtmd_cap.video || (mtmd_cap.vision && init_opt.video_params.decoder != nullptr);

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_sched.slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // kv-offload buffer decoupling: startup validation.
        // (3c) a decoupled buffer (B > 0) smaller than the logical context only
        //      works with the eviction mechanism (kv_offload + focus-memory host);
        //      without it nothing can be evicted, so a prompt longer than B
        //      hard-fails in find_slot - reject. (B >= n_ctx needs no eviction:
        //      the buffer covers the whole position range.)
        if (params_base.kv_cache_size > 0 &&
                params_base.kv_cache_size < params_base.n_ctx &&
                !(params_base.kv_offload && !params_base.focus_memory_host.empty())) {
            SRV_ERR("kv_offload: buffer %d cells < n_ctx %d without the eviction path (--kv-offload-threshold + --fm-offload host) - prompts longer than the buffer hard-fail (find_slot); enable eviction or raise --kv-cache-size\n",
                    params_base.kv_cache_size, params_base.n_ctx);
            return false;
        }
        // (a) the buffer must hold the hot window (threshold) + headroom even with
        //     retain = 0; below that no retain value is safe - reject.
        if (params_base.kv_offload && params_base.kv_cache_size > 0 &&
                params_base.kv_cache_size < params_base.kv_offload_threshold + 16384) {
            SRV_ERR("kv_offload: buffer %d cells < threshold %d + headroom 16384 - the hot window alone exceeds the buffer (find_slot failure); raise --kv-cache-size or lower --kv-offload-threshold\n",
                    params_base.kv_cache_size, params_base.kv_offload_threshold);
            return false;
        }
        // (c) the buffer cannot hold threshold + retain + headroom: the runtime
        //     clamp shrinks the effective retain - warn with the clamped value.
        if (params_base.kv_offload && params_base.kv_cache_size > 0 &&
                params_base.kv_cache_size <
                params_base.kv_offload_threshold + params_base.kv_retain_tokens + 16384) {
            SRV_WRN("kv_offload: buffer %d cells < threshold %d + retain %d + headroom 16384 - effective retain clamped to %d (find_slot failure risk if pin is large)\n",
                    params_base.kv_cache_size, params_base.kv_offload_threshold,
                    params_base.kv_retain_tokens,
                    params_base.kv_cache_size - 16384 - params_base.kv_offload_threshold);
        }
        // (d) the buffer cannot even hold threshold + gen tail + headroom: the
        //     clamp degenerates (effective retain -> 0) and the reject gate fails
        //     every request regardless of eviction - warn with the minimum
        //     required buffer. The gen tail is the capped reservation (the runtime
        //     gate/clamp reserve min(n_predict, KV_GEN_TAIL_CAP), not the full
        //     n_predict).
        {
            const int64_t gen_tail = params_base.n_predict > 0 ?
                std::min<int64_t>(params_base.n_predict, KV_GEN_TAIL_CAP) : 0;
            if (params_base.kv_offload && params_base.kv_cache_size > 0 &&
                    params_base.kv_cache_size <
                    params_base.kv_offload_threshold + gen_tail + 16384) {
                SRV_WRN("kv_offload: buffer %d cells < threshold %d + gen tail %d + headroom 16384 - minimum required buffer is %d; even with perfect eviction the generation tail exceeds the budget (400 on every request); raise --kv-cache-size to >= %d or lower --kv-offload-threshold/--n-predict\n",
                        params_base.kv_cache_size, params_base.kv_offload_threshold,
                        (int) gen_tail,
                        (int) (params_base.kv_offload_threshold + gen_tail + 16384),
                        (int) (params_base.kv_offload_threshold + gen_tail + 16384));
            }
        }
        // (e) high/low watermark (holes mode): the high watermark must be >= the
        //     low (drain) target, and a request just below the high watermark
        //     (the idle band's worst case) must still fit the prefill peak
        //     (resident + n_batch in holes mode) plus the gen tail under the
        //     buffer cap - otherwise the request would be rejected (400) before
        //     eviction can trigger. high == 0 = legacy per-turn gate (no check).
        if (params_base.kv_offload && params_base.kv_offload_high > 0) {
            if (params_base.kv_offload_high < params_base.kv_offload_threshold) {
                SRV_ERR("kv_offload: --kv-offload-high %d < --kv-offload-threshold %d - the high watermark must be >= the low (drain) target\n",
                        params_base.kv_offload_high, params_base.kv_offload_threshold);
                return false;
            }
            if (!params_base.kv_offload_holes) {
                SRV_WRN("kv_offload: --kv-offload-high %d is set but --kv-offload-holes is off - the watermark is holes-mode only (Option C needs the evict plan every request to reduce the prompt text) - ignoring --kv-offload-high (legacy per-turn gate)\n",
                        params_base.kv_offload_high);
            } else if (params_base.kv_cache_size > 0) {
                const int64_t gen_tail = params_base.n_predict > 0 ?
                    std::min<int64_t>(params_base.n_predict, KV_GEN_TAIL_CAP) : 0;
                if ((int64_t) params_base.kv_offload_high + params_base.n_batch + gen_tail > (int64_t) params_base.kv_cache_size - 16384) {
                    SRV_ERR("kv_offload: --kv-offload-high %d + batch %d + gen tail %d exceeds buffer %d - headroom 16384 - a request just below the high watermark would be rejected before eviction can trigger; lower --kv-offload-high to <= %d or raise --kv-cache-size\n",
                            params_base.kv_offload_high, params_base.n_batch, (int) gen_tail,
                            params_base.kv_cache_size,
                            (int) (params_base.kv_cache_size - 16384 - params_base.n_batch - gen_tail));
                    return false;
                }
            }
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                // In --kv-unified mode the 2-stream (B) DA path reserves n_da_b_slots
                // extra sequence ids (n_slots..2*n_slots, see apply_da_b) that both ctxs
                // accept (context n_seq_max = 2*n_parallel, bumped above). The
                // speculative per-seq state (dparams + each impl's per-seq vectors)
                // must cover all of them too, or a B slot drafting on da_seq hits the
                // `seq_id < dparams.size()` assert. The extra ids stay idle
                // (drafting=false) until B slots use them, so non-B / non-unified
                // paths are unaffected.
                const uint32_t n_seq_spec =
                        (uint32_t) params_base.n_parallel + (params_base.kv_unified ? (uint32_t) n_da_b_slots : 0u);
                spec.reset(common_speculative_init(params_base.speculative, n_seq_spec));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();
            slot.da_measure_only = params_base.da_measure_only;
            slot.da_tail_keep    = params_base.da_tail_keep;
            slot.da_b_min_remove = params_base.da_b_min_remove;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            if (has_cap_tts()) {
                slot.tts.ctx.init(ctx_tgt, mctx);
            }

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.callback_on_kv_holes_change = [this](const server_slot & slot) {
                kv_session_cut_update(slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(ctx_tgt, std::max(n_batch, params_base.n_parallel), n_embd);
        }

        // decision requests rarely repeat a state, the prompt cache is opt-in for decision models
        if (decision.type != COMMON_DECISION_TYPE_NONE && !params_base.cache_ram_explicit) {
            SRV_INF("%s", "decision model: prompt cache disabled by default, use `--cache-ram N` to enable it\n");
            params_base.cache_ram_mib    = 0;
            params_base.cache_idle_slots = false;
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            try {
                asr_prompt = server_chat_get_asr_prompt(model_tgt);
            } catch (const std::exception & e) {
                SRV_ERR("%s: \"asr\" chat template error: %s\n", __func__, e.what());
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mtmd_cap.vision,
                /* allow_audio           */ mtmd_cap.audio,
                /* allow_video           */ mtmd_cap.video,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* media_allow_private   */ params_base.media_allow_private_urls,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning").get<bool>();
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    std::vector<server_slot_info> get_slot_infos() const {
        std::vector<server_slot_info> res;
        for (const server_slot & slot : slots) {
            res.push_back({ slot.id, slot.task.get(), slot.is_processing(), slot.t_last_used, &slot.prompt });
        }
        return res;
    }

    // call only when the task is sure to launch on the slot
    void update_prompt_cache(server_slot & slot, const server_task & task) {
        SRV_TRC("%s", "updating prompt cache\n");

        const int64_t t_start = ggml_time_us();

        slot.prompt_save(*prompt_cache);

        if (!slot.prompt_load(*prompt_cache, task.tokens)) {
            slot.prompt_clear();
        }

        prompt_cache->update();

        SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // per-request lora if given, otherwise the global config
        auto task_loras = task.params.lora.empty() ? params_base.lora_adapters : construct_lora_list(task.params.lora);
        if (!are_lora_equal(task_loras, slot.lora)) {
            // if lora has changed, check to see if the cache should be cleared
            if (lora_should_clear_cache(slot.lora, task_loras)) {
                SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task_loras.size());
                slot.prompt.clear();
            } else {
                SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
            }
            slot.lora = task_loras;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        if (task.type == SERVER_TASK_TYPE_TTS) {
            GGML_ASSERT(has_cap_tts()); // should already checked in route handler
            if (!slot.tts.is_supported()) {
                slot.tts.ctx.init(ctx_tgt, slot.mctx);
            }
            // mtmd_helper always re-eval the whole prompt
            slot.prompt_clear();
            task.tts_inp.data.seq_id = slot.id;
            if (slot.tts.ctx.set_input(task.tts_inp.get()) != 0) {
                send_error(task, "failed to process TTS prompt", ERROR_TYPE_SERVER);
                return false;
            }
        }

        // realtime transcription: the rows come from the audio stream, the helper may bias some tokens
        if (task.type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
            auto sparams = mtmd_audio_stream_params_default();
            sparams.prompt   = asr_prompt.data();
            sparams.n_prompt = asr_prompt.size();
            slot.audio_stream.reset(mtmd_audio_stream_init(mctx, sparams));
            if (!slot.audio_stream) {
                send_error(task, "failed to initialize the audio stream", ERROR_TYPE_SERVER);
                return false;
            }
            if (!asr_prompt.empty()) {
                for (const auto & b : mtmd_helper::audio_stream_logit_bias(mctx, model_tgt)) {
                    task.params.sampling.logit_bias.push_back(b);
                }
            }
        }

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: check verify if this actually works with TTS
            use_backend_sampling &= task.type != SERVER_TASK_TYPE_TTS;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        // kv-offload-holes: learn the session owning this slot's KV so ledger
        // mutations (including prompt_clear on an idle slot) can refresh the
        // context's kv_session_cut entry. A takeover by another session drops
        // the old session's entry now: its holes will be expired/cleared by
        // the prefill, and a stale entry would understate its resident.
        if (!task.params.kv_offload_session.empty() && task.params.kv_offload_session != slot.kv_session) {
            if (!slot.kv_session.empty()) {
                std::lock_guard<std::mutex> lk(kv_session_cut_mutex);
                kv_session_cut[slot.kv_session] = 0;
            }
            slot.kv_session = task.params.kv_offload_session;
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        if (slot.task->type == SERVER_TASK_TYPE_TTS) {
            slot.state = SLOT_STATE_PROCESSING_PROMPT;
        } else {
            slot.state = slot.task->is_child()
                ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
                : SLOT_STATE_STARTED;
        }

        // no prompt, the rows come from the audio stream, which audio controls reach from now on
        if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
            slot.prompt_clear();
            slot.init_sampler();
            slot.sampled = LLAMA_TOKEN_NULL;
            slot.i_batch = -1;
            slot.stats.update_prompt_start();
            slot.state = SLOT_STATE_GENERATING;
            send_partial_response(slot, {}, false, true);
        }

        // no prompt either, the session appends text and audio and asks for responses with its controls
        if (slot.task->type == SERVER_TASK_TYPE_REALTIME) {
            slot.rt = std::make_unique<server_realtime_slot>(ctx_tgt, mctx, slot.id, slot.n_ctx, slot.task->params.sampling,
                                                             queue_results, slot.task->id, slot.task->index);
            slot.prompt_clear();
            slot.stats.update_prompt_start();
            slot.state      = SLOT_STATE_GENERATING;
            slot.audio_wait = true;
            send_partial_response(slot, {}, false, true);
        }

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    static size_t tool_closing_hold(const std::string & s) {
        static const std::string cls[] = {"</tool_call>","</function>","</parameter>"};
        for (const auto & c : cls)
            for (size_t k = std::min(s.size(), c.size()-1); k > 0; k--)
                if (s.compare(s.size()-k, k, c, 0, k) == 0) return k;
        return 0;
    }
    // da_hold = std::max(da_tag_hold_len(...), tool_closing_hold(unsent));

    bool process_token(completion_token_output & result, server_slot & slot) {
        // L4: per-token trace right after a tag closed (both the spec-accept
        // and the single-token paths funnel through here)
        if (slot.da_trace_left > 0) {
            da_trace("slot=%d task=%d L4 tok n_gen=%d id=%d piece='%s' prompt_tokens=%d mode=%d seq=%d\n",
                     (int) slot.id, (int) slot.task->id,
                     (int) slot.stats.n_gen, (int) result.tok, dbg_esc(result.text_to_send, 40).c_str(),
                     (int) slot.prompt.n_tokens(), (int) slot.da_mode, (int) slot.kv_seq());
            slot.da_trace_left--;
        }

        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated.append(token_str, result.tok);

        // DA tag parser: the model's own <focus magic_chunks="N"> tag drives the
        // attention restriction. Scan the accumulated text for the first
        // complete tag; on tag close the removals run exactly once, mid-decode
        // (no-op unless the client sent da_chunks, and after it has applied).
        apply_da_tag(slot);
        scrub_headless_tool_closings(slot);

        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated.text) < slot.generated.text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated.text.size());

            const std::string str_test = slot.generated.text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated.truncate(pos + stop_pos);
                pos = std::min(slot.n_sent_text, slot.generated.text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response. DA tag holdback: a
                // trailing fragment that could still grow into a control tag
                // stays unsent until it closes (apply_da_tag then erases it)
                // or is proven to be plain text. An EOG token drops the held
                // fragment instead of flushing it: generation is over, so it
                // can no longer grow into a tag - a partial tag must not leak
                // to the client. The resize removes it from generated_text as
                // well, so non-streaming responses lose it too.
                // Recomputed here because the stop-word erase above may have
                // removed the previously held fragment.
                const bool da_eog = llama_vocab_is_eog(vocab, result.tok);
                const std::string unsent = slot.generated.text.substr(
                        std::min(slot.n_sent_text, slot.generated.text.size()));
                size_t da_hold = 0;
                if (!slot.task->params.da_chunks.empty()) {
                    da_hold = da_tag_hold_len(unsent, slot.da_mode);
                }
                // tool-call 닫힘 홀드백: DA 활성 여부와 무관하게 항상 적용
                const size_t hold = std::max(da_hold, tool_closing_hold(unsent));
                if (da_eog && hold > 0) {
                    slot.generated.text.resize(slot.generated.text.size() - hold);
                }
                const size_t send_end = slot.generated.text.size() - (da_eog ? 0 : hold);

                result.text_to_send = slot.generated.text.substr(pos, send_end - pos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context.
        // Decoupled-KV mode: the physical buffer (kv_cache_size) fills before the
        // logical n_ctx. (2026-10-10, incident 2026-10-10-kv-ledger-loss) The
        // buffer check now uses the PHYSICAL free-cell count
        // (llama_memory_n_free_cells) instead of the ledger-based resident
        // projection: the hole ledger can disagree with the physical KV (a slot
        // cleared by the idle path and re-prefilled reports "holes cut 0" while
        // ~101k cells are resident), and only the physical count sees that.
        // Backends without n_free_cells support (return -1) fall back to the
        // old ledger-based projection.
        {
            const int64_t buf_cap = params_base.kv_cache_size > 0 ?
                (int64_t) params_base.kv_cache_size - 16384 : (int64_t) slot.n_ctx;
            const int64_t ctx_limit = std::min<int64_t>(slot.n_ctx, buf_cap);
            int64_t holes_cut = 0;
            for (const auto & r : slot.kv_hole_ranges) holes_cut += (int64_t) (r.second - r.first);
            const int32_t n_free = llama_memory_n_free_cells(llama_get_memory(ctx_tgt));
            const bool ctx_full = (int64_t) slot.prompt.n_tokens() + (int64_t) slot.stats.n_gen + 1 >= slot.n_ctx;
            const bool buf_full = params_base.kv_cache_size > 0 && n_free >= 0 && n_free < 2048;
            // Fallback estimate (n_free unavailable): the ledger-based resident
            // projection. In holes mode (Option B) the evicted text stays in the
            // prompt while its KV cells are cut (kv_hole_ranges ledger), so the
            // prompt length overstates the resident by the cut span (2026-10-07:
            // prompt 59358 / resident 46865, 12055 cells free).
            int64_t resident_base = (int64_t) slot.prompt.n_tokens() - holes_cut;
            if (holes_cut == 0 && slot.kv_holes_active && slot.kv_hole_peak_resident > 0) {
                resident_base = slot.kv_hole_peak_resident;
            }
            const bool est_full = n_free < 0 &&
                resident_base + (int64_t) slot.stats.n_gen + 1 >= ctx_limit;
            if (!params_base.ctx_shift && (ctx_full || buf_full || est_full)) {
                slot.truncated      = true;
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                if (buf_full) {
                    // INFO (not DBG): the client sees finish_reason=length and
                    // halts, so the journal must record that the physical KV
                    // buffer (not the model n_ctx) capped generation.
                    SRV_INF("slot %d: generation stopped at KV buffer cap - free cells %d < 2048 (prompt %d, holes cut %d, n_gen %d, buffer %d, n_ctx %d) -> finish_reason=length\n",
                            (int) slot.id, n_free, (int) slot.prompt.n_tokens(), (int) holes_cut,
                            (int) slot.stats.n_gen, params_base.kv_cache_size, slot.n_ctx);
                } else if (est_full) {
                    SRV_INF("slot %d: generation stopped at KV buffer cap (estimate, n_free unavailable) - resident %d (prompt %d - holes cut %d) + n_gen %d >= buffer %d - 16384 (n_ctx %d) -> finish_reason=length\n",
                            (int) slot.id, (int) resident_base, (int) slot.prompt.n_tokens(), (int) holes_cut,
                            (int) slot.stats.n_gen, params_base.kv_cache_size, slot.n_ctx);
                } else {
                    SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                            slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
                }
            }
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated.text.size() && (slot.generated.text[pos] == ' ' || slot.generated.text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated.text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated.text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(), tkn.text_to_send.size());
            res->tokens.assign(1, tkn.tok);
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_tts_result(server_slot & slot, int32_t sample_rate, const char * data, size_t data_len, bool final) {
        auto res = std::make_unique<server_task_result_tts>();

        res->id          = slot.task->id;
        res->index       = slot.task->index;
        res->sample_rate = sample_rate;
        res->audio.assign(data, data_len);
        res->final       = final;

        queue_results.send(std::move(res));
    }

    // headless tool-call closing scrubber (DA erase와 동일 패턴)
    static void scrub_headless_tool_closings(server_slot & slot) {
        static const std::pair<std::string,std::string> fam[] = {
            {"<tool_call>", "</tool_call>"},
            {"<function=",  "</function>"},
            {"<parameter=", "</parameter>"},
        };
        std::string & g = slot.generated.text;
        for (const auto & [op, cl] : fam) {
            int bal = 0; size_t i = 0;
            while (i < g.size()) {
                if (g.compare(i, op.size(), op) == 0) { bal++; i += op.size(); continue; }
                if (g.compare(i, cl.size(), cl) == 0) {
                    if (bal > 0) { bal--; i += cl.size(); continue; }  // 정상 닫힘
                    const size_t len = cl.size();
                    g.erase(i, len);                                    // headless → 제거
                    if (slot.n_sent_text > i)
                        slot.n_sent_text = (slot.n_sent_text >= i+len) ? slot.n_sent_text-len : i;
                    continue;
                }
                i++;
            }
        }
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated.text;
        }

        // DA: log the full (tag-erased) generation + final DA state so the
        // server log is self-documenting for diagnosing answer-generation
        // failures. The client never sees the focus tags (they are erased from
        // generated_text before being sent), so this is the only place the raw
        // output and the DA end-state (mode / da_seq / keep / removed) are
        // captured together for a request.
        if (!slot.task->params.da_chunks.empty() || slot.da_applied) {
            SLT_INF_DBG(slot, "da: request complete - n_gen=%d, mode=%s, da_seq=%d, da_bound=%d, keep=%zu, removed=%u, n_full=%d\n",
                    (int) slot.stats.n_gen,
                    slot.da_mode == DA_MODE_GLOBAL ? "GLOBAL"
                        : (slot.da_mode == DA_MODE_FOCUS ? "FOCUS" : "LOCAL"),
                    (int) slot.da_seq, (int) slot.da_bound,
                    slot.da_keep_count, slot.n_da_removed,
                    (int) slot.prompt.n_tokens());

            // L5: leak detection at request end (before generated_text is
            // moved into the response) - unbalanced tool-call tags are the
            // 14:21 derailment signature (a headless </parameter></function>
            // tail that no client can parse as a functionCall)
            {
                const std::string & g = slot.generated.text;
                const size_t E = g.size();

                // close > open is the derail signature (a headless closing
                // tail). open > close (the model mentioning the tags as text)
                // is NOT a symptom - counting it flagged ~52% of turns.
                const bool unbalanced =
                    dbg_count(g, "<tool_call>", E) != dbg_count(g, "</tool_call>", E) ||
                    dbg_count(g, "<function=", E) != dbg_count(g, "</function>", E) ||
                    dbg_count(g, "<parameter=", E) != dbg_count(g, "</parameter>", E);
                //SLT_INF(slot, "da_trace: final n_gen=%d stop=%d unbalanced=%d tail='%s'\n",
                //        (int) slot.stats.n_gen, (int) slot.stop, (int) unbalanced, dbg_esc(g.substr(E > 300 ? E - 300 : 0), 320).c_str());
                if (unbalanced) SLT_WRN_DBG(slot, "%s", "da_trace: DERAIL-SUSPECT (tool-call tags unbalanced)\n");
            }

#if 0  // disabled: re-logged the entire (17K+ char) generation on every DA request,
      // flooding the journal. The compact "da: request complete" state line above
      // stays. Re-enable for one-off answer-generation diagnosis only.
            // single line (newlines escaped) so a line-based journal filter can
            // capture the whole generation in one record
            std::string gtext = slot.generated.text;
            std::string escaped;
            escaped.reserve(gtext.size());
            for (char c : gtext) {
                if (c == '\n')      escaped += "\\n";
                else if (c == '\r') escaped += "\\r";
                else                escaped += c;
            }
            SLT_INF_DBG(slot, "da: generated_text (%zu chars): %s\n",
                    gtext.size(), escaped.c_str());
#endif
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = {};
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.tokens[i].seq_id);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;

        if (decision.pooled) {
            const float * embd = llama_get_embeddings_seq(slot.ctx_tgt, slot.id);
            if (embd == nullptr) {
                send_error(slot, "failed to get the class scores", ERROR_TYPE_SERVER);
                return;
            }
            res->scores.assign(embd, embd + llama_model_n_embd_out(model_tgt));
        } else if (!decision.labels.empty()) {
            const float * logits = llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == nullptr) {
                send_error(slot, "failed to get logits", ERROR_TYPE_SERVER);
                return;
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : decision.labels) {
                GGML_ASSERT(label >= 0 && label < n_vocab);
                res->scores.push_back(logits[label]);
            }
            if (!decision.label_groups.empty()) {
                std::vector<float> scores;
                size_t i = 0;
                for (const int32_t n : decision.label_groups) {
                    GGML_ASSERT(n > 0 && i + n <= res->scores.size());
                    scores.push_back(*std::max_element(res->scores.begin() + i, res->scores.begin() + i + n));
                    i += n;
                }
                res->scores = std::move(scores);
            }
        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            std::vector<int32_t> idx;
            for (int i = 0; i < batch.size(); ++i) {
                if (batch.tokens[i].output && batch.tokens[i].seq_id == slot.id) {
                    idx.push_back(i);
                }
            }
            const int32_t pos_first = slot.prompt.n_tokens() - (int32_t) idx.size();
            auto get_embd = [&](int32_t pos) -> const float * {
                const int32_t i = pos - pos_first;
                return i >= 0 && i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
            };

            // joint head (decision model): the scores are the first rows
            for (int32_t i = 0; i < decision.n_scores; i++) {
                const float * embd = i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
                if (embd == nullptr) {
                    send_error(slot, "failed to get embeddings", ERROR_TYPE_SERVER);
                    return;
                }
                res->scores.push_back(embd[0]);
            }

            const int32_t n_embd_out = llama_model_n_embd_out(model_tgt);
            const int32_t n_pointer  = n_embd_out / 2;
            const float * embd_q = decision.pointer >= 0 ? get_embd(decision.pointer) : nullptr;
            GGML_ASSERT(decision.column >= 0 && decision.column < n_embd_out);

            for (const int32_t marker : decision.markers) {
                const float * embd = get_embd(marker);
                if (embd == nullptr || (decision.pointer >= 0 && embd_q == nullptr)) {
                    send_error(slot, "failed to get embeddings, the question and its options must fit in one batch", ERROR_TYPE_SERVER);
                    return;
                }
                if (decision.pointer < 0) {
                    res->scores.push_back(embd[decision.column]);
                    continue;
                }
                float dot = 0.0f;
                for (int32_t i = 0; i < n_pointer; i++) {
                    dot += embd_q[i] * embd[n_pointer + i];
                }
                res->scores.push_back(dot / sqrtf((float) n_pointer));
            }
        }

        SLT_DBG(slot, "%s", "sending decision result\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.tokens[i].seq_id);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (int id : slot_sched.get_free_slots(get_slot_infos(), n_slots_needed, exclude_id_slot)) {
            free_slots.push_back(get_slot_by_id(id));
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            const auto & cur = **it;
            if (cur.id_task != id_task && last >= 0 && cur.n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = cur.n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = *slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if ((*it)->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", (*it)->n_tokens);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto ckpt = std::make_shared<common_prompt_checkpoint>();
        auto & cur = *ckpt;

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        slot.prompt.checkpoints.push_back(std::move(ckpt));

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION = 1;

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    template <typename Buf>
    static bool ckpt_read_buf(std::ifstream & ifs, Buf & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    template <typename Buf>
    static void ckpt_write_buf(std::ofstream & ofs, const Buf & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        if (slot.prompt.checkpoints.empty()) {
            return true;
        }
        std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) slot.prompt.checkpoints.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto & ckpt : slot.prompt.checkpoints) {
            const auto & cur = *ckpt;
            ckpt_write(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
            ckpt_write(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
            ckpt_write(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
            ckpt_write_buf(ofs, cur.data_tgt,  n_written);
            ckpt_write_buf(ofs, cur.data_dft,  n_written);
            ckpt_write_buf(ofs, cur.data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable appendix
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_CKPT_VERSION ||
            !ckpt_read(ifs, &count,   sizeof(count),   n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        std::list<std::shared_ptr<common_prompt_checkpoint>> checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            auto & cur = *ckpt;
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.empty()) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            checkpoints.push_back(std::move(ckpt));
            if (checkpoints.size() > (size_t) params_base.n_ctx_checkpoints) {
                checkpoints.pop_front();
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        if (ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back()->data_dft.empty()) {
            const bool ok = checkpoints.back()->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            if (!ok) {
                SLT_WRN(slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", filepath.c_str());
                for (auto & cur : checkpoints) {
                    cur->clear_dft();
                }
            }
        }
        slot.prompt.checkpoints.assign(checkpoints.begin(), checkpoints.end());
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'\n", slot.prompt.checkpoints.size(), filepath.c_str());
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only
    static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (ctx_dft != nullptr) {
            data_dft.resize(llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
            llama_state_seq_get_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        common_speculative_get_state(spec.get(), slot.id, data_spec);
        if (data_dft.empty() && data_spec.empty()) {
            return true;
        }
        std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_DRAFT_MAGIC;
        const uint32_t version = SLOT_DRAFT_VERSION;
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, data_dft,  n_written);
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB) to '%s'\n", (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable draft state
    // the draft starts empty unless the file holds a draft state that fits the draft context
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        if (ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_DRAFT_VERSION ||
            !ckpt_read_buf(ifs, data_dft,  n_avail, n_read) ||
            !ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        if (!data_dft.empty()) {
            if (ctx_dft == nullptr ||
                llama_state_seq_set_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.size()) {
                if (ctx_dft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                }
                SLT_WRN(slot, "draft state in '%s' does not match the draft context - dropped\n", filepath.c_str());
                return n_read;
            }
        }
        if (!data_spec.empty()) {
            common_speculative_set_state(spec.get(), slot.id, data_spec);
        }
        SLT_INF(slot, "restored the draft state (%.3f MiB) from '%s'\n", (float) n_read / 1024 / 1024, filepath.c_str());
        return n_read;
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
            case SERVER_TASK_TYPE_REALTIME:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
            case SERVER_TASK_TYPE_TTS:
            case SERVER_TASK_TYPE_DECISION:
                {
                    if (task.need_sampling() && !decision.can_generate()) {
                        send_error(task, "This model is a decision model without text generation, use /v1/systemone", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    const auto plan = slot_sched.select(get_slot_infos(), task);

                    server_slot * slot = plan.id_slot == -1 ? nullptr : get_slot_by_id(plan.id_slot);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    std::vector<server_slot *> child_slots;
                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                    }

                    // the task is not deferred past this point
                    if (plan.update_cache && prompt_cache) {
                        update_prompt_cache(*slot, task);
                    }

                    if (task.is_parent()) {
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %d on unknown completion id=%s, no live slot\n",
                                (int) task.params.control_action, task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == SERVER_CONTROL_ACTION_REASONING_END) {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else if (slot->rt) {
                        // a cancel acts now, everything else runs after what is already queued
                        if (task.params.control_action == SERVER_CONTROL_ACTION_RESPONSE_CANCEL) {
                            slot->rt->cancel();
                        } else {
                            slot->rt->push({
                                task.params.control_action,
                                std::move(task.params.control_text),
                                std::move(task.params.control_audio),
                                task.params.control_response,
                            });
                        }
                        slot->audio_wait = false;
                        res->success = true;
                    } else if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END) {
                        if (!slot->audio_stream) {
                            res->success = false;
                            res->message = "not a realtime transcription slot";
                        } else {
                            auto * stream = slot->audio_stream.get();
                            if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND) {
                                mtmd_audio_stream_push(stream, task.params.control_audio.data(), task.params.control_audio.size());
                            } else {
                                const int32_t ret = mtmd_audio_stream_flush(stream, task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END);
                                // a non-final commit completes its item once the rows of its audio are decoded
                                if (ret == 0 && task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT) {
                                    slot->audio_commit_rows.push_back(mtmd_audio_stream_get_n_rows_flushed(stream));
                                }
                            }
                            slot->audio_wait = false;
                            res->success = true;
                        }
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (int pass = 0; pass < 2; ++pass) {
            for (auto & slot : slots) {
                const bool is_child = slot.is_processing() && slot.task->is_child();
                if (is_child != (pass == 1)) {
                    continue;
                }
                try {
                    callback(slot);
                } catch (const std::exception & e) {
                    SLT_ERR(slot, "got exception: %s\n", e.what());
                    send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                    slot.release();
                }
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // fill slot.inp_embd with the next rows of the audio stream
    // returns false if the slot has nothing to decode now
    bool slot_audio_step(server_slot & slot) {
        if (!slot.inp_embd.empty()) {
            return true; // rows from a previous step, not decoded yet
        }

        const int32_t n = mtmd_audio_stream_step(slot.audio_stream.get(), slot.sampled);

        // complete the commits whose rows have their text sampled, before the new rows: their text goes to the next item
        const int64_t n_rows = mtmd_audio_stream_get_n_rows(slot.audio_stream.get());
        while (!slot.audio_commit_rows.empty() && slot.audio_commit_rows.front() <= n_rows) {
            slot.audio_commit_rows.pop_front();
            auto res = std::make_unique<server_task_result_cmpl_partial>();
            res->id                   = slot.task->id;
            res->index                = slot.task->index;
            res->n_decoded            = slot.stats.n_gen;
            res->n_prompt_tokens      = slot.task->n_tokens();
            res->res_type             = slot.task->params.res_type;
            res->oaicompat_model      = slot.task->params.oaicompat_model;
            res->oaicompat_cmpl_id    = slot.task->params.oaicompat_cmpl_id;
            res->audio_commit_drained = true;
            queue_results.send(std::move(res));
        }

        if (n > 0) {
            // new sequence (e.g. live ASR restart): drop the old one, the text goes on in a new line
            if (mtmd_audio_stream_is_seq_start(slot.audio_stream.get())) {
                slot.prompt_clear();
                if (slot.n_sent_text > 0) {
                    completion_token_output sep;
                    sep.tok          = LLAMA_TOKEN_NULL;
                    sep.text_to_send = "\n";
                    slot.generated.append(common_chat_input(sep.text_to_send));
                    slot.n_sent_text += sep.text_to_send.size();
                    send_partial_response(slot, sep, false);
                }
            }
            const float * rows = mtmd_audio_stream_get_output_embd(slot.audio_stream.get());
            slot.inp_embd.assign(rows, rows + (size_t) n * batch.n_embd);
            slot.audio_wait = false;
            return true;
        }
        if (n == 0) {
            slot.audio_wait = true;
            return false;
        }
        if (n == -2) {
            send_error(slot, "audio stream error", ERROR_TYPE_SERVER);
            slot.release();
            return false;
        }

        // end of stream
        slot.stop           = STOP_TYPE_EOS;
        slot.has_next_token = false;
        slot.print_timings();
        send_final_response(slot);
        slot.release();
        return false;
    }

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        // a realtime slot waiting for audio has no work, the next audio control task wakes the loop
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing() && !slot.audio_wait) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        // note: TTS slots bypass the shared batch entirely
        try {
            process_tts_slots();
        } catch (const std::exception & e) {
            SRV_ERR("process_tts_slots() failed: %s\n", e.what());
            abort_all_slots("process_tts_slots() failed: " + std::string(e.what()));
        }

        // so do the realtime conversations
        try {
            process_realtime_slots();
        } catch (const std::exception & e) {
            SRV_ERR("process_realtime_slots() failed: %s\n", e.what());
            abort_all_slots("process_realtime_slots() failed: " + std::string(e.what()));
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch.render(off, n_tokens);
                bool ok = decode(n_batch, off);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);

                    // kv-offload-holes: cut the pending holes whose KV this
                    // sub-batch completed (progressive apply - keeps the
                    // prefill peak near the projected resident instead of the
                    // raw prompt length)
                    kv_holes_progressive_apply(off_next);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    // realtime conversations (SERVER_TASK_TYPE_REALTIME) drive their own decode loop, see server_realtime_slot
    void process_realtime_slots() {
        iterate(slots, [&](server_slot & slot) {
            if (!slot.is_processing() || !slot.rt) {
                return;
            }
            if (!slot.rt->update()) {
                send_error(slot, slot.rt->error, slot.rt->error_type);
                slot.release();
                return;
            }
            slot.audio_wait = slot.rt->is_idle();
        });
    }

    void process_tts_slots() {
        iterate(slots, [&](server_slot & slot) {
            if (!slot.is_processing() || slot.task->type != SERVER_TASK_TYPE_TTS) {
                return;
            }

            llama_set_embeddings(ctx_tgt, true);

            if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                const int32_t ret = slot.tts.ctx.step_prompt(llama_n_batch(ctx_tgt));
                if (ret < 0) {
                    send_error(slot, "TTS prompt processing failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (ret == 0) {
                    // done prompt, do sample and save h_state_prompt
                    slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                    common_sampler_accept(slot.smpl.get(), slot.tts.sampled, true);
                    const float * h_embd = llama_get_embeddings_ith(ctx_tgt, -1);
                    slot.tts.h_state_prompt.assign(h_embd, h_embd + llama_model_n_embd(model_tgt));
                    slot.tts.h_state = slot.tts.h_state_prompt.data();
                    slot.state = SLOT_STATE_GENERATING;
                }
                return;
            }

            const int32_t n_predict = slot.n_predict_max > 0 ? slot.n_predict_max : 512;
            if (slot.tts.stop || slot.tts.n_decoded >= n_predict) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                // generation truly ends here: force out any sub-window remainder still buffered
                if (slot.tts.ctx.flush() != 0 || slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "failed to finalize TTS output", ERROR_TYPE_SERVER);
                } else {
                    send_tts_result(slot, sample_rate, data, data_len, true);
                }
                slot.release();
                return;
            }

            const float * h_state_next = nullptr;
            bool          stop         = false;
            if (slot.tts.ctx.step_gen(slot.tts.sampled, slot.tts.h_state, &h_state_next, &stop) != 0) {
                send_error(slot, "TTS generation failed", ERROR_TYPE_SERVER);
                slot.release();
                return;
            }
            if (h_state_next == nullptr) {
                // end-of-speech without a new frame (e.g. pocket-tts eos head)
                slot.tts.stop = true;
                return;
            }
            slot.tts.h_state = h_state_next;
            slot.tts.n_decoded++;
            slot.tts.stop = stop;

            if (!stop) {
                slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                common_sampler_accept(slot.smpl.get(), slot.tts.sampled, true);
            }

            if (slot.task->params.stream) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                if (slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "TTS streaming output failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (data_len > 0) {
                    send_tts_result(slot, sample_rate, data, data_len, false);
                }
            }
        });
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.task && (slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME)) {
                // TTS and realtime conversation slots drive their own decode loop, never enter the shared batch
                return;
            }
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                // kv_seq(): for a 2-stream (B) slot the live KV is on the
                // second sequence; shifting the original one would leave the
                // live sequence overflowing
                slot.mem.seq_rm (slot.kv_seq(), n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.kv_seq(), n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME) {
                return;
            }

            if (slot.audio_stream && !slot_audio_step(slot)) {
                return; // waiting for audio, or the stream ended
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                // kv_seq(): a 2-stream (B) slot drafts on its second sequence
                // after the switch (P6)
                common_speculative_get_draft_params(spec.get(), slot.kv_seq()).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                // DA tag gate (P6): while a control tag could still close
                // (the generated tail is a tag prefix) or for the first step
                // right after a tag closed (da_skip_draft, set by
                // apply_da_tag()), decode without a new draft. The tag tokens
                // and the first post-switch token then follow the
                // single-token trajectory (identical to spec-off) instead of
                // being batch-verified: a verification batch that closes the
                // tag carries post-tag tokens the batch cut discards, and a
                // batch-computed token at the mode transition can diverge
                // numerically from the single-token one and drop the answer.
                // The replay branch below is exempt - its leftover draft was
                // already cut and carries no post-tag tokens.
                bool da_draft_gate = false;
                if (n_draft_max > 0 && slot.spec_draft.empty() &&
                    !slot.task->params.da_chunks.empty()) {
                    const bool da_just_closed = slot.da_skip_draft;
                    da_draft_gate = da_just_closed ||
                            da_tag_inflight(slot.generated.text, slot.da_mode);
                    slot.da_skip_draft = false;
                    if (da_draft_gate) {
                        if (da_just_closed) {
                            SLT_INF_DBG(slot, "%s", "da_tag: draft skipped (first step after tag close)\n");
                        } else {
                            SLT_DBG(slot, "%s", "da_tag: draft skipped (tag in flight)\n");
                        }
                    }
                }

                if (n_draft_max > 0 && !da_draft_gate) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.kv_seq()),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.kv_seq()));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.kv_seq(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        const bool spec_reject = slot.use_spec_rejection();

                        common_speculative_get_draft_params(spec.get(), slot.kv_seq()) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .temp     = */ slot.task->params.sampling.temp,
                            /* .seed     = */ slot.task->params.sampling.seed,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, slot.kv_seq(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.kv_seq(), ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", (int) slot.kv_seq());
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.kv_seq(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.kv_seq(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing() || slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    SLT_DBG(slot, "%s", "__TEST_TAG_WAIT_PARENT_SLOT__\n");
                    return;
                }

                // no prompt, the rows come from the audio stream
                if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        // the tag path (da_chunks) supersedes the static da_rm
                        // removals: the model's own tag decides the ranges
                        slot.da_rm_pending = !slot.task->params.da_rm.empty() && slot.task->params.da_chunks.empty();
                        slot.da_rm_gate_diag_done = false;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // da_rm diagnostics: full input state at request start, so the
                        // journal shows whether the fields arrived and whether the
                        // mid-prefill gate can run. Note: has_mtmd is true for
                        // text-only prompts too when the server has an mmproj loaded
                        // (mctx != nullptr), so n_media distinguishes real media.
                        if (slot.da_rm_pending || slot.task->params.da_rm_at >= 0) {
                            const size_t n_total = slot.task->tokens.size();
                            const size_t n_text  = slot.task->tokens.get_text_tokens().size();
                            slot.da_rm_text_only = (n_text == n_total);
                            SLT_INF_DBG(slot, "da_rm: request start - %zu range(s), da_rm_at=%d, da_b=%d, n_tokens=%zu, has_mtmd=%d, n_text=%zu, n_media=%zu\n",
                                    slot.task->params.da_rm.size(), slot.task->params.da_rm_at,
                                    (int) slot.task->params.da_b,
                                    n_total, (int) slot.task->tokens.has_mtmd, n_text, n_total - n_text);
                            for (const auto & range : slot.task->params.da_rm) {
                                SLT_INF_DBG(slot, "da_rm:   range [%d, %d)\n", range.first, range.second);
                            }
                        }

                        // da_tag diagnostics: the model's own <focus
                        // magic_chunks="N"> tag drives the removals; the client
                        // only supplies the chunk layout.
                        if (!slot.task->params.da_chunks.empty()) {
                            SLT_INF_DBG(slot, "da_tag: request start - %zu chunk(s), da_b=%d, filler=%s\n",
                                    slot.task->params.da_chunks.size(), (int) slot.task->params.da_b,
                                    slot.task->params.da_filler.first >= 0
                                        ? string_format("[%d, %d)", slot.task->params.da_filler.first, slot.task->params.da_filler.second).c_str()
                                        : "none");
                            for (size_t n = 0; n < slot.task->params.da_chunks.size(); n++) {
                                SLT_INF_DBG(slot, "da_tag:   chunk %zu [%d, %d)\n", n + 1,
                                        slot.task->params.da_chunks[n].first, slot.task->params.da_chunks[n].second);
                            }
                        }

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;
                        // Option B (--kv-offload-holes): set if the n_cache_reuse
                        // shift block moved KV this request - the pending hole
                        // coordinates (recorded in the final token space) would be
                        // stale, so the holes are deferred rather than applied.
                        // Slot member: the progressive (mid-prefill) path in
                        // decode() reads it too. The progressive progress ledger
                        // is request-scoped - fresh for every (re-)prefill.
                        slot.kv_hole_shift_blocked = false;
                        slot.kv_hole_progress.clear();
                        slot.kv_hole_peak_resident = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        // a decision reads its logits from the one batch of its prompt
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt) && slot.task->type != SERVER_TASK_TYPE_DECISION) {
                            send_error(slot, "the current context does not support logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // the outputs of a decision are read from one batch
                            const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;
                            if (n_decision_first >= 0 && slot.task->n_tokens() - n_decision_first > n_batch) {
                                send_error(slot,
                                           string_format("the question and its options (%d tokens) are too large to process. "
                                                         "increase the batch size (current batch size: %d)",
                                                         slot.task->n_tokens() - n_decision_first, n_batch),
                                           ERROR_TYPE_INVALID_REQUEST);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // the children start from the shared prefix, do not go past it
                                if (slot.task->n_tokens_shared > 0) {
                                    n_past = std::min(n_past, slot.task->n_tokens_shared);
                                }

                                // the outputs of a decision are read from the batch, so its first position must be evaluated
                                if (n_decision_first >= 0) {
                                    n_past = std::min(n_past, n_decision_first);
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                if (n_past > 0) {
                                    SLT_INF(slot, "cache reuse: n_past = %d\n", n_past);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            // Option B: the shift moved KV, so any pending
                                            // hole coordinates (recorded in the pre-shift
                                            // token space) are stale - defer them.
                                            slot.kv_hole_shift_blocked = true;

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // number of cached tokens shared with the new prompt, before any checkpoint is restored
                            const int n_prefix = n_past;

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                auto reuse = slot.prompt.get_reuse(n_past, has_new_tokens, n_swa, pos_min);

                                if (reuse.ckpt) {
                                    const auto & cur = *reuse.ckpt;

                                    // restore the context checkpoint
                                    if (!cur.load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                                        !cur.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                                        if (cur.id_task != -1) {
                                            GGML_ABORT("failed to restore context checkpoint\n");
                                        }
                                        // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                                        SLT_WRN(slot, "%s", "failed to load context checkpoint restored from a slot file\n");
                                        reuse = {};
                                    } else {
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, cur.data_spec);

                                        // validate the restored state (hybrid/recurrent memory only):
                                        // a context checkpoint serializes the recurrent state, but its
                                        // position label was recorded as min(attn, recr). If the attention
                                        // side was hole-cut - or left stale by an earlier PARTIAL_ONLY
                                        // restore - when the checkpoint was saved, the label is lower than
                                        // the recurrent position the state actually holds. Resuming at the
                                        // label is impossible (the recurrent state cannot roll back beyond
                                        // n_rs_seq), so discard the checkpoint and re-prefill from 0
                                        // (plans/2026-10-07-kv-400-incident2.md).
                                        // The label is compared against the actual recurrent position, not
                                        // n_tokens - 1: the label == n_tokens - 1 identity holds in the
                                        // incident logs, but it is not a general invariant - hole/evict
                                        // paths can diverge n_tokens from position, which would
                                        // false-positive healthy checkpoints.
                                        const llama_pos recr_pos = llama_memory_seq_pos_max_recr(llama_get_memory(ctx_tgt), slot.id);
                                        if (recr_pos >= 0 && recr_pos != cur.pos_max) {
                                            SLT_WRN(slot, "context checkpoint label (pos_max = %d) does not match the restored recurrent position (%d) - discarding checkpoint, wiping sequence and re-prefilling from 0\n",
                                                    cur.pos_max, (int) recr_pos);
                                            slot.mem.seq_rm_checked(slot.id, -1, -1);
                                            reuse = {};
                                        } else {
                                            SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, reuse.n_past, (float) cur.size() / 1024 / 1024);
                                        }
                                    }
                                }

                                if (reuse.n_past == 0 && !reuse.ckpt) {
                                    SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                            "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                }

                                pos_next = reuse.pos_next;
                                n_past   = reuse.n_past;
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next, or that cover tokens not shared with the new prompt
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = **it;
                                    if (cur.pos_max > pos_next || cur.n_tokens > n_prefix) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // Option B (--kv-offload-holes):
                        // (a) n_past == 0 -> the whole sequence is wiped by the
                        //     tail seq_rm below, so the applied holes are gone;
                        //     clear the ledger. Covers every path that zeroes
                        //     n_past (cache_prompt off, checkpoint do_reset, slot
                        //     takeover by another session). The pending holes are
                        //     request-scoped (task_params::kv_hole_pending) and
                        //     are not applied here (there is no KV to cut) - the
                        //     gate re-plans them on the next request.
                        // (b) apply the pending holes (planned by the eviction
                        //     gate this request, request-scoped) now that the
                        //     matched-prefix KV is known to exist: each hole must
                        //     lie fully inside [0, n_past) (its KV was created by
                        //     an earlier prefill) and the shift block must not
                        //     have moved the KV. Idempotent: the gate re-plans
                        //     the same holes every request (the evicted text is
                        //     kept, so the token count stays over the threshold),
                        //     so holes already in the applied ledger are skipped.
                        //     Per-hole: a hole is cut iff its KV is inside the
                        //     matched prefix (hi <= n_past); the rest are deferred
                        //     and retried on the next request (disjoint ranges -
                        //     no cross-hole dependency). All-or-nothing livelocked
                        //     a growing session: one hole in the re-prefilled tail
                        //     blocked all ~80 (2026-09-26 production journal).
                        // Item 4: expire stale ledger entries before (re)applying.
                        // A hole with hi > n_past sits in the re-prefilled region, so
                        // its KV was freshly written and is not reliably cut. Keeping
                        // it inflates the resident projection (n_past - ledger_span)
                        // and lets the dedup below skip a re-plan that needs the cut.
                        // Expire it; the gate re-plans and the apply path re-cuts
                        // once hi <= n_past.
                        if (n_past > 0 && !slot.kv_hole_ranges.empty()) {
                            // 2026-10-12: the shared expire rule (kv_holes_survive) -
                            // the same function the gate's kv_session_valid_cut uses
                            // with its simulated n_past.
                            const auto valid = kv_holes_survive(slot.kv_hole_ranges, n_past);
                            if (valid.size() != slot.kv_hole_ranges.size()) {
                                for (const auto & r : slot.kv_hole_ranges) {
                                    if (r.second > n_past) {
                                        SLT_INF_DBG(slot, "kv-offload-holes: expiring stale hole [%d, %d) (hi > n_past=%d) - KV re-prefilled, re-plan will re-apply\n",
                                                (int) r.first, (int) r.second, n_past);
                                    }
                                }
                                slot.kv_hole_ranges = std::move(valid);
                            }
                            if (slot.kv_hole_ranges.empty()) {
                                slot.kv_holes_active = false;
                            }
                        }
                        // kv-recut: re-apply every remaining ledger range. All have
                        // hi <= n_past here (stale ones expired above), so the
                        // recurrent tail (at n_past) is untouched and the op is an
                        // idempotent metadata no-op on already-cut ranges. A
                        // checkpoint restore may re-materialize cells the ledger
                        // believes are cut (2026-10-10 2nd incident: all 48694
                        // cut cells survived a PARTIAL_ONLY restore), and the
                        // n_past path would otherwise never re-cut them (the
                        // pending dedup skips ranges already in the ledger; an
                        // idle gate plans nothing). Measure the freed cells so
                        // the journal confirms the ledger matches the physical
                        // pool; a rejected range is dropped so the gate's
                        // resident projection stays honest.
                        if (n_past > 0 && !slot.kv_hole_shift_blocked && !slot.kv_hole_ranges.empty()) {
                            auto * m = llama_get_memory(ctx_tgt);
                            const int32_t f0 = llama_memory_n_free_cells(m);
                            int span = 0;
                            for (auto it = slot.kv_hole_ranges.begin(); it != slot.kv_hole_ranges.end(); ) {
                                if (!slot.mem.seq_rm_checked(slot.kv_seq(), it->first, it->second)) {
                                    SLT_WRN_DBG(slot, "kv-recut: [%d, %d) rejected - dropping from ledger\n", (int) it->first, (int) it->second);
                                    it = slot.kv_hole_ranges.erase(it);
                                    continue;
                                }
                                span += it->second - it->first;
                                ++it;
                            }
                            if (slot.kv_hole_ranges.empty()) {
                                slot.kv_holes_active = false;
                            }
                            SLT_INF_DBG(slot, "kv-recut: %zu range(s), span=%d, freed=%d\n",
                                    slot.kv_hole_ranges.size(), span, llama_memory_n_free_cells(m) - f0);
                        }
                        if (n_past == 0) {
                            if (slot.kv_holes_active || !slot.kv_hole_ranges.empty()) {
                                SLT_INF_DBG(slot, "kv-offload-holes: n_past=0 (sequence wiped) - clearing hole ledger (%zu applied)\n",
                                        slot.kv_hole_ranges.size());
                                slot.kv_hole_ranges.clear();
                                slot.kv_holes_active = false;
                            }
                        } else if (!slot.kv_hole_shift_blocked && !slot.task->params.kv_hole_pending.empty()) {
                            std::vector<std::pair<int32_t, int32_t>> fresh;
                            for (const auto & r : slot.task->params.kv_hole_pending) {
                                // Item 4: overlap-aware dedup. The gate re-plans the
                                // same holes every request; a re-plan whose range
                                // overlaps an already-applied entry (even with a
                                // +-1 coordinate shift) must be skipped - re-cutting
                                // is a no-op, but recording it again would
                                // double-count the span (the projection assumes a
                                // larger cut than physically happened, so it can
                                // admit a request that overflows the buffer).
                                bool dup = false;
                                for (const auto & a : slot.kv_hole_ranges) {
                                    if (a.first < r.second && r.first < a.second) {
                                        dup = true;
                                        break;
                                    }
                                }
                                if (!dup) {
                                    fresh.push_back(r);
                                }
                            }
                            if (fresh.size() != slot.task->params.kv_hole_pending.size()) {
                                SLT_INF_DBG(slot, "kv-offload-holes: %zu pending hole(s) already applied - skipping re-apply\n",
                                        slot.task->params.kv_hole_pending.size() - fresh.size());
                            }
                            if (!fresh.empty()) {
                                int span = 0;
                                size_t n_applied = 0;
                                size_t n_deferred = 0;
                                size_t n_noop = 0;
                                for (const auto & r : fresh) {
                                    if (r.first < 0 || r.second <= r.first || r.second > n_past || (size_t) r.second > input_tokens.size()) {
                                        n_deferred++;
                                        continue;
                                    }
                                    // fail-open (kv-offload never aborts the server): a range that is no longer
                                    // removable is typically already cut (e.g. a checkpoint restored the KV with
                                    // the hole while the ledger entry was lost or differs) - re-cutting is a no-op,
                                    // so record it to keep the per-request re-plan idempotent.
                                    if (!slot.mem.seq_rm_checked(slot.kv_seq(), r.first, r.second)) {
                                        SLT_WRN_DBG(slot, "kv-offload-holes: seq_rm no-op for [%d, %d) in main seq %d (n_past=%d) - recording as applied\n",
                                                r.first, r.second, (int) slot.kv_seq(), n_past);
                                        slot.kv_hole_ranges.push_back(r);
                                        n_noop++;
                                        continue;
                                    }
                                    slot.kv_hole_ranges.push_back(r);
                                    span += r.second - r.first;
                                    n_applied++;
                                }
                                if (n_applied + n_noop > 0) {
                                    slot.kv_holes_active = true;
                                }
                                if (n_applied > 0) {
                                    SLT_INF_DBG(slot, "kv-offload-holes: applied %zu hole(s), %d token span, in main seq %d (n_past=%d)\n",
                                            n_applied, span, (int) slot.kv_seq(), n_past);
                                }
                                if (n_noop > 0) {
                                    SLT_INF_DBG(slot, "kv-offload-holes: %zu hole(s) already cut (no-op), recorded in ledger\n", n_noop);
                                }
                                if (n_deferred > 0) {
                                    SLT_INF_DBG(slot, "kv-offload-holes: deferring %zu pending hole(s) (n_past=%d, need hi<=n_past)\n",
                                            n_deferred, n_past);
                                }
                            }
                        }

                        // kv-offload-holes: the ledger just changed (expire /
                        // apply / n_past=0 clear) - refresh the session's
                        // actual-cut entry so the next gate sees the real
                        // resident.
                        kv_session_cut_update(slot);
                        // kv-probe: physical occupancy vs the ledger, so the
                        // journal shows whether the holes are really cut
                        // (used ≈ prompt - ledger_cut) or the ledger lags the
                        // physical state (2026-10-10: holes cut 0 with ~101k
                        // resident after an idle-slot clear).
                        /*
                        if (params_base.kv_cache_size > 0) {
                            const int32_t n_free = llama_memory_n_free_cells(llama_get_memory(ctx_tgt));
                            int64_t probe_cut = 0;
                            for (const auto & r : slot.kv_hole_ranges) {
                                probe_cut += (int64_t) (r.second - r.first);
                            }
                            SLT_INF_DBG(slot, "kv-probe: used=%d free=%d ledger_cut=%d prompt=%d buffer=%d\n",
                                    n_free >= 0 ? (int) (params_base.kv_cache_size - n_free) : -1, n_free,
                                    (int) probe_cut, (int) slot.prompt.n_tokens(), params_base.kv_cache_size);
                        }
                        */

                        // progressive apply: seed the peak tracker with the
                        // resident at prefill start (the matched prefix minus
                        // the holes already cut out of it). The per-batch
                        // progressive cut (kv_holes_progressive_apply) raises
                        // it as the head advances; the end-of-prefill summary
                        // reports it.
                        {
                            int32_t ledger_span = 0;
                            size_t n_stale = 0;
                            for (const auto & r : slot.kv_hole_ranges) {
                                ledger_span += r.second - r.first;
                                if (r.second > n_past) {
                                    n_stale++;
                                }
                            }
                            if (ledger_span > n_past || n_stale > 0) {
                                SLT_WRN_DBG(slot, "kv-offload-holes: ledger polluted - span=%d, n_past=%d, %zu stale hole(s) with hi>n_past (overlapping/stale entries inflate the resident projection)\n",
                                        ledger_span, n_past, n_stale);
                            }
                            slot.kv_hole_peak_resident = n_past - ledger_span;
                        }

                        // (2026-10-12, incident 2026-10-12) Secondary safety net:
                        // the gate's estimate (HTTP thread) and the actual n_past
                        // (here, after the checkpoint restore + hole
                        // expire/recut/apply) can diverge - the gate ran
                        // milliseconds before the prior task's release, a
                        // checkpoint restore lands below the LCP, and the
                        // wipe path clears the ledger. If the projected prefill
                        // peak cannot fit the physical buffer, fail fast with a
                        // 503 (retryable: the client compacts and retries
                        // smaller) instead of the ~60s prefill + find_slot
                        // cascade + slot abort + KV clear. The progressive cut
                        // keeps the live-cell peak at the final resident plus
                        // one in-flight sub-batch, so
                        //   peak = prompt - (applied + pending cuts) + n_batch.
                        // The prefill has not started, so the slot's KV is still
                        // the consistent n_past prefix - release() keeps it for
                        // the next request's LCP match (no prompt_clear).
                        if (params_base.kv_cache_size > 0) {
                            int64_t ledger_cut = 0;
                            for (const auto & r : slot.kv_hole_ranges) {
                                ledger_cut += (int64_t) (r.second - r.first);
                            }
                            int64_t pending_cut = 0;
                            if (!slot.kv_hole_shift_blocked) {
                                // the n_cache_reuse shift moved the KV: the pending
                                // coordinates are stale and the progressive path
                                // skips them this prefill - do not count the cut
                                // (when uncertain, bias to the evict side).
                                for (const auto & r : slot.task->params.kv_hole_pending) {
                                    pending_cut += (int64_t) (r.second - r.first);
                                }
                            }
                            const int64_t prefill_peak = (int64_t) slot.task->n_tokens() - ledger_cut - pending_cut + n_batch;
                            if (prefill_peak > (int64_t) params_base.kv_cache_size) {
                                SLT_ERR(slot, "kv_offload: prefill peak %d exceeds buffer %d (n_past=%d, ledger_cut=%d, pending_cut=%d) - rejecting with 503 (retryable: client compacts and retries)\n",
                                        (int) prefill_peak, params_base.kv_cache_size, n_past,
                                        (int) ledger_cut, (int) pending_cut);
                                // The message intentionally matches the client's
                                // context-overflow patterns (and carries the real
                                // token counts for parseTokenCounts): reactive
                                // compaction is the recovery path.
                                send_error(slot,
                                           string_format("Context length exceeded: %d tokens > %d limit",
                                                         (int) prefill_peak, params_base.kv_cache_size),
                                           ERROR_TYPE_UNAVAILABLE);
                                slot.release();
                                return;
                            }
                        }

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!slot.mem.seq_rm_checked(slot.kv_seq(), p0, -1)) {
                        // fail-open (the server never aborts on a cache inconsistency): the
                        // recurrent state cannot be rolled back to n_past - e.g. a context
                        // checkpoint whose recurrent state is newer than its position label
                        // (see plans/2026-10-07-kv-400-incident2.md). The cached prefix is
                        // unusable: wipe the whole sequence (target + draft) and re-prefill
                        // from 0. Full removal always succeeds (rm_all).
                        SLT_WRN(slot, "truncate seq_rm failed (p0=%d) - recurrent state cannot roll back to n_past; wiping sequence and re-prefilling from 0\n", p0);
                        slot.mem.seq_rm_checked(slot.kv_seq(), -1, -1);
                        slot.prompt.tokens.clear();
                        slot.stats.n_prompt_cached = 0;
                        // the wiped KV invalidates the applied holes; the request-scoped
                        // pending holes survive and are re-cut progressively during the
                        // re-prefill
                        slot.kv_hole_ranges.clear();
                        slot.kv_holes_active = false;
                        slot.kv_hole_peak_resident = 0;
                        kv_session_cut_update(slot);
                    }

                    // shared prompt prefix: once it is processed, the children continue from it with their own prompt
                    bool wait_shared = false;
                    if (slot.task->n_tokens_shared > 0) {
                        const bool is_shared_done = slot.prompt.n_tokens() == slot.task->n_tokens_shared;
                        for (auto & other : slots) {
                            if (other.state != SLOT_STATE_WAIT_OTHER || other.task->id_parent != slot.task->id) {
                                continue;
                            }
                            if (is_shared_done) {
                                SLT_TRC(slot, " - copying shared prompt (%d tokens) to child %d\n", slot.prompt.n_tokens(), other.id);
                                slot.copy_prompt_to(other);
                                other.state = SLOT_STATE_STARTED;
                            } else {
                                wait_shared = true;
                            }
                        }
                    }

                    // kv-probe (post-trim): occupancy after the tail trim, so the
                    // used value is comparable to the ledger (the STARTED-block
                    // probe runs before this trim and overstates used by the
                    // trimmed tail - 2026-10-10). Hole sessions only.
                    if (params_base.kv_cache_size > 0 && !slot.kv_hole_ranges.empty()) {
                        const int32_t n_free = llama_memory_n_free_cells(llama_get_memory(ctx_tgt));
                        int64_t probe_cut = 0;
                        for (const auto & r : slot.kv_hole_ranges) {
                            probe_cut += (int64_t) (r.second - r.first);
                        }
                        SLT_INF_DBG(slot, "kv-probe (post-trim): used=%d free=%d ledger_cut=%d prompt=%d\n",
                                n_free >= 0 ? (int) (params_base.kv_cache_size - n_free) : -1, n_free,
                                (int) probe_cut, (int) slot.prompt.n_tokens());
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    // TODO: do the same for all models, then remove process_mtmd_chunk()
                    if (use_mixed_batch() && !slot.can_split() && input_tokens.has_mtmd) {
                        if (!add_prompt_mixed(slot)) {
                            return;
                        }
                    }

                    // Declarative Attention: apply the pending removals once prompt
                    // prefill has crossed da_rm_at. The KV for [0, da_rm_at) is
                    // complete at this point (tokens pushed in earlier rounds have
                    // been decoded), so the remaining prompt is prefilled without
                    // the removed ranges. Mid-prefill application is skipped for
                    // prompts that contain media (the boundary must not fall
                    // inside an mtmd chunk); those fall through to the
                    // post-prefill path. has_mtmd is not the right test here: it
                    // is true whenever an mmproj is loaded, even for text-only
                    // prompts, so the guard uses da_rm_text_only (the prompt has
                    // no media chunks).
                    if (slot.da_rm_pending && slot.task->params.da_rm_at >= 0 &&
                            slot.prompt.n_tokens() >= slot.task->params.da_rm_at) {
                        if (slot.da_rm_text_only) {
                            if (slot.task->params.da_b) {
                                apply_da_b(slot, slot.task->params.da_rm, da_static_bound(slot), "at da_rm_at");
                            } else {
                                apply_da_rm(slot, slot.task->params.da_rm, da_static_bound(slot), "at da_rm_at");
                            }
                            slot.da_rm_pending = false;
                        } else if (!slot.da_rm_gate_diag_done) {
                            slot.da_rm_gate_diag_done = true;
                            const size_t n_total = slot.task->tokens.size();
                            const size_t n_text  = slot.task->tokens.get_text_tokens().size();
                            SLT_INF_DBG(slot, "da_rm: mid-prefill boundary reached but skipped (media present) - da_rm_at=%d has_mtmd=%d n_tokens=%d n_text=%zu n_media=%zu; applying after prefill instead (the boundary may fall inside an mtmd chunk)\n",
                                    slot.task->params.da_rm_at, (int) slot.task->tokens.has_mtmd, (int) slot.prompt.n_tokens(),
                                    n_text, n_total - n_text);
                        }
                    }

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // stop at the end of the shared prefix, the children are started from this state
                        if (wait_shared && slot.prompt.n_tokens() == slot.task->n_tokens_shared) {
                            break;
                        }

                        // the outputs of a decision are read from one batch, do not split them
                        if (slot.prompt.n_tokens() == n_decision_first && batch.size() + slot.task->n_tokens() - n_decision_first > n_batch) {
                            break;
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        // kv_seq(): after a 2-stream (B) switch the remaining
                        // prompt (e.g. the question) is prefilled on the new
                        // sequence.
                        add_ok &= batch.add(slot.kv_seq(),
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        if (!slot.task->decision.order.empty()) {
                            batch.set_decision_order(batch.size() - 1, slot.task->decision.order[slot.prompt.n_tokens()]);
                        }
                        slot.prompt.tokens.push_back(cur_tok);

                        // Declarative Attention: stop filling at the removal boundary
                        // so the removals can run before the remaining prompt
                        // (e.g. the question) is prefilled. Without this break the
                        // checkpoint offset (4 tokens before the end) would push the
                        // first batch past da_rm_at and the removal would apply too
                        // late. da_rm_text_only, not has_mtmd: the latter is true
                        // for text-only prompts on an mmproj server.
                        if (slot.da_rm_pending && slot.task->params.da_rm_at >= 0 &&
                                slot.da_rm_text_only &&
                                slot.prompt.n_tokens() >= slot.task->params.da_rm_at) {
                            // fires at most once: da_rm_pending is cleared when the
                            // gate applies the removals in the next prefill round
                            SLT_INF_DBG(slot, "da_rm: batch fill stopped at boundary - n_tokens=%d, da_rm_at=%d, %d token(s) left to prefill after removal\n",
                                    (int) slot.prompt.n_tokens(), slot.task->params.da_rm_at,
                                    (int) (slot.task->n_tokens() - slot.prompt.n_tokens()));
                            break;
                        }

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back()->n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    // the label a PARTIAL_ONLY checkpoint must record: the position at which
                    // the serialized state can be resumed. For hybrid memory the state is the
                    // recurrent state only, so this is the recurrent position - not
                    // seq_pos_max() (min(attn, recr)), which is lower whenever the attention
                    // side is hole-cut or truncated to n_past by a checkpoint restore and
                    // would poison the label below the state (plans/2026-10-07-kv-400-incident2.md)
                    const auto pos_max = llama_memory_seq_pos_max_partial(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back()->n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // see https://github.com/ggml-org/llama.cpp/pull/29969
    // TODO @ngxson : maybe remove this once we use "mixed" batch everywhere
    bool use_mixed_batch() const {
        return !llama_get_memory(ctx_tgt) || !llama_get_causal_attn(ctx_tgt);
    }

    // add the rest of the prompt to the batch, the mtmd chunks are added as embeddings next to the text tokens
    // the caller makes sure that it fits in the batch
    // returns false on error, the slot is then released
    bool add_prompt_mixed(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        const auto n_tokens_prev = batch.size();

        while (slot.prompt.n_tokens() < slot.task->n_tokens()) {
            const auto cur_token_idx = slot.prompt.n_tokens();
            const llama_token cur_tok = input_tokens[cur_token_idx];

            if (cur_tok != LLAMA_TOKEN_NULL) {
                const bool add_ok = batch.add(slot.id,
                    cur_tok,
                    /* pos       = */ slot.prompt.tokens.pos_next(),
                    /* output    = */ slot.need_embd(),
                    /* is_prompt = */ true);
                GGML_ASSERT(add_ok);
                if (!slot.task->decision.order.empty()) {
                    batch.set_decision_order(batch.size() - 1, slot.task->decision.order[cur_token_idx]);
                }
                slot.prompt.tokens.push_back(cur_tok);
                continue;
            }

            const auto & chunk = input_tokens.find_chunk(cur_token_idx);

            float * embd = slot.mbatch ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            if (!embd) {
                // encode on the worker thread, so we can still handle metrics tasks
                int32_t res = 0;
                queue_tasks.yield_to_queue([&]() {
                    res = encode_mtmd_chunk(slot, slot.mbatch, cur_token_idx);
                });
                embd = res == 0 ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            }

            if (!embd || !add_mtmd_chunk(slot, chunk.get(), embd, batch)) {
                SLT_ERR(slot, "%s", "failed to process mtmd chunk\n");
                // the batch must not keep the entries of a released slot
                batch.truncate(n_tokens_prev);
                send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }

            slot.prompt.tokens.push_back_placeholder(chunk.get());
        }

        return true;
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd()) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }

        // DA: push the n_kv_max bound for this window before the decode
        // (no-op while the value is unchanged; 0 = dense)
        update_da_n_kv_max(off, batch.view.size());

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    // L6: who holds the KV at the moment of the decode fail
                    // (progressive-cut verification: applied holes + pending
                    // holes + observed peak vs the buffer)
                    for (auto & s : slots) {
                        if (!s.is_processing()) continue;
                        int span = 0;
                        for (auto & r : s.kv_hole_ranges) span += (int) (r.second - r.first);
                        SRV_ERR_DBG("da_trace: decode fail slot=%d prompt=%d cached=%d holes_applied=%zu span=%d progress=%zu pending=%zu peak=%d buffer=%d da_seq=%d\n",
                                s.id, (int) s.prompt.n_tokens(), (int) s.stats.n_prompt_cached, s.kv_hole_ranges.size(), span,
                                s.kv_hole_progress.size(), s.task ? s.task->params.kv_hole_pending.size() : 0,
                                s.kv_hole_peak_resident, params_base.kv_cache_size, (int) s.da_seq);
                    }

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch.view.size(), has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);
                    // children with their own prompt are started at the end of the shared prefix
                    GGML_ASSERT(slot.task->n_tokens_shared == 0);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    // kv-offload-holes progressive apply (prefill-time hole cutting): after
    // each successful prompt sub-batch, cut the pending holes whose KV has
    // been written so far. Without this, a (re-)prefill of a prompt whose
    // evicted text is kept (Option B) writes every token to cells before the
    // holes are cut at the next request's n_past check - the prefill peak is
    // the raw prompt length, which exceeds a decoupled buffer B < n_ctx and
    // hard-fails in find_slot ("Context size has been exceeded", 2026-10-06
    // production journal, 66025 tokens vs B = 65536). Cutting as the head
    // passes keeps the peak near the projected resident + one batch.
    //
    // Per hole, [max(lo, applied_until), min(hi, done)) is cut, where done is
    // this slot's processed prefix length (its max processed position + 1).
    // The batch path (STARTED block) already cut the holes fully inside the
    // matched prefix (hi <= n_past) - they are in the applied ledger and are
    // excluded here; this path handles the holes in the freshly re-prefilled
    // region that the batch path deferred (hi > n_past).
    //
    // The cut reuses the production hole path (common_memory::seq_rm_checked,
    // fail-open): a rejected cut - e.g. a recurrent-state hybrid whose state
    // cannot roll back that far (rollback bounded by n_rs_seq) - keeps the
    // progress and is retried as the head passes the hole (once the head is
    // past it the cut is a plain untag). If it never cuts, the n_past path
    // applies it on the next request (the hole's KV is then fully inside the
    // matched prefix). A fully cut hole joins the applied ledger (drives the
    // release() cache policy, the dense n_kv_max override, and the
    // per-request re-plan idempotency).
    void kv_holes_progressive_apply(int32_t n_processed) {
        iterate(slots, [&](server_slot & slot) {
            if (slot.kv_hole_shift_blocked) {
                return; // the shift moved the KV - the hole coordinates are stale
            }
            if (slot.state != SLOT_STATE_PROCESSING_PROMPT && slot.state != SLOT_STATE_DONE_PROMPT) {
                return; // generation batches carry no prompt tokens
            }
            if (!slot.task || slot.task->params.kv_hole_pending.empty()) {
                return; // no holes planned - zero overhead for plain traffic
            }

            // this slot's processed prefix length: the max position of its
            // prompt tokens in the processed part of the batch (+1). The
            // matched prefix [0, n_past) is not in the batch (its KV already
            // exists), so the first token of the round sits at n_past.
            llama_pos max_pos = -1;
            for (int32_t i = 0; i < n_processed; ++i) {
                const auto & t = batch.tokens[i];
                if (t.id_slot == slot.kv_seq() && t.is_prompt && t.pos[0] > max_pos) {
                    max_pos = t.pos[0];
                }
            }
            if (max_pos < 0) {
                return; // nothing of this slot's prompt processed yet
            }
            const llama_pos done = max_pos + 1;

            // lazily build the per-hole progress ledger for this request:
            // the pending holes minus the ones already in the applied ledger
            // (cut by the batch path this request or in a previous one).
            if (slot.kv_hole_progress.empty()) {
                for (const auto & r : slot.task->params.kv_hole_pending) {
                    bool applied = false;
                    for (const auto & a : slot.kv_hole_ranges) {
                        if (a.first == r.first && a.second == r.second) {
                            applied = true;
                            break;
                        }
                    }
                    if (!applied) {
                        slot.kv_hole_progress.push_back({ r.first, r.second, r.first });
                    }
                }
            }

            // resident = processed - cut so far (applied ledger + partial
            // progress cuts); each hole is counted exactly once
            int32_t cut_total = 0;
            for (const auto & a : slot.kv_hole_ranges) {
                cut_total += a.second - a.first;
            }
            for (const auto & h : slot.kv_hole_progress) {
                cut_total += (int32_t) (h.applied_until - h.lo);
            }

            // the cell pressure is at write time: this batch's cells are
            // already resident, and the cuts below only free cells for the
            // next batch - measure the peak before cutting
            const int32_t resident_pre = (int32_t) done - cut_total;
            if (resident_pre > slot.kv_hole_peak_resident) {
                slot.kv_hole_peak_resident = resident_pre;
            }

            for (size_t i = 0; i < slot.kv_hole_progress.size(); ++i) {
                auto & h = slot.kv_hole_progress[i];
                const llama_pos a = std::max(h.lo, h.applied_until);
                // cut up to done-1, not done: a range with p1 > cell.pos
                // (cell.pos = done-1 after this batch) straddles the
                // recurrent tail and enters the rollback branch, which a
                // hybrid rejects for big holes (rollback bounded by
                // n_rs_seq). The lagging last token is picked up by the next
                // round or the final n_past cut.
                const llama_pos b = std::min(h.hi, done - 1);
                if (b <= a) {
                    continue;
                }
                if (!slot.mem.seq_rm_checked(slot.kv_seq(), a, b)) {
                    // fail-open: keep the progress, retry as the head passes
                    if (!h.fail_logged) {
                        h.fail_logged = true;
                        SLT_WRN_DBG(slot, "kv-offload-holes: progressive cut of [%d, %d) rejected at done=%d (seq %d) - retrying as the head passes; if it never cuts, the n_past path applies it on the next request\n",
                                (int) a, (int) b, (int) done, (int) slot.kv_seq());
                    }
                    continue;
                }
                h.applied_until = b;
                cut_total += (int32_t) (b - a);
                if (b == h.hi) {
                    // fully cut - join the applied ledger and stop tracking
                    slot.kv_hole_ranges.push_back({ h.lo, h.hi });
                    slot.kv_holes_active = true;
                    slot.kv_hole_progress.erase(slot.kv_hole_progress.begin() + i);
                    --i;
                    kv_session_cut_update(slot);
                    SLT_INF_DBG(slot, "kv-offload-holes: progressive cut completed [%d, %d) at done=%d (seq %d)\n",
                            (int) h.lo, (int) h.hi, (int) done, (int) slot.kv_seq());
                }
            }

            const int32_t resident = (int32_t) done - cut_total;
            if (resident > slot.kv_hole_peak_resident) {
                slot.kv_hole_peak_resident = resident;
            }
            // FOCUS_SPARSE_DEBUG-gated: fires once per prompt sub-batch.
            if (focus_sparse_debug()) {
                SLT_INF_DBG(slot, "kv-offload-holes: prefill progress - done=%d resident=%d (cut %d, peak %d, buffer %d)\n",
                        (int) done, resident, cut_total, slot.kv_hole_peak_resident, params_base.kv_cache_size);
            }
        });
    }

    // Declarative Attention (W2 physical reduction): the per-batch n_kv_max
    // bound pushed to the target context before each decode. Backends with a
    // sparse flash-attention path (CUDA MMA_F16, Metal, Vulkan) use it to
    // gather only the finite (attended) KV cells of each mask row, so the DA
    // holes - physically still in the stream (A: cells untagged from the
    // sequence, B: cells still tagged on the original sequence) - are skipped
    // instead of read.
    //
    // A mask row's finite set grows monotonically with the token position
    // (each row attends to a superset of the previous row's cells), so the
    // last token of the batch carries the maximum count. Per slot it is the
    // logical read set the DA logs already track:
    //   A path: n_full - n_da_removed
    //   B path: da_keep_count + (n_full - da_bound)
    //   plain:  n_full (no holes, the whole causal row is finite)
    // where n_full = slot.prompt.n_tokens() already includes this batch's
    // tokens (handle_last_sampled_token adds them before decode() runs).
    // The bound is set only when at least one slot has holes: for plain
    // traffic it stays 0, so the flash-attn nodes (and the cached graphs)
    // are untouched. When holes are present, every processing slot must be
    // counted - the bound must never under-estimate any row's finite count,
    // or the excess cells are silently dropped. The value is bucketed to 512
    // because it is baked into the flash-attn nodes: a change rebuilds the
    // cached graph, and 512 bounds that to ~1 rebuild per 512 generated
    // tokens. Prefill batches stay dense (0): the bound must never be set
    // while prompt tokens are being written, and one tile per query row is
    // not worth it there.
    void update_da_n_kv_max(int32_t off, int32_t n_tokens) {
        // FOCUS_DA_DENSE=1: keep the dense path even with DA holes (A/B
        // verification of the sparse kernel against the dense reference)
        static const bool dense_override = []() {
            const char * e = getenv("FOCUS_DA_DENSE");
            if (e && atoi(e) != 0) {
                // log once: without this, a leftover FOCUS_DA_DENSE=1 in the
                // service env forces dense for the whole process with no trace
                SRV_INF_DBG("%s", "da: FOCUS_DA_DENSE active - n_kv_max stays 0 (dense forced for A/B verification)\n");
            }
            return e && atoi(e) != 0;
        }();
        if (dense_override) {
            return;
        }

        // Option B (--kv-offload-holes): mid-sequence holes are not supported by
        // the sparse (n_kv_max) path in v1 (R3 - unverified at 100K context).
        // Force the dense path whenever any slot carries kv holes: the generic
        // KQ -inf mask handles the holes (the same physical state the DA A-path
        // runs in production). Reset the bound to 0 so a prior sparse bound (from
        // a co-existing DA slot) does not persist. Scans all slots: an idle slot's
        // holes must not be under-estimated by a sparse bound set for a co-tenant.
        static bool r3_forced = false;
        for (auto & slot : slots) {
            if (slot.kv_holes_active) {
                if (!r3_forced) {
                    // the bound -> 0 log below only fires on a non-zero transition;
                    // log the forcing itself on entry, so an always-0 bound is not
                    // a silent dense (the kernel journal then shows only "n_kv_max=0")
                    r3_forced = true;
                    SRV_INF("da: R3 forcing dense - slot %d carries kv-offload holes (%zu range(s)) - n_kv_max stays 0, sparse gate cannot engage\n",
                            (int) slot.kv_seq(), slot.kv_hole_ranges.size());
                }
                if (da_n_kv_max_last != 0) {
                    SRV_INF("da: n_kv_max %lld -> 0 (kv-offload-holes: forcing dense, R3)\n", (long long) da_n_kv_max_last);
                    da_n_kv_max_last = 0;
                    llama_set_n_kv_max(ctx_tgt, 0);
                }
                return;
            }
        }
        if (r3_forced) {
            r3_forced = false;
            SRV_INF("%s", "da: R3 lifted - no slot with kv-offload holes, n_kv_max computation resumes\n");
        }

        int64_t n_kv_max = 0;

        bool prefill = false;
        for (int32_t i = off; i < off + n_tokens; ++i) {
            if (batch.tokens[i].is_prompt) {
                prefill = true;
                break;
            }
        }

        if (!prefill) {
            bool any_holes = false;
            for (auto & slot : slots) {
                if (slot.is_processing() && (slot.da_seq >= 0 || slot.da_removed_a)) {
                    any_holes = true;
                    break;
                }
            }
            if (any_holes) {
                for (auto & slot : slots) {
                    if (!slot.is_processing()) {
                        continue;
                    }
                    const int32_t n_full = (int32_t) slot.prompt.n_tokens();
                    int32_t n_read;
                    if (slot.da_seq >= 0) {
                        // B: the slot decodes on da_seq - the compact
                        // sequence (kept prefix + generated tail), every
                        // cell finite
                        n_read = (int32_t) slot.da_keep_count + (n_full - slot.da_bound);
                    } else if (slot.da_removed_a) {
                        // A: holes in the original sequence, finite = the rest
                        n_read = std::max(0, n_full - (int32_t) slot.n_da_removed);
                    } else {
                        // plain co-tenant: the whole causal row is finite
                        n_read = n_full;
                    }
                    n_kv_max = std::max(n_kv_max, ((int64_t) n_read + 511) / 512 * 512);
                }
            }
        }

        if (n_kv_max != da_n_kv_max_last) {
            SRV_INF_DBG("da: n_kv_max %lld -> %lld%s\n", (long long) da_n_kv_max_last, (long long) n_kv_max, prefill ? " (prefill, dense)" : "");
            da_n_kv_max_last = n_kv_max;
        }
        llama_set_n_kv_max(ctx_tgt, n_kv_max);
        // the draft context is intentionally left dense: its prompt length is
        // not exposed by common_speculative, and the bound must never under-
        // estimate a row's finite count (the excess cells would be silently
        // dropped)
    }

    // The removal boundary for the static (client-supplied da_rm) path: the
    // da_rm_at position clamped to the prompt length, or the prompt end when
    // da_rm_at < 0. (The tag path computes its own boundary - the current
    // decode position - in apply_da_tag.)
    int32_t da_static_bound(const server_slot & slot) const {
        const int32_t n_prompt = slot.task->n_tokens();
        return slot.task->params.da_rm_at >= 0
            ? std::min(slot.task->params.da_rm_at, n_prompt)
            : n_prompt;
    }

    // Subtract [sub_lo, sub_hi) from a list of token ranges. For each range
    // that overlaps the subtraction range, the overlapping part is removed
    // (splitting the range into left and right parts when the subtraction
    // range is in the middle). Ranges that do not overlap are kept unchanged.
    // Used to guarantee that the Σ anchor range is NOT removed (stays in the
    // keep set) regardless of the layout scan's classification.
    static void da_subtract_range(std::vector<std::pair<int32_t, int32_t>> & ranges, int32_t sub_lo, int32_t sub_hi) {
        if (sub_hi <= sub_lo) {
            return;
        }
        std::vector<std::pair<int32_t, int32_t>> out;
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

    // Total span covered by a range list. Exact on merged lists; on
    // overlapping lists an overlap is counted once per range (fine for the
    // newly-kept estimate in da_apply_keep_floor - the floor range is
    // disjoint from the Sigma anchor, so both call sites see near-merged
    // ranges).
    static size_t da_ranges_total(const std::vector<std::pair<int32_t, int32_t>> & ranges) {
        size_t total = 0;
        for (const auto & r : ranges) {
            total += (size_t) (r.second - r.first);
        }
        return total;
    }

    // DA tail floor (plans/continous_work2.md): subtract [bound-K, bound)
    // from the removal ranges so the most recent K tokens before the removal
    // boundary are always kept. The keep set is the union of the tag
    // selection, the Sigma anchor and the tail: without the floor, the work
    // region the model was just in (not covered by any magic_chunks tag) is
    // masked out at a B switch and the model forgets what it just did (the
    // 10-09 12:43 amnesia: ~21k tokens of [42060,63370) masked whole).
    // Reuses da_subtract_range; the floor is clamped to [0, bound], so
    // bound < K keeps everything. K <= 0 or bound <= 0: no-op (legacy
    // behavior). Logs the applied floor, the newly-kept token count and the
    // remaining removals bucketed by distance from the tail (bound), so the
    // floor's read-reduction trade is visible in the same log group as the
    // switch summary.
    static void da_apply_keep_floor(const server_slot & slot,
                                    std::vector<std::pair<int32_t, int32_t>> & rm_ranges,
                                    int32_t bound,
                                    const char * when) {
        const int32_t K = slot.da_tail_keep;
        if (K <= 0 || bound <= 0) {
            return;
        }
        const int32_t lo = std::max(bound - K, 0);
        const int32_t hi = bound;
        const size_t before = da_ranges_total(rm_ranges);
        da_subtract_range(rm_ranges, lo, hi);
        const size_t newly_kept = before - da_ranges_total(rm_ranges);
        SLT_INF_DBG(slot, "da: tail_keep=[%d, %d) - %zu token(s) newly kept (K=%d, %s)\n",
                lo, hi, newly_kept, K, when);
        // bucket the remaining removals by the distance of the range end
        // from the tail: how much of what was removed sat close to the work
        // in progress (0 = touching the boundary)
        if (!rm_ranges.empty()) {
            const int32_t edges[] = { 1024, 4096, 8192, 16384, 32768 };
            size_t bkt[6] = { 0, 0, 0, 0, 0, 0 };
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
            SLT_INF_DBG(slot, "da: removed by distance from tail (%s): <1k=%zu 1-4k=%zu 4-8k=%zu 8-16k=%zu 16-32k=%zu 32k+=%zu\n",
                    when, bkt[0], bkt[1], bkt[2], bkt[3], bkt[4], bkt[5]);
        }
    }

    // Declarative Attention: remove the given prompt token ranges from the KV
    // cache of the target context (and the draft context, when speculative
    // decoding is active). The removal is a no-op for positions whose KV does
    // not exist yet, so callers must invoke it once the covered positions are
    // fully prefilled (or, for the tag path, mid-decode, where the ranges are
    // mid-range and the covered positions already exist). Ranges are clamped
    // to the removal boundary `bound`: the next token is placed at `bound`, so
    // position bound - 1 must stay in the cache for contiguity.
    void apply_da_rm(server_slot & slot, const std::vector<std::pair<int32_t, int32_t>> & ranges, int32_t bound, const char * when) {
        if (ranges.empty()) {
            return;
        }

        if (slot.da_measure_only) {
            SLT_INF_DBG(slot, "da_rm: SKIPPED (measure-only) - %zu range(s) not applied to KV (%s); output stays vanilla\n",
                    ranges.size(), when);
            return;
        }

        const int32_t n_prompt = slot.task->n_tokens();

        SLT_INF_DBG(slot, "da_rm: applying %zu range(s) (%s) - bound=%d, n_prompt=%d\n",
                ranges.size(), when, bound, n_prompt);

        // Subtract the Σ anchor range from the removal ranges so it is
        // guaranteed to stay (not removed). The Σ block is server-injected
        // text (not in the client history) that must remain visible to the
        // model after a removal, regardless of how the layout scan classifies
        // the region after the filler.
        std::vector<std::pair<int32_t, int32_t>> rm_ranges = ranges;
        {
            const auto & sig = slot.task->params.da_sigma;
            if (sig.first >= 0) {
                const int32_t s_lo = std::max(sig.first, 0);
                const int32_t s_hi = std::min(sig.second, bound);
                if (s_hi > s_lo) {
                    da_subtract_range(rm_ranges, s_lo, s_hi);
                    SLT_INF_DBG(slot, "da_rm: sigma tokens [%d, %d) kept (subtracted from removal ranges)\n", s_lo, s_hi);
                }
            }
        }

        // Tail floor (plans/continous_work2.md): the most recent K tokens
        // before the boundary are always kept, next to the Sigma anchor.
        da_apply_keep_floor(slot, rm_ranges, bound, when);

        for (const auto & range : rm_ranges) {
            int32_t lo = range.first;
            int32_t hi = range.second;

            if (lo >= bound) {
                SLT_WRN_DBG(slot, "da_rm: range [%d, %d) starts at/after the removal boundary %d - skipping\n", lo, hi, bound);
                continue;
            }

            // The next token (the prompt continuation, or the first generated
            // token) is placed at position `bound`. The attention KV cache
            // requires position contiguity, so the last position of the
            // prefilled prefix (bound - 1) must remain in the cache. Clamp the
            // range end to leave it intact; a range that reaches the boundary
            // would otherwise drop seq_pos_max and make the continuation
            // non-contiguous (llama_decode returns "Invalid input batch").
            if (hi > bound - 1) {
                SLT_WRN_DBG(slot, "da_rm: range [%d, %d) reaches the removal boundary %d - clamping end to %d (position %d must stay for contiguity)\n",
                        lo, hi, bound, bound - 1, bound - 1);
                hi = bound - 1;
            }

            if (hi <= lo) {
                SLT_WRN_DBG(slot, "da_rm: range [%d, %d) is empty after clamping - skipping\n", lo, hi);
                continue;
            }

            bool ok = true;
            auto * mem_tgt = llama_get_memory(ctx_tgt);
            if (mem_tgt) {
                ok = llama_memory_seq_rm(mem_tgt, slot.id, lo, hi) && ok;
            }
            if (ctx_dft) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                if (mem_dft) {
                    ok = llama_memory_seq_rm(mem_dft, slot.id, lo, hi) && ok;
                }
            }
            if (ok) {
                slot.n_da_removed += (uint32_t) (hi - lo);
                slot.da_applied    = true;
                slot.da_removed_a  = true;
            }
            SLT_INF_DBG(slot, "da_rm: [%d, %d) %s (%s)\n", lo, hi, ok ? "removed" : "rejected", when);
        }
    }

    // Declarative Attention 2-stream (B): instead of removing the da_rm ranges
    // from the slot's own sequence, copy the KEEP ranges (the complement of
    // da_rm in [0, bound)) to a second sequence via llama_memory_seq_cp and
    // switch the slot's decode to that sequence. The original sequence is left
    // intact - B's unique value over the logical removal (A): a global return
    // (attending the full prompt again) reuses the original sequence without
    // a re-prefill - the close-tag handler (apply_da_tag case 2-5) copies the
    // generated tail back and frees da_seq. A's removals stay irreversible.
    //
    // Read-reduction semantics: in the unified KV pool, seq_cp only retags
    // cell metadata - the cells are not moved and the attention scan range
    // [0, n_kv) is not shortened. The decode's LOGICAL read set (the tokens
    // it attends to) shrinks to the kept prefix plus the tokens added after
    // the boundary; the removed cells stay in place, masked with -inf. The
    // physical KV load IS reduced where the FA kernel skips fully-masked
    // chunks (Metal: 32-cell decode / 64-block prefill, CPU: per-position;
    // CUDA unverified), so the reduction is real but chunk-quantized.
    //
    // Preconditions (checked here; on violation, fall back to the logical
    // removal apply_da_rm):
    //   - the KV must be unified (--kv-unified): a partial-range seq_cp across
    //     KV streams aborts (llama_kv_cache::seq_cp), while within one stream
    //     it only retags cell metadata.
    //   - a sequence id beyond the slot ids must be available: the server
    //     reserves n_slots extra ids at context creation (n_seq_max = 2*n_slots
    //     in --kv-unified mode), so up to n_slots B slots can run at once.
    //     Ids below n_slots are never used - they may hold idle slots' cached
    //     prompts, which prompt_clear() would destroy.
    //   - the prefilled prompt must not exceed the removal boundary: the keep
    //     ranges only cover [0, bound), so tokens in [bound, n_prompt) would
    //     be missing from the new sequence (the media-blocked after-prefill
    //     path).
    //
    // Read-reduction logs: INF at the switch (keep ranges, logical read set
    // "N of M"), DBG per decode step (handle_last_sampled_token), INF summary
    // in release().
    //
    // `ranges` are the removal ranges and `bound` the removal boundary, both
    // supplied by the caller. The static path passes the client's da_rm ranges
    // and the da_rm_at/prompt-end boundary (before the first generated token).
    // The tag path (apply_da_tag) passes the tag-derived ranges and the current
    // decode position: the keep ranges then also cover the already-generated
    // tokens [n_prompt, bound), which are shared into the second sequence so
    // the switch stays position-contiguous.
    void apply_da_b(server_slot & slot, const std::vector<std::pair<int32_t, int32_t>> & ranges, int32_t bound, const char * when) {
        if (ranges.empty()) {
            return;
        }

        if (slot.da_measure_only) {
            SLT_INF_DBG(slot, "da_b: SKIPPED (measure-only) - %zu range(s) not applied to KV (%s); da_seq stays -1, output stays vanilla\n",
                    ranges.size(), when);
            return;
        }

        if (bound <= 0) {
            SLT_WRN_DBG(slot, "da_b: removal boundary is empty (bound=%d) - falling back to logical removal (%s)\n", bound, when);
            apply_da_rm(slot, ranges, bound, when);
            return;
        }

        // the keep ranges only cover [0, bound); if the prefilled prompt is
        // longer than the boundary (the mid-prefill gate was blocked, e.g. by
        // media, and the removals are applied after prefill), the tokens in
        // [bound, n_prompt) would be missing from the new sequence
        if (slot.prompt.n_tokens() > bound) {
            SLT_WRN_DBG(slot, "da_b: prefilled prompt (%d tokens) exceeds the removal boundary %d - falling back to logical removal (%s)\n",
                    slot.prompt.n_tokens(), bound, when);
            apply_da_rm(slot, ranges, bound, when);
            return;
        }

        // partial-range seq_cp across KV streams aborts; only the unified pool
        // (one stream for all sequences) supports it via metadata retagging
        if (!llama_kv_unified(ctx_tgt) || (ctx_dft && !llama_kv_unified(ctx_dft))) {
            SLT_WRN_DBG(slot, "da_b: requires --kv-unified (partial-range seq_cp across KV streams is unsupported) - falling back to logical removal (%s)\n", when);
            apply_da_rm(slot, ranges, bound, when);
            return;
        }

        // pick a sequence id beyond the slot ids: ids below n_slots may hold
        // idle slots' cached prompts, which this slot's prompt_clear() would
        // destroy. The server reserves n_slots extra ids at context creation
        // (n_seq_max = 2*n_slots in --kv-unified mode).
        const int n_seq_max = (int) llama_n_seq_max(ctx_tgt);
        const int n_slots   = (int) slots.size();
        llama_seq_id da_seq = -1;
        for (int i = n_slots; i < n_seq_max; ++i) {
            bool taken = false;
            for (auto & other : slots) {
                // only a processing slot can hold live KV on the id: an idle
                // slot's da_seq KV was already cleared by prompt_clear() in
                // release() (and da_seq reset there too), so reusing the id
                // under it is safe. Counting idle slots would burn the single
                // reserved id on the first finished request.
                if (other.is_processing() && other.da_seq == i) {
                    taken = true;
                    break;
                }
            }
            if (!taken) {
                da_seq = (llama_seq_id) i;
                break;
            }
        }
        if (da_seq < 0) {
            SLT_WRN_DBG(slot, "da_b: no sequence id beyond the slot ids (n_seq_max=%d, n_slots=%d) - falling back to logical removal (%s)\n", n_seq_max, n_slots, when);
            apply_da_rm(slot, ranges, bound, when);
            return;
        }

        // build the keep ranges: the complement of the removal ranges in
        // [0, bound). The clamping mirrors apply_da_rm: a range reaching the
        // boundary is cut at bound - 1 so the last position of the prefilled
        // prefix stays in the new sequence (position contiguity for the
        // continuation).
        //
        // The removal ranges may be unsorted or overlapping (e.g.
        // [[100,120],[10,20]]). Clamping, sorting by lo and merging overlaps
        // BEFORE taking the complement - a forward-only sweep over the raw
        // ranges would let an out-of-order range corrupt the keep set.
        std::vector<std::pair<int32_t, int32_t>> rm;
        for (const auto & range : ranges) {
            int32_t lo = range.first;
            int32_t hi = range.second;

            if (lo >= bound) {
                SLT_WRN_DBG(slot, "da_b: range [%d, %d) starts at/after the removal boundary %d - ignoring for the copy\n", lo, hi, bound);
                continue;
            }
            if (hi > bound - 1) {
                SLT_WRN_DBG(slot, "da_b: range [%d, %d) reaches the removal boundary %d - clamping end to %d (position %d must stay for contiguity)\n",
                        lo, hi, bound, bound - 1, bound - 1);
                hi = bound - 1;
            }
            lo = std::max(lo, 0);
            if (hi <= lo) {
                continue;
            }
            rm.emplace_back(lo, hi);
        }
        std::sort(rm.begin(), rm.end());
        std::vector<std::pair<int32_t, int32_t>> rm_merged;
        for (const auto & r : rm) {
            if (!rm_merged.empty() && r.first <= rm_merged.back().second) {
                rm_merged.back().second = std::max(rm_merged.back().second, r.second);
            } else {
                rm_merged.emplace_back(r);
            }
        }

        // Subtract the Σ anchor range from the removal ranges so it is
        // guaranteed to stay in the keep set. The Σ block is server-injected
        // text (not in the client history) that must remain visible to the
        // model after a B switch, regardless of how the layout scan classifies
        // the region after the filler.
        {
            const auto & sig = slot.task->params.da_sigma;
            if (sig.first >= 0) {
                const int32_t s_lo = std::max(sig.first, 0);
                const int32_t s_hi = std::min(sig.second, bound);
                if (s_hi > s_lo) {
                    da_subtract_range(rm_merged, s_lo, s_hi);
                    SLT_INF_DBG(slot, "da_b: sigma tokens [%d, %d) kept (subtracted from removal ranges)\n", s_lo, s_hi);
                }
            }
        }

        // Tail floor (plans/continous_work2.md): the most recent K tokens
        // before the boundary are always kept, next to the Sigma anchor.
        da_apply_keep_floor(slot, rm_merged, bound, when);

        std::vector<std::pair<int32_t, int32_t>> keep;
        {
            int32_t cur = 0;
            for (const auto & r : rm_merged) {
                if (r.first > cur) {
                    keep.emplace_back(cur, std::min(r.first, bound));
                }
                cur = std::max(cur, r.second);
            }
            if (cur < bound) {
                keep.emplace_back(cur, bound);
            }
        }

        size_t n_keep = 0;
        for (const auto & r : keep) {
            n_keep += (size_t) (r.second - r.first);
        }
        const size_t n_remove = (size_t) bound - n_keep;

        if (n_remove == 0) {
            SLT_WRN_DBG(slot, "da_b: removal ranges cover nothing in [0, %d) after clamping - nothing to remove, staying on seq %d (%s)\n", bound, slot.id, when);
            return;
        }

        // no-benefit skip (plans/continous_work2.md, optional): when the
        // removal is tiny, the switch cost (seq_cp, a second sequence id,
        // the restore copy at release) outweighs the read reduction - stay
        // on the original sequence, output stays vanilla. The caller only
        // moves the mode state machine when da_seq >= 0, so the slot stays
        // GLOBAL. Opt-in via --da-b-min-remove (0 = always switch, so the
        // small-prompt da_b_smoke mechanism checks keep passing).
        if (slot.da_b_min_remove > 0 && n_remove < (size_t) slot.da_b_min_remove) {
            SLT_WRN_DBG(slot, "da_b: only %zu token(s) would be removed (< %d) - skipping the switch, staying on seq %d (%s)\n",
                    n_remove, slot.da_b_min_remove, slot.id, when);
            return;
        }

        // copy each keep range into the fresh sequence (target + draft memory)
        for (const auto & r : keep) {
            slot.mem.seq_cp(slot.id, da_seq, r.first, r.second);
        }

        // switch the slot to the new sequence
        slot.da_seq        = da_seq;
        slot.da_bound      = bound;
        slot.da_keep_count = n_keep;
        // Mark the removal as applied so release() takes the B-restore
        // branch (copy the generated tail back to the original sequence and
        // free da_seq). Without it, a B response that ends mid-FOCUS
        // (max_tokens hit, no </focus>) leaves da_seq's KV cells in place;
        // the next request reuses that sequence id and llama_decode rejects
        // the non-contiguous append ("Invalid input batch").
        slot.da_applied    = true;

        SLT_INF_DBG(slot, "da_b: switched decode to seq %d (%s) - logical read set now %zu of %d prefix token(s) (%zu removed, -%.1f%%)\n",
                da_seq, when, n_keep, bound, n_remove, 100.0 * (double) n_remove / (double) bound);
        for (const auto & r : keep) {
            SLT_INF_DBG(slot, "da_b:   keep [%d, %d) %zu token(s)\n", r.first, r.second, (size_t) (r.second - r.first));
        }
        SLT_INF_DBG(slot, "da_b: original seq %d untouched (%d token(s)) - rollback/reference\n", slot.id, bound);
    }

    // ---------------------------------------------------------------------
    // Declarative Attention (DA) tag state machine
    // ---------------------------------------------------------------------
    // The model emits control tags in its own output; the server enforces the
    // matching attention scope mid-decode:
    //   <focus magic_chunks="N">   enter FOCUS: keep chunk N (1-based) + scaffold
    //   <local>                   enter LOCAL: scaffold + generated only
    //   </focus> / </local>       return to GLOBAL (full attention)
    //   <global> / </global>      (re)enter GLOBAL - no-op when already there
    // where scaffold = every prompt token outside the chunk/filler ranges
    // (system, the question, the injected instruction); the generated tail
    // is always attended.
    //
    // B path (task da_b): the decode switches to a second sequence (da_seq)
    // holding the kept ranges; the original sequence stays intact, so a
    // switch/return is a seq_rm/seq_cp pair on da_seq (reversible).
    // A path: seq_rm holes in the original sequence - monotonic, a chunk
    // removed once is gone for the rest of the request.
    //
    // Consumed tags are erased from generated_text (they never reach the
    // user); a trailing fragment that could still grow into a tag is held
    // back from the output stream by process_token() via da_tag_hold_len().

    // DA tag state machine (P3): process every control tag that has closed
    // since the last call and apply the mode transition to the KV cache.
    // Called from process_token() after each generated token is appended to
    // slot.generated.text. No-op unless the client supplied the chunk layout
    // (task params da_chunks). The removal boundary is the current decode
    // position, so the B switch (or the A removal) stays position-contiguous
    // for the next token.
    void apply_da_tag(server_slot & slot) {
        if (slot.task->params.da_chunks.empty()) {
            return;
        }

        const auto & chunks = slot.task->params.da_chunks;
        const auto & filler = slot.task->params.da_filler;
        const bool   is_b   = slot.task->params.da_b;

        for (;;) {
            da_tag_t tag = scan_da_tag(slot.generated.text, slot.da_tag_scan_pos, slot.da_mode);
            if (tag.type < 0) {
                break;  // no more complete tags
            }

            int32_t n_full   = (int32_t) slot.prompt.n_tokens();
            const bool    in_focus = slot.da_mode == DA_MODE_FOCUS;
            const bool    in_local = slot.da_mode == DA_MODE_LOCAL;
            // Tool-call / parameter state at tag close, computed on the
            // pre-erase generated_text (the tag is still in it here). When a
            // tool-call or parameter is still open at tag close, a recall
            // re-prefilled here would land INSIDE it (the opening tag is in
            // the prompt, only the closing tail would be generated). The
            // in-tool gate below (and the B2 gate in case 0) use these.
            // Tag names are assembled from parts: literal tags do not
            // survive editors / uploads.
            const std::string tc_open  = std::string("<")  + "tool_call>";
            const std::string tc_close = std::string("</") + "tool_call>";
            const std::string pr_open  = std::string("<")  + "parameter=";
            const std::string pr_close = std::string("</") + "parameter>";
            const std::string & g_txt = slot.generated.text;
            const int in_tool  = dbg_count(g_txt, tc_open,  tag.start) > dbg_count(g_txt, tc_close, tag.start) ? 1 : 0;
            const int in_param = dbg_count(g_txt, pr_open,  tag.start) > dbg_count(g_txt, pr_close, tag.start) ? 1 : 0;

            // L1: tag-closed snapshot - the state the refill decision runs
            // on. B1 check: prompt_tokens == seq_pos_max + 1 means the
            // sampled closing token is not in the KV yet (it commits in the
            // next handle_last_sampled_token). in_tool = 1 means the tag
            // fired mid tool call (B2). in_tool/in_param are computed above
            // on the pre-erase generated_text (g_txt).
            {
                const size_t b0 = tag.start > 240 ? tag.start - 240 : 0;

                da_trace("slot=%d task=%d L1 tag closed type=%d nums=%zu n_gen=%d mode=%d seq=%d prompt_tokens=%d seq_pos_max=%d sampled=%d spec_draft=%zu in_tool=%d in_param=%d before='%s' tag='%s'\n",
                         (int) slot.id, (int) slot.task->id,
                         tag.type, tag.keep_nums.size(), (int) slot.stats.n_gen, (int) slot.da_mode, (int) slot.kv_seq(),
                         (int) slot.prompt.n_tokens(),
                         (int) llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.kv_seq()),
                         (int) slot.sampled, slot.spec_draft.size(),
                         in_tool, in_param,
                         dbg_esc(g_txt.substr(b0, tag.start - b0), 260).c_str(),
                         dbg_esc(g_txt.substr(tag.start, tag.end - tag.start)).c_str());
                // T1-f: a DA control tag that closes while a tool-call/parameter
                // is still open means the pending recall would re-prefill inside
                // an open tool-call (a derail risk). The L1 da_trace above already
                // carries in_tool/in_param but is DBG-gated; promote the signal
                // to SRV_INF and warn when either is set (B2 gate would defer).
                if (in_tool || in_param) {
                    SRV_INF("kv_offload L1: slot=%d task=%d control tag closed with open tool-call: in_tool=%d in_param=%d n_gen=%d mode=%d (recall would re-prefill inside an open tool-call)\n",
                            (int) slot.id, (int) slot.task->id, in_tool, in_param,
                            (int) slot.stats.n_gen, (int) slot.da_mode);
                    SLT_WRN_DBG(slot, "kv_offload L1: tag closed while in_tool=%d in_param=%d - B2 gate would defer this recall\n", in_tool, in_param);
                }

                // arm the per-token trace (L4) for the next 24 tokens
                slot.da_trace_left = 24;
            }

            // In-tool gate (before any erase / mode handling): a tag that
            // closes while a tool-call or parameter is still open sits inside
            // the JSON the model is emitting - it is data, not a control
            // signal. Erasing it would corrupt the tool call (the client
            // already has the opening, and the text offsets would shift).
            // Preserve the text, leave the mode unchanged for all types 0~5,
            // and advance the scan past the tag so it is never re-scanned.
            if (in_tool || in_param) {
                SLT_INF(slot, "da_tag: type=%d SKIPPED inside tool call (in_tool=%d in_param=%d) n_gen=%d mode=%d - text preserved, mode unchanged, scan advanced\n",
                        tag.type, in_tool, in_param, (int) slot.stats.n_gen, (int) slot.da_mode);
                slot.da_tag_scan_pos = tag.end;
                continue;
            }

            // consume the tag: erase it from the user-visible text and
            // restart the scan where the tag began (nothing complete can sit
            // before it - the scanner returns the earliest tag)
            const size_t da_erased = tag.end - tag.start;
            slot.generated.text.erase(tag.start, da_erased);
            // The in-progress tag is held back from the output stream by
            // da_tag_hold_len() (the whole <focus magic_chunks="N> prefix,
            // not just the opener), so by the time the tag closes it has
            // normally NOT been sent: n_sent_text == tag.start and the
            // branch below is a no-op. It remains as a safety net for the
            // edge where part of the tag leaked (e.g. a literal tag the
            // model emitted as data) - reconcile the pointer so the send
            // path (substr(n_sent_text)) stays in range: fully sent - shift
            // back by the tag length; partially sent - clamp to the start.
            if (slot.n_sent_text > tag.start) {
                slot.n_sent_text = (slot.n_sent_text >= tag.end)
                        ? slot.n_sent_text - da_erased
                        : tag.start;
            }
            slot.da_tag_scan_pos = tag.start;

            // <local> block: the model emits <local> from learned habit even
            // after the prompt txt is fixed, and the mode transition would
            // still run and pollute the experiment. Ignore it: the tag is
            // already erased (not in the output) and the mode is left
            // unchanged. </local> (type 3) is NOT ignored - no slot is ever
            // in LOCAL state, so its da_set_mode(GLOBAL) is a no-op anyway.
            // Default on; FOCUS_DA_NO_LOCAL=0 disables it for A/B comparison.
            static const bool da_no_local = []() {
                const char * e = getenv("FOCUS_DA_NO_LOCAL");
                return !e || atoi(e) != 0;
            }();
            if (tag.type == 1 && da_no_local) {
                SLT_INF(slot, "da_tag: type=1 (<local>) IGNORED - mode unchanged n_gen=%d mode=%d\n",
                        (int) slot.stats.n_gen, (int) slot.da_mode);
                continue;
            }

            // The first token after a mode transition must follow the
            // single-token trajectory (see the DA tag gate in the draft
            // decision) - the next draft step of this slot is skipped.
            slot.da_skip_draft = true;

            // Phase 0 (g measurement): remember where the first restriction
            // tag (<focus>/<local>) closed. release() combines this with the
            // total n_gen to report g = first_tag / total, the global-step
            // share before the attention restriction kicked in. The non-restriction
            // tags (</focus>/</local>/<global>/</global>, type 2-5) do not start
            // the restricted phase, so they are not recorded here.
            if ((tag.type == 0 || tag.type == 1) && slot.da_first_tag_gen < 0) {
                slot.da_first_tag_gen = (int32_t) slot.stats.n_gen;
            }

            // Applied-tag log: every tag that passed the in-tool gate and now
            // runs the mode transition below (SLT_INF so it is visible in
            // production, not just under FOCUS_SPARSE_DEBUG).
            SLT_INF(slot, "da_tag: type=%d applied (in_tool=%d in_param=%d) n_gen=%d mode=%d seq=%d\n",
                    tag.type, in_tool, in_param, (int) slot.stats.n_gen, (int) slot.da_mode, (int) slot.kv_seq());

            switch (tag.type) {
                case 0: { // <focus magic_chunks="N"> (or "1,3") : enter/switch FOCUS
                    // Resolve the emitted chunk numbers to layout indices:
                    // the hook numbers chunks monotonically per session, so
                    // the current block may start at da_chunk_base > 1.
                    const int32_t base     = slot.task->params.da_chunk_base;
                    const int32_t n_chunks = (int32_t) chunks.size();
                    const auto & offloaded = slot.task->params.da_offloaded;
                    std::vector<size_t> keep_idx;
                    bool did_refill = false;
                    for (int32_t n : tag.keep_nums) {
                        // kv-offload (get-on-focus): n points to an evicted (offloaded)
                        // chunk (number >= base + n_chunks). Fetch its text from the
                        // FocusMemory store and re-prefill it at the tail so the model
                        // can read it. No KV restriction is applied (the chunk is not in
                        // the active layout) - the model reads the re-prefilled content
                        // in the current mode. Fail-open on any store error.
                        if (n >= base + n_chunks) {
                            // measure-only (Phase 0): observe the tag position without
                            // mutating KV - skip the store GET and re-prefill so the
                            // output stays vanilla (mirrors the da_rm/da_b guards).
                            if (slot.da_measure_only) {
                                SLT_INF(slot, "kv_offload: get-on-focus chunk %d SKIPPED (measure-only) - no store GET / KV refill; output stays vanilla\n", n);
                                continue;
                            }
                            bool found = false;
                            for (const auto & seg : offloaded) {
                                if (seg.chunk_id != n) continue;
                                found = true;
                                // Diagnostic: the model is asking for an evicted
                                // chunk - log the fetch so get-on-focus is traceable.
                                SLT_INF(slot, "kv_offload: get-on-focus chunk %d (key=%s) - fetching from FocusMemory\n",
                                        n, seg.key.c_str());
                                std::string text;
                                // The GET can block up to the read timeout (5s) on a slow or
                                // unreachable store. Yield to the queue so metrics/cancel tasks
                                // are still handled while we wait (fail-open either way).
                                bool got = false;
                                queue_tasks.yield_to_queue([&]() {
                                    got = kv_offload_get(slot.task->params.kv_offload_host,
                                                         slot.task->params.kv_offload_token,
                                                         slot.task->params.kv_offload_session,
                                                         seg.key, text);
                                });
                                if (got && slot.task->params.kv_offload_vocab) {
                                    SLT_INF(slot, "kv_offload: GET ok chunk %d - %zu chars, re-prefilling\n",
                                            n, text.size());
                                    const llama_tokens toks =
                                            common_tokenize(slot.task->params.kv_offload_vocab, text, true, true);
                                    // B1: queue the refill instead of decoding it here -
                                    // the sampled closing token is not in the KV yet
                                    // (it commits in the next handle_last_sampled_token),
                                    // so a tail decode now would land BEFORE it and split
                                    // the model's own tag. post_decode() applies the
                                    // queue after the commit and samples the next token
                                    // from the refill's last logits row.
                                    slot.da_pending_refill.insert(slot.da_pending_refill.end(),
                                                                  toks.begin(), toks.end());
                                    did_refill = true;
                                } else {
                                    SLT_WRN(slot, "kv_offload: GET failed for offloaded chunk %d - fail-open\n", n);
                                }
                                break;
                            }
                            if (!found) {
                                SLT_WRN_DBG(slot, "da_tag: magic_chunks number %d is not an active or offloaded chunk - ignored\n", n);
                            }
                            continue;
                        }
                        if (n < base || n - base >= n_chunks) {
                            SLT_WRN_DBG(slot, "da_tag: magic_chunks number %d out of range (%d..%d) - ignored\n",
                                    n, base, base + n_chunks - 1);
                            continue;
                        }
                        const size_t idx = (size_t) (n - base);
                        if (std::find(keep_idx.begin(), keep_idx.end(), idx) == keep_idx.end()) {
                            keep_idx.push_back(idx);
                        }
                    }
                    // kv-offload (Option B, --kv-offload-holes): a focused real chunk may
                    // overlap an applied KV hole (its evicted message's KV was seq_rm'd from
                    // the main sequence), so the B-switch seq_cp would copy an empty range and
                    // the model could not read the evicted content. Re-prefill the holed
                    // overlap at the tail (Option B keeps the text in the prompt, so no store
                    // GET is needed) so the model can read it; the B switch then copies the
                    // refilled tail onto da_seq, exactly like the Option C get-on-focus path.
                    // Fail-open. measure-only (Phase 0) skips the re-prefill so the KV is
                    // not mutated and the output stays vanilla (mirrors the da_rm/da_b guards).
                    if (slot.kv_holes_active && !keep_idx.empty() && !slot.da_measure_only) {
                        SLT_INF(slot, "kv_offload: checking %zu keep chunk(s) against %zu hole range(s)\n",
                            keep_idx.size(), slot.kv_hole_ranges.size());
                        const int32_t n_prompt = (int32_t) slot.prompt.tokens.size();
                        std::string keep_nums, hole_str;
                        for (size_t n : keep_idx) {
                            if (!keep_nums.empty()) keep_nums += ",";
                            keep_nums += std::to_string((int) n);
                        }
                        for (const auto & hole : slot.kv_hole_ranges) {
                            if (!hole_str.empty()) hole_str += " ";
                            hole_str += "[" + std::to_string(hole.first) + "," + std::to_string(hole.second) + ")";
                        }
                        SLT_INF(slot, "kv_offload: keep = {%s}, holes = {%s}, prompt tokens = %d\n",
                            keep_nums.c_str(), hole_str.c_str(), n_prompt);
                        for (size_t n : keep_idx) {
                            const auto & cr = chunks[n];
                            SLT_INF(slot, "kv_offload:   chunk %zu range [%d,%d)\n", n, cr.first, cr.second);
                            for (const auto & hole : slot.kv_hole_ranges) {
                                const int32_t olo = std::max(cr.first, (int32_t) hole.first);
                                int32_t       ohi = std::min(cr.second, (int32_t) hole.second);
                                if (ohi <= olo) {
                                    SLT_INF(slot, "kv_offload:   chunk %zu x hole [%d,%d) - no overlap, skip\n",
                                        n, hole.first, hole.second);
                                    continue;
                                }
                                if (ohi > n_prompt) {
                                    SLT_INF(slot, "kv_offload:   chunk %zu x hole [%d,%d) - ohi clamped to prompt (%d)\n",
                                        n, hole.first, hole.second, n_prompt);
                                    ohi = n_prompt;
                                }
                                if (ohi <= olo) {
                                    SLT_INF(slot, "kv_offload:   chunk %zu x hole [%d,%d) - no overlap after clamp, skip\n",
                                        n, hole.first, hole.second);
                                    continue;
                                }
                                llama_tokens toks;
                                for (int32_t p = olo; p < ohi; ++p) {
                                    toks.push_back(slot.prompt.tokens[p]);
                                }
                                SLT_INF(slot, "kv_offload: focus on holed chunk %zu - re-prefilling %d token(s) at [%d,%d)\n",
                                        n, (int) toks.size(), olo, ohi);
                                // B1: queue (see the get-on-focus path above) -
                                // post_decode() decodes it after the closing
                                // token is committed.
                                slot.da_pending_refill.insert(slot.da_pending_refill.end(),
                                                              toks.begin(), toks.end());
                                did_refill = true;
                            }
                        }
                    }
                    // B1: the queued refill has NOT grown the prompt yet (it
                    // decodes in the next post_decode, after the closing token
                    // commits), so the bound stays at the current decode
                    // position. The recalled block then lands on the tail of
                    // kv_seq() (da_seq after a B switch) BEHIND the bound,
                    // where the B read set already reaches (tokens after the
                    // boundary are read). Kept as a defensive recompute in
                    // case a future path refills synchronously.
                    if (did_refill) {
                        n_full = (int32_t) slot.prompt.n_tokens();
                    }
                    if (keep_idx.empty()) {
                        if (did_refill) {
                            SLT_INF(slot, "%s", "kv_offload: focus on offloaded chunk(s) - re-prefilled at tail, model reads in current mode");
                        } else {
                            SLT_WRN_DBG(slot, "%s", "da_tag: no valid magic_chunks number - tag consumed, mode unchanged\n");
                        }
                        break;
                    }
                    std::sort(keep_idx.begin(), keep_idx.end());

                    const auto is_kept = [&keep_idx](size_t n) {
                        return std::find(keep_idx.begin(), keep_idx.end(), n) != keep_idx.end();
                    };
                    const auto is_old = [&slot](size_t n) {
                        return std::find(slot.da_keep_chunks.begin(), slot.da_keep_chunks.end(), (int32_t) n) != slot.da_keep_chunks.end();
                    };
                    std::string keep_str;
                    for (size_t n : keep_idx) {
                        if (!keep_str.empty()) {
                            keep_str += ",";
                        }
                        keep_str += std::to_string((int) (n + base));
                    }

                    if (is_b) {
                        if (slot.da_seq < 0) {
                            // GLOBAL -> FOCUS: fresh da_seq = scaffold + keep + tail
                            std::vector<std::pair<int32_t, int32_t>> ranges;
                            for (size_t n = 0; n < chunks.size(); n++) {
                                if (!is_kept(n)) {
                                    ranges.push_back(chunks[n]);
                                }
                            }
                            if (filler.first >= 0) {
                                ranges.push_back(filler);
                            }
                            SLT_INF_DBG(slot, "da_tag: <focus magic_chunks=\"%s\"> closed at generated token %d - B switch, removing %zu range(s), bound=%d\n",
                                    keep_str.c_str(), (int) slot.stats.n_gen, ranges.size(), n_full);
                            apply_da_b(slot, ranges, n_full, "at tag close");
                            if (slot.da_seq >= 0) {
                                slot.da_set_mode(DA_MODE_FOCUS);
                                slot.da_keep_chunks.assign(keep_idx.begin(), keep_idx.end());
                            }
                        } else if (in_focus) {
                            // FOCUS -> FOCUS: drop the old keep(s) that left
                            // the set, copy in the new keep(s)
                            for (int32_t n : slot.da_keep_chunks) {
                                if (!is_kept((size_t) n)) {
                                    slot.mem.seq_rm(slot.da_seq, chunks[n].first, chunks[n].second);
                                    slot.da_keep_count -= (size_t) (chunks[n].second - chunks[n].first);
                                }
                            }
                            for (size_t n : keep_idx) {
                                if (!is_old(n)) {
                                    slot.mem.seq_cp(slot.id, slot.da_seq, chunks[n].first, chunks[n].second);
                                    slot.da_keep_count += (size_t) (chunks[n].second - chunks[n].first);
                                }
                            }
                            slot.da_keep_chunks.assign(keep_idx.begin(), keep_idx.end());
                            SLT_INF_DBG(slot, "da_tag: FOCUS switch - now keeping chunk(s) %s\n", keep_str.c_str());
                        } else {
                            // LOCAL -> FOCUS: copy the keep(s) into da_seq
                            for (size_t n : keep_idx) {
                                slot.mem.seq_cp(slot.id, slot.da_seq, chunks[n].first, chunks[n].second);
                                slot.da_keep_count += (size_t) (chunks[n].second - chunks[n].first);
                            }
                            slot.da_keep_chunks.assign(keep_idx.begin(), keep_idx.end());
                            slot.da_set_mode(DA_MODE_FOCUS);
                            SLT_INF_DBG(slot, "da_tag: LOCAL -> FOCUS - keeping chunk(s) %s\n", keep_str.c_str());
                        }
                    } else {
                        // A path: monotonic. The chunks still present in the
                        // sequence are all of them before the first removal.
                        std::vector<size_t> present;
                        if (!slot.da_applied && slot.da_mode == DA_MODE_GLOBAL) {
                            for (size_t n = 0; n < chunks.size(); n++) {
                                present.push_back(n);
                            }
                        } else {
                            for (int32_t n : slot.da_keep_chunks) {
                                present.push_back((size_t) n);
                            }
                        }
                        bool all_present = true;
                        for (size_t n : keep_idx) {
                            if (std::find(present.begin(), present.end(), n) == present.end()) {
                                SLT_WRN_DBG(slot, "da_tag: A path is monotonic - chunk %d was already removed, cannot switch keep (staying %s)\n",
                                        (int) (n + base), in_focus ? "FOCUS" : (in_local ? "LOCAL" : "GLOBAL"));
                                all_present = false;
                                break;
                            }
                        }
                        if (all_present) {
                            std::vector<std::pair<int32_t, int32_t>> ranges;
                            for (size_t n : present) {
                                if (!is_kept(n)) {
                                    ranges.push_back(chunks[n]);
                                }
                            }
                            if (!slot.da_applied && filler.first >= 0) {
                                ranges.push_back(filler);
                            }
                            SLT_INF_DBG(slot, "da_tag: <focus magic_chunks=\"%s\"> closed at generated token %d - A removal, %zu range(s), bound=%d\n",
                                    keep_str.c_str(), (int) slot.stats.n_gen, ranges.size(), n_full);
                            apply_da_rm(slot, ranges, n_full, "at tag close");
                            slot.da_keep_chunks.assign(keep_idx.begin(), keep_idx.end());
                            slot.da_set_mode(DA_MODE_FOCUS);
                        }
                    }
                    // L3: are the kept chunks actually alive on da_seq? The
                    // 4-turn loop signature is holed == len with did_refill = 0
                    // - the model cannot read a chunk whose KV was evicted and
                    // never recalled.
                    if (is_b && slot.da_seq >= 0) {
                        auto * mem = llama_get_memory(ctx_tgt);
                        for (size_t n : keep_idx) {
                            int32_t holed = 0;
                            for (const auto & h : slot.kv_hole_ranges) {
                                holed += std::max(0, std::min(chunks[n].second, (int32_t) h.second) - std::max(chunks[n].first, (int32_t) h.first));
                            }
                            if (holed > chunks[n].second - chunks[n].first) {
                                SLT_WRN_DBG(slot, "da: L3 holed=%d > len=%d for keep chunk %d [%d,%d) - ledger spans exceed the chunk (stale/overlapping holes)\n",
                                        holed, chunks[n].second - chunks[n].first, (int) (n + base), chunks[n].first, chunks[n].second);
                            }
                            da_trace("slot=%d task=%d L3 B keep chunk=%d [%d,%d) len=%d holed=%d did_refill=%d da_seq=[%d,%d] bound=%d n_full=%d\n",
                                     (int) slot.id, (int) slot.task->id,
                                     (int) (n + base), chunks[n].first, chunks[n].second, chunks[n].second - chunks[n].first, holed, (int) did_refill,
                                     (int) llama_memory_seq_pos_min(mem, slot.da_seq), (int) llama_memory_seq_pos_max(mem, slot.da_seq),
                                     (int) slot.da_bound, (int) n_full);
                        }
                    }
                    break;
                }
                case 1: { // <local> : enter LOCAL
                    if (is_b) {
                        if (slot.da_seq < 0) {
                            // GLOBAL -> LOCAL: fresh da_seq = scaffold + tail
                            std::vector<std::pair<int32_t, int32_t>> ranges(chunks.begin(), chunks.end());
                            if (filler.first >= 0) {
                                ranges.push_back(filler);
                            }
                            SLT_INF_DBG(slot, "da_tag: <local> closed at generated token %d - B switch, removing %zu range(s), bound=%d\n",
                                    (int) slot.stats.n_gen, ranges.size(), n_full);
                            apply_da_b(slot, ranges, n_full, "at tag close");
                            if (slot.da_seq >= 0) {
                                slot.da_set_mode(DA_MODE_LOCAL);
                                slot.da_keep_chunks.clear();
                            }
                        } else if (in_focus) {
                            // FOCUS -> LOCAL: drop the kept chunk from da_seq
                            for (int32_t n : slot.da_keep_chunks) {
                                slot.mem.seq_rm(slot.da_seq, chunks[n].first, chunks[n].second);
                                slot.da_keep_count -= (size_t) (chunks[n].second - chunks[n].first);
                            }
                            slot.da_keep_chunks.clear();
                            slot.da_set_mode(DA_MODE_LOCAL);
                            SLT_INF_DBG(slot, "%s", "da_tag: FOCUS -> LOCAL - keep chunk(s) dropped\n");
                        }
                        // LOCAL -> LOCAL: no-op
                    } else {
                        // A path: remove every chunk still present
                        std::vector<size_t> present;
                        if (!slot.da_applied && slot.da_mode == DA_MODE_GLOBAL) {
                            for (size_t n = 0; n < chunks.size(); n++) {
                                present.push_back(n);
                            }
                        } else {
                            for (int32_t n : slot.da_keep_chunks) {
                                present.push_back((size_t) n);
                            }
                        }
                        std::vector<std::pair<int32_t, int32_t>> ranges;
                        for (size_t n : present) {
                            ranges.push_back(chunks[n]);
                        }
                        if (!slot.da_applied && filler.first >= 0) {
                            ranges.push_back(filler);
                        }
                        SLT_INF_DBG(slot, "da_tag: <local> closed at generated token %d - A removal, %zu range(s), bound=%d\n",
                                (int) slot.stats.n_gen, ranges.size(), n_full);
                        apply_da_rm(slot, ranges, n_full, "at tag close");
                        slot.da_keep_chunks.clear();
                        slot.da_set_mode(DA_MODE_LOCAL);
                    }
                    break;
                }
                case 2:
                case 3:
                case 4:
                case 5: { // </focus> / </local> / <global> / </global> : return to GLOBAL
                    if (in_local && tag.type == 2) {
                        SLT_WRN_DBG(slot, "%s", "da_tag: </focus> while in LOCAL - treating as </local>\n");
                    }
                    if (slot.da_seq >= 0) {
                        // B: copy the generated tail back to the original
                        // sequence and free da_seq - the slot decodes the full
                        // context again without a re-prefill
                        if (n_full > slot.da_bound) {
                            slot.mem.seq_cp(slot.da_seq, slot.id, slot.da_bound, n_full);
                        }
                        slot.mem.seq_rm(slot.da_seq, -1, -1);
                        SLT_INF_DBG(slot, "da_tag: returned to GLOBAL - %d generated token(s) copied back to seq %d, da_seq %d freed\n",
                                std::max(0, n_full - slot.da_bound), (int) slot.id, (int) slot.da_seq);
                        slot.da_seq        = -1;
                        slot.da_bound      = -1;
                        slot.da_keep_count = 0;
                        slot.da_keep_chunks.clear();
                    } else if (slot.da_applied) {
                        SLT_WRN_DBG(slot, "da_tag: A path removals are irreversible - attention stays restricted (%zu chunk(s) kept), mode nominal GLOBAL\n",
                                slot.da_keep_chunks.size());
                    }
                    // !da_applied: nothing was removed - a clean return
                    slot.da_set_mode(DA_MODE_GLOBAL);
                    break;
                }
            }
        }
    }

    void post_decode(int32_t n_batch_tokens, int32_t off) {
        // for checking if a given batch index is inside the current sub-batch
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        // A deferred refill runs llama_decode inside the slot loop below and
        // overwrites the ctx output buffer. Set when one ran in this pass so
        // the stale-logits probe can name the cause in its error.
        bool refill_ran_this_iter = false;

        // Shared sampling path (probe + sample + accept + stats + process_token)
        // for the main (non-spec) decode row. Extracted so the deferred-refill
        // pass (below) can sample a refilling slot from the refill's last row
        // without duplicating the logic. Returns false if the slot was aborted
        // or released (the caller must then return from its iterate).
        auto sample_main_path = [&](server_slot & slot, int32_t tok_idx) -> bool {
            // Defensive line (2026-10-09 12:43 crash): a decode that ran after
            // this slot's main-batch row was computed overwrote the ctx output
            // buffer, so the row no longer resolves and common_sampler's
            // GGML_ASSERT would abort the whole server. Probe first and abort
            // this slot with an error instead. The step is unrecoverable:
            // re-decoding the token is unsafe in the cell-based KV cache
            // (double-counted K/V).
            if (llama_get_logits_ith(slot.ctx_tgt, tok_idx) == nullptr) {
                SLT_ERR(slot, "kv_offload: stale logits row at sampling (a decode ran after this slot's batch row)%s - aborting slot\n",
                        refill_ran_this_iter ? " (a deferred refill ran this iteration)" : "");
                send_error(slot, "kv_offload: stale logits row - the slot was aborted (the in-flight step was dropped)");
                slot.da_pending_refill.clear();
                slot.i_batch = -1;
                slot.da_refill_out_idx = -1;
                slot.release();
                slot.prompt_clear();
                return false;
            }

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;
            slot.da_refill_out_idx = -1;  // consumed

            // L4: top-5 probabilities of the token just sampled - the B1
            // check reads the continuation right after [End of Recalled Chunk]
            if (slot.da_trace_left > 0) {
                std::string s;
                for (const auto & c : get_token_probabilities(slot.ctx_tgt, tok_idx, 5)) {
                    s += string_format("%d:'%s':%.3f ", (int) c.id, dbg_esc(common_token_to_piece(slot.ctx_tgt, c.id, true), 20).c_str(), c.p);
                }
                da_trace("slot=%d task=%d L4 top5 sampled=%d [%s]\n",
                         (int) slot.id, (int) slot.task->id, (int) id, s.c_str());
            }

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;
            slot.da_count_generated_token();

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();
                return false;
            }

            slot.print_timings_tg();
            return true;
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_DECISION) {
                    send_decision(slot, batch.view, slot.i_batch - off);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                // kv-offload-holes: one-line summary of the prefill's KV peak
                // (the progressive cut's success criterion:
                // peak <= buffer - 16384 headroom).
                if (!slot.task->params.kv_hole_pending.empty()) {
                    SLT_INF_DBG(slot, "kv-offload-holes: prefill done - peak resident %d (buffer %d, prompt %d, applied holes %zu)\n",
                            slot.kv_hole_peak_resident, params_base.kv_cache_size,
                            (int) slot.prompt.n_tokens(), slot.kv_hole_ranges.size());
                }

                // Declarative Attention: apply the removals that were not applied
                // mid-prefill (da_rm_at < 0, prompts containing media, or the
                // boundary coincides with the end of the prompt). The first generated token
                // is sampled from the already-computed prompt logits; subsequent
                // decode steps see the reduced KV.
                if (slot.da_rm_pending) {
                    if (slot.task->params.da_rm_at >= 0) {
                        const size_t n_total = slot.task->tokens.size();
                        const size_t n_text  = slot.task->tokens.get_text_tokens().size();
                        SLT_INF_DBG(slot, "da_rm: mid-prefill boundary never applied (da_rm_at=%d has_mtmd=%d n_tokens=%d n_text=%zu n_media=%zu) - applying after prefill\n",
                                slot.task->params.da_rm_at, (int) slot.task->tokens.has_mtmd, (int) slot.prompt.n_tokens(),
                                n_text, n_total - n_text);
                    }
                    if (slot.task->params.da_b) {
                        apply_da_b(slot, slot.task->params.da_rm, da_static_bound(slot), "after prefill");
                    } else {
                        apply_da_rm(slot, slot.task->params.da_rm, da_static_bound(slot), "after prefill");
                    }
                    slot.da_rm_pending = false;
                }

                // A 2-stream (B) slot that has already switched decodes on its
                // second sequence (da_seq) - spec was started before the
                // switch and follows it via kv_seq().
                // DA tag mode (P6): spec starts before the tag fires. When a
                // control tag closes mid-decode, the accept loop cuts the
                // draft batch at the closing token and the mode transition
                // runs; decoding (and the draft) then continues on kv_seq().
                if (slot.can_speculate() && slot.da_seq < 0) {
                    common_speculative_begin(spec.get(), slot.kv_seq(), slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // B1 (2026-10-09 12:43 crash root cause): a slot with a queued
            // refill (apply_da_tag) is DEFERRED to the refill pass below. Its
            // sampling must run AFTER the refill (it samples from the refill's
            // last row), and the refill must run AFTER every other slot has
            // sampled - the refill's llama_decode overwrites the ctx output
            // buffer, and running it inside this loop stale the other slots'
            // main-batch rows (the GGML_ASSERT abort). Skip it here; the refill
            // pass (after the spec iterate) runs the refill and samples it.
            if (!slot.da_pending_refill.empty()) {
                return;
            }

            // sample from the main batch's row (no refill ran for this slot)
            if (!sample_main_path(slot, slot.i_batch - off)) {
                return;
            }
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // batch indices of the draft tokens, used to get the token probabilities below
            auto spec_i_batch = std::move(slot.spec_i_batch);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));

                // Defensive line (2026-10-09 12:43 crash): the backtrace died
                // in sample_and_accept_n on idx 1 - the first draft row of the
                // slot that did NOT refill. A deferred refill in another slot's
                // iterate overwrote the ctx output buffer, so every main-batch
                // row (the whole spec_i_batch) is stale and common_sampler's
                // GGML_ASSERT would abort the whole server. Probe before
                // sampling and abort this slot with an error instead. The step
                // is unrecoverable: re-decoding the token is unsafe in the
                // cell-based KV cache (double-counted K/V).
                int stale_idx = -1;
                for (int32_t idx : spec_i_batch) {
                    if (llama_get_logits_ith(slot.ctx_tgt, idx) == nullptr) {
                        stale_idx = idx;
                        break;
                    }
                }
                if (stale_idx >= 0) {
                    SLT_ERR(slot, "kv_offload: stale logits row at spec sampling (idx=%d, a decode ran after this slot's batch row)%s - aborting slot\n",
                            stale_idx, refill_ran_this_iter ? " (a deferred refill ran this iteration)" : "");
                    send_error(slot, "kv_offload: stale logits row - the slot was aborted (the in-flight step was dropped)");
                    slot.da_pending_refill.clear();
                    slot.i_batch = -1;
                    slot.da_refill_out_idx = -1;
                    slot.spec_i_batch.clear();
                    slot.release();
                    slot.prompt_clear();
                    return;
                }
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // drafters that fill no distribution fall back here
                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                }

                GGML_ASSERT(accepted.size() >= 1);

                // DA tag batch cut (P6): if a control tag closes inside the
                // accepted batch, discard everything after the closing token
                // - the mode transition (apply_da_tag in process_token) must
                // run before the following tokens are committed to the
                // output stream and the KV cache. The cut point is the token
                // whose text contains the tag's closing '>'.
                if (!slot.task->params.da_chunks.empty()) {
                    const size_t n_acc_full = accepted.size();
                    const size_t gen_len    = slot.generated.text.size();
                    std::string preview;
                    for (llama_token tok : accepted) {
                        preview += common_token_to_piece(slot.ctx_tgt, tok, accept_special_token(slot, tok));
                    }
                    // a tag may start in the held-back tail of generated_text
                    // (process_token holds back partial tag openers) and close
                    // inside this batch; only the closing position matters
                    const da_tag_t tag = scan_da_tag(slot.generated.text + preview, slot.da_tag_scan_pos, slot.da_mode);
                    if (tag.type >= 0 && tag.end > gen_len) {
                        const size_t cut = tag.end - gen_len;  // offset into the preview
                        size_t off = 0;
                        for (size_t i = 0; i < accepted.size(); i++) {
                            const std::string piece = common_token_to_piece(slot.ctx_tgt, accepted[i], accept_special_token(slot, accepted[i]));
                            if (off + piece.size() >= cut) {
                                accepted.resize(i + 1);
                                SLT_INF_DBG(slot, "da_tag: batch cut - %zu of %zu accepted token(s) discarded (tag closed mid-batch)\n",
                                        n_acc_full - (i + 1), n_acc_full);
                                break;
                            }
                            off += piece.size();
                        }
                    }
                }

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                // the recurrent snapshots are only guaranteed for batches of up to n_rs_seq + 1 tokens, so a longer draft uses its checkpoint
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.kv_seq(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                        if (slot.ctx_dft) {
                            GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.kv_seq(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        slot.mem.seq_rm(slot.kv_seq(), ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.kv_seq(), accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.kv_seq(), slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // post_sampling_probs is not supported with speculative decoding
                if (slot.task->params.sampling.n_probs > 0 && !slot.task->params.post_sampling_probs) {
                    populate_token_probs(slot, result, false, params_base.special, spec_i_batch[i]);
                }

                slot.stats.n_gen += 1;
                slot.da_count_generated_token();

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });

        // Refill pass (B1, 2026-10-09 12:43 crash root cause): run the deferred
        // refills AFTER every slot (main + spec) has sampled from the main
        // batch's rows. The refill's llama_decode overwrites the ctx output
        // buffer, so running it inside the slot loop above stale the other
        // slots' main-batch rows (the GGML_ASSERT abort). Now all main-batch
        // rows are consumed before any refill runs, so the other slots are safe.
        // Each refilling slot samples from its own refill's last row (or the
        // main batch's row if the refill was rejected by the free-cell gate).
        // ctx_out_clobbered tracks whether a decode loop already ran in this
        // pass: once any slot's refill decoded, the ctx output buffer no longer
        // holds the main batch's rows, so a LATER slot's gate rejection can no
        // longer fail open (its main-batch row is stale) and must abort.
        bool ctx_out_clobbered = false;
        iterate(slots, [&](server_slot & slot) {
            if (slot.da_pending_refill.empty()) {
                return;
            }
            // A spec slot's refill is NOT applied here (mirrors the original
            // behavior: the spec check in the first iterate returns before the
            // refill, so a spec slot's refill was silently dropped). Skipping
            // it here also avoids a double sample - the slot already sampled in
            // the spec iterate above.
            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                slot.da_pending_refill.clear();
                return;
            }
            // The slot was deferred in the first iterate; it must still be
            // generating. If it was released or moved to another state, drop
            // the recall.
            if (slot.state != SLOT_STATE_GENERATING) {
                slot.da_pending_refill.clear();
                return;
            }
            // Not in this sub-batch's view: post_decode runs per sub-batch and
            // the first iterate skips out-of-view slots, so the recall must be
            // kept for the sub-batch that contains it. Do NOT clear it here.
            if (!is_inside_view(slot.i_batch)) {
                return;
            }
            int32_t last_idx = -1;
            bool safe_failopen = false;
            const bool ok = kv_offload_refill(slot.ctx_tgt, slot, slot.da_pending_refill, &last_idx,
                                              [this] { return try_clear_idle_slots(); }, &safe_failopen);
            if (!safe_failopen) {
                // the decode loop ran (success or mid-decode failure) - the ctx
                // output buffer no longer holds the main batch's rows
                ctx_out_clobbered = true;
            }
            if (ok) {
                slot.da_refill_out_idx = last_idx;
                refill_ran_this_iter = true;
            } else if (safe_failopen && !ctx_out_clobbered) {
                // Rejected BEFORE any decode ran (the free-cell hard gate) AND
                // no earlier refill in this pass decoded - the ctx output buffer
                // still holds the MAIN batch's rows, so the closing token's row
                // (slot.i_batch - off) is valid. Fail-open: drop the recall and
                // sample that row; the slot continues without the recalled chunk
                // (keeps its cache).
                SLT_INF(slot, "%s", "kv_offload: refill skipped (no free cells) - continuing without the recalled chunk\n");
                // da_refill_out_idx stays -1 -> tok_idx = slot.i_batch - off.
            } else {
                // Either a failed refill (mid-decode) OR a gate rejection after
                // an earlier slot's refill already decoded in this pass: the ctx
                // logits buffer no longer holds the main batch's rows, so the
                // closing token's row (slot.i_batch - off) is stale. Re-decoding
                // the closing token is NOT safe in the cell-based KV cache
                // (double-counted K/V). Abort the slot instead of sampling a
                // stale row.
                if (safe_failopen) {
                    SLT_ERR(slot, "%s", "kv_offload: refill rejected (no free cells) after another slot's refill already decoded - aborting slot (the main-batch logits row is stale)\n");
                    send_error(slot, "kv_offload: refill rejected after the ctx output was clobbered - the slot was aborted (the recalled block was dropped to avoid sampling a stale logits row)");
                } else {
                    SLT_ERR(slot, "%s", "kv_offload: deferred refill failed - aborting slot (dropping the recalled block to avoid sampling a stale logits row)\n");
                    send_error(slot, "kv_offload: deferred refill failed - the slot was aborted (the recalled block was dropped to avoid sampling a stale logits row)");
                }
                slot.i_batch = -1;
                slot.release();
                slot.prompt_clear();
                slot.da_pending_refill.clear();
                return;
            }
            slot.da_pending_refill.clear();
            // sample from the refill's last row (or the main batch's row if the
            // refill was rejected by the free-cell gate)
            const int tok_idx = slot.da_refill_out_idx >= 0 ? slot.da_refill_out_idx : slot.i_batch - off;
            if (!sample_main_path(slot, tok_idx)) {
                return;
            }
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    std::vector<std::string> tts_voices;
    if (impl->mctx) {
        for (int32_t i = 0; i < mtmd_gen_audio_get_info(impl->mctx).n_voices; i++) {
            tts_voices.push_back(mtmd_gen_audio_voice_name(impl->mctx, i));
        }
    }

    return server_context_meta {
        /* build_info              */ std::string(llama_build_info()),
        /* model_name              */ impl->model_name,
        /* model_aliases           */ impl->model_aliases,
        /* model_tags              */ impl->model_tags,
        /* model_path              */ impl->params_base.model.path,
        /* model_output_modalities */ server_model_output_modalities(impl->model_tgt, impl->has_cap_chat(), impl->has_cap_tts()),
        /* has_mtmd                */ impl->mctx != nullptr,
        /* has_inp_image           */ impl->chat_params.allow_image,
        /* has_inp_audio           */ impl->chat_params.allow_audio,
        /* has_inp_video           */ impl->chat_params.allow_video,
        /* has_cap_chat            */ impl->has_cap_chat(),
        /* has_cap_tts             */ impl->has_cap_tts(),
        /* has_audio_stream        */ impl->mtmd_cap.audio_stream,
        /* tts_voices              */ std::move(tts_voices),
        /* json_ui_settings        */ impl->json_ui_settings,
        /* slot_n_ctx              */ impl->n_ctx_slot(),
        /* pooling_type            */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params             */ impl->chat_params,
        /* chat_template_caps      */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str           */ bos_token_str,
        /* eos_token_str           */ eos_token_str,
        /* fim_pre_token           */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token           */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token           */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token           */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token           */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token           */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog          */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type        */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens    */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train       */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp        */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params          */ llama_model_n_params(impl->model_tgt),
        /* model_size              */ llama_model_size(impl->model_tgt),
        /* model_ftype             */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_queue * queue_awake = nullptr; // set while this response keeps the server awake
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
            queue_awake = &queue_tasks;
        }
    }
    ~server_res_generator() {
        if (queue_awake) {
            queue_awake->release_inflight();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

void server_context::set_video_decoder(const mtmd_helper_video_decoder * decoder) {
    impl->init_opt.video_params.decoder = decoder;
}

// Roll back a partial kv_offload_refill: remove [n_full, -1) (the tail from the
// pre-refill length onward) on kv_seq(), restoring KV<->prompt.tokens
// consistency. Runs on every failure (any error class), not only the first
// sub-batch: a later sub-batch failure leaves the earlier ones' tokens in the
// KV and the rollback must remove them all. If the removal is a no-op (e.g. a
// recurrent-state hybrid cannot be partially rolled back), the slot's KV stays
// inconsistent with prompt.tokens -> mark it poisoned so release()
// prompt_clear()s it instead of keeping the prompt cache. Harmless when
// nothing was written (the range contains no cells).
static void kv_offload_refill_rollback(server_slot & slot, int32_t n_full, int32_t off) {
    if (!slot.mem.seq_rm_checked(slot.kv_seq(), n_full, -1)) {
        slot.refill_poisoned = true;
        SRV_WRN("kv_offload: refill rollback no-op for [%d, -1) in seq %d after %d token(s) - slot poisoned, prompt cache will be cleared on release\n",
                n_full, (int) slot.kv_seq(), off);
    }
}

// Re-prefill an offloaded chunk at the tail of the slot's sequence
// (get-on-focus). Appends the tokens at positions [n_full, n_full+len) on
// kv_seq() and runs a prefill decode so the model can attend to the chunk.
// Returns true on success; false on any error (fail-open: the caller
// proceeds without the chunk). The slot's prompt tokens are advanced so
// pos_next() reflects the new length. HIGH RISK: mid-decode prefill - must be
// validated by the offload probe before trusting.
static bool kv_offload_refill(llama_context * ctx, server_slot & slot, const llama_tokens & toks, int32_t * out_last_idx,
                              const std::function<bool()> & on_retry_idle, bool * out_safe_failopen) {
    // No decode has run yet - a failure at this point leaves the ctx output
    // buffer holding the main batch's rows, so the caller can fail-open.
    // Flipped to false once the decode loop starts (see below).
    if (out_safe_failopen) *out_safe_failopen = true;
    const int32_t n_raw = (int32_t) toks.size();
    if (n_raw <= 0 || ctx == nullptr) return false;

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    std::string raw_text = sanitize_recalled_text(common_detokenize(ctx, toks),
                                                recall_tag_names(vocab));

    std::string sanitized_wrapped_text =
        "\n[Recalled Chunk Start]\n" + raw_text + "\n[End of Recalled Chunk]\n";

    //SRV_INF("sanitize_recalled_text: %s\n", sanitized_wrapped_text.c_str());

    llama_tokens wrapped_toks = common_tokenize(ctx, sanitized_wrapped_text, false, true);

    const int32_t n = (int32_t) wrapped_toks.size();
    const int32_t n_full = (int32_t) slot.prompt.n_tokens();
    const llama_seq_id seq = slot.kv_seq();
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);

    if (n_batch <= 0) return false;

    // L2: the refill plan - where it lands (n_full / pos_next / seq_pos_max)
    // and whether sanitize leaked control or EOG tokens into the wrapped
    // block (both must be 0; ctrl > 0 = a tag sanitize missed, eog > 0 = the
    // model may stop mid-recall).
    {
        int n_ctrl = 0, n_eog = 0;
        for (llama_token t : wrapped_toks) {
            if (llama_vocab_get_attr(vocab, t) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) n_ctrl++;
            if (llama_vocab_is_eog(vocab, t)) n_eog++;
        }
        std::string head, tail;
        for (size_t i = 0; i < std::min<size_t>(8, wrapped_toks.size()); i++) {
            head += string_format("%d:'%s' ", (int) wrapped_toks[i], dbg_esc(common_token_to_piece(ctx, wrapped_toks[i], true), 24).c_str());
        }
        for (size_t i = wrapped_toks.size() > 8 ? wrapped_toks.size() - 8 : 0; i < wrapped_toks.size(); i++) {
            tail += string_format("%d:'%s' ", (int) wrapped_toks[i], dbg_esc(common_token_to_piece(ctx, wrapped_toks[i], true), 24).c_str());
        }
        da_trace("slot=%d task=%d L2 refill plan raw=%d wrapped=%d ctrl=%d eog=%d n_full=%d pos_next=%d seq_pos_max=%d sampled=%d head=[%s] tail=[%s]\n",
                 (int) slot.id, (int) (slot.task ? slot.task->id : -1),
                 n_raw, n, n_ctrl, n_eog, n_full, (int) slot.prompt.tokens.pos_next(),
                 (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq), (int) slot.sampled, head.c_str(), tail.c_str());
    }

    // Hoisted out of the try so the catch path can roll back the partial
    // refill: number of tokens successfully written to the KV before the failure.
    int32_t off = 0;
    const int64_t t_refill_start = ggml_time_us();
    try {
        // Hard gate (2026-10-09 12:43 crash): reject the refill BEFORE any
        // decode runs when the wrapped block cannot fit the physical KV. The
        // previous warning-only check let the refill start, run out of cells
        // mid-way (used=76304), halve the sub-batch 10x, roll back (no-op ->
        // slot poisoned) and clear the 66k prompt cache - and the failed
        // decodes overwrote the ctx output buffer, killing the other slot's
        // in-flight row (the GGML_ASSERT abort). Rejecting here leaves the
        // buffer holding the MAIN batch's rows, so the caller fails open (drops
        // the recall, samples the closing token's row, keeps the slot + cache).
        {
            const int32_t n_free = llama_memory_n_free_cells(llama_get_memory(ctx));
            da_trace("slot=%d task=%d L2 refill free cells: n_free=%d n_wrapped=%d\n",
                     (int) slot.id, (int) (slot.task ? slot.task->id : -1), n_free, n);
            if (n_free >= 0 && n_free < n) {
                SLT_WRN(slot, "kv_offload: refill rejected - not enough free cells (n_free=%d < n_wrapped=%d); failing open (the recalled chunk is dropped, the slot continues without it)\n", n_free, n);
                return false; // out_safe_failopen is already true
            }
        }
        // From here on the decode loop runs and overwrites the ctx output
        // buffer - a failure can no longer fail-open (the main batch's rows are
        // gone), so the caller must abort the slot.
        if (out_safe_failopen) *out_safe_failopen = false;
        // Dense forcing: the refill is a prefill and this function calls
        // llama_decode directly (bypassing update_da_n_kv_max, which the main
        // decode runs before each decode). A stale sparse bound from the last
        // main decode must not persist into the refill - force n_kv_max=0
        // (dense). The next main decode recomputes the bound via
        // update_da_n_kv_max.
        llama_set_n_kv_max(ctx, 0);
        // The re-prefilled chunk can be larger than n_batch, and llama_decode
        // asserts n_tokens <= n_batch (it does not sub-batch), so split the
        // refill into n_batch-sized sub-batches.
        int32_t sub_batch = n_batch;
        while (off < n) {
            const int32_t m = std::min(sub_batch, n - off);

            // B1: the final sub-batch's last row is the only output token of
            // the whole refill - the caller samples the next token from it
            // (see the da_refill_out_idx use in post_decode).
            if (out_last_idx && off + m == n) {
                *out_last_idx = m - 1;
            }

            common_batch batch(ctx);

            // common_batch::add appends to the token mirror and the underlying
            // llama_batch_ext; the batch owns its memory (no manual free).
            for (int32_t i = 0; i < m; i++) {
                batch.add(wrapped_toks[off + i], n_full + off + i, seq,
                          (off + i == n - 1));
            }

            int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());

            // Keep the draft (MTP) context in sync with the refill: a
            // target-only decode leaves the draft KV empty over the
            // refilled region, silently dropping draft acceptance for the
            // rest of the session. The impls decode the draft without
            // outputs (logits nulled), so the existing output flags (last
            // token only) are sufficient - no n_outputs_max impact.
            bool spec_ok = true;
            if (ret == 0 && slot.spec != nullptr) {
                spec_ok = common_speculative_process(slot.spec, batch);
            }

            SRV_INF("kv_offload: refill sub-batch [%d,%d) of %d token(s) ret=%d\n",
                    off, off + m, n, ret);

            if (ret != 0 || !spec_ok) {
                // Retry (mirrors decode()): free an idle slot's cells, or
                // halve the sub-batch. Terminal guard: sub_batch == 1 and no
                // slot to clear -> fail-open (rollback + return false).
                bool cleared = on_retry_idle ? on_retry_idle() : false;
                if (cleared) {
                    SRV_WRN("kv_offload: refill %s failed (ret=%d, tokens %d..%d of %d) - cleared idle slot, retrying\n",
                            ret != 0 ? "decode" : "draft process", ret, off, off + m, n);
                    continue;
                }
                if (sub_batch > 1) {
                    sub_batch /= 2;
                    SRV_WRN("kv_offload: refill %s failed (ret=%d, tokens %d..%d of %d) - halving sub_batch to %d, retrying\n",
                            ret != 0 ? "decode" : "draft process", ret, off, off + m, n, sub_batch);
                    continue;
                }
                SRV_WRN("kv_offload: refill %s failed (ret=%d, tokens %d..%d of %d, sub_batch=%d) - fail-open\n",
                        ret != 0 ? "decode" : "draft process", ret, off, off + m, n, sub_batch);
                kv_offload_refill_rollback(slot, n_full, off);
                return false;
            }
            off += m;
        }

        // advance the slot's prompt tokens so pos_next() reflects the refill.
        // insert() appends directly: get_tokens() asserts !has_mtmd (true on any
        // server with mmproj loaded, even for text-only prompts) and the clear()
        // round-trip would drop the media index map.
        const size_t old_size = slot.prompt.tokens.size();
        slot.prompt.tokens.insert(wrapped_toks);

        SRV_INF("kv_offload: re-prefilled %d token(s) at [%d, %d) on seq %d prompt tokens %zu -> %zu, seq_pos_max %d, took %.0f ms\n",
                n, n_full, n_full + n, (int) seq, old_size, old_size + (size_t) n,
                (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq), (ggml_time_us() - t_refill_start) / 1000.0);

        return true;
    } catch (const std::exception & e) {
        SRV_WRN("kv_offload: refill exception: %s - fail-open\n", e.what());
        kv_offload_refill_rollback(slot, n_full, off);
        return false;
    }
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    if (!ctx_server.has_cap_chat()) {
        res->error(format_error_response("this server does not support chat/completions", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // DA injection guard (code-level opt-out, plans/bug.md "scaffold
        // re-reflection"): a request may opt out of ALL DA scaffold
        // injection - via the "da_disable" JSON field (passed through by
        // oaicompat_chat_params_parse on the chat path) or a control-
        // character-fenced marker (U+001C FILE SEPARATOR around
        // "DA-DISABLE") in the prompt text (injected by a client hook).
        // The marker is stripped before tokenization so it never reaches
        // the model. When disabled, both DA paths (marker scan +
        // auto-chunking incl. kv-offload) are skipped for this request and
        // it stays vanilla.
        // NOTE: the marker must stay control-char fenced. A plain text
        // marker self-triggers: a session that discusses this feature has
        // the literal string in its (re-sent) history and silently loses
        // DA for the rest of the session (observed 2026-09-24 on 123).
        static const std::string da_disable_marker = "\x1c" "DA-DISABLE" "\x1c";
        const json j_da_disable = json_value(data, "da_disable", json());
        bool da_disable = !j_da_disable.is_null() && j_da_disable != false; // lenient: "true" strings count
        json da_guard_prompt = prompt;
        if (da_guard_prompt.is_string()) {
            std::string s = da_guard_prompt.get<std::string>();
            const size_t pos = s.find(da_disable_marker);
            if (pos != std::string::npos) {
                s.erase(pos, da_disable_marker.size());
                da_guard_prompt = std::move(s);
                da_disable = true;
            }
        }
        if (da_disable) {
            SRV_INF_DBG("%s", "da: injection disabled for this request (da_disable) - staying vanilla");
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, da_guard_prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, da_guard_prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.apply_chat_session(chat_session);

            // Both DA paths need a strict char->token mapping of the rendered
            // prompt string. An mtmd prompt that contains media has
            // LLAMA_TOKEN_NULL placeholders that map to no character (and
            // get_tokens() asserts on has_mtmd) - skip DA for those. A
            // text-only mtmd prompt (mmproj loaded, no media) tokenizes
            // exactly like plain text, so DA applies; use the assert-free
            // get_text_tokens() for the mapping in that case.
            const bool da_media = task.tokens.has_mtmd &&
                    task.tokens.find_next_media_chunk(0).first != nullptr;

            // (2026-10-13) A media prompt used to skip EVERY budget gate (both
            // DA paths below required !da_media, and the kv-offload reject gate
            // is nested inside the P4 branch), so an over-buffer media prompt
            // would hard-fail mid-prefill. It can now be saved by the P4
            // evict+holes path (the P4 gate no longer excludes da_media): the
            // evicted segments are pure text (old messages before the last
            // user's image), so their KV is cut as holes while the image stays
            // resident. da_media_enters_p4 is exactly the P4 gate (minus the
            // now-removed !da_media) plus the kv-offload config that makes the
            // evict+holes branch actually run. When it is false, no other gate
            // will protect an over-buffer media prompt, so reject immediately
            // (the media path has no recall; the headroom only covers the gen
            // tail (capped) plus a fixed safety margin - smaller than the text
            // path's 16384).
            const bool da_media_enters_p4 = !da_disable && params.da_auto && da_guard_prompt.is_string() &&
                    task.params.da_chunks.empty() && (int32_t) task.tokens.size() >= params.da_min_ctx &&
                    params.kv_unified && params.kv_offload && !params.focus_memory_host.empty() &&
                    params.kv_offload_holes;
            if (da_media && params.kv_cache_size > 0 && !da_media_enters_p4) {
                const int64_t n_predict_eff_m = task.params.n_predict != -1 ? task.params.n_predict : params.n_predict;
                const int64_t gen_cost_m = n_predict_eff_m > 0 ? std::min<int64_t>(n_predict_eff_m, KV_GEN_TAIL_CAP) : 0;
                const int64_t media_headroom = gen_cost_m + 4096;
                const int64_t media_cap = (int64_t) params.kv_cache_size - media_headroom;
                // No holes cut (the P4 evict+holes path is not reachable for
                // this prompt), so every prompt cell (text + media) stays
                // resident: the prefill peak is the full KV plus one in-flight
                // sub-batch (same model as the holes-mode HARD tier in
                // reject_if_over_buffer below).
                const int64_t prefill_peak = (int64_t) task.tokens.pos_next(-1) + params.n_batch;
                if (prefill_peak > media_cap) {
                    SRV_WRN("kv_offload: media request rejected - prefill peak %d tokens is above buffer %d - media headroom %d (gen %d + margin 4096), no recall\n",
                            (int) prefill_peak, params.kv_cache_size, (int) media_headroom, (int) gen_cost_m);
                    // NOTE: message wording is constrained - the client (qwen-code) pattern-matches
                    // the error text for context overflow ("input|prompt|message|context" + "tokens"
                    // + "exceed" within 120 chars triggers reactive compaction with a wrong
                    // originalTokenCount). Keep those word combinations out.
                    res->error(format_error_response(
                        string_format("conversation is larger than this server's KV buffer capacity: %d tokens > %d cell budget (--kv-cache-size %d minus %d media headroom). Start a new session with the image, describe it in text instead, or raise --kv-cache-size.",
                                      (int) prefill_peak, (int) media_cap, params.kv_cache_size, (int) media_headroom),
                        ERROR_TYPE_EXCEED_CONTEXT_SIZE));
                    return res;
                }
            }

            // DA prompt scan (P1): the client hook (FocusMemory) may have
            // appended a marker block to the rendered prompt; recover the
            // chunk layout so the tag state machine can drive the attention
            // restriction. String prompts only (the hook renders one string);
            // fail-open: any mismatch leaves the request vanilla.
            if (!da_disable && params.da_prompt_scan && da_guard_prompt.is_string() && !da_media) {
                const llama_tokens da_tokens = task.tokens.has_mtmd
                        ? task.tokens.get_text_tokens()
                        : task.tokens.get_tokens();
                da_scan_prompt(ctx_server.vocab, da_guard_prompt.get<std::string>(),
                        da_tokens, params.kv_unified, task.params);
            }

            // DA auto-chunking (P4): no client markers and the prompt is long
            // enough - split the rendered chat prompt into magic chunks
            // ourselves, re-tokenize, and map the layout to token ranges.
            // Any mapping failure leaves the original tokens untouched.
            //
            // da-auto is only safe with the reversible B-path (--kv-unified):
            // the A-path (seq_rm) is irreversible, so a wrong-focus permanently
            // deletes the answer chunk with no return (the "ZEBRA" incident).
            // Without --kv-unified, stay vanilla rather than risk that.
            if (!da_disable && params.da_auto && da_guard_prompt.is_string() && !da_media &&
                    task.params.da_chunks.empty() &&
                    (int32_t) task.tokens.size() >= params.da_min_ctx &&
                    !params.kv_unified) {
                static std::atomic<bool> da_auto_needs_kvunified_warned{false};
                if (!da_auto_needs_kvunified_warned.exchange(true)) {
                    SRV_WRN_DBG("%s", "da_auto: requires --kv-unified (reversible B-path) - staying VANILLA to avoid irreversible seq_rm deletion. Add --kv-unified to enable da-auto.\n");
                }
            }
            // (2026-10-13) da_media is no longer excluded: a media prompt is
            // routed through this evict+holes path so the old text's KV is cut
            // while the image (last user, protected) stays resident. The
            // da_auto_chunk re-tokenization below is skipped for media (it would
            // drop the image) - see the da_media guard on the da_auto_chunk call.
            if (!da_disable && params.da_auto && da_guard_prompt.is_string() &&
                    task.params.da_chunks.empty() &&
                    (int32_t) task.tokens.size() >= params.da_min_ctx &&
                    params.kv_unified) {
                // kv-offload: before chunking, evict the oldest middle messages to the
                // FocusMemory store so the reduced prompt fits the window. A failed PUT
                // keeps its segment in the prompt (fail-open). The evicted segments
                // become virtual chunks (M+1..M+K) exposed in the DA instruction.
                std::string da_prompt = da_guard_prompt.get<std::string>();
                std::vector<std::string> offloaded_hints;
                std::vector<kv_offload_evict_segment> evicted_segs;
                std::string kv_session;  // FocusMemory session id (set if kv_offload ran)
                // True once the session has ANY cumulative evicted (the PUT ledger
                // is non-empty), even on a watermark-idle turn where no new evict
                // ran. Drives the Sigma anchor fetch (the DA scaffold must carry
                // the state anchor while history is offloaded, not only on the
                // turn that evicts).
                bool kv_has_offloaded_history = false;
                if (params.kv_offload && !params.focus_memory_host.empty()) {
                    kv_session = kv_offload_resolve_session(req, data);
                    // Record the session's logical (full prompt) token count for
                    // GET /kv_state. resident is derived as logical - offloaded.
                    // Per-session (keyed by kv_session), so parallel slots running
                    // different sessions never mix their values.
                    {
                        std::lock_guard<std::mutex> lk(kv_offload.put_mutex);
                        kv_offload.session_logical[kv_session] = (int64_t) task.tokens.size();
                    }
                    // 2026-10-12: the original prompt tokens, captured before the
                    // da_auto re-tokenization below replaces task.tokens. Once the
                    // request passes the gate it becomes the session's last-prompt
                    // copy (kv_last_prompt_set) - the next gate computes the LCP
                    // against it (the slot's residual prompt is queue-thread owned
                    // and must not be read from the HTTP thread).
                    // get_tokens() asserts on has_mtmd (true whenever an mmproj is
                    // loaded, even for text-only prompts - the P1 DA scan above
                    // uses the same split). A real media prompt never reaches this
                    // block (the P4 branch requires !da_media), so
                    // get_text_tokens() drops no placeholders here.
                    const llama_tokens kv_orig_tokens = task.tokens.has_mtmd
                            ? task.tokens.get_text_tokens()
                            : task.tokens.get_tokens();
                    // B4 pin release: the FocusMemory state worker sets the
                    // session's pin_released flag when the user cancelled or
                    // superseded the original first task. Sticky: a cached
                    // true is never re-queried; a false is re-queried on the
                    // next eviction plan (a local GET, sub-millisecond). Any
                    // query failure keeps the pin (fail-open = current
                    // behavior).
                    bool release_pin = false;
                    {
                        std::lock_guard<std::mutex> lk(kv_offload.put_mutex);
                        release_pin = kv_offload.pin_released.count(kv_session) > 0;
                    }
                    if (!release_pin &&
                            kv_offload_session_get(params.focus_memory_host, params.focus_memory_token,
                                                   kv_session, release_pin)) {
                        if (release_pin) {
                            std::lock_guard<std::mutex> lk(kv_offload.put_mutex);
                            kv_offload.pin_released.insert(kv_session);
                            SRV_INF("kv_offload: pin released (session=%s) - first user message becomes evictable\n",
                                    kv_session.c_str());
                        }
                    }
                    // task 15: clamp the evict stop target (threshold + retain) to the
                    // physical buffer capacity minus headroom, so we never plan to keep
                    // more cells in the KV than the buffer can hold. B = kv_cache_size
                    // (0 = default, no clamp); headroom = 16384. The clamped target is
                    // fed back as an effective retain (target - threshold) for
                    // kv_offload_evict.
                    const int32_t kv_target_unclamped = params.kv_offload_threshold + params.kv_retain_tokens;
                    // gen budget: same expression as the reject lambda below (request
                    // n_predict if > 0, else params.n_predict). The reserved tail is
                    // capped at KV_GEN_TAIL_CAP - reserving the full n_predict (often
                    // 32768) makes resident + recall + gen exceed the buffer for most
                    // real prompts (spurious 400s). An over-long generation is ended
                    // by the buffer-cap graceful stop (process_token) with
                    // finish_reason=length, so the full n_predict need not be reserved.
                    const int64_t n_predict_eff = task.params.n_predict != -1 ? task.params.n_predict : params.n_predict;
                    const int64_t gen_cost = n_predict_eff > 0 ? std::min<int64_t>(n_predict_eff, KV_GEN_TAIL_CAP) : 0;
                    // 10-06 fix: lower bound at the threshold (N) - when B - 16384 < N
                    // the clamp must not push the stop target below the hot window
                    // (evicting below the threshold would break the gate).
                    int32_t kv_target;
                    if (params.kv_cache_size > 0) {
                        // 10-07 fix: the clamp must also reserve the gen tail so the
                        // eviction target matches the reject gate's projection
                        // (target + gen <= B - 16384); without it a prompt with
                        // resident in (B - 16384 - gen, N + M] skipped eviction
                        // (nothing to recall) and was then rejected.
                        const int64_t resident_ceiling = (int64_t) params.kv_cache_size - 16384 - gen_cost;
                        kv_target = std::max((int32_t) std::min<int64_t>(kv_target_unclamped, resident_ceiling),
                                             params.kv_offload_threshold);
                    } else {
                        kv_target = kv_target_unclamped;
                    }
                    if (kv_target < kv_target_unclamped) {
                        static std::atomic<bool> kv_offload_target_clamped_warned{false};
                        if (!kv_offload_target_clamped_warned.exchange(true)) {
                            SRV_INF("kv_offload: retain target clamped %d -> %d (buffer=%d, headroom=16384) - buffer limits KV-resident tokens\n",
                                    kv_target_unclamped, kv_target, params.kv_cache_size);
                        }
                    }
                    int32_t kv_retain_eff = kv_target - params.kv_offload_threshold;
                    // (2026-10-10, incident 2026-10-10-kv-ledger-loss) Resident
                    // estimate for the watermark gate: the tokens actually cut
                    // from this session's KV, NOT the cumulative PUT total.
                    // (2026-10-12, incident 2026-10-12) The raw cut-ledger total
                    // is still wrong, in the other direction: the client
                    // rewrites/compacts the history between turns, so this
                    // request's n_past (LCP with the previous prompt) is far
                    // below the previous prompt size, and every hole with
                    // hi > n_past expires at task start (its KV is re-prefilled).
                    // Projecting resident = prompt - ledger_cut then pins the
                    // gate "idle" while the physical KV re-fills to the buffer
                    // cap (13:35-13:54: 11x full re-prefill of 139-167k
                    // prompts, ~55s each, then failure). Recompute the VALID
                    // cut: LCP of the current prompt against the session's last
                    // accepted prompt (stored copy - see kv_last_prompt), then
                    // the shared expire rule (kv_holes_survive) with an
                    // effective n_past below the LCP: a checkpoint restore can
                    // land below the LCP by up to the checkpoint spacing, and
                    // the da_auto re-tokenization can shift the chunk headers by
                    // up to one chunk. When uncertain, bias to the evict-plan
                    // side - over-planning is a few idempotent PUTs,
                    // under-planning is a 55s prefill + failure.
                    const int64_t kv_gate_margin = std::max<int64_t>(params.checkpoint_min_step, params.da_chunk_tokens);
                    size_t kv_lcp = 0;
                    const int64_t kv_valid_cut = ctx_server.kv_session_valid_cut(kv_session, kv_orig_tokens, kv_gate_margin, &kv_lcp);
                    const int64_t kv_resident = (int64_t) task.tokens.size() - kv_valid_cut;
                    // The Sigma anchor fetch below still keys off the PUT
                    // ledger: it answers "has this session ever offloaded
                    // text", which the cut ledger cannot express.
                    int64_t kv_put_total = 0;
                    {
                        std::lock_guard<std::mutex> lk(kv_offload.put_mutex);
                        auto it = kv_offload.uploaded.find(kv_session);
                        if (it != kv_offload.uploaded.end()) {
                            for (const auto & kv : it->second) kv_put_total += kv.second;
                        }
                    }
                    kv_has_offloaded_history = kv_put_total > 0;
                    // Watermark idle (holes mode only): while the resident is below
                    // the high watermark, skip the per-turn evict plan / PUT / hole
                    // re-apply. The already-cut holes persist in the KV (the
                    // kv_hole_ranges ledger), so the resident stays low; recall is
                    // still available via the Sigma anchor (offloaded-chunks note).
                    const bool kv_watermark_idle =
                            params.kv_offload_holes && params.kv_offload_high > 0 &&
                            kv_resident < (int64_t) params.kv_offload_high;
                    // Diagnostic: config state + current token count vs threshold, so
                    // the journal shows exactly why eviction does or does not fire
                    // (was the flag parsed? threshold reached? host set? pin released?).
                    // task 17: gate log now carries the effective retain/target and the
                    // physical buffer so the journal shows the full evict budget.
                    // Watermark: resident (the KV-resident count) and high (the
                    // trigger watermark; 0 = legacy per-turn gate).
                    SRV_INF("kv_offload: gate - threshold=%d retain=%d target=%d buffer=%d tokens=%d resident=%d lcp=%zu margin=%d valid_cut=%d high=%d host=%s session=%s pin_released=%d\n",
                            params.kv_offload_threshold, kv_retain_eff, kv_target, params.kv_cache_size,
                            (int) task.tokens.size(), (int) kv_resident, kv_lcp, (int) kv_gate_margin, (int) kv_valid_cut,
                            params.kv_offload_high,
                            params.focus_memory_host.c_str(),
                            kv_session.c_str(), (int) release_pin);
                    // Returns true if the request was rejected (res already holds the error).
                    // Two tiers (10-07):
                    //  - HARD: the prefill peak (the maximum number of KV cells LIVE at
                    //    any moment during the prompt prefill) cannot fit the physical
                    //    buffer. No eviction plan can fix it, so reject before it
                    //    hard-fails in find_slot mid-stream (slot abort + KV clear,
                    //    worse than a clean 400). The peak is a live-cell count, not a
                    //    physical-memory figure (the cell pool is fixed at B; seq_rm
                    //    only recycles slots, it never shrinks the pool).
                    //    * Option C (default): the evicted text is removed from the
                    //      prompt, so the peak is the resident (prompt - evicted).
                    //    * Holes mode (Option B): the evicted text stays in the prompt,
                    //      but the progressive cut (kv_holes_progressive_apply) cuts
                    //      each hole's KV as the prefill head passes it, so the live
                    //      count never reaches the raw prompt length - the peak is the
                    //      final resident plus at most one in-flight sub-batch. The old
                    //      "peak = full prompt" model predated the progressive cut
                    //      (9a06c0385) and rejected every request whose prompt crossed
                    //      B - 16384 even though the live cells stayed near the evict
                    //      target (2026-10-07: prompt 69287 / evicted 23485 / actual
                    //      peak ~46.8k vs cap 68920 - spurious 400 loop, see
                    //      plans/2026-10-07-kv-400-incident.md).
                    //  - SOFT: the steady-state projection (resident + largest evicted
                    //    segment + gen tail) exceeds the cap, but the prefill fits.
                    //    Proceed - every runtime overflow is graceful: generation stops
                    //    at the buffer cap (finish_reason=length, process_token), a
                    //    recall whose refill cannot fit rolls back (kv_offload_refill),
                    //    and a store GET failure fails open (get-on-focus). A 400 here
                    //    only interrupted the client turn (and pre-301cf5abc drove the
                    //    qwen-code compaction loop). The store-down case (PUT failed,
                    //    segment kept resident) degrades the same way: no recall for the
                    //    failed segments, generation capped by the buffer-cap stop.
                    auto reject_if_over_buffer = [&](int64_t evicted_tokens, int32_t planned_tokens, bool put_failed) -> bool {
                        if (params.kv_cache_size <= 0) return false;
                        const int64_t cap = (int64_t) params.kv_cache_size - 16384;
                        // (2026-10-13) media: the prompt's KV footprint is the KV
                        // position count (pos_next(-1)), not the array size - media
                        // chunks occupy more KV cells than array tokens. The evicted
                        // segments are pure text (their KV == their token count), so
                        // resident = KV - evicted_tokens is the post-hole resident.
                        const int64_t prompt = da_media ? (int64_t) task.tokens.pos_next(-1) : (int64_t) task.tokens.size();
                        const int64_t resident = prompt - evicted_tokens;
                        // Holes mode: progressive cut keeps the live-cell peak at the
                        // final resident + one in-flight sub-batch (NOT the raw prompt).
                        // Option C: the evicted text is gone, so the peak is the resident.
                        const int64_t prefill_peak = params.kv_offload_holes ? resident + params.n_batch : resident;
                        if (prefill_peak > cap) {
                            SRV_WRN("kv_offload: rejecting request - prefill peak %d tokens is above buffer %d - headroom 16384 (holes=%d, evicted %d of planned %d, put_failed=%d, session=%s)\n",
                                    (int) prefill_peak, params.kv_cache_size, (int) params.kv_offload_holes,
                                    (int) evicted_tokens, (int) planned_tokens, (int) put_failed, kv_session.c_str());
                            // 2026-10-12: text path = 503 (retryable), NOT 400
                            // (fail-fast). The message intentionally matches the
                            // client's context-overflow patterns and carries the
                            // real token counts (parseTokenCounts): reactive
                            // compaction is the recovery path - the client shrinks
                            // the prompt and retries, and the retry fits. The
                            // 400 (ERROR_TYPE_EXCEED_CONTEXT_SIZE) stays reserved
                            // for the media path, where a retry cannot fix the
                            // prompt size.
                            if (da_media) {
                                // media: a retry (compaction) cannot fix a prompt
                                // that still exceeds the cap after evict+holes -
                                // fail fast with 400. The message avoids the
                                // client's context-overflow trigger words (which
                                // would loop a useless compaction).
                                res->error(format_error_response(
                                    string_format("conversation is larger than this server's KV buffer capacity: %d tokens > %d cell budget (--kv-cache-size %d). Start a new session with the image, describe it in text instead, or raise --kv-cache-size.",
                                                  (int) prefill_peak, (int) cap, params.kv_cache_size),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE));
                            } else {
                                res->error(format_error_response(
                                    string_format("Context length exceeded: %d tokens > %d limit",
                                                  (int) prefill_peak, (int) cap),
                                    ERROR_TYPE_UNAVAILABLE));
                            }
                            return true;
                        }
                        int64_t recall_cost = 0;
                        for (const auto & seg : evicted_segs) {
                            if (seg.tokens > recall_cost) recall_cost = seg.tokens;
                        }
                        // n_predict_eff / gen_cost are the hoisted clamp-scope values
                        // (single source of truth shared with the eviction clamp).
                        // resident is defined above (shared with the HARD peak).
                        const int64_t projected = resident + recall_cost + gen_cost;
                        if (projected > cap) {
                            SRV_INF("kv_offload: projected demand %d (resident %d + max recall %d + gen %d) above buffer cap %d - proceeding (buffer-cap stop + refill rollback + get fail-open are the runtime safety nets; evicted %d of planned %d, put_failed=%d, session=%s)\n",
                                    (int) projected, (int) resident, (int) recall_cost, (int) gen_cost,
                                    (int) cap, (int) evicted_tokens, (int) planned_tokens,
                                    (int) put_failed, kv_session.c_str());
                        }
                        return false;
                    };
                    std::vector<kv_offload_evict_segment> evict_segs;
                    // Optimistic pre-PUT projection (all planned segments evicted).
                    // The authoritative capacity check after the evict if/else uses
                    // the actually-evicted count instead, so a failed PUT (segment
                    // kept in the prompt, fail-open) is reflected in the judgment.
                    int32_t plan_tokens = 0;
                    if (kv_watermark_idle) {
                        // Watermark idle (holes mode): the resident is below the high
                        // watermark, so no eviction this turn. The already-cut holes
                        // persist in the KV (kv_hole_ranges ledger), keeping the
                        // resident low; recall stays available via the Sigma anchor.
                        // Run the capacity check with the actual resident (the
                        // cumulative evicted) so a store-down / pin overflow is
                        // still caught.
                        SRV_INF("kv_offload: watermark idle - resident=%d < high=%d (low=%d) - no eviction this turn (holes persist, session=%s)\n",
                                (int) kv_resident, params.kv_offload_high, params.kv_offload_threshold, kv_session.c_str());
                        if (reject_if_over_buffer(kv_valid_cut, 0, false)) {
                            return res;
                        }
                    } else if (kv_offload_evict(ctx_server.vocab, da_prompt, (int32_t) task.tokens.size(),
                                         params.kv_offload_threshold, evict_segs, release_pin,
                                         kv_retain_eff)) {
                        // Diagnostic: the eviction plan (n segments, tokens each).
                        for (auto & seg : evict_segs) plan_tokens += seg.tokens;
                        // 10-07 fix (2nd 400, session 051dfdf9): the clamp above
                        // reserves the gen tail but not the max recall segment -
                        // focus recall re-prefills the largest evicted message on
                        // top of the resident KV, so the reject gate's projection
                        // (resident + max_seg + gen) can exceed B - 16384 even when
                        // the evict target (<= B - 16384 - gen) was met. Observed:
                        // B=75304, resident 42442 + max_seg 8381 + gen 8192 =
                        // 59015 > 58920 (short by 95). Re-plan with a target that
                        // also reserves the largest planned segment. Each pass
                        // evicts a superset (the target strictly decreases), so
                        // max_seg only grows and the loop converges in 1-2 passes
                        // (cap: 3). If the hot window (threshold) alone cannot
                        // carry the projection, the gate rejects below.
                        if (params.kv_cache_size > 0) {
                            for (int pass = 0; pass < 3 && !evict_segs.empty(); ++pass) {
                                int64_t max_seg = 0;
                                for (const auto & seg : evict_segs) max_seg = std::max<int64_t>(max_seg, seg.tokens);
                                const int64_t resident_after = (int64_t) task.tokens.size() - plan_tokens;
                                if (resident_after + max_seg + gen_cost <= (int64_t) params.kv_cache_size - 16384) break;
                                const int64_t deeper_ceiling = (int64_t) params.kv_cache_size - 16384 - gen_cost - max_seg;
                                const int32_t kv_target2 = std::max((int32_t) std::min<int64_t>(kv_target, deeper_ceiling),
                                                                    params.kv_offload_threshold);
                                if (kv_target2 >= kv_target) break;  // infeasible at the hot window - the gate rejects
                                SRV_INF("kv_offload: evict re-plan pass %d - target %d -> %d (resident %d + max recall %d + gen %d > buffer %d - headroom 16384)\n",
                                        pass + 1, kv_target, kv_target2, (int) resident_after, (int) max_seg,
                                        (int) gen_cost, params.kv_cache_size);
                                kv_target = kv_target2;
                                std::vector<kv_offload_evict_segment> replan_segs;
                                if (!kv_offload_evict(ctx_server.vocab, da_prompt, (int32_t) task.tokens.size(),
                                                      params.kv_offload_threshold, replan_segs, release_pin,
                                                      kv_target2 - params.kv_offload_threshold)) break;
                                evict_segs = std::move(replan_segs);
                                plan_tokens = 0;
                                for (auto & seg : evict_segs) plan_tokens += seg.tokens;
                            }
                        }
                        SRV_INF("kv_offload: evict plan - %zu segment(s), ~%d token(s) to offload\n",
                                evict_segs.size(), plan_tokens);
                        std::vector<std::pair<size_t, size_t>> evicted_ranges;
                        // Per-session uploaded-key cache (Option B re-PUT elimination):
                        // the PUT is idempotent (key = FNV-1a content hash of the segment
                        // text), so a key already uploaded for this session is a cache hit -
                        // skip the network call and count it as evicted. The lock is not
                        // held across the network call: a concurrent miss on the same key
                        // just re-PUTs once, which is safe (idempotent).
                        for (auto & seg : evict_segs) {
                            bool put_ok = false;
                            bool was_cached = false;
                            {
                                std::lock_guard<std::mutex> lk(kv_offload.put_mutex);
                                auto it = kv_offload.uploaded.find(kv_session);
                                if (it != kv_offload.uploaded.end() && it->second.count(seg.key) > 0) {
                                    was_cached = true;
                                }
                            }
                            if (was_cached) {
                                put_ok = true;
                                //SRV_INF("kv_offload: PUT skip (cached) key=%s tokens=%d (%s)\n",
                                //        seg.key.c_str(), seg.tokens, seg.hint.c_str());
                            } else {
                                put_ok = kv_offload_put(params.focus_memory_host, params.focus_memory_token,
                                                        kv_session, seg.key, seg.text, seg.tokens);
                                if (put_ok) {
                                    std::lock_guard<std::mutex> lk(kv_offload.put_mutex);
                                    auto & keys = kv_offload.uploaded[kv_session];
                                    // Memory ceiling: bound the per-session key map. Evict
                                    // the lexicographically smallest key (std::map is
                                    // ordered by key) - worst case that key is re-PUT
                                    // later (idempotent).
                                    if (keys.size() >= 4096) keys.erase(keys.begin());
                                    keys[seg.key] = seg.tokens;
                                    // FOCUS_SPARSE_DEBUG-gated: fires on every segment PUT.
                                    SRV_INF_DBG("kv_offload: PUT ok key=%s tokens=%d (%s)\n",
                                                seg.key.c_str(), seg.tokens, seg.hint.c_str());
                                }
                            }
                            if (put_ok) {
                                evicted_ranges.push_back({ seg.range_lo, seg.range_hi });
                                // Option B keeps the evicted text in the prompt (as a
                                // real chunk), so no virtual-chunk hint is exposed.
                                if (!params.kv_offload_holes) {
                                    offloaded_hints.push_back(seg.hint);
                                }
                                evicted_segs.push_back(seg);
                            } else {
                                SRV_WRN("kv_offload: PUT failed for key=%s - keeping segment in prompt\n", seg.key.c_str());
                                // 3a (10-06): store-down + decoupled buffer - the
                                // failed segment stays in the prompt (fail-open), so
                                // the full prompt must fit the physical buffer B.
                                // With B < n_ctx a long prompt hard-fails in
                                // find_slot: the store is a hard dependency in
                                // decoupled mode. One-shot warning.
                                if (params.kv_cache_size > 0) {
                                    static std::atomic<bool> kv_offload_store_down_warned{false};
                                    if (!kv_offload_store_down_warned.exchange(true)) {
                                        SRV_WRN("kv_offload: store down with a decoupled buffer (B=%d) - failed segments stay in the prompt and the full prompt must fit B (find_slot failure risk); FocusMemory store is a hard dependency in decoupled mode\n", params.kv_cache_size);
                                    }
                                }
                            }
                        }
                        // Authoritative post-PUT capacity check (replaces the
                        // optimistic pre-PUT projection): only the segments whose
                        // PUT succeeded (or were cached) are actually evicted; a
                        // failed PUT keeps its segment in the prompt (fail-open),
                        // so the projected KV-resident count is larger than the
                        // plan assumed. If it exceeds the buffer minus the 16k
                        // headroom, the request cannot fit the decoupled KV
                        // buffer - reject now with a clear error instead of dying
                        // later in find_slot ("no slot available"). Covers both
                        // the store-down case (PUTs fail, full prompt must fit B)
                        // and the pin+hot+delta > B case (nothing evictable is
                        // enough).
                        int64_t evicted_tokens = 0;
                        for (auto & seg : evicted_segs) evicted_tokens += seg.tokens;
                        // Temporary diagnostic: verify evicted_segs is cumulative (cache-hit
                        // keys re-planned each request must be counted) - planned == evicted
                        // in normal sessions. Remove once confirmed.
                        SRV_INF("kv_offload: evict result - planned=%d evicted=%d segs=%zu/%zu\n",
                                (int) plan_tokens, (int) evicted_tokens, evicted_segs.size(), evict_segs.size());
                        if (reject_if_over_buffer(evicted_tokens, plan_tokens, evicted_tokens < plan_tokens)) {
                            return res;
                        }
                        if (!evicted_ranges.empty()) {
                            if (params.kv_offload_holes) {
                                // Option B: keep the evicted text in the prompt. The
                                // holes are cut from the KV at the next request's
                                // n_past check (not from the text), so the client can
                                // keep sending the full prompt and re-match at
                                // n_past=full (1-token re-prefill instead of the
                                // evicted tail).
                                SRV_INF("kv_offload-holes: %zu segment(s) evicted to FocusMemory (session=%s), prompt text kept (%zu chars) - holes will be cut from KV\n",
                                        evicted_ranges.size(), kv_session.c_str(), da_prompt.size());
                            } else {
                                // Option C: remove the evicted text from the prompt
                                std::sort(evicted_ranges.begin(), evicted_ranges.end(),
                                          [](const auto & a, const auto & b) { return a.first > b.first; });
                                const size_t before_chars = da_prompt.size();
                                std::string reduced = da_prompt;
                                for (auto & [lo, hi] : evicted_ranges) {
                                    reduced.erase(lo, hi - lo);
                                }
                                da_prompt = std::move(reduced);
                                SRV_INF("kv_offload: evicted %zu segment(s) to FocusMemory (session=%s), prompt %zu->%zu chars\n",
                                        evicted_ranges.size(), kv_session.c_str(), before_chars, da_prompt.size());
                            }
                        }
                    } else {
                        // Diagnostic: considered but nothing planned (under threshold,
                        // or no evictable middle messages) - the common steady state.
                        SRV_INF("kv_offload: no eviction planned (tokens=%d, threshold=%d)\n",
                                (int) task.tokens.size(), params.kv_offload_threshold);
                        // Nothing evicted: everything stays resident (pin / last turn
                        // alone may exceed B) - run the same capacity check.
                        if (reject_if_over_buffer(0, 0, false)) {
                            return res;
                        }
                    }
                    // 2026-10-12: the request passed the gate - record its original
                    // prompt as the session's last accepted prompt (the next gate's
                    // LCP reference). A rejected request never reaches here (its KV
                    // state did not change, so the reference must not move either).
                    ctx_server.kv_last_prompt_set(kv_session, kv_orig_tokens);
                } else {
                    // Diagnostic (one-time): kv-offload configured off - the flag was
                    // not parsed (was the old --kv-offload collision) or the host is
                    // empty, so eviction can never fire.
                    static std::atomic<bool> kv_offload_disabled_warned{false};
                    if (!kv_offload_disabled_warned.exchange(true)) {
                        SRV_WRN("kv_offload: disabled (flag=%d, host_empty=%d) - add --fm-offload and --focus-memory-host to enable eviction\n",
                                (int) params.kv_offload, (int) params.focus_memory_host.empty());
                    }
                }
                // (2026-10-13) media: the evicted segments are pure text (old
                // messages before the last user's image). Cut their KV as holes
                // (kv_hole_pending), keeping the prompt (and the image) intact.
                // No DA scaffold (no magic headers) - the progressive hole cut
                // (kv_holes_progressive_apply) does the cutting as the prefill
                // head passes. The image (last user) is protected by
                // kv_offload_evict, so its KV is never holed. The char ranges
                // are in the original da_prompt text space; the evicted messages
                // precede the last user's media marker, so their positions are
                // unchanged by the marker. text_toks (get_text_tokens) drops the
                // NULL media placeholders; text_to_arr recovers each text token's
                // array index, and pos_next maps it to a KV position (media
                // chunks occupy more KV cells than array tokens).
                if (da_media && params.kv_offload_holes && !evicted_segs.empty()) {
                    const llama_tokens text_toks = task.tokens.get_text_tokens();
                    std::vector<int64_t> text_to_arr(text_toks.size());
                    int64_t t = 0;
                    for (size_t a = 0; a < task.tokens.size() && t < (int64_t) text_toks.size(); a++) {
                        if (task.tokens[a] != LLAMA_TOKEN_NULL) text_to_arr[t++] = (int64_t) a;
                    }
                    const std::vector<size_t> offs = da_token_offsets_lenient(ctx_server.vocab, da_prompt, text_toks);
                    if (offs.empty() || (size_t) t != text_toks.size()) {
                        // Fail-closed: without holes the full prompt (text + media)
                        // would exceed the buffer mid-prefill. A retry cannot fix a
                        // media prompt, so reject with 400 (the message avoids the
                        // client's context-overflow trigger words).
                        SRV_WRN("kv_offload-holes(media): char->token mapping failed (drift) - rejecting, no holes to cut (session=%s)\n", kv_session.c_str());
                        res->error(format_error_response(
                            string_format("conversation is larger than this server's KV buffer capacity: %d tokens > %d cell budget (--kv-cache-size %d). Start a new session with the image, describe it in text instead, or raise --kv-cache-size.",
                                          (int) task.tokens.pos_next(-1), (int) (params.kv_cache_size - 16384), params.kv_cache_size),
                            ERROR_TYPE_EXCEED_CONTEXT_SIZE));
                        return res;
                    }
                    for (auto & seg : evicted_segs) {
                        const int32_t t_lo = da_char_to_token_nearest(offs, seg.range_lo);
                        const int32_t t_hi = da_char_to_token_nearest(offs, seg.range_hi);
                        if (t_lo < 0 || t_hi < 0 || t_hi <= t_lo ||
                                t_lo >= (int32_t) text_to_arr.size() || t_hi >= (int32_t) text_to_arr.size()) {
                            SRV_WRN("kv_offload-holes(media): failed to map evicted segment %s to a token range - skipping its hole\n", seg.key.c_str());
                            continue;
                        }
                        const int32_t kv_lo = (int32_t) task.tokens.pos_next(text_to_arr[t_lo]);
                        const int32_t kv_hi = (int32_t) task.tokens.pos_next(text_to_arr[t_hi]);
                        task.params.kv_hole_pending.push_back({ kv_lo, kv_hi });
                        SRV_INF("kv_offload-holes(media): pending hole %s -> kv [%d, %d) (%d)\n",
                                seg.key.c_str(), kv_lo, kv_hi, kv_hi - kv_lo);
                    }
                    // The slot learns its kv_session from this (the cut ledger is
                    // keyed by it); host/token/vocab are needed for PUT/recall.
                    task.params.kv_offload_session = kv_session;
                    task.params.kv_offload_host    = params.focus_memory_host;
                    task.params.kv_offload_token   = params.focus_memory_token;
                    task.params.kv_offload_vocab   = ctx_server.vocab;
                }
                // Option B (--kv-offload-holes): pass the evicted messages' char
                // ranges + hints so da_auto_chunk gives each evicted message its
                // own chunk (the hole boundary then coincides with the chunk
                // boundary) and exposes the holed chunk numbers in the DA
                // instruction. Empty otherwise - the packing is unchanged.
                std::vector<da_evict_bound> evict_bounds;
                if (params.kv_offload_holes) {
                    evict_bounds.reserve(evicted_segs.size());
                    for (const auto & seg : evicted_segs) {
                        evict_bounds.push_back({ seg.range_lo, seg.range_hi, seg.hint });
                    }
                }
                // kv-offload (mid-turn Σ, 2026-10-09): while history is
                // offloaded the offloaded-chunks note is non-empty — fetch the
                // session's state anchor (200ms cap, fail-open) and append it
                // to the DA scaffold so a long autonomous turn keeps its
                // ground. Nothing offloaded → the full history is in the KV →
                // no anchor needed (no fetch, no cost).
                std::string sigma_anchor;
                // (2026-10-13) media skips the DA scaffold (no magic chunks), so
                // the sigma anchor (appended to the scaffold) is not needed.
                if (!da_media && (kv_has_offloaded_history || !offloaded_hints.empty() || !evict_bounds.empty()) &&
                        params.kv_offload && !params.focus_memory_host.empty() && !kv_session.empty()) {
                    if (kv_offload_get_sigma(params.focus_memory_host, params.focus_memory_token, kv_session, sigma_anchor)) {
                        SRV_INF("kv_offload: sigma anchor for DA scaffold (%zu chars, session=%s)\n",
                                sigma_anchor.size(), kv_session.c_str());
                    }
                }
                // (2026-10-13) media: skip da_auto_chunk - its common_tokenize
                // re-tokenization is pure text and would drop the image. The
                // default-constructed layout has ok=false, so the tokenize /
                // set-tokens path below is skipped and the original mtmd
                // task.tokens (with the image) is kept.
                const da_auto_layout layout =
                        da_media ? da_auto_layout()
                                 : da_auto_chunk(ctx_server.vocab, da_prompt, params.da_chunk_tokens, offloaded_hints, evict_bounds, sigma_anchor, params.da_scaffold_file);
                if (layout.ok) {
                    const llama_tokens  toks = common_tokenize(ctx_server.vocab, layout.modified, true, true);
                    const std::vector<size_t> offs = da_token_offsets_lenient(ctx_server.vocab, layout.modified, toks);
                    bool ok = !offs.empty();

                    // chunk c = [header_c, header_{c+1}) ; last = [header_last, tail)
                    // header_pos_mod (modified-text coordinates) - header_pos is
                    // original-text and would drift left by the earlier headers'
                    // lengths as c grows.
                    std::vector<std::pair<int32_t, int32_t>> chunks;
                    if (ok) {
                        chunks.reserve(layout.header_pos_mod.size());
                        for (size_t c = 0; c < layout.header_pos_mod.size(); c++) {
                            const size_t next = c + 1 < layout.header_pos_mod.size() ? layout.header_pos_mod[c + 1] : layout.tail_pos;
                            const int32_t lo = da_char_to_token_nearest(offs, layout.header_pos_mod[c]);
                            const int32_t hi = da_char_to_token_nearest(offs, next);
                            if (lo < 0 || hi < 0 || hi <= lo) {
                                ok = false;
                                break;
                            }
                            // (start, end) token range, per the da_chunks contract
                            // (server-task.h) and the P1 hook path: apply_da_b/
                            // apply_da_rm consume these as [lo, hi) removal ranges.
                            chunks.push_back({ lo, hi });
                        }
                    }
                    // Diagnostic (guards the header_pos_mod coordinate fix):
                    // each chunk's token range must start at its [Magic Chunk N]
                    // header. The nearest-boundary snap can shift the start by a
                    // token or two, so the check is "the header appears within
                    // the first 20 detokenized tokens of the chunk", not an
                    // exact prefix match.
                    if (ok) {
                        size_t n_mismatch = 0;
                        size_t first_mismatch = 0;  // 1-based chunk number, 0 = none
                        size_t n_logged = 0;
                        for (size_t c = 0; c < chunks.size(); c++) {
                            const int32_t lo = chunks[c].first;
                            std::string piece;
                            for (int32_t t = lo; t < lo + 20 && t < (int32_t) toks.size(); t++) {
                                piece += common_token_to_piece(ctx_server.vocab, toks[t], true);
                            }
                            // first 24 chars, backed off to a UTF-8 boundary
                            size_t n = piece.size() < 24 ? piece.size() : 24;
                            while (n > 0 && n < piece.size() && ((unsigned char) piece[n] & 0xC0) == 0x80) {
                                n--;
                            }
                            std::string head = piece.substr(0, n);
                            for (char & ch : head) {
                                if ((unsigned char) ch < 0x20 || ch == 0x7F) ch = ' ';
                            }
                            const std::string expected = "[Magic Chunk " + std::to_string(c + 1) + "]";
                            if (piece.find(expected) == std::string::npos) {
                                n_mismatch++;
                                if (first_mismatch == 0) first_mismatch = c + 1;
                                if (n_logged < 5) {
                                    SRV_INF("da_auto: chunk %zu header mismatch head='%s' lo=%d\n",
                                            c + 1, head.c_str(), lo);
                                    n_logged++;
                                }
                            }
                        }
                        SRV_INF("da_auto: chunk header check - n_chunks=%zu mismatch=%zu first_mismatch=%zu\n",
                                chunks.size(), n_mismatch, first_mismatch);
                    }
                    std::pair<int32_t, int32_t> filler = { -1, -1 };
                    if (ok && layout.filler.second > layout.filler.first) {
                        const int32_t lo = da_char_to_token_nearest(offs, layout.filler.first);
                        const int32_t hi = da_char_to_token_nearest(offs, layout.filler.second);
                        if (lo < 0 || hi < 0 || hi <= lo) {
                            ok = false;
                        } else {
                            filler = { lo, hi };
                        }
                    }
                    std::pair<int32_t, int32_t> sigma = { -1, -1 };
                    if (ok && layout.sigma.second > layout.sigma.first) {
                        const int32_t lo = da_char_to_token_nearest(offs, layout.sigma.first);
                        const int32_t hi = da_char_to_token_nearest(offs, layout.sigma.second);
                        if (lo < 0 || hi < 0 || hi <= lo) {
                            // A sigma mapping failure must not disable DA for
                            // the whole request (keeping vanilla): drop the
                            // sigma protection ({-1,-1}) and continue with the
                            // rest of the layout.
                            SRV_WRN_DBG("%s", "da_auto: sigma range does not map to token boundaries - continuing without sigma protection\n");
                        } else {
                            sigma = { lo, hi };
                        }
                    }
                    if (!ok) {
                        SRV_WRN_DBG("%s", "da_auto: layout does not map to token boundaries - keeping vanilla\n");
                    } else {
                        task.tokens  = server_tokens(toks, false);
                        task.params.da_chunks = std::move(chunks);
                        task.params.da_filler = filler;
                        task.params.da_sigma  = sigma;
                        task.params.da_b      = params.kv_unified;
                        // kv-offload (Option C): assign the virtual chunk numbers to
                        // the evicted segments (M+1..M+K) so get-on-focus can look
                        // them up. Option B keeps the evicted text as real magic
                        // chunks (in the prompt), so no virtual chunk IDs are
                        // assigned and get-on-focus re-prefill is not needed for them.
                        if (!params.kv_offload_holes) {
                            const int32_t n_active = (int32_t) task.params.da_chunks.size();
                            for (size_t i = 0; i < evicted_segs.size(); i++) {
                                task.params.da_offloaded.push_back(
                                        { evicted_segs[i].key, evicted_segs[i].hint,
                                          task.params.da_chunk_base + n_active + (int32_t) i });
                            }
                        }
                        // (2026-10-10, incident 2026-10-10-kv-ledger-loss) Always
                        // tag the task with its session id, not only on evicting
                        // turns: the slot learns its kv_session from this (the
                        // cut ledger is keyed by it), and a watermark-idle turn
                        // is exactly when the slot keeps running. host/token/
                        // vocab are only needed when there is something to PUT.
                        if (!kv_session.empty()) {
                            task.params.kv_offload_session = kv_session;
                            if (!evicted_segs.empty()) {
                                task.params.kv_offload_host    = params.focus_memory_host;
                                task.params.kv_offload_token   = params.focus_memory_token;
                                task.params.kv_offload_vocab   = ctx_server.vocab;
                            }
                        }
                        // Option B (--kv-offload-holes): map each evicted segment's
                        // char range (original text space) to a token range (final
                        // space) and record it as a pending KV hole. The range must
                        // be shifted by the [Magic Chunk N] headers that
                        // da_auto_chunk inserted into layout.modified: a header at
                        // original position h shifts every position p >= h right by
                        // the header length. A header can sit inside an evicted
                        // segment (at the segment's content start), so range_lo and
                        // range_hi get different shifts. The holes are applied at
                        // the next request's n_past check, once the prompt KV is
                        // known to exist.
                        if (params.kv_offload_holes && !evicted_segs.empty()) {
                            auto header_len = [](int32_t num) -> size_t {
                                return (size_t) (15 + (int) std::to_string(num).size()); // "[Magic Chunk N]\n"
                            };
                            auto mod_pos = [&](size_t p) -> size_t {
                                size_t shift = 0;
                                for (size_t i = 0; i < layout.header_pos.size(); i++) {
                                    if (layout.header_pos[i] <= p) {
                                        shift += header_len((int32_t) (i + 1));
                                    }
                                }
                                return p + shift;
                            };
                            for (auto & seg : evicted_segs) {
                                const int32_t lo = da_char_to_token_nearest(offs, mod_pos(seg.range_lo));
                                const int32_t hi = da_char_to_token_nearest(offs, mod_pos(seg.range_hi));
                                if (lo < 0 || hi < 0 || hi <= lo) {
                                    SRV_WRN_DBG("kv-offload-holes: failed to map evicted segment %s to a token range - skipping its hole\n", seg.key.c_str());
                                    continue;
                                }
                                task.params.kv_hole_pending.push_back({ lo, hi });
                                SRV_INF_DBG("kv-offload-holes: pending hole %s -> tokens [%d, %d) (%d tokens)\n",
                                        seg.key.c_str(), lo, hi, hi - lo);
                            }
                        }
                        SRV_INF_DBG("da_auto: %zu chunk(s) from %zu source message(s), %zu token(s) - %s path%s%s\n",
                                task.params.da_chunks.size(), layout.n_source_msgs,
                                task.tokens.get_tokens().size(),
                                task.params.da_b ? "B" : "A",
                                task.params.da_offloaded.empty() ? "" : " + kv-offload",
                                params.kv_offload_holes && !task.params.kv_hole_pending.empty()
                                    ? string_format(" + %zu kv-hole(s)", task.params.kv_hole_pending.size()).c_str() : "");
                    }
                }
            }

            // message delimiters for checkpointing (fork feature; the b12412
            // rebase dropped the function-level declaration, re-parse per task)
            common_chat_msg_delimiters delimiters;
            {
                const json delims = json_value(data, "message_delimiters", json::array());
                if (delims.is_array()) {
                    delimiters.delimiters.reserve(delims.size());
                    for (const auto & d : delims) {
                        if (!d.is_object()) continue;
                        delimiters.delimiters.push_back({
                            common_chat_role_from_string(d.value("role", std::string())),
                            d.value("delimiter", std::string()),
                        });
                    }
                }
                delimiters.tokenize(ctx_server.vocab);
            }
            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks), chat_session);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & usage   = arr[0]["usage"];
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }

                // OAI counts generated tokens across all choices
                int64_t n_completion_tokens = 0;
                for (auto & item : arr) {
                    n_completion_tokens += item["usage"]["completion_tokens"].get<int64_t>();
                }
                usage["completion_tokens"] = n_completion_tokens;
                usage["total_tokens"]      = usage["prompt_tokens"].get<int64_t>() + n_completion_tokens;

                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", server_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };

    if (!params.server_connect_code.empty()) {
        props["connect_code"] = params.server_connect_code;
    }
    if (!meta.tts_voices.empty()) {
        props["voices"] = meta.tts_voices;
    }
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

// body of /tts and /v1/audio/speech; multipart form fields arrive as strings, the numeric and boolean params get their types back
static json parse_tts_body(const server_http_req & req) {
    json body = json::parse(req.body);
    for (const char * k : { "top_k", "n_predict", "max_tokens", "seed" }) {
        if (body.contains(k) && body[k].is_string()) {
            body[k] = std::stoll(body[k].get<std::string>());
        }
    }
    for (const char * k : { "top_p", "temperature", "repeat_penalty", "speed" }) {
        if (body.contains(k) && body[k].is_string()) {
            body[k] = std::stof(body[k].get<std::string>());
        }
    }
    if (body.contains("stream") && body["stream"].is_string()) {
        body["stream"] = body["stream"] == "true";
    }
    return body;
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->get_kv_state = [this](const server_http_req & req) {
        // Per-session KV state for the FocusMemory client. Reads in-memory
        // state only (no model / task queue), so it works during sleep too.
        auto res = create_response(true);
        const std::string session = req.get_param("session");
        std::lock_guard<std::mutex> lk(kv_offload.put_mutex);
        if (session.empty()) {
            // No session: return the state of every known session.
            json sessions = json::object();
            std::set<std::string> all;
            for (const auto & kv : kv_offload.uploaded) all.insert(kv.first);
            for (const auto & kv : kv_offload.session_logical)  all.insert(kv.first);
            for (const auto & s : all) {
                sessions[s] = kv_offload_build_state_json(kv_offload, s, (int64_t) params.kv_cache_size);
            }
            res->ok(sessions);
            return res;
        }
        json state = kv_offload_build_state_json(kv_offload, session, (int64_t) params.kv_cache_size);
        if (state.is_null()) {
            res->error(format_error_response("kv_state: session not found: " + session, ERROR_TYPE_NOT_FOUND));
            return res;
        }
        res->ok(state);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = SERVER_CONTROL_ACTION_REASONING_END;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR,
            session);
    };

    // realtime transcription over websocket, modeled after the OpenAI realtime API
    // audio is PCM16 mono at the model sample rate, sent as base64 in input_audio_buffer.append events
    // input_audio_buffer.commit with "final": true ends the input, closing the socket also ends it
    this->ws_realtime = [this](const server_http_req & req, server_http_context::ws_conn & conn) {
        auto send_event = [&conn](json ev) {
            ev["event_id"] = "event_" + random_string();
            conn.send(safe_json_to_str(ev));
        };
        auto send_error_event = [&send_event](const std::string & message, const std::string & type) {
            send_event(json {
                {"type", "error"},
                {"error", {
                    {"type",    type},
                    {"code",    nullptr},
                    {"message", message},
                    {"param",   nullptr},
                }},
            });
        };

        // models that answer after the audio hold a conversation, see server-realtime.h
        // live transcription: Voxtral Realtime (no chat), or an "asr" model in a session opened with ?intent=transcription
        const bool live_asr = !ctx_server.asr_prompt.empty() && req.get_param("intent") == "transcription";
        if (ctx_server.mtmd_cap.audio_stream && ctx_server.mtmd_cap.chat && !live_asr) {
            server_realtime_conversation({queue_tasks, queue_results, *meta, params, ctx_server.mctx, ctx_server.mtmd_cap, ctx_server.vocab}, conn);
            return;
        }

        if (!meta->has_audio_stream) {
            send_error_event("The current model does not support realtime transcription.", "invalid_request_error");
            conn.close();
            return;
        }

        const std::string session_id = "sess_" + random_string();
        const int model_rate = mtmd_get_audio_sample_rate(ctx_server.mctx);
        json session_config = json {
            {"id",     session_id},
            {"object", "realtime.transcription_session"},
            {"type",   "transcription"},
            {"audio",  {{"input", {
                {"format",          {{"type", "audio/pcm"}, {"rate", model_rate}}},
                {"transcription",   {{"model", meta->model_name}}},
                {"turn_detection",  nullptr},
                {"noise_reduction", nullptr},
            }}}},
        };

        auto res = create_response();
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_TRANSCRIBE_RT);
            task.id = rd.get_new_id();
            task.params.sampling          = params.sampling;
            task.params.stream            = true;
            task.params.oaicompat_cmpl_id = session_id;
            task.params.oaicompat_model   = meta->model_name;
            rd.post_task(std::move(task));
        }

        // the slot answers once its audio stream is live, the session starts there
        {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result || result->is_error()) {
                if (result) {
                    send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                }
                conn.close();
                return;
            }
        }
        send_event(json {{"type", "session.created"}, {"session", session_config}});

        // one item per committed audio buffer, deltas are attributed to the current item
        std::mutex  mutex_item;
        std::string item_id = "item_" + random_string();
        std::string item_prev; // the committed item before item_id, empty for the first one
        std::string item_text;

        // called with mutex_item held
        auto send_completed = [&]() {
            send_event(json {
                {"type",          "conversation.item.input_audio_transcription.completed"},
                {"item_id",       item_id},
                {"content_index", 0},
                {"transcript",    item_text},
            });
            item_text.clear();
        };

        auto post_control = [&](server_control_action action, std::vector<float> && audio) {
            server_response_reader ctl(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id                     = ctl.get_new_id();
            task.params.control_action  = action;
            task.params.control_cmpl_id = session_id;
            task.params.control_audio   = std::move(audio);
            ctl.post_task(std::move(task));
            auto result = ctl.next([&conn]() { return !conn.is_open(); });
            auto * control = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
            return control && control->success;
        };

        // client events are read on this thread, the transcript is sent from the thread below
        std::atomic<bool> input_ended{false};
        std::thread reader([&]() {
            int                        input_rate = model_rate;
            mtmd_helper::resampler_ptr resampler; // input_rate -> model_rate
            std::string msg;
            while (conn.read(msg) != 0) {
                json ev;
                try {
                    ev = json::parse(msg);
                } catch (const std::exception &) {
                    send_error_event("invalid JSON", "invalid_request_error");
                    continue;
                }
                const std::string type = json_value(ev, "type", std::string());
                if (type == "session.update" || type == "transcription_session.update") {
                    const json session = json_value(ev, "session", json::object());
                    const json audio   = session.is_object() ? session.value("audio", json::object()) : json::object();
                    const json in      = audio.is_object() ? audio.value("input", json::object()) : json::object();
                    if (in.is_object() && in.contains("format") && in["format"].is_object()) {
                        const int rate = json_value(in["format"], "rate", input_rate);
                        if (rate < 8000 || rate > 192000) {
                            send_error_event("session.update: the input audio rate must be between 8000 and 192000 Hz", "invalid_request_error");
                            continue;
                        }
                        if (rate != input_rate) {
                            input_rate = rate;
                            resampler.reset();
                            session_config["audio"]["input"]["format"]["rate"] = rate;
                        }
                    }
                    send_event(json {{"type", "session.updated"}, {"session", session_config}});
                } else if (type == "input_audio_buffer.append") {
                    std::string pcm;
                    try {
                        pcm = base64::decode(json_value(ev, "audio", std::string()));
                    } catch (const std::exception &) {
                        send_error_event("invalid base64 audio", "invalid_request_error");
                        continue;
                    }
                    std::vector<float> samples(pcm.size() / 2);
                    for (size_t i = 0; i < samples.size(); i++) {
                        int16_t v;
                        memcpy(&v, pcm.data() + 2 * i, sizeof(v));
                        samples[i] = v / 32768.0f;
                    }
                    if (!resampler) {
                        resampler.reset(mtmd_helper_resampler_init(input_rate, model_rate));
                        if (!resampler) {
                            send_error_event("failed to initialize the audio resampler", "server_error");
                            break;
                        }
                    }
                    const float * out = nullptr;
                    const int64_t n   = mtmd_helper_resampler_process(resampler.get(), samples.data(), samples.size(), &out);
                    if (n < 0) {
                        send_error_event("failed to resample the audio", "server_error");
                        break;
                    }
                    if (n == 0) {
                        continue;
                    }
                    if (out != samples.data()) {
                        samples.assign(out, out + n);
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_APPEND, std::move(samples))) {
                        break;
                    }
                } else if (type == "input_audio_buffer.commit") {
                    // "final": true ends the input, the last item is completed when the stream drains
                    if (json_value(ev, "final", false)) {
                        input_ended = true;
                        post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
                        continue;
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_COMMIT, {})) {
                        break;
                    }
                    // completed is sent once the rows of the committed audio are decoded (audio_commit_drained below)
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_event(json {
                        {"type",             "input_audio_buffer.committed"},
                        {"previous_item_id", item_prev.empty() ? json(nullptr) : json(item_prev)},
                        {"item_id",          item_id},
                    });
                } else if (type == "input_audio_buffer.clear") {
                    send_event(json {{"type", "input_audio_buffer.cleared"}});
                } else {
                    send_error_event("unknown event type: " + type, "invalid_request_error");
                }
            }
            if (!input_ended.exchange(true)) {
                post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
            }
            // the result loop waits on the socket, nobody reads it from here on
            conn.close();
        });

        while (true) {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result) {
                break; // connection closed
            }
            if (result->is_error()) {
                send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                break;
            }
            if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                if (partial->audio_commit_drained) {
                    // complete the committed item and open the next one
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_completed();
                    item_prev = item_id;
                    item_id   = "item_" + random_string();
                } else if (!partial->content.text.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_item);
                    item_text += partial->content.text;
                    send_event(json {
                        {"type",          "conversation.item.input_audio_transcription.delta"},
                        {"item_id",       item_id},
                        {"content_index", 0},
                        {"delta",         partial->content.text},
                    });
                }
            } else if (result->is_stop()) {
                std::lock_guard<std::mutex> lock(mutex_item);
                send_completed();
                break;
            }
        }

        input_ended = true;
        conn.close();
        reader.join();
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        return handle_rerank_impl(req);
    };

    this->post_tts = [this](const server_http_req & req) {
        return handle_tts_impl(req, parse_tts_body(req));
    };

    this->post_speech_oai = [this](const server_http_req & req) {
        json body = convert_speech_to_tts(parse_tts_body(req));
        SRV_DBG("%s\n", "Request converted: OpenAI Speech -> TTS");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        return handle_tts_impl(req, std::move(body));
    };

    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.decision;
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);

        std::vector<raw_buffer> files;
        const json state = decision.prepare_state(decision.parse_state(body, files));
        if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image)) {
            res->error(format_error_response("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // one task per variant of each question, or one task for all the questions
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, files, ctx_server.mctx, ctx_server.init_opt, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }
            if (decision.can_share_prompt()) {
                tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
        }

        auto all_results = rd.wait_for_all(req.should_stop);

        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json answers = json::object();
        int32_t n_tokens = 0;
        size_t i_result = 0;
        size_t i_score  = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                // one result with the scores of all the questions, in order
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[0].get());
                GGML_ASSERT(result != nullptr && i_score + question.options.size() <= result->scores.size());
                scores.emplace_back(result->scores.begin() + i_score, result->scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
                n_tokens = result->n_tokens;
            }
            for (size_t variant = 0; !decision.is_joint() && variant < decision.n_variants(question); variant++) {
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[i_result++].get());
                GGML_ASSERT(result != nullptr);
                scores.push_back(result->scores);
                n_tokens += result->n_tokens;
            }
            answers[question.id] = decision.format_answer(question, scores, !files.empty());
        }

        res->ok(json{
            {"model",   meta->model_name},
            {"answers", answers},
            {"usage",   {
                {"input_tokens",  n_tokens},
                {"output_tokens", 0},
            }},
        });
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.vocab, ctx_server.mctx, meta->chat_params, p.at("content"), true, true, ctx_server.init_opt);
        }
        return tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt);
    };

    // dispatch to the appropriate tokenizer based on the shape of the input
    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_rerank_impl(const server_http_req & req) {
    auto res = create_response();
    if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const json body = json::parse(req.body);

    // if true, use TEI API format, otherwise use Jina API format
    // Jina: https://jina.ai/reranker/
    // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    bool is_tei_format = body.contains("texts");

    // expand one rerank input item: plain string or {"text": str|[str], "image": url|data-uri|raw-base64|[urls]}
    auto parse_item = [&](const json & item, std::string & out_text, std::vector<raw_buffer> & out_files) {
        if (item.is_string()) {
            out_text = item.get<std::string>();
            return;
        }
        if (!item.is_object()) {
            throw std::invalid_argument("rerank input must be a string or an object of the form {\"text\": ..., \"image\": ...}");
        }
        if (item.contains("video") || item.contains("fps") || item.contains("max_frames")) {
            throw std::invalid_argument("video/fps/max_frames are not supported for rerank input");
        }
        json text = item.value("text", json());
        if (text.is_string()) {
            out_text += text.get<std::string>();
        } else if (text.is_array()) {
            for (const auto & t : text) {
                if (!t.is_string()) {
                    throw std::invalid_argument("\"text\" array elements must be strings");
                }
                out_text += t.get<std::string>();
            }
        } else if (!text.is_null()) {
            throw std::invalid_argument("\"text\" must be a string or an array of strings");
        }
        json image = item.value("image", json());
        auto add_image = [&](const std::string & url) {
            // marker is appended by format_prompt_rerank, one per decoded file
            try {
                handle_media(out_files, url, params.media_path, params.media_allow_private_urls);
            } catch (const std::runtime_error & e) {
                throw std::invalid_argument(e.what());
            }
        };
        if (image.is_string()) {
            add_image(image.get<std::string>());
        } else if (image.is_array()) {
            for (const auto & u : image) {
                if (!u.is_string()) {
                    throw std::invalid_argument("\"image\" array elements must be strings");
                }
                add_image(u.get<std::string>());
            }
        } else if (!image.is_null()) {
            throw std::invalid_argument("\"image\" must be a string or an array of strings");
        }
    };

    // Parse query
    std::string query_str;
    std::vector<raw_buffer> query_files;
    if (body.count("query") == 1) {
        try {
            parse_item(body.at("query"), query_str, query_files);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else {
        res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Resolve Jina/TEI format ambiguity
    json documents_json = json_value(body, "documents",
        json_value(body, "texts", json::array()));
    if (!documents_json.is_array() || documents_json.empty()) {
        res->error(format_error_response("\"documents\" must be a non-empty array of strings or objects", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Parse documents
    std::vector<std::string>          documents(documents_json.size());
    std::vector<std::vector<raw_buffer>> document_files(documents_json.size());
    for (size_t i = 0; i < documents_json.size(); i++) {
        try {
            parse_item(documents_json[i], documents[i], document_files[i]);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int top_n = json_value(body, "top_n", (int)documents.size());
    if (top_n < 0) {
        res->error(format_error_response("\"top_n\" must be >= 0", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        tasks.reserve(documents.size());
        for (size_t i = 0; i < documents.size(); i++) {
            auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query_str, documents[i], query_files, document_files[i], ctx_server.init_opt);
            server_task task = server_task(SERVER_TASK_TYPE_RERANK);
            task.id     = rd.get_new_id();
            task.tokens = std::move(tmp);
            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = format_response_rerank(
        body,
        meta->model_name,
        responses,
        is_tei_format,
        documents,
        top_n);

    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_tts_impl(const server_http_req & req, json body) {
    auto res = create_response();
    res->set_req(&req); // will also set spipe if needed

    if (!ctx_server.has_cap_tts()) {
        res->error(format_error_response("this server does not support audio generation", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const auto info = mtmd_gen_audio_get_info(ctx_server.mctx);

    std::string prompt = json_value(body, "input", json_value(body, "prompt", std::string()));
    if (prompt.empty()) {
        res->error(format_error_response("\"input\" must be a non-empty string", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const std::string response_format = json_value(body, "response_format", std::string("wav"));
    const bool pcm16 = response_format == "pcm16";

    server_task task(SERVER_TASK_TYPE_TTS);
    server_schema::eval_tts_schema(params, task, body);
    const bool stream = task.params.stream;
    // a model with a "tts" chat template (e.g. LFM3-Audio) speaks the prompt rendered from it
    const std::string lang       = json_value(body, "lang", params.tts_lang);
    // speaker_ref in the request overrides the default voice
    const bool has_speaker_ref   = req.files.count("speaker_ref") > 0 || !json_value(body, "speaker_ref_b64", std::string()).empty();
    const std::string voice      = json_value(body, "voice", has_speaker_ref ? std::string() : params.tts_voice);
    const std::string tts_prompt = common_chat_get_tts_prompt(meta->chat_params.tmpls.get(), prompt, lang, voice);
    // a model with predefined voices only speaks with one of them
    mtmd::bitmap_ptr voice_ref;
    if (!voice.empty() && info.n_voices > 0) {
        voice_ref.reset(mtmd_gen_audio_get_voice(ctx_server.mctx, voice.c_str()));
        if (!voice_ref) {
            res->error(format_error_response("unknown voice \"" + voice + "\", see \"voices\" in GET /props", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }
    task.tts_inp.set_prompt(tts_prompt.empty() ? prompt : tts_prompt);
    task.tts_inp.set_lang(lang);
    task.tts_inp.data.stream    = stream;
    task.tts_inp.data.out_type  = response_format == "pcm" || pcm16
        ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM
        : MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    // note: -1 is clamped to 0 by the sampler, it will disable the penalty
    task.params.sampling.penalty_last_n = task.params.n_predict > 0 ? task.params.n_predict : 512;
    task.params.sampling.seed = task.tts_inp.data.seed; // same seed for the codec/vocoder RNG
    if (task.tts_inp.data.top_k > 0) {
        task.params.sampling.top_k = task.tts_inp.data.top_k;
    }
    if (task.tts_inp.data.top_p > 0) {
        task.params.sampling.top_p = task.tts_inp.data.top_p;
    }
    if (task.tts_inp.data.temp >= 0) {
        task.params.sampling.temp = task.tts_inp.data.temp;
    }

    // speaker reference: either an uploaded form file ("speaker_ref") or a base64 JSON field ("speaker_ref_b64")
    const unsigned char * speaker_ref_data = nullptr;
    size_t speaker_ref_len  = 0;
    std::string speaker_ref_b64_decoded;

    auto speaker_ref_file = req.files.find("speaker_ref");
    if (speaker_ref_file != req.files.end()) {
        speaker_ref_data = speaker_ref_file->second.data.data();
        speaker_ref_len  = speaker_ref_file->second.data.size();
    } else {
        std::string speaker_ref_b64 = json_value(body, "speaker_ref_b64", std::string());
        if (!speaker_ref_b64.empty()) {
            try {
                speaker_ref_b64_decoded = base64::decode(speaker_ref_b64);
            } catch (const std::exception &) {
                res->error(format_error_response("\"speaker_ref_b64\" is not valid base64", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            speaker_ref_data = (const unsigned char *) speaker_ref_b64_decoded.data();
            speaker_ref_len  = speaker_ref_b64_decoded.size();
        }
    }

    if (speaker_ref_len > 0) {
        if (voice_ref) {
            res->error(format_error_response("\"voice\" and \"speaker_ref\" cannot be used together", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!mtmd_support_audio(ctx_server.mctx)) {
            res->error(format_error_response("this model has no voice encoder, \"speaker_ref\" cannot be used", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, speaker_ref_data, speaker_ref_len, false, ctx_server.init_opt);
        if (!wrapper.bitmap) {
            res->error(format_error_response("failed to decode \"speaker_ref\"", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(wrapper.bitmap));
    } else if (voice_ref) {
        task.tts_inp.set_speaker_ref(std::move(voice_ref));
    } else if (info.n_voices > 0) {
        // without a voice prompt the model produces near silence: the first voice is the default
        const char * first = mtmd_gen_audio_voice_name(ctx_server.mctx, 0);
        SRV_WRN("no voice given, speaking with \"%s\"\n", first);
        task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(mtmd_gen_audio_get_voice(ctx_server.mctx, first)));
    } else {
        SRV_WRN("%s", "no speaker reference provided, the model may behave randomly\n");
    }

    auto & rd = res->rd;
    task.id = rd.get_new_id();
    rd.post_task(std::move(task));

    // raw LE mono samples; audio/L16 would imply 16-bit big-endian (RFC 2586)
    const std::string pcm_type = "audio/pcm;rate=" + std::to_string(info.sample_rate);
    const std::string content_type = pcm16 ? pcm_type + ";encoding=signed-int;bits=16"
        : response_format == "pcm" ? pcm_type + ";encoding=float;bits=32"
        : "audio/wav";

    // the pipeline outputs float32 samples
    auto take_audio = [pcm16](server_task_result_tts * tts_res) {
        const std::string & audio = tts_res->audio;
        return pcm16 ? pcm_f32_to_s16((const float *) audio.data(), audio.size() / sizeof(float)) : std::move(tts_res->audio);
    };

    if (!stream) {
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
        GGML_ASSERT(tts_res != nullptr);
        res->status       = 200;
        res->content_type = content_type;
        res->data         = take_audio(tts_res);
        return res;
    } else {
        auto first_result = rd.next(req.should_stop);
        if (!first_result) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }
        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }
        auto * first_tts_res = dynamic_cast<server_task_result_tts *>(first_result.get());
        GGML_ASSERT(first_tts_res != nullptr);

        res->status       = 200;
        res->content_type = content_type;
        res->data          = take_audio(first_tts_res);
        bool is_done       = first_tts_res->final;

        res->set_next([res_this = res.get(), is_done, take_audio](std::string & output) mutable -> bool {
            // flush buffered audio before is_done - the first result can already be final
            if (!res_this->data.empty()) {
                output = std::move(res_this->data);
                res_this->data.clear();
                return true;
            }
            if (is_done) {
                return false;
            }
            if (res_this->should_stop()) {
                return false;
            }

            server_response_reader & rd = res_this->rd;
            if (!rd.has_next()) {
                return false;
            }
            auto result = rd.next([&res_this]() { return res_this->should_stop(); });
            if (!result || result->is_error()) {
                return false;
            }
            auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
            GGML_ASSERT(tts_res != nullptr);
            output  = take_audio(tts_res);
            is_done = tts_res->final;
            return true;
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
