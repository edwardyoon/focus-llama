#pragma once

// Recall-text tag sanitizer, extracted from server-context.cpp.
//
// kv_offload_refill re-prefills an offloaded chunk by detokenizing it,
// sanitizing the recalled text (replacing LLM tool/think tags with inert
// [past_...] markers so the model cannot act on them), and re-tokenizing the
// wrapped block. recall_tag_names builds the set of tag names to replace: a
// manual list (custom tags not present in the vocab) plus the vocab's control
// / user-defined tokens.
//
// Free functions, no per-request state. Depends on llama.h (llama_vocab + the
// token attr/text getters); this header forward-declares llama_vocab so it
// stays light, and the real include lives in recall-sanitize.cpp only.

#include <cstddef>
#include <string>
#include <unordered_set>

struct llama_vocab;

// Tag names to replace: the manual list (custom tags absent from the vocab)
// plus the vocab's control / user-defined tokens. Cached per vocab pointer, so
// a model reload (new vocab) rebuilds the set instead of reusing the stale
// one (the old function-local `static` was built once and stayed stale after
// a reload).
const std::unordered_set<std::string> & recall_tag_names(const llama_vocab * vocab);

// Replace recognized tags with inert [past_...] markers (fail-open: an
// unrecognized tag is kept verbatim). `names` is recall_tag_names(vocab).
std::string sanitize_recalled_text(const std::string & s,
                                   const std::unordered_set<std::string> & names);
