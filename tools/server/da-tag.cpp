#include "da-tag.h"

bool da_tag_start_allowed(const std::string & text, size_t lt, da_mode_t mode) {
    (void) mode;  // the start rule no longer depends on the mode (S1-CLOSE)
    if (lt == 0) {
        return true;
    }
    return (unsigned char) text[lt - 1] == '\n';
}

namespace {
// The single <focus magic_chunks=...> grammar, shared by scan_da_tag() and
// da_tag_prefix() so the consume side and the hold side can never diverge on
// what counts as a focus tag (the 0602ad47c sync rule). The model is
// instructed to emit the canonical <focus magic_chunks="N"> form, but it
// occasionally imitates variants - single quotes, extra whitespace (even a
// newline) in the head, no quotes at all. Every variant the scanner accepts
// must be held by the prefix while it is being emitted, or the tag text
// leaks to the client before the tag is consumed. Hence the lenient grammar:
//   <focus W* magic_chunks W* = W* Q? N (, W* N)* Q? W* >
//   W = any isspace char, Q = " or ' (a matching pair, optional),
//   N = a run of digits (the tag may keep several chunks: "12,13").
// A comma opens the next run, which may still be empty - the hold must not
// release at the comma (09-25 19:29, task 5952).
enum focus_parse_t { FOCUS_TAG_NONE, FOCUS_TAG_OPEN, FOCUS_TAG_DONE };

focus_parse_t parse_focus_tag(const std::string & s, size_t from,
                              std::vector<int32_t> & nums, size_t & end) {
    static const char head[] = "<focus";
    static const char attr[] = "magic_chunks";
    const size_t n = s.size();
    size_t i = from;
    // <focus - a shorter suffix at end-of-string is a live prefix
    if (n - i < 6) {
        if (s.compare(i, n - i, head, n - i) != 0) return FOCUS_TAG_NONE;
        return FOCUS_TAG_OPEN;
    }
    if (s.compare(i, 6, head) != 0) return FOCUS_TAG_NONE;
    i += 6;
    // a longer tag name (<focusx ...) is data, not a focus tag
    if (i < n && (std::isalnum((unsigned char) s[i]) || s[i] == '_' || s[i] == '-' || s[i] == '.')) {
        return FOCUS_TAG_NONE;
    }
    while (i < n && std::isspace((unsigned char) s[i])) i++;
    // magic_chunks - a partial name at end-of-string is still open
    if (n - i < 12) {
        if (s.compare(i, n - i, attr, n - i) != 0) return FOCUS_TAG_NONE;
        return FOCUS_TAG_OPEN;
    }
    if (s.compare(i, 12, attr) != 0) return FOCUS_TAG_NONE;
    i += 12;
    while (i < n && std::isspace((unsigned char) s[i])) i++;
    if (i == n) return FOCUS_TAG_OPEN;
    if (s[i] != '=') return FOCUS_TAG_NONE;
    i++;
    while (i < n && std::isspace((unsigned char) s[i])) i++;
    if (i == n) return FOCUS_TAG_OPEN;
    char quote = 0;
    if (s[i] == '"' || s[i] == '\'') {
        quote = s[i];
        i++;
    }
    if (i == n) return FOCUS_TAG_OPEN;
    bool closed  = false;
    bool has_num = false;
    while (i < n) {
        unsigned char c = (unsigned char) s[i];
        if (std::isdigit(c)) {
            int32_t v = 0;
            do {
                v = v * 10 + (c - '0');
                i++;
                c = (i < n) ? (unsigned char) s[i] : 0;
            } while (std::isdigit(c));
            nums.push_back(v);
            has_num = true;
            continue;
        }
        if (c == ',' || std::isspace(c)) {
            i++;
            continue;
        }
        if (quote != 0 && !closed && c == (unsigned char) quote) {
            closed = true;
            i++;
            continue;
        }
        if (c == '>' && has_num) {
            end = i + 1;
            return FOCUS_TAG_DONE;
        }
        return FOCUS_TAG_NONE;
    }
    return FOCUS_TAG_OPEN;
}
}  // namespace

da_tag_t scan_da_tag(const std::string & text, size_t from, da_mode_t mode) {
    da_tag_t best;
    for (size_t lt = text.find('<', from); lt != std::string::npos; lt = text.find('<', lt + 1)) {
        if (!da_tag_start_allowed(text, lt, mode)) {
            continue;
        }
        da_tag_t cand;
        cand.start = lt;
        if (text.compare(lt, 8, "</focus>") == 0) {
            cand.type = 2;
            cand.end  = lt + 8;
        } else if (text.compare(lt, 8, "</local>") == 0) {
            cand.type = 3;
            cand.end  = lt + 8;
        } else if (text.compare(lt, 9, "</global>") == 0) {
            cand.type = 5;
            cand.end  = lt + 9;
        } else if (text.compare(lt, 8, "<global>") == 0) {
            cand.type = 4;
            cand.end  = lt + 8;
        } else if (text.compare(lt, 7, "<local>") == 0) {
            cand.type = 1;
            cand.end  = lt + 7;
        } else if (text.compare(lt, 6, "<focus") == 0) {
            std::vector<int32_t> nums;
            size_t end = 0;
            if (parse_focus_tag(text, lt, nums, end) == FOCUS_TAG_DONE) {
                cand.type      = 0;
                cand.end       = end;
                cand.keep_nums = std::move(nums);
            }
        }
        if (cand.type >= 0 && (best.type < 0 || cand.end < best.end)) {
            best = cand;
        }
    }
    return best;
}

bool da_tag_prefix(const std::string & s) {
    static const char * const fixed[] = { "<local>", "</focus>", "</local>", "<global>", "</global>" };
    for (const char * t : fixed) {
        const size_t tl = std::strlen(t);
        if (s.size() < tl && s.compare(0, s.size(), t, s.size()) == 0) {
            return true;
        }
    }
    std::vector<int32_t> nums;
    size_t end = 0;
    return parse_focus_tag(s, 0, nums, end) == FOCUS_TAG_OPEN;
}

size_t da_tag_hold_len(const std::string & unsent, da_mode_t mode) {
    size_t lt = std::string::npos;
    for (size_t i = unsent.find('<'); i != std::string::npos; i = unsent.find('<', i + 1)) {
        if (!da_tag_start_allowed(unsent, i, mode)) {
            continue;
        }
        lt = i;
    }
    return (lt != std::string::npos && da_tag_prefix(unsent.substr(lt)))
            ? (unsent.size() - lt) : 0;
}

bool da_tag_inflight(const std::string & text, da_mode_t mode) {
    size_t lt = std::string::npos;
    for (size_t i = text.find('<'); i != std::string::npos; i = text.find('<', i + 1)) {
        if (!da_tag_start_allowed(text, i, mode)) {
            continue;
        }
        lt = i;
    }
    return lt != std::string::npos && da_tag_prefix(text.substr(lt));
}
