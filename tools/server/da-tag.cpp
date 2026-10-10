#include "da-tag.h"

bool da_tag_start_allowed(const std::string & text, size_t lt, da_mode_t mode) {
    (void) mode;  // the start rule no longer depends on the mode (S1-CLOSE)
    if (lt == 0) {
        return true;
    }
    return (unsigned char) text[lt - 1] == '\n';
}

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
            const size_t close = text.find('>', lt + 6);
            if (close != std::string::npos) {
                const std::string tag = text.substr(lt, close - lt + 1);
                const size_t q0 = tag.find("magic_chunks");
                if (q0 != std::string::npos) {
                    // every number run after the attribute name - the
                    // tag may keep several chunks (magic_chunks="1,3")
                    for (size_t q = q0 + 14; q < tag.size(); q++) {
                        if (std::isdigit((unsigned char) tag[q])) {
                            int32_t v = 0;
                            do {
                                v = v * 10 + (tag[q] - '0');
                                q++;
                            } while (q < tag.size() && std::isdigit((unsigned char) tag[q]));
                            q--;  // the loop increment steps past the run
                            cand.keep_nums.push_back(v);
                        }
                    }
                    if (!cand.keep_nums.empty()) {
                        cand.type = 0;
                        cand.end  = close + 1;
                    }
                }
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
    static const char head[] = "<focus magic_chunks=";
    const size_t hl = sizeof(head) - 1;
    if (s.size() < hl && s.compare(0, s.size(), head, s.size()) == 0) {
        return true;
    }
    if (s.size() >= hl && s.compare(0, hl, head) == 0) {
        size_t i = hl;
        if (i < s.size() && s[i] == '"') {
            i++;
        }
        if (i == s.size()) {
            return true;  // attribute name done, waiting for the number
        }
        // Number list: digit runs separated by commas, mirroring the
        // grammar scan_da_tag() parses (magic_chunks="12,13" keeps
        // several chunks). A comma opens the next run, which may still
        // be empty; a closing quote is only valid once a run has
        // arrived, and only the final '>' may follow it. 09-25 19:29
        // (task 5952): the old single-run check released the hold at
        // the comma, streaming the tag head to the client before the
        // complete tag could be erased at the closing token.
        bool in_list = false;  // a digit or comma has arrived
        for (; i < s.size(); i++) {
            const unsigned char c = (unsigned char) s[i];
            if (std::isdigit(c) || c == ',') {
                in_list = true;
                continue;  // inside the (possibly multi-run) list
            }
            if (c == '"' && in_list) {
                // closing quote - only the final '>' may follow
                return i + 1 == s.size();
            }
            return false;  // a '>' (complete) or any other character
        }
        return true;  // number list open, the tag may still close
    }
    return false;
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
