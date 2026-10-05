#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>

static inline bool rc_is_ident(unsigned char c) { 
    return std::isalnum(c) || c == '_' || c == ':' || c == '-'; 
}
static inline bool rc_is_blank(unsigned char c) { return c == ' ' || c == '\t'; }

static std::string rc_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// "<tool_call>" / "</think>" / "<function=" -> "tool_call" / "think" / "function"
static std::string rc_tag_ident(const char * t) {
    size_t p = 1;
    if (t[p] == '/') p++;
    size_t q = p;
    while (t[q] && rc_is_ident((unsigned char) t[q])) q++;
    return std::string(t + p, q - p);
}

// 치환 대상 태그 이름: 수동 목록(vocab에 없는 커스텀 태그 포함) + vocab의 control/user-defined 토큰
static const std::unordered_set<std::string> & recall_tag_names(const llama_vocab * vocab) {
    static const std::unordered_set<std::string> names = [vocab]() {
        std::unordered_set<std::string> s = {
            "tool_call", "tool_response", "tools", "function", "parameter",
            "think", "thinking", "invoke",
            "function_calls", "function_results", "tool_use", "tool_result",
            "focus", "local", "global", "recap", "recall",
            "qwen:user-prompt-submit-context"
        };
        const int32_t n = llama_vocab_n_tokens(vocab);
        for (int32_t i = 0; i < n; i++) {
            const auto attr = llama_vocab_get_attr(vocab, i);
            if (!(attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED))) continue;
            const char * t = llama_vocab_get_text(vocab, i);
            if (!t || t[0] != '<' || t[1] == '\0' || t[1] == '|') continue;  // <|...|>는 스캐너가 일괄 처리
            std::string id = rc_tag_ident(t);
            if (id.size() >= 3) s.insert(rc_lower(id));  // <s>, <br> 같은 짧은 이름은 오탐 방지로 제외
        }
        return s;
    }();
    return names;
}

// s[i] == '<' 위치에서 태그를 시도. 성공하면 out에 치환 결과를 붙이고 next에 다음 위치를 돌려준다.
static bool rc_sanitize_tag_at(const std::string & s, size_t i,
                               const std::unordered_set<std::string> & names,
                               std::string & out, size_t & next) {
    const size_t n = s.size();

    // 1) <|im_end|>, <|im_start|> 등 전부
    if (i + 1 < n && s[i + 1] == '|') {
        size_t j = i + 2;
        while (j < n && j - i < 64 && rc_is_ident((unsigned char) s[j])) j++;
        if (j > i + 2 && j + 1 < n && s[j] == '|' && s[j + 1] == '>') {
            out += "[past_";
            out.append(s, i + 2, j - (i + 2));
            out += ']';
            next = j + 2;
            return true;
        }
        return false;
    }

    const bool closing = (i + 1 < n && s[i + 1] == '/');
    const size_t p = i + 1 + (closing ? 1 : 0);
    size_t q = p;
    while (q < n && rc_is_ident((unsigned char) s[q])) q++;
    // 식별자를 끝까지 읽어서 비교하므로 <locale>, <thinking2> 등은 자동으로 제외된다
    if (q == p || !names.count(rc_lower(s.substr(p, q - p)))) return false;

    // 2) 닫는 태그: </focus>
    if (closing) {
        size_t k = q;
        while (k < n && rc_is_blank((unsigned char) s[k])) k++;
        if (k < n && s[k] == '>') {
            out += "[past_end_";
            out.append(s, p, q - p);
            out += ']';
            next = k + 1;
            return true;
        }
        return false;
    }

    // 3) 여는 태그: 식별자 바로 뒤의 '<'는 제네릭/비교식으로 보고 제외 (vector<function>, a<global)
    if (i > 0 && (rc_is_ident((unsigned char) s[i - 1]) || s[i - 1] == ':')) return false;

    size_t k = q;
    if (k < n && s[k] == '=') {
        // <function=name>, <parameter=key>
        k++;
        while (k < n && s[k] != '>' && s[k] != '<' && !std::isspace((unsigned char) s[k])) k++;
    } else {
        // <focus>, <focus magic_chunks="3" ...>
        while (true) {
            size_t a = k;
            while (a < n && rc_is_blank((unsigned char) s[a])) a++;
            if (a == k) break;
            size_t b = a;
            while (b < n && rc_is_ident((unsigned char) s[b])) b++;
            if (b == a || b + 1 >= n || s[b] != '=' || s[b + 1] != '"') break;
            size_t c = b + 2;
            while (c < n && s[c] != '"' && s[c] != '\n') c++;
            if (c >= n || s[c] != '"') break;
            k = c + 1;
        }
    }
    size_t e = k;
    while (e < n && rc_is_blank((unsigned char) s[e])) e++;
    if (e >= n || s[e] != '>') return false;  // `a<global && b>c` 같은 코드는 여기서 탈락

    out += "[past_";
    out.append(s, p, k - p);   // 이름 + 속성 원문 그대로
    out += ']';
    next = e + 1;
    return true;
}

static std::string sanitize_recalled_text(const std::string & s,
                                          const std::unordered_set<std::string> & names) {
    std::string out;
    out.reserve(s.size() + s.size() / 16);
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        if (s[i] != '<') {
            size_t j = s.find('<', i);
            if (j == std::string::npos) j = n;
            out.append(s, i, j - i);
            i = j;
            continue;
        }
        size_t next = 0;
        if (rc_sanitize_tag_at(s, i, names, out, next)) {
            i = next;
        } else {
            out += '<';
            i++;
        }
    }
    return out;
}