#include "gbnf-to-peg.h"

#include "peg-parser.h"

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ──────────────────────────────────────────────────────────────────────────────
// Minimal GBNF lexer / parser
//
// GBNF format: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
//   rule-name ::= body
//   body: sequence ("|" sequence)*
//   sequence: item+
//   item: atom suffix?
//   suffix: "*" | "+" | "?"
//   atom: NAME | "literal" | [char-class] | "(" body ")"
// ──────────────────────────────────────────────────────────────────────────────

struct GbnfParser {
    const std::string &        src;
    size_t                     pos = 0;
    common_peg_parser_builder & builder;

    GbnfParser(const std::string & s, common_peg_parser_builder & b) : src(s), builder(b) {}

    char peek(size_t off = 0) const {
        return pos + off < src.size() ? src[pos + off] : '\0';
    }

    void advance() { if (pos < src.size()) ++pos; }

    void skip_ws() {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' ||
               src[pos] == '\r' || src[pos] == '\n')) {
            advance();
        }
    }

    void skip_ws_no_nl() {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\r')) {
            advance();
        }
    }

    void skip_line() {
        while (pos < src.size() && src[pos] != '\n') advance();
    }

    bool at_end() const { return pos >= src.size(); }

    std::string read_name() {
        // GBNF rule names can contain a-z, A-Z, 0-9, '-', '_'
        std::string result;
        while (pos < src.size() &&
               (std::isalnum((unsigned char)src[pos]) || src[pos] == '-' || src[pos] == '_')) {
            result += src[pos]; advance();
        }
        return result;
    }

    std::string read_string() {
        // Opening '"' already consumed
        std::string result;
        while (pos < src.size()) {
            char c = src[pos];
            if (c == '"') { advance(); break; }
            if (c == '\\') {
                advance();
                if (at_end()) break;
                char esc = src[pos];
                switch (esc) {
                    case 'n':  result += '\n'; break;
                    case 't':  result += '\t'; break;
                    case 'r':  result += '\r'; break;
                    case '\\': result += '\\'; break;
                    case '"':  result += '"';  break;
                    default:   result += '\\'; result += esc; break;
                }
                advance();
            } else {
                result += c; advance();
            }
        }
        return result;
    }

    // Read character class content: everything inside [ ... ]
    std::string read_char_class() {
        // Opening '[' already consumed
        std::string result = "[";
        int depth = 1;
        while (pos < src.size() && depth > 0) {
            char c = src[pos];
            if (c == ']') {
                --depth;
                result += c; advance();
            } else if (c == '\\') {
                result += c; advance();
                if (!at_end()) { result += src[pos]; advance(); }
            } else {
                if (c == '[') ++depth;
                result += c; advance();
            }
        }
        return result;
    }

    // Parse alternatives (top-level rule body)
    common_peg_parser parse_alternatives() {
        std::vector<common_peg_parser> alts;
        alts.push_back(parse_sequence());

        skip_ws_no_nl();
        while (!at_end() && peek() == '|') {
            advance(); // '|'
            skip_ws_no_nl();
            alts.push_back(parse_sequence());
            skip_ws_no_nl();
        }

        if (alts.size() == 1) return alts[0];
        return builder.choice(alts);
    }

    common_peg_parser parse_sequence() {
        std::vector<common_peg_parser> items;

        skip_ws_no_nl();
        while (!at_end()) {
            char c = peek();
            // Stop at '|', ')', or end of line (but not inside a group)
            if (c == '|' || c == ')' || c == '\n') break;
            // Also stop at '#' comment
            if (c == '#') { skip_line(); break; }

            items.push_back(parse_item());
            skip_ws_no_nl();
        }

        if (items.empty())     return builder.eps();
        if (items.size() == 1) return items[0];
        return builder.sequence(items);
    }

    common_peg_parser parse_item() {
        auto atom = parse_atom();

        // Suffix
        if (!at_end()) {
            char c = peek();
            if (c == '*') { advance(); return builder.zero_or_more(atom); }
            if (c == '+') { advance(); return builder.one_or_more(atom); }
            if (c == '?') { advance(); return builder.optional(atom); }
        }

        return atom;
    }

    common_peg_parser parse_atom() {
        skip_ws_no_nl();
        if (at_end()) {
            throw std::runtime_error("gbnf-to-peg: unexpected end of input");
        }

        char c = peek();

        if (c == '"') {
            advance();
            std::string val = read_string();
            if (val.empty()) return builder.eps();
            return builder.literal(val);
        }

        if (c == '[') {
            advance();
            std::string cls = read_char_class();
            return builder.chars(cls);
        }

        if (c == '(') {
            advance();
            skip_ws_no_nl();
            auto inner = parse_alternatives();
            skip_ws_no_nl();
            if (at_end() || peek() != ')') {
                throw std::runtime_error("gbnf-to-peg: expected ')' to close group");
            }
            advance();
            return inner;
        }

        if (std::isalpha((unsigned char)c) || c == '_') {
            std::string name = read_name();
            return builder.ref(name);
        }

        throw std::runtime_error(
            std::string("gbnf-to-peg: unexpected character '") + c + "' at pos " + std::to_string(pos));
    }

    void parse_all() {
        std::string start_rule;
        bool        root_set = false;

        skip_ws();
        while (!at_end()) {
            // Skip blank lines and comments
            if (peek() == '\n') { advance(); skip_ws(); continue; }
            if (peek() == '#')  { skip_line(); skip_ws(); continue; }

            // Read rule name
            if (!std::isalpha((unsigned char)peek()) && peek() != '_') {
                skip_line(); skip_ws(); continue;
            }

            std::string name = read_name();
            if (name.empty()) { skip_line(); skip_ws(); continue; }

            skip_ws_no_nl();

            // Expect '::='
            if (pos + 2 >= src.size() || src.substr(pos, 3) != "::=") {
                skip_line(); skip_ws(); continue;
            }
            pos += 3;
            skip_ws_no_nl();

            // Parse body (may span multiple lines via '|' continuation)
            // We read until end-of-line; continuation lines start with whitespace then '|'
            common_peg_parser body = parse_alternatives();

            // Handle multi-line rules: next line may start with '|' (with leading whitespace)
            while (!at_end() && peek() == '\n') {
                advance(); // consume newline
                // Peek: is this a continuation (whitespace then '|')?
                size_t saved = pos;
                while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t')) advance();
                if (!at_end() && peek() == '|') {
                    // Parse continuation alternatives
                    std::vector<common_peg_parser> alts = {body};
                    while (!at_end() && peek() == '|') {
                        advance(); // '|'
                        skip_ws_no_nl();
                        alts.push_back(parse_sequence());
                        if (!at_end() && peek() == '\n') {
                            advance();
                            size_t s2 = pos;
                            while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t')) advance();
                            if (at_end() || peek() != '|') { pos = (peek() == '\n') ? pos : s2; break; }
                        } else {
                            break;
                        }
                    }
                    body = alts.size() == 1 ? alts[0] : builder.choice(alts);
                } else {
                    pos = saved; // restore — not a continuation
                    break;
                }
            }

            auto rule_ref = builder.rule(name, body);

            bool is_terminal = !name.empty() && std::isupper((unsigned char)name[0]);
            if (!root_set && !is_terminal) {
                start_rule = name;
                root_set   = true;
                builder.set_root(rule_ref);
            }

            skip_ws();
        }
        (void)start_rule; // suppresses unused-variable warning
    }
};

} // namespace

common_peg_arena common_gbnf_to_peg(const std::string & gbnf_grammar) {
    common_peg_parser_builder builder;
    GbnfParser                parser(gbnf_grammar, builder);
    parser.parse_all();
    return builder.build();
}
