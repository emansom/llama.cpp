#include "lark-to-peg.h"

#include "peg-parser.h"

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ──────────────────────────────────────────────────────────────────────────────
// Lexer
// ──────────────────────────────────────────────────────────────────────────────

enum class LarkTok {
    END,
    NEWLINE,
    COLON,
    PIPE,
    STAR,
    PLUS,
    QMARK,
    LPAREN,
    RPAREN,
    TILDE,
    DOT,
    DOTDOT,
    NUMBER,
    STRING,
    REGEX,
    NAME,
    PERCENT,
    COMMA,
};

struct LarkToken {
    LarkTok     type;
    std::string value;
    size_t      line = 0;
    size_t      col  = 0;
};

struct LarkLexer {
    const std::string &    src;
    size_t                 pos   = 0;
    size_t                 line  = 1;
    size_t                 col   = 1;
    std::vector<LarkToken> tokens;
    size_t                 tok_pos = 0;

    explicit LarkLexer(const std::string & s) : src(s) {}

    char peek(size_t offset = 0) const {
        size_t p = pos + offset;
        return p < src.size() ? src[p] : '\0';
    }

    void advance() {
        if (pos < src.size()) {
            if (src[pos] == '\n') { ++line; col = 1; }
            else                  { ++col; }
            ++pos;
        }
    }

    void skip_whitespace_inline() {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\r')) {
            advance();
        }
    }

    void skip_to_eol() {
        while (pos < src.size() && src[pos] != '\n') advance();
    }

    std::string read_string(char delim) {
        std::string result;
        while (pos < src.size()) {
            char c = src[pos];
            if (c == delim) { advance(); break; }
            if (c == '\\') {
                advance();
                if (pos < src.size()) {
                    char esc = src[pos];
                    switch (esc) {
                        case 'n':  result += '\n'; break;
                        case 't':  result += '\t'; break;
                        case 'r':  result += '\r'; break;
                        case '\\': result += '\\'; break;
                        case '"':  result += '"';  break;
                        case '\'': result += '\''; break;
                        case '|':  result += '|';  break;
                        default:   result += '\\'; result += esc; break;
                    }
                    advance();
                }
            } else {
                result += c; advance();
            }
        }
        return result;
    }

    std::string read_regex() {
        std::string result;
        while (pos < src.size()) {
            char c = src[pos];
            if (c == '/') { advance(); break; }
            if (c == '\\') {
                result += c; advance();
                if (pos < src.size()) { result += src[pos]; advance(); }
            } else {
                result += c; advance();
            }
        }
        return result;
    }

    std::string read_name() {
        std::string result;
        while (pos < src.size() && (std::isalnum((unsigned char)src[pos]) || src[pos] == '_')) {
            result += src[pos]; advance();
        }
        return result;
    }

    std::string read_number() {
        std::string result;
        while (pos < src.size() && std::isdigit((unsigned char)src[pos])) {
            result += src[pos]; advance();
        }
        return result;
    }

    std::string read_directive_rest() {
        // Read until end of logical line (handling {...} blocks)
        std::string result;
        int brace = 0;
        while (pos < src.size()) {
            char c = src[pos];
            if (c == '\n' && brace == 0) break;
            if (c == '{') { ++brace; }
            if (c == '}') { --brace; }
            result += c; advance();
        }
        return result;
    }

    void tokenize() {
        while (pos < src.size()) {
            size_t tok_line = line;
            size_t tok_col  = col;
            char   c        = src[pos];

            if (c == ' ' || c == '\t' || c == '\r') { skip_whitespace_inline(); continue; }

            if (c == '#') { skip_to_eol(); continue; }

            if (c == '\n') {
                tokens.push_back({LarkTok::NEWLINE, "\n", tok_line, tok_col});
                advance();
                continue;
            }

            switch (c) {
                case ':': tokens.push_back({LarkTok::COLON,  ":",  tok_line, tok_col}); advance(); continue;
                case '|': tokens.push_back({LarkTok::PIPE,   "|",  tok_line, tok_col}); advance(); continue;
                case '*': tokens.push_back({LarkTok::STAR,   "*",  tok_line, tok_col}); advance(); continue;
                case '+': tokens.push_back({LarkTok::PLUS,   "+",  tok_line, tok_col}); advance(); continue;
                case '?': tokens.push_back({LarkTok::QMARK,  "?",  tok_line, tok_col}); advance(); continue;
                case '(': tokens.push_back({LarkTok::LPAREN, "(",  tok_line, tok_col}); advance(); continue;
                case ')': tokens.push_back({LarkTok::RPAREN, ")",  tok_line, tok_col}); advance(); continue;
                case '~': tokens.push_back({LarkTok::TILDE,  "~",  tok_line, tok_col}); advance(); continue;
                case ',': tokens.push_back({LarkTok::COMMA,  ",",  tok_line, tok_col}); advance(); continue;
                default: break;
            }

            if (c == '.') {
                advance();
                if (pos < src.size() && src[pos] == '.') {
                    advance();
                    tokens.push_back({LarkTok::DOTDOT, "..", tok_line, tok_col});
                } else {
                    tokens.push_back({LarkTok::DOT, ".", tok_line, tok_col});
                }
                continue;
            }

            if (c == '"' || c == '\'') {
                advance();
                std::string val = read_string(c);
                tokens.push_back({LarkTok::STRING, val, tok_line, tok_col});
                continue;
            }

            if (c == '/') {
                advance();
                std::string val = read_regex();
                tokens.push_back({LarkTok::REGEX, val, tok_line, tok_col});
                continue;
            }

            if (c == '%') {
                advance();
                std::string directive = read_name();
                skip_whitespace_inline();
                std::string rest = read_directive_rest();
                tokens.push_back({LarkTok::PERCENT, directive + " " + rest, tok_line, tok_col});
                continue;
            }

            if (std::isdigit((unsigned char)c)) {
                tokens.push_back({LarkTok::NUMBER, read_number(), tok_line, tok_col});
                continue;
            }

            if (std::isalpha((unsigned char)c) || c == '_') {
                tokens.push_back({LarkTok::NAME, read_name(), tok_line, tok_col});
                continue;
            }

            // Skip unknown characters
            advance();
        }
        tokens.push_back({LarkTok::END, "", line, col});
    }

    bool at_end() const {
        return tok_pos >= tokens.size() || tokens[tok_pos].type == LarkTok::END;
    }

    LarkToken & current() { return tokens[tok_pos]; }

    LarkToken consume() {
        LarkToken t = tokens[tok_pos];
        if (tok_pos < tokens.size()) ++tok_pos;
        return t;
    }

    void skip_newlines() {
        while (!at_end() && tokens[tok_pos].type == LarkTok::NEWLINE) ++tok_pos;
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Recursive-descent parser
// ──────────────────────────────────────────────────────────────────────────────

struct LarkParser {
    LarkLexer &                lex;
    common_peg_parser_builder & builder;
    size_t                     tok_end; // stop parsing at this token index

    LarkParser(LarkLexer & l, common_peg_parser_builder & b, size_t end)
        : lex(l), builder(b), tok_end(end) {}

    bool stopped() const { return lex.at_end() || lex.tok_pos >= tok_end; }

    LarkToken & cur() { return lex.tokens[lex.tok_pos]; }

    bool peek_is(LarkTok t) const {
        return !lex.at_end() && lex.tok_pos < tok_end && lex.tokens[lex.tok_pos].type == t;
    }

    LarkToken consume() { return lex.consume(); }

    // Peek at next non-NEWLINE token type
    bool peek_pipe_after_newlines() const {
        size_t j = lex.tok_pos;
        while (j < lex.tokens.size() && lex.tokens[j].type == LarkTok::NEWLINE) ++j;
        return j < lex.tokens.size() && j < tok_end && lex.tokens[j].type == LarkTok::PIPE;
    }

    common_peg_parser parse_alternatives() {
        std::vector<common_peg_parser> alts;
        alts.push_back(parse_sequence());

        while (!stopped() && (peek_is(LarkTok::PIPE) || peek_pipe_after_newlines())) {
            lex.skip_newlines();
            if (stopped() || !peek_is(LarkTok::PIPE)) break;
            consume(); // '|'
            lex.skip_newlines();
            if (stopped()) { alts.push_back(builder.eps()); break; }
            alts.push_back(parse_sequence());
        }

        if (alts.size() == 1) return alts[0];
        return builder.choice(alts);
    }

    common_peg_parser parse_sequence() {
        std::vector<common_peg_parser> items;

        while (!stopped()) {
            LarkTok t = cur().type;
            if (t == LarkTok::PIPE || t == LarkTok::RPAREN || t == LarkTok::NEWLINE) break;
            items.push_back(parse_item());
        }

        if (items.empty())        return builder.eps();
        if (items.size() == 1)    return items[0];
        return builder.sequence(items);
    }

    common_peg_parser parse_item() {
        auto atom = parse_atom();

        if (stopped()) return atom;
        LarkTok t = cur().type;

        if (t == LarkTok::STAR)  { consume(); return builder.zero_or_more(atom); }
        if (t == LarkTok::PLUS)  { consume(); return builder.one_or_more(atom); }
        if (t == LarkTok::QMARK) { consume(); return builder.optional(atom); }
        if (t == LarkTok::TILDE) {
            consume();
            if (stopped() || cur().type != LarkTok::NUMBER) {
                throw std::runtime_error("lark-to-peg: expected number after ~");
            }
            int n = std::stoi(consume().value);
            if (!stopped() && cur().type == LarkTok::DOTDOT) {
                consume();
                if (!stopped() && cur().type == LarkTok::NUMBER) {
                    int m = std::stoi(consume().value);
                    return builder.repeat(atom, n, m);
                }
                return builder.repeat(atom, n, -1);
            }
            return builder.repeat(atom, n, n);
        }

        return atom;
    }

    common_peg_parser parse_atom() {
        if (stopped()) {
            throw std::runtime_error("lark-to-peg: unexpected end of rule");
        }

        LarkToken tok = cur();

        if (tok.type == LarkTok::STRING) {
            consume();
            if (tok.value.empty()) return builder.eps();
            return builder.literal(tok.value);
        }

        if (tok.type == LarkTok::REGEX) {
            consume();
            const std::string & pat = tok.value;
            // Non-greedy .* patterns and complex alternation — map to rest()
            // These are used for llguidance constraint only; PEG extraction uses lenient mode.
            if (pat == "(.|\n)*?" || pat == "[\\s\\S]*?" || pat == "(.|\n)+" || pat == ".*") {
                return builder.rest();
            }
            // Complex regex (contains top-level alternation or grouping) — map to rest()
            // The precise constraint is enforced by llguidance during generation.
            if (!pat.empty() && (pat[0] == '(' || pat.find('|') != std::string::npos)) {
                return builder.rest();
            }
            // Sequence of character classes (and optional simple literals/groups)
            // Walk the pattern token by token, decoding:
            //   - '[…][quant?]'  -> chars(class) with quantifier
            //   - 'literal'      -> literal(string) (a single non-special char)
            //   - '\\.'          -> literal(string) for the escaped char
            //   - '(\.[0-9]+)?'  -> recursively parse the inner shape with quant
            // Anything we can't decode falls back to rest() so llguidance still
            // enforces the precise shape at sampling time.
            std::function<bool(const std::string &, std::vector<common_peg_parser>&)> parse_sequence_shape =
                [&](const std::string & p, std::vector<common_peg_parser>& parts) -> bool {
                size_t i = 0;
                while (i < p.size()) {
                    int  min_n = 1;
                    int  max_n = 1;
                    auto apply_quant = [&](size_t & next) {
                        if (next < p.size()) {
                            if      (p[next] == '*') { min_n = 0; max_n = -1; ++next; }
                            else if (p[next] == '+') { min_n = 1; max_n = -1; ++next; }
                            else if (p[next] == '?') { min_n = 0; max_n = 1;  ++next; }
                        }
                    };
                    if (p[i] == '[') {
                        size_t end = p.find(']', i + 1);
                        if (end == std::string::npos) return false;
                        std::string cls = p.substr(i, end - i + 1);
                        size_t next = end + 1;
                        apply_quant(next);
                        parts.push_back(builder.chars(cls, min_n, max_n));
                        i = next;
                    } else if (p[i] == '(') {
                        size_t depth = 1;
                        size_t end   = i + 1;
                        while (end < p.size() && depth > 0) {
                            if      (p[end] == '\\' && end + 1 < p.size()) end += 2;
                            else if (p[end] == '(') { ++depth; ++end; }
                            else if (p[end] == ')') { --depth; ++end; }
                            else                    ++end;
                        }
                        if (depth != 0) return false;
                        std::string inner = p.substr(i + 1, end - i - 2);
                        size_t next = end;
                        apply_quant(next);
                        std::vector<common_peg_parser> inner_parts;
                        if (!parse_sequence_shape(inner, inner_parts)) return false;
                        common_peg_parser inner_p = inner_parts.size() == 1
                            ? inner_parts[0]
                            : builder.sequence(inner_parts);
                        if (min_n == 1 && max_n == 1) {
                            parts.push_back(inner_p);
                        } else {
                            parts.push_back(builder.repeat(inner_p, min_n, max_n));
                        }
                        i = next;
                    } else if (p[i] == '\\' && i + 1 < p.size()) {
                        std::string lit(1, p[i + 1]);
                        size_t next = i + 2;
                        apply_quant(next);
                        common_peg_parser lit_p = builder.literal(lit);
                        if (min_n == 1 && max_n == 1) parts.push_back(lit_p);
                        else                          parts.push_back(builder.repeat(lit_p, min_n, max_n));
                        i = next;
                    } else if (p[i] != '*' && p[i] != '+' && p[i] != '?' && p[i] != '|' && p[i] != '.') {
                        std::string lit(1, p[i]);
                        size_t next = i + 1;
                        apply_quant(next);
                        common_peg_parser lit_p = builder.literal(lit);
                        if (min_n == 1 && max_n == 1) parts.push_back(lit_p);
                        else                          parts.push_back(builder.repeat(lit_p, min_n, max_n));
                        i = next;
                    } else {
                        return false;
                    }
                }
                return true;
            };
            std::vector<common_peg_parser> parts;
            if (!pat.empty() && parse_sequence_shape(pat, parts) && !parts.empty()) {
                if (parts.size() == 1) return parts[0];
                return builder.sequence(parts);
            }
            // Single character class with no quantifier — use chars()
            return builder.chars(pat);
        }

        if (tok.type == LarkTok::NAME) {
            consume();
            const std::string & raw = tok.value;
            // Magic markers injected by chat_grammar_to_peg() for PEG extraction.
            // The chat-grammar runtime substitutes {{TOOL_SCHEMA}} and
            // {{RESPONSE_SCHEMA}} placeholders with these names so the
            // post-generation PEG parser matches any JSON object / value
            // (the precise schema constraint is enforced by llguidance/GBNF
            // at sampling time).
            if (raw == "__JSON_OBJECT__") return builder.json_object();
            if (raw == "__JSON_VALUE__")  return builder.json();
            // Normalize underscores to hyphens to match builder.rule() clean_name behavior
            std::string name = raw;
            for (char & c : name) { if (c == '_') c = '-'; }
            return builder.ref(name);
        }

        if (tok.type == LarkTok::LPAREN) {
            consume();
            lex.skip_newlines();
            auto inner = parse_alternatives();
            lex.skip_newlines();
            if (stopped() || cur().type != LarkTok::RPAREN) {
                throw std::runtime_error("lark-to-peg: expected ')' to close group");
            }
            consume();
            return inner;
        }

        if (tok.type == LarkTok::DOT) {
            consume();
            return builder.any();
        }

        throw std::runtime_error(
            std::string("lark-to-peg: unexpected token '") + tok.value +
            "' (type " + std::to_string((int)tok.type) + ") at line " + std::to_string(tok.line));
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Two-pass rule collector
// ──────────────────────────────────────────────────────────────────────────────

struct RuleDef {
    std::string name;
    size_t      tok_start; // first token of rule body
    size_t      tok_end;   // one past last token of rule body
    bool        is_terminal;
};

// Returns start rule name and fills rule_defs
std::string collect_rules(LarkLexer & lex, std::vector<RuleDef> & rule_defs) {
    std::string            start_rule;
    auto &                 toks = lex.tokens;
    size_t                 i    = 0;

    auto skip_nl = [&]() {
        while (i < toks.size() && toks[i].type == LarkTok::NEWLINE) ++i;
    };

    auto skip_to_nl = [&]() {
        while (i < toks.size() && toks[i].type != LarkTok::NEWLINE && toks[i].type != LarkTok::END) ++i;
    };

    // Looks ahead from position j (skipping newlines) to check if NAME COLON starts
    auto is_rule_start = [&](size_t j) -> bool {
        while (j < toks.size() && toks[j].type == LarkTok::NEWLINE) ++j;
        if (j >= toks.size() || toks[j].type == LarkTok::END) return false;
        if (toks[j].type != LarkTok::NAME) return false;
        size_t k = j + 1;
        // Skip optional .alias
        if (k < toks.size() && toks[k].type == LarkTok::DOT) k += 2;
        return k < toks.size() && toks[k].type == LarkTok::COLON;
    };

    skip_nl();

    while (i < toks.size() && toks[i].type != LarkTok::END) {
        skip_nl();
        if (i >= toks.size() || toks[i].type == LarkTok::END) break;

        // Directive
        if (toks[i].type == LarkTok::PERCENT) {
            skip_to_nl();
            continue;
        }

        // Rule definition
        if (toks[i].type != LarkTok::NAME) { skip_to_nl(); continue; }

        std::string name = toks[i].value;
        ++i;

        // Optional .alias modifier
        if (i < toks.size() && toks[i].type == LarkTok::DOT) {
            ++i; // '.'
            if (i < toks.size() && toks[i].type == LarkTok::NAME) ++i;
        }

        if (i >= toks.size() || toks[i].type != LarkTok::COLON) {
            skip_to_nl();
            continue;
        }
        ++i; // ':'

        bool is_terminal = !name.empty() && std::isupper((unsigned char)name[0]);
        if (start_rule.empty() && !is_terminal) {
            start_rule = name;
        }

        size_t body_start = i;

        // Find end: next NEWLINE followed by a rule-start or directive or END
        // Multi-line rules can continue with '|' on the next line
        while (i < toks.size() && toks[i].type != LarkTok::END) {
            if (toks[i].type == LarkTok::NEWLINE) {
                // Check: is next non-NL token a new rule or directive?
                size_t j = i + 1;
                while (j < toks.size() && toks[j].type == LarkTok::NEWLINE) ++j;
                if (j >= toks.size() || toks[j].type == LarkTok::END) {
                    break; // end of file
                }
                if (toks[j].type == LarkTok::PERCENT) {
                    break; // directive
                }
                if (is_rule_start(j)) {
                    break; // new rule
                }
                // '|' continuation — include this newline in body
                ++i;
                continue;
            }
            ++i;
        }

        rule_defs.push_back({name, body_start, i, is_terminal});

        // Consume the terminating newline
        if (i < toks.size() && toks[i].type == LarkTok::NEWLINE) ++i;
    }

    if (start_rule.empty() && !rule_defs.empty()) {
        for (auto & r : rule_defs) {
            if (!r.is_terminal) { start_rule = r.name; break; }
        }
    }
    if (start_rule.empty() && !rule_defs.empty()) {
        start_rule = rule_defs[0].name;
    }

    return start_rule;
}

} // namespace

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

common_peg_arena common_lark_to_peg(const std::string & lark_grammar) {
    LarkLexer lex(lark_grammar);
    lex.tokenize();

    std::vector<RuleDef> rule_defs;
    std::string          start_rule = collect_rules(lex, rule_defs);

    common_peg_parser_builder builder;

    // Parse and register each rule
    common_peg_parser root_parser = builder.eps();
    bool              root_set    = false;

    for (auto & rdef : rule_defs) {
        lex.tok_pos = rdef.tok_start;
        LarkParser rp(lex, builder, rdef.tok_end);
        lex.skip_newlines();

        common_peg_parser body = builder.eps();
        try {
            if (!rp.stopped()) {
                body = rp.parse_alternatives();
            }
        } catch (const std::exception & e) {
            throw std::runtime_error(
                std::string("lark-to-peg: error parsing rule '") + rdef.name + "': " + e.what());
        }

        // Normalize underscores to hyphens to match builder clean_name behavior
        std::string normalized_name = rdef.name;
        for (char & c : normalized_name) { if (c == '_') c = '-'; }

        // Apply semantic tags based on conventional rule names (drives common_chat_peg_mapper)
        const std::string & n = normalized_name;
        if (n == "tool-call") {
            body = builder.tag("tool", body);
        } else if (n == "tool-open") {
            body = builder.tag("tool-open", body);
        } else if (n == "tool-close") {
            body = builder.tag("tool-close", body);
        } else if (n == "func-name" || n == "tool-name" || n == "function-name") {
            body = builder.tag("tool-name", body);
        } else if (n == "tool-id") {
            body = builder.tag("tool-id", body);
        } else if (n == "tool-args" || n == "arguments" || n == "args") {
            body = builder.tag("tool-args", body);
        } else if (n == "tool-arg-name" || n == "arg-name") {
            body = builder.tag("tool-arg-name", body);
        } else if (n == "tool-arg-value" || n == "arg-value") {
            body = builder.tag("tool-arg-value", body);
        } else if (n == "content" || n == "analysis-content" || n == "response-content") {
            // `analysis-content` is the no-reasoning variant: the rule body
            // covers `[THINK]…[/THINK]` text that should surface as content
            // (markers preserved verbatim) rather than as reasoning.
            // `response-content` is the structured response_format payload
            // (e.g. JSON between code fences) that surfaces as content.
            body = builder.tag("content", body);
        } else if (n == "reasoning" || n == "thought") {
            body = builder.tag("reasoning", body);
        }

        auto rule_ref = builder.rule(normalized_name, body);
        if (rdef.name == start_rule && !root_set) {
            root_parser = rule_ref;
            root_set    = true;
        }
    }

    if (root_set) {
        builder.set_root(root_parser);
    }

    return builder.build();
}
