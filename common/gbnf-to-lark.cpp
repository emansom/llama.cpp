#include "gbnf-to-lark.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// A 1:1 port of llguidance's python/llguidance/gbnf_to_lark.py. Structure,
// names and ordering follow the original so the two can be diffed; see the
// header for why fidelity is the goal rather than elegance.

namespace {

// ── source position, for error messages ───────────────────────────────────────

struct Position {
    const std::string * text;
    size_t              pos;

    char current() const { return pos < text->size() ? (*text)[pos] : '\0'; }

    std::string peek(size_t n) const {
        return text->substr(std::min(pos, text->size()), n);
    }

    Position advance(size_t n = 1) const { return Position{ text, pos + n }; }

    std::string describe() const {
        size_t line = 1;
        for (size_t i = 0; i < pos && i < text->size(); i++) {
            if ((*text)[i] == '\n') { line++; }
        }
        return "line " + std::to_string(line) + ", near \"" +
               text->substr(pos, 20) + "\"";
    }
};

[[noreturn]] void fail(const Position & pos, const std::string & msg) {
    throw std::invalid_argument("gbnf-to-lark: " + msg + " at " + pos.describe());
}

// ── AST ───────────────────────────────────────────────────────────────────────

struct Rule;

enum class NK { LITERAL, REGEX, REF, REP, SEQ, ALT };

struct Node;
using NodePtr = std::shared_ptr<Node>;

struct Node {
    NK                   kind;
    std::string          text;       // LITERAL body / REGEX source / REF name
    std::vector<NodePtr> kids;       // REP: one; SEQ/ALT: many
    int                  min_times = 0;
    int                  max_times = -1;  // -1 == unlimited
    Rule *               target    = nullptr;  // REF, once resolved

    bool is_atomic() const { return kind != NK::SEQ && kind != NK::ALT; }
};

struct Rule {
    std::string name;
    NodePtr     body;
    std::string comment;
    bool        rule_is_terminal = false;
    size_t      order            = 0;
};

NodePtr make(NK kind) {
    auto n  = std::make_shared<Node>();
    n->kind = kind;
    return n;
}

bool node_is_terminal(const NodePtr & n) {
    switch (n->kind) {
        case NK::LITERAL:
        case NK::REGEX:
            return true;
        case NK::REF:
            // Reads the target's FLAG rather than recursing, which is what
            // makes the promotion fixpoint below terminate on a cyclic grammar.
            return n->target != nullptr && n->target->rule_is_terminal;
        default:
            break;
    }
    for (const auto & k : n->kids) {
        if (!node_is_terminal(k)) { return false; }
    }
    return true;
}

std::string node_to_string(const NodePtr & n);

std::string join(const std::vector<std::string> & parts, const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i) { out += sep; }
        out += parts[i];
    }
    return out;
}

std::string node_to_string(const NodePtr & n) {
    switch (n->kind) {
        case NK::LITERAL:
            return "\"" + n->text + "\"";
        case NK::REGEX:
            return "/" + n->text + "/";
        case NK::REF:
            return n->target != nullptr ? n->target->name : n->text;
        case NK::REP: {
            std::string inner = node_to_string(n->kids[0]);
            if (!n->kids[0]->is_atomic()) { inner = "(" + inner + ")"; }
            if (n->min_times == 0 && n->max_times < 0) { return inner + "*"; }
            if (n->min_times == 1 && n->max_times < 0) { return inner + "+"; }
            if (n->min_times == 0 && n->max_times == 1) { return inner + "?"; }
            // llguidance accepts `expr{M,N}` as a spelling of Lark's `expr~M..N`
            // (docs/syntax.md, "Minor syntax changes"), so the GBNF spelling
            // carries over unchanged.
            return inner + "{" + std::to_string(n->min_times) + "," +
                   (n->max_times < 0 ? "" : std::to_string(n->max_times)) + "}";
        }
        case NK::SEQ: {
            if (n->kids.empty()) { return "\"\""; }
            std::vector<std::string> parts;
            for (const auto & k : n->kids) { parts.push_back(node_to_string(k)); }
            return join(parts, " ");
        }
        case NK::ALT: {
            std::vector<std::string> parts;
            for (const auto & k : n->kids) { parts.push_back(node_to_string(k)); }
            return "(" + join(parts, " | ") + ")";
        }
    }
    return {};
}

// A rule body prints without the wrapping parentheses an inline alternation gets.
std::string node_to_top_string(const NodePtr & n) {
    if (n->kind != NK::ALT) { return node_to_string(n); }
    std::vector<std::string> parts;
    for (const auto & k : n->kids) { parts.push_back(node_to_string(k)); }
    return join(parts, "\n     | ");
}

NodePtr simplify(const NodePtr & n) {
    for (auto & k : n->kids) { k = simplify(k); }
    if ((n->kind == NK::SEQ || n->kind == NK::ALT) && n->kids.size() == 1) {
        return n->kids[0];
    }
    return n;
}

// ── parser ────────────────────────────────────────────────────────────────────

bool is_word_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
}

bool is_hex(const std::string & s) {
    if (s.empty()) { return false; }
    for (char c : s) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) { return false; }
    }
    return true;
}

// Leading zeros are stripped, matching the reference converter. An all-zero
// escape would strip to nothing there; "0" is kept here so the output stays a
// valid escape.
std::string strip_leading_zeros(const std::string & hex) {
    size_t i = hex.find_first_not_of('0');
    return i == std::string::npos ? "0" : hex.substr(i);
}

struct GbnfParser {
    std::string curr_comment;

    // One source character, with GBNF's escapes preserved AS escapes: the
    // output is Lark/regex source, so `\n` stays two characters.
    std::pair<std::string, Position> parse_char(Position pos) {
        if (pos.current() == '\\') {
            if (pos.peek(2).size() < 2) { fail(pos, "incomplete escape sequence"); }
            pos    = pos.advance();
            char c = pos.current();
            if (c == '"' || c == '\\' || c == '[' || c == ']' || c == 'n' || c == 'r' || c == 't') {
                return { std::string("\\") + c, pos.advance() };
            }
            if (c == 'x' || c == 'u' || c == 'U') {
                const size_t want = (c == 'x') ? 2 : (c == 'u') ? 4 : 8;
                std::string  hex  = pos.peek(want + 1).substr(1);
                if (hex.size() != want || !is_hex(hex)) {
                    fail(pos, std::string("invalid \\") + c + " escape sequence");
                }
                pos = pos.advance(want + 1);
                if (c == 'x') { return { "\\x" + hex, pos }; }
                return { std::string("\\") + c + strip_leading_zeros(hex), pos };
            }
            fail(pos, std::string("invalid escape sequence \\") + c);
        }
        if (pos.current() == '\0' && pos.pos >= pos.text->size()) {
            fail(pos, "unexpected end of input");
        }
        return { std::string(1, pos.current()), pos.advance() };
    }

    std::pair<NodePtr, Position> parse_char_class(Position pos) {
        if (pos.current() != '[') { fail(pos, "expected '['"); }
        std::string rx = "[";
        pos            = pos.advance();
        while (true) {
            auto [c, next] = parse_char(pos);
            pos            = next;
            // Escaped for the REGEX this becomes: '/' would close the literal
            // and '[' opens a nested class in Rust's regex syntax.
            rx += (c == "/" || c == "[") ? "\\" + c : c;
            if (c == "]") { break; }
        }
        auto n  = make(NK::REGEX);
        n->text = rx;
        return { n, pos };
    }

    std::pair<NodePtr, Position> parse_literal(Position pos) {
        if (pos.current() != '"') { fail(pos, "expected '\"'"); }
        pos = pos.advance();
        std::string body;
        while (true) {
            auto [c, next] = parse_char(pos);
            pos            = next;
            if (c == "\"") { break; }
            body += c;
        }
        auto n  = make(NK::LITERAL);
        n->text = body;
        return { n, pos };
    }

    static std::pair<std::string, Position> parse_name(Position pos) {
        const size_t start = pos.pos;
        while (is_word_char(pos.current())) { pos = pos.advance(); }
        if (pos.pos == start) { fail(pos, "expected a rule name"); }
        return { pos.text->substr(start, pos.pos - start), pos };
    }

    static std::pair<int, Position> parse_int(Position pos) {
        const size_t start = pos.pos;
        while (std::isdigit(static_cast<unsigned char>(pos.current()))) { pos = pos.advance(); }
        if (pos.pos == start) { fail(pos, "expected an integer"); }
        return { std::stoi(pos.text->substr(start, pos.pos - start)), pos };
    }

    static Position skip_newline(Position pos) {
        if (pos.current() == '\r') {
            pos = pos.advance();
            if (pos.current() == '\n') { pos = pos.advance(); }
        } else if (pos.current() == '\n') {
            pos = pos.advance();
        }
        return pos;
    }

    // GBNF comments are carried over as Lark comments attached to the next rule.
    Position skip_space(Position pos, bool allow_newlines) {
        while (pos.pos < pos.text->size()) {
            const char c = pos.current();
            if (c == ' ' || c == '\t') {
                pos = pos.advance();
            } else if (allow_newlines && (c == '\r' || c == '\n')) {
                pos = skip_newline(pos);
            } else if (c == '#') {
                pos              = pos.advance();
                std::string cmt  = "//";
                while (pos.pos < pos.text->size() && pos.current() != '\r' && pos.current() != '\n') {
                    cmt += pos.current();
                    pos = pos.advance();
                }
                curr_comment += cmt + "\n";
            } else {
                break;
            }
        }
        return pos;
    }

    Position parse_repetition(Position pos, std::vector<NodePtr> & nodes) {
        if (nodes.empty()) { return pos; }
        auto wrap = [&](int lo, int hi) {
            auto n       = make(NK::REP);
            n->kids      = { nodes.back() };
            n->min_times = lo;
            n->max_times = hi;
            nodes.back() = n;
        };
        switch (pos.current()) {
            case '*': wrap(0, -1); return pos.advance();
            case '+': wrap(1, -1); return pos.advance();
            case '?': wrap(0, 1);  return pos.advance();
            case '{': break;
            default:  return pos;
        }
        pos                 = skip_space(pos.advance(), true);
        auto [min_times, p] = parse_int(pos);
        pos                 = skip_space(p, true);
        if (pos.current() == '}') {
            wrap(min_times, min_times);
            return pos.advance();
        }
        if (pos.current() != ',') { fail(pos, "expected ',' or '}'"); }
        pos          = skip_space(pos.advance(), true);
        int max_times = -1;
        if (std::isdigit(static_cast<unsigned char>(pos.current()))) {
            auto [m, p2] = parse_int(pos);
            max_times    = m;
            pos          = p2;
        }
        pos = skip_space(pos, true);
        if (pos.current() != '}') { fail(pos, "expected '}'"); }
        wrap(min_times, max_times);
        return pos.advance();
    }

    std::pair<NodePtr, Position> parse_group(Position pos, bool is_nested) {
        if (pos.current() != '(') { fail(pos, "expected '('"); }
        pos              = skip_space(pos.advance(), true);
        auto [alts, next] = parse_alternatives(pos, /* is_nested = */ true);
        pos              = next;
        if (pos.current() != ')') { fail(pos, "expected ')'"); }
        return { alts, skip_space(pos.advance(), is_nested) };
    }

    std::pair<NodePtr, Position> parse_sequence(Position pos, bool is_nested) {
        std::vector<NodePtr> nodes;
        while (pos.pos < pos.text->size()) {
            const char c = pos.current();
            if (c == '|' || c == ')') { break; }
            if (!is_nested && (c == '\r' || c == '\n')) { break; }

            if (c == '"') {
                auto [n, p] = parse_literal(pos);
                nodes.push_back(n);
                pos = p;
            } else if (c == '[') {
                auto [n, p] = parse_char_class(pos);
                nodes.push_back(n);
                pos = p;
            } else if (c == '(') {
                auto [n, p] = parse_group(pos, is_nested);
                nodes.push_back(n);
                pos = p;
            } else if (c == '.') {
                nodes.push_back(make(NK::REGEX));
                nodes.back()->text = ".";
                pos                = pos.advance();
            } else if (is_word_char(c)) {
                auto [name, p] = parse_name(pos);
                auto n         = make(NK::REF);
                n->text        = name;
                nodes.push_back(n);
                pos = p;
            } else {
                break;
            }

            pos = skip_space(pos, is_nested);
            pos = parse_repetition(pos, nodes);
            pos = skip_space(pos, is_nested);
        }
        auto seq  = make(NK::SEQ);
        seq->kids = nodes;
        return { seq, pos };
    }

    std::pair<NodePtr, Position> parse_alternatives(Position pos, bool is_nested) {
        std::vector<NodePtr> alts;
        while (true) {
            auto [seq, p] = parse_sequence(pos, is_nested);
            alts.push_back(seq);
            pos = skip_space(p, is_nested);
            if (pos.current() != '|') { break; }
            pos = skip_space(pos.advance(), true);
        }
        auto alt  = make(NK::ALT);
        alt->kids = alts;
        return { alt, pos };
    }

    std::pair<Rule, Position> parse_rule(Position pos) {
        auto [name, p] = parse_name(pos);
        pos            = skip_space(p, false);
        if (pos.peek(3) != "::=") { fail(pos, "expected '::='"); }
        pos               = skip_space(pos.advance(3), true);
        auto [body, next] = parse_alternatives(pos, /* is_nested = */ false);
        pos               = skip_newline(next);

        Rule r;
        r.name       = name;
        r.body       = body;
        r.comment    = curr_comment;
        curr_comment = {};
        return { r, pos };
    }
};

// ── resolve ───────────────────────────────────────────────────────────────────

struct Grammar {
    std::vector<std::unique_ptr<Rule>>   rules;   // definition order
    std::unordered_map<std::string, Rule *> by_name;

    void rename(Rule * r, const std::string & to) {
        if (to == r->name) { return; }
        if (by_name.count(to)) {
            throw std::invalid_argument("gbnf-to-lark: renaming rule '" + r->name +
                                     "' to '" + to + "' collides with an existing rule");
        }
        by_name.erase(r->name);
        r->name        = to;
        by_name[to]    = r;
    }
};

void for_each_node(const NodePtr & n, const std::function<void(const NodePtr &)> & fn) {
    fn(n);
    for (const auto & k : n->kids) { for_each_node(k, fn); }
}

// fooBar-baz -> foo_bar_baz
std::string snake_case(const std::string & name) {
    std::string out;
    for (size_t i = 0; i < name.size(); i++) {
        const char c = name[i];
        if (c == '-') {
            out += '_';
            continue;
        }
        const bool boundary = i > 0 && std::isupper(static_cast<unsigned char>(c)) &&
                              std::islower(static_cast<unsigned char>(name[i - 1]));
        if (boundary) { out += '_'; }
        out += c;
    }
    return out;
}

std::string to_lower(std::string s) {
    for (char & c : s) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    return s;
}

std::string to_upper(std::string s) {
    for (char & c : s) { c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }
    return s;
}

void resolve(Grammar & g) {
    for (size_t i = 0; i < g.rules.size(); i++) {
        g.rules[i]->order = i;
        g.rules[i]->body  = simplify(g.rules[i]->body);
    }

    for (auto & r : g.rules) {
        for_each_node(r->body, [&](const NodePtr & n) {
            if (n->kind != NK::REF) { return; }
            auto it = g.by_name.find(n->text);
            if (it == g.by_name.end()) {
                throw std::invalid_argument("gbnf-to-lark: rule '" + n->text + "' is referenced but never defined");
            }
            n->target = it->second;
        });
    }

    auto root = g.by_name.find("root");
    if (root == g.by_name.end()) {
        throw std::invalid_argument("gbnf-to-lark: no 'root' rule found");
    }
    g.rename(root->second, "start");

    // A rule whose body contains no context-free structure becomes a LEXEME
    // (uppercase). llguidance's docs call this critical for performance;
    // promoting to a fixpoint is how a chain of such rules is caught.
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto & r : g.rules) {
            if (r->name != "start" && !r->rule_is_terminal && node_is_terminal(r->body)) {
                r->rule_is_terminal = true;
                changed             = true;
            }
        }
    }

    for (auto & r : g.rules) {
        std::string n = snake_case(r->name);
        n             = r->rule_is_terminal ? to_upper(n) : to_lower(n);
        g.rename(r.get(), n);
    }
}

}  // namespace

std::string common_gbnf_to_lark(const std::string & gbnf) {
    GbnfParser parser;
    Grammar    g;

    Position pos{ &gbnf, 0 };
    pos = parser.skip_space(pos, /* allow_newlines = */ true);
    while (pos.pos < gbnf.size()) {
        auto [rule, next] = parser.parse_rule(pos);
        if (g.by_name.count(rule.name)) {
            throw std::invalid_argument("gbnf-to-lark: rule '" + rule.name + "' is defined twice");
        }
        g.rules.push_back(std::make_unique<Rule>(std::move(rule)));
        g.by_name[g.rules.back()->name] = g.rules.back().get();
        pos                             = parser.skip_space(next, /* allow_newlines = */ true);
    }
    if (g.rules.empty()) {
        throw std::invalid_argument("gbnf-to-lark: grammar is empty");
    }

    resolve(g);

    std::vector<Rule *> ordered;
    ordered.reserve(g.rules.size());
    for (auto & r : g.rules) { ordered.push_back(r.get()); }
    std::sort(ordered.begin(), ordered.end(),
              [](const Rule * a, const Rule * b) { return a->order < b->order; });

    std::string out = "%llguidance {}\n\n";
    for (const Rule * r : ordered) {
        out += r->comment + r->name + ": " + node_to_top_string(r->body) + "\n";
    }
    return out;
}
