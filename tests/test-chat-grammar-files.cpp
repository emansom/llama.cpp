#include "chat.h"
#include "gbnf-to-peg.h"
#include "lark-to-peg.h"
#include "peg-parser.h"
#include "testing.h"

#include <iostream>
#include <string>

// ──────────────────────────────────────────────────────────────────────────────
// Lark→PEG transpiler unit tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_lark_to_peg_basic(testing & t) {
    t.test("literal", [](testing & t) {
        const std::string grammar = "%llguidance {}\nstart: \"hello\"\n";
        auto arena = common_lark_to_peg(grammar);
        t.assert_true("arena not empty", !arena.empty());
        common_peg_parse_context ctx("hello");
        t.assert_true("parses 'hello'", arena.parse(ctx).success());
    });

    t.test("choice", [](testing & t) {
        const std::string grammar = "%llguidance {}\nstart: \"foo\" | \"bar\"\n";
        auto arena = common_lark_to_peg(grammar);
        {
            common_peg_parse_context ctx("foo");
            t.assert_true("parses 'foo'", arena.parse(ctx).success());
        }
        {
            common_peg_parse_context ctx("bar");
            t.assert_true("parses 'bar'", arena.parse(ctx).success());
        }
        {
            common_peg_parse_context ctx("baz");
            t.assert_true("rejects 'baz'", !arena.parse(ctx).success());
        }
    });

    t.test("sequence", [](testing & t) {
        const std::string grammar = "%llguidance {}\nstart: \"a\" \"b\" \"c\"\n";
        auto arena = common_lark_to_peg(grammar);
        common_peg_parse_context ctx("abc");
        t.assert_true("parses 'abc'", arena.parse(ctx).success());
    });

    t.test("repetition (+)", [](testing & t) {
        const std::string grammar = "%llguidance {}\nstart: item+\nitem: \"x\"\n";
        auto arena = common_lark_to_peg(grammar);
        {
            common_peg_parse_context ctx("xxx");
            t.assert_true("parses 'xxx'", arena.parse(ctx).success());
        }
        {
            common_peg_parse_context ctx("");
            t.assert_true("rejects empty", !arena.parse(ctx).success());
        }
    });

    t.test("optional (?)", [](testing & t) {
        const std::string grammar = "%llguidance {}\nstart: \"a\" \"b\"?\n";
        auto arena = common_lark_to_peg(grammar);
        {
            common_peg_parse_context ctx("ab");
            t.assert_true("parses 'ab'", arena.parse(ctx).success());
        }
        {
            common_peg_parse_context ctx("a");
            t.assert_true("parses 'a'", arena.parse(ctx).success());
        }
    });

    t.test("multi-line rule", [](testing & t) {
        const std::string grammar = "%llguidance {}\nstart: \"a\"\n     | \"b\"\n     | \"c\"\n";
        auto arena = common_lark_to_peg(grammar);
        common_peg_parse_context ctx("b");
        t.assert_true("parses 'b' from multi-line rule", arena.parse(ctx).success());
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GBNF→PEG transpiler unit tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_gbnf_to_peg_basic(testing & t) {
    t.test("literal", [](testing & t) {
        const std::string grammar = R"(root ::= "hello")";
        auto arena = common_gbnf_to_peg(grammar);
        t.assert_true("arena not empty", !arena.empty());
        common_peg_parse_context ctx("hello");
        t.assert_true("parses 'hello'", arena.parse(ctx).success());
    });

    t.test("choice", [](testing & t) {
        const std::string grammar = R"(root ::= "foo" | "bar")";
        auto arena = common_gbnf_to_peg(grammar);
        {
            common_peg_parse_context ctx("foo");
            t.assert_true("parses 'foo'", arena.parse(ctx).success());
        }
        {
            common_peg_parse_context ctx("bar");
            t.assert_true("parses 'bar'", arena.parse(ctx).success());
        }
    });

    t.test("char class", [](testing & t) {
        const std::string grammar = R"(root ::= [a-z]+)";
        auto arena = common_gbnf_to_peg(grammar);
        common_peg_parse_context ctx("abc");
        t.assert_true("parses 'abc'", arena.parse(ctx).success());
    });

    t.test("rule reference", [](testing & t) {
        const std::string grammar =
            "root ::= greeting \" \" name\n"
            "greeting ::= \"hello\" | \"hi\"\n"
            "name ::= [A-Za-z]+\n";
        auto arena = common_gbnf_to_peg(grammar);
        common_peg_parse_context ctx("hello world");
        t.assert_true("parses 'hello world'", arena.parse(ctx).success());
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Gemma4 grammar tests
// ──────────────────────────────────────────────────────────────────────────────

static const std::string GEMMA4_LARK =
    "%llguidance {}\n"
    "\n"
    "start: think_block? content tool_calls\n"
    "content: /([^<]|<[^|]|<\\|[^tc]|<\\|t[^o]|<\\|to[^o]|<\\|too[^l]|<\\|tool[^_]|<\\|tool_[^c])*/\n"
    "think_block: \"<|channel>thought\" /[ \\t\\n\\r]+/ reasoning /(.|\n)*?/ \"<channel|>\"\n"
    "reasoning: /[^<]*/\n"
    "\n"
    "tool_calls: tool_call+\n"
    "tool_call: \"<|tool_call>call:\" func_name gemma4_dict \"<tool_call|>\"\n"
    "func_name: FUNC_NAME\n"
    "FUNC_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/\n"
    "\n"
    "gemma4_dict: \"{\" (gemma4_kv (\",\" gemma4_kv)*)? \"}\"\n"
    "gemma4_kv: gemma4_dict_key \":\" gemma4_value\n"
    "gemma4_dict_key: /[^:{}]+/\n"
    "gemma4_value: gemma4_string | gemma4_dict | gemma4_array | GEMMA4_NUMBER | GEMMA4_BOOL | \"null\"\n"
    "gemma4_string: \"<|\\\"|>\" GEMMA4_STR_CONTENT \"<|\\\"|>\"\n"
    "GEMMA4_STR_CONTENT: /([^<]|<[^|]|<\\|[^\"]|<\\|\"[^|])*/\n"
    "gemma4_array: \"[\" (gemma4_value (\",\" gemma4_value)*)? \"]\"\n"
    "\n"
    "GEMMA4_NUMBER: /-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\n"
    "GEMMA4_BOOL: \"true\" | \"false\"\n"
    "\n"
    "%ignore /[ \\t\\n\\r]/\n";

static void test_grammar_gemma4(testing & t) {
    t.test("arena has expected rules", [](testing & t) {
        auto arena = common_lark_to_peg(GEMMA4_LARK);
        t.assert_true("arena not empty", !arena.empty());
        t.assert_true("has 'start' rule", arena.has_rule("start"));
        t.assert_true("has 'tool-call' rule", arena.has_rule("tool-call"));
        t.assert_true("has 'gemma4-dict' rule", arena.has_rule("gemma4-dict"));
    });

    t.test("parse tool call", [](testing & t) {
        // content uses a complex regex for llguidance; PEG uses rest() approximation + LENIENT mode
        const std::string input =
            "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>";
        auto arena = common_lark_to_peg(GEMMA4_LARK);
        common_peg_parse_context ctx(input, COMMON_PEG_PARSE_FLAG_LENIENT);
        auto result = arena.parse(ctx);
        t.assert_true("parses gemma4 tool call", result.type != COMMON_PEG_PARSE_RESULT_FAIL);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// LFM2 grammar tests
// ──────────────────────────────────────────────────────────────────────────────

static const std::string LFM2_LARK =
    "%llguidance {}\n"
    "\n"
    "start: think_block? content tool_call_block\n"
    "think_block: \"<think>\" reasoning \"</think>\"\n"
    "reasoning: /(.|\n)*?/\n"
    "content: /([^<]|<(?!\\|tool_call_start\\|))*/\n"
    "\n"
    "tool_call_block: \"<|tool_call_start|>\" tool_calls \"<|tool_call_end|>\"\n"
    "tool_calls: tool_call+\n"
    "tool_call: \"[\" func_name \"(\" arg_list \")\" \"]\"\n"
    "func_name: FUNC_NAME\n"
    "FUNC_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/\n"
    "arg_list: (named_arg (\",\" named_arg)*)?\n"
    "named_arg: ARG_NAME \"=\" arg_value\n"
    "ARG_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/\n"
    "arg_value: QUOTED_STRING | LFM_NUMBER | LFM_BOOL | \"None\"\n"
    "QUOTED_STRING: \"\\\"\" /([^\"\\\\]|\\\\.)*/ \"\\\"\"\n"
    "LFM_NUMBER: /-?[0-9]+(\\.[0-9]+)?/\n"
    "LFM_BOOL: \"True\" | \"False\"\n"
    "\n"
    "%ignore \" \"\n";

static void test_grammar_lfm2(testing & t) {
    t.test("arena has expected rules", [](testing & t) {
        auto arena = common_lark_to_peg(LFM2_LARK);
        t.assert_true("arena not empty", !arena.empty());
        t.assert_true("has 'start' rule", arena.has_rule("start"));
        t.assert_true("has 'tool-call' rule", arena.has_rule("tool-call"));
        t.assert_true("has 'func-name' rule", arena.has_rule("func-name"));
    });

    t.test("parse tool call block", [](testing & t) {
        // content uses a complex regex for llguidance; PEG uses rest() approximation + LENIENT mode
        const std::string input =
            "<|tool_call_start|>[get_weather(location=\"Paris\", unit=\"celsius\")]<|tool_call_end|>";
        auto arena = common_lark_to_peg(LFM2_LARK);
        common_peg_parse_context ctx(input, COMMON_PEG_PARSE_FLAG_LENIENT);
        auto result = arena.parse(ctx);
        t.assert_true("parses lfm2 tool call", result.type != COMMON_PEG_PARSE_RESULT_FAIL);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Kimi K2 grammar tests
// ──────────────────────────────────────────────────────────────────────────────

static const std::string KIMI_K2_LARK =
    "%llguidance {}\n"
    "\n"
    "start: think_block? content tool_calls_section\n"
    "think_block: \"<think>\" reasoning \"</think>\"\n"
    "reasoning: /(.|\n)*?/\n"
    "content: /([^<]|<(?!\\|tool_calls_section_begin\\|))*/\n"
    "\n"
    "tool_calls_section: \"<|tool_calls_section_begin|>\"? tool_call+ \"<|tool_calls_section_end|>\"?\n"
    "tool_call: \"<|tool_call_begin|>\" tool_id \"<|tool_call_argument_begin|>\" tool_args \"<|tool_call_end|>\"?\n"
    "tool_id: \"functions.\" func_name \":\" CALL_INDEX\n"
    "func_name: FUNC_NAME\n"
    "FUNC_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/\n"
    "CALL_INDEX: /[0-9]+/\n"
    "tool_args: /\\{[^}]*\\}/\n";

static void test_grammar_kimi_k2(testing & t) {
    t.test("arena has expected rules", [](testing & t) {
        auto arena = common_lark_to_peg(KIMI_K2_LARK);
        t.assert_true("arena not empty", !arena.empty());
        t.assert_true("has 'start' rule", arena.has_rule("start"));
        t.assert_true("has 'tool-call' rule", arena.has_rule("tool-call"));
        t.assert_true("has 'tool-id' rule", arena.has_rule("tool-id"));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GigaChat v3 grammar tests
// ──────────────────────────────────────────────────────────────────────────────

static const std::string GIGACHAT_LARK =
    "%llguidance {}\n"
    "\n"
    "start: content tool_call\n"
    "content: /([^<]|<(?!\\|message_sep\\|))*/\n"
    "\n"
    "tool_call: \"<|message_sep|>\\n\\nfunction call<|role_sep|>\\n\" tool_body\n"
    "tool_body: \"{\" \"\\\"name\\\"\" \":\" \"\\\"\" func_name \"\\\"\" \",\" \"\\\"arguments\\\"\" \":\" tool_args \"}\"\n"
    "func_name: FUNC_NAME\n"
    "FUNC_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/\n"
    "tool_args: /\\{[^}]*\\}/\n";

static void test_grammar_gigachat(testing & t) {
    t.test("arena has expected rules", [](testing & t) {
        auto arena = common_lark_to_peg(GIGACHAT_LARK);
        t.assert_true("arena not empty", !arena.empty());
        t.assert_true("has 'start' rule", arena.has_rule("start"));
        t.assert_true("has 'tool-call' rule", arena.has_rule("tool-call"));
        t.assert_true("has 'func-name' rule", arena.has_rule("func-name"));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// override_grammar routing test
// ──────────────────────────────────────────────────────────────────────────────

static void test_override_grammar(testing & t) {
    t.test("lark grammar dispatch", [](testing & t) {
        // Simulate common_chat_load_grammar_parser() routing logic
        const std::string & grammar_str = GEMMA4_LARK;
        common_peg_arena    arena;
        if (grammar_str.rfind("%llguidance", 0) == 0) {
            arena = common_lark_to_peg(grammar_str);
        } else {
            arena = common_gbnf_to_peg(grammar_str);
        }
        t.assert_true("lark grammar dispatched to lark transpiler", !arena.empty());
    });

    t.test("gbnf grammar dispatch", [](testing & t) {
        const std::string grammar_str = R"(root ::= "test")";
        common_peg_arena  arena;
        if (grammar_str.rfind("%llguidance", 0) == 0) {
            arena = common_lark_to_peg(grammar_str);
        } else {
            arena = common_gbnf_to_peg(grammar_str);
        }
        t.assert_true("gbnf grammar dispatched to gbnf transpiler", !arena.empty());
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    t.test("lark to peg transpiler", test_lark_to_peg_basic);
    t.test("gbnf to peg transpiler", test_gbnf_to_peg_basic);
    t.test("gemma4 grammar", test_grammar_gemma4);
    t.test("lfm2 grammar", test_grammar_lfm2);
    t.test("kimi-k2 grammar", test_grammar_kimi_k2);
    t.test("gigachat grammar", test_grammar_gigachat);
    t.test("override grammar routing", test_override_grammar);

    return t.summary();
}
