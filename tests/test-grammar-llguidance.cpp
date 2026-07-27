#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "chat.h"
#include "chat-formats/gemma4-format.h"
#include "lark-to-peg.h"
#include "sampling.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <set>
#include <string>
#include <vector>

static const llama_vocab * vocab;

static bool match_string(const std::string & input, llama_sampler * grammar) {
    llama_sampler_reset(grammar);
    // parse_special=true, because this stands in for text the MODEL GENERATED.
    // A model emits token ids, and a marker it was trained on is one of them; the
    // grammar masks that id directly. Tokenizing with parse_special=false models
    // a user typing the marker's spelling into a prompt instead, which splits it
    // into ordinary text and asks the grammar about a token sequence the model
    // would never produce.
    //
    // It is not a distinction without a difference here. `tokenizer_st_partition`
    // skips CONTROL tokens when parse_special is false but keeps USER_DEFINED
    // ones, and Gemma 4's markers are split across both classes: `<|tool_call>`
    // (48) is USER_DEFINED and survived, `<|tool_response>` (50) and `<|tool>`
    // (46) carry CONTROL and did not. So the false setting silently tested two of
    // the format's markers as prose -- which reads as a grammar that rejects a
    // valid string, and cost real time to chase.
    auto tokens = common_tokenize(vocab, input, false, true);

    auto n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token_data> cur;
    cur.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token) n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{ token_id, 0.0f, 0.0f });
    }
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    for (const auto token : tokens) {
        for (llama_token token_id = 0; token_id < (llama_token) n_vocab; token_id++) {
            cur[token_id].logit = 0.0f;
        }
        llama_sampler_apply(grammar, &tok_arr);
        if (cur[token].logit < 0.0f) {
            return false;
        }
        llama_sampler_accept(grammar, token);
    }

    // do we allow EOS at the end? if so the grammar is accepting

    auto tok_eos = llama_vocab_eot(vocab);
    if (tok_eos == LLAMA_TOKEN_NULL) {
        tok_eos = llama_vocab_eos(vocab);
    }

    cur[tok_eos].logit = 0.0f;
    llama_sampler_apply(grammar, &tok_arr);

    return cur[tok_eos].logit >= 0.0f;
}

static void test(const std::string & test_desc, const std::string & grammar_str,
                 const std::vector<std::string> & passing_strings, const std::vector<std::string> & failing_strings) {
    fprintf(stderr, "⚫ Testing %s\n%s\n", test_desc.c_str(), grammar_str.c_str());
    fflush(stderr);

    auto * grammar = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());

    fprintf(stderr, "  🔵 Valid strings:\n");

    // Passing strings
    for (const auto & test_string : passing_strings) {
        fprintf(stderr, "    \"%s\" ", test_string.c_str());
        fflush(stderr);

        bool matched = match_string(test_string, grammar);

        if (!matched) {
            fprintf(stderr, "❌ (failed to match)\n");

            // DEBUG: Write strings to files so that we can analyze more easily with gbnf-validator program to see exactly where things failed.
            // DEBUG: Write the grammar_str to test-grammar-integration.grammar.gbnf
            FILE * grammar_file = fopen("test-grammar-integration.grammar.gbnf", "w");
            if (grammar_file) {
                fprintf(grammar_file, "%s", grammar_str.c_str());
                fclose(grammar_file);
            }

            // DEBUG: Write the test string to test-grammar-integration.string.txt
            FILE * string_file = fopen("test-grammar-integration.string.txt", "w");
            if (string_file) {
                fprintf(string_file, "%s", test_string.c_str());
                fclose(string_file);
            }

            fprintf(stderr,
                    "\n NOTE: Debug grammar file generated. To analyze this failure in detail, run the following "
                    "command:     ./test-gbnf-validator test-grammar-integration.grammar.gbnf "
                    "test-grammar-integration.string.txt\n\n");
        } else {
            fprintf(stdout, "✅︎\n");
        }

        assert(matched);
    }

    fprintf(stderr, "  🟠 Invalid strings:\n");

    // Failing strings
    for (const auto & test_string : failing_strings) {
        fprintf(stderr, "    \"%s\" ", test_string.c_str());
        fflush(stderr);

        bool matched = match_string(test_string, grammar);

        if (matched) {
            fprintf(stderr, "❌ (incorrectly matched)\n");
        } else {
            fprintf(stdout, "✅︎\n");
        }
        assert(!matched);
    }

    llama_sampler_free(grammar);
}

static void test_grammar(const std::string & test_desc, const std::string & grammar_str,
                         const std::vector<std::string> & passing_strings,
                         const std::vector<std::string> & failing_strings) {
    test(test_desc + ". Grammar: " + grammar_str, grammar_str, passing_strings, failing_strings);
}

static void test_schema(const std::string & test_desc, const std::string & schema_str,
                        const std::vector<std::string> & passing_strings,
                        const std::vector<std::string> & failing_strings) {
    test(test_desc + ". Schema: " + schema_str, "%llguidance {}\nstart: %json " + schema_str, passing_strings,
         failing_strings);
}

static void test_simple_grammar() {
    test_schema("min 0",
                R"""({
            "type": "integer",
            "minimum": 0
        })""",
                // Passing strings
                {
                    "0",
                    "10",
                    "12",
                    "10000",
                },
                // Failing strings
                {
                    "-1",
                    "-10",
                    "-10000",
                    "-100000000000000000000000000000000",
                    // "100000000000000000000000000000000",
                    "00",
                    "01",
                    "-0",
                });
    test_schema("min 2",
                // Schema
                R"""({
            "type": "integer",
            "minimum": 2
        })""",
                // Passing strings
                {
                    "2",
                    "3",
                    "4",
                    "10",
                    "20",
                    "1234567890000000",
                },
                // Failing strings
                {
                    "0", "1", "-1", "-100", "0", "1", "01", "02",
                    // "12345678900000000",
                });
    test_schema("min 456",
                R"""({
            "type": "integer",
            "minimum": 456
        })""",
                // Passing strings
                {
                    "456",
                    "4560",
                    "457",
                    "460",
                    "500",
                },
                // Failing strings
                {
                    "455",
                    "356",
                    "50",
                    "050",
                    "-1",
                    "-456",
                });
    test_schema("min -123",
                R"""({
            "type": "integer",
            "minimum": -123
        })""",
                // Passing strings
                {
                    "-123",
                    "-122",
                    "-11",
                    "-1",
                    "0",
                    "1",
                    "123",
                    "1234",
                    "2345",
                },
                // Failing strings
                {
                    "-1234",
                    "-124",
                });

    test_schema("max 9999",
                // Schema
                R"""({
            "type": "integer",
            "maximum": 9999
        })""",
                // Passing strings
                {
                    "-99999",
                    "0",
                    "9999",
                },
                // Failing strings
                {
                    "10000",
                    "99991",
                });
    test_schema("max -9999",
                // Schema
                R"""({
            "type": "integer",
            "maximum": -9999
        })""",
                // Passing strings
                {
                    "-10000",
                    "-9999",
                },
                // Failing strings
                {
                    "-9998",
                    "0",
                    "9999",
                });
    test_schema("min 5 max 30",
                // Schema
                R"""({
            "type": "integer",
            "minimum": 5,
            "maximum": 30
        })""",
                // Passing strings
                {
                    "5",
                    "10",
                    "30",
                },
                // Failing strings
                {
                    "05",
                    "4",
                    "-1",
                    "31",
                    "123",
                    "0123",
                });
    test_schema("min -1 max 1",
                R"""({
            "type": "integer",
            "minimum": -1,
            "maximum": 1
        })""",
                // Passing strings
                {
                    "-1",
                    "0",
                    "1",
                },
                // Failing strings
                {
                    "-11",
                    "-10",
                    "-2",
                    "2",
                    "10",
                    "11",
                });
    test_schema("min -123 max 42",
                R"""({
            "type": "integer",
            "minimum": -123,
            "maximum": 42
        })""",
                // Passing strings
                {
                    "-123",
                    "-122",
                    "-13",
                    "-11",
                    "-2",
                    "-1",
                    "0",
                    "1",
                    "5",
                    "10",
                    "39",
                    "40",
                    "42",
                },
                // Failing strings
                {
                    "-0123",
                    "-124",
                    "-1123",
                    "-200",
                    "43",
                    "123",
                    "0123",
                });
    test_schema("exclusive min / max",
                // Schema
                R"""({
            "type": "integer",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 10000
        })""",
                // Passing strings
                {
                    "1",
                    "9999",
                },
                // Failing strings
                {
                    "0",
                    "01",
                    "10000",
                    "99999",
                });

    // Test case for a simple grammar
    test_grammar("simple grammar",
                 R"""(
            start: expr
            expr: term ("+" term)*
            term: number
            number: /[0-9]+/ )""",
                 // Passing strings
                 {
                     "42",
                     "1+2+3+4+5",
                     "123+456",
                 },
                 // Failing strings
                 {
                     "+",
                     "/ 3",
                     "1+2+3+4+5+",
                     "12a45",
                 });
}

static void test_complex_grammar() {
    // Test case for a more complex grammar, with both failure strings and success strings
    test_grammar("medium complexity grammar",
                 // Grammar
                 R"""(
            start: expression
            expression: term ws (("+"|"-") ws term)*
            term: factor ws (("*"|"/") ws factor)*
            factor: number | variable | "(" expression ")" | function-call
            number: /[0-9]+/
            variable: /[a-zA-Z_][a-zA-Z0-9_]*/
            function-call: variable ws "(" (expression ("," ws expression)*)? ")"
            ws: /[ \t\n\r]?/ )""",
                 // Passing strings
                 { "42",
                   "1*2*3*4*5",
                   "x",
                   "x+10",
                   "x1+y2",
                   "(a+b)*(c-d)",
                   "func()",
                   "func(x,y+2)",
                   "a*(b+c)-d/e",
                   "f(g(x),h(y,z))",
                   "x + 10",
                   "x1 + y2",
                   "(a + b) * (c - d)",
                   "func()",
                   "func(x, y + 2)",
                   "a * (b + c) - d / e",
                   "f(g(x), h(y, z))",
                   "123+456",
                   "123*456*789-123/456+789*123",
                   "123+456*789-123/456+789*123-456/789+123*456-789/123+456*789-123/456+789*123-456" },
                 // Failing strings
                 {
                     "+",
                     "/ 3x",
                     "x + + y",
                     "a * / b",
                     "func(,)",
                     "func(x y)",
                     "(a + b",
                     "x + y)",
                     "a + b * (c - d",
                     "42 +",
                     "x +",
                     "x + 10 +",
                     "(a + b) * (c - d",
                     "func(",
                     "func(x, y + 2",
                     "a * (b + c) - d /",
                     "f(g(x), h(y, z)",
                     "123+456*789-123/456+789*123-456/789+123*456-789/123+456*789-123/456+789*123-456/",
                 });
}

static void test_special_chars() {
    // A collection of tests to exercise special characters such as "."
    test_grammar("special characters",
                 // Grammar
                 R"""(
            start: /.../ "abc" /.../
            )""",
                 // Passing strings
                 { "abcabcabc", "aaaabcccc",
                   // NOTE: Also ensures that multi-byte characters still count as a single character
                   "🔵🟠✅abc❌🟠🔵" },
                 // Failing strings
                 { "aaabcccc", "aaaaabcccc", "aaaabccc", "aaaabccccc", "🔵🟠✅❌abc❌✅🟠🔵", "🔵🟠abc🟠🔵" });
}

static void test_quantifiers() {
    // A collection of tests to exercise * + and ? quantifiers

    test_grammar(
        "* quantifier",
        // Grammar
        R"""(start: "a"*)""",
        // Passing strings
        { "", "a", "aaaaa", "aaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
        // Failing strings
        { "b", "ab", "aab", "ba", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" });
    test_grammar(
        "+ quantifier",
        // Grammar
        R"""(start: "a"+)""",
        // Passing strings
        { "a", "aaaaa", "aaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
        // Failing strings
        { "", "b", "ab", "aab", "ba", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" });
    test_grammar("? quantifier",
                 // Grammar
                 R"""(start: "a"?)""",
                 // Passing strings
                 { "", "a" },
                 // Failing strings
                 {
                     "b",
                     "ab",
                     "aa",
                     "ba",
                 });
    test_grammar("mixed quantifiers",
                 // Grammar
                 R"""(
            start: cons+ vowel* cons? (vowel cons)*
            vowel: /[aeiouy]/
            cons: /[bcdfghjklmnpqrstvwxyz]/
            )""",
                 // Passing strings
                 {
                     "yes",
                     "no",
                     "noyes",
                     "crwth",
                     "four",
                     "bryyyy",
                 },
                 // Failing strings
                 {
                     "yess",
                     "yesno",
                     "forty",
                     "catyyy",
                 });
    test_grammar("simple exact repetition",
                 // Grammar
                 R"""(
            start: /[ab]{4}/
        )""",
                 // Passing strings
                 {
                     "aaaa",
                     "bbbb",
                     "abab",
                 },
                 // Failing strings
                 {
                     "a",
                     "b",
                     "aaaaa",
                 });
    test_grammar("simple min repetition",
                 // Grammar
                 R"""(
            start: /[ab]{4,}/
        )""",
                 // Passing strings
                 {
                     "aaaa",
                     "aaaaab",
                     "bbbb",
                     "ababab",
                 },
                 // Failing strings
                 {
                     "",
                     "aba",
                 });
    test_grammar("simple max repetition",
                 // Grammar
                 R"""(
            start: /[ab]{0,4}/
        )""",
                 // Passing strings
                 {
                     "",
                     "a",
                     "aa",
                     "aaa",
                     "aaab",
                 },
                 // Failing strings
                 {
                     "aaaaa",
                 });
    // test_grammar("min / max repetition",
    //              // Grammar
    //              R"""(
    //         start: ("0x" /[A-F0-9]{2}/ " "?){3,5}
    //     )""",
    //              // Passing strings
    //              {
    //                  "0xFF 0x12 0xAB",
    //                  "0xFF 0x12 0xAB 0x00 0x00",
    //              },
    //              // Failing strings
    //              {
    //                  "",
    //                  "0xFF",
    //                  "0xFF 0x12",
    //                  "0xFF 0x12 0xAB 0x00 0x00 0x00",
    //              });
}

static void test_json_schema() {
    // Note that this is similar to the regular grammar tests,
    //  but we convert each json schema to a grammar before parsing.
    // Otherwise, this test structure is the same.

    test_schema("empty schema (object)",
                // Schema
                R"""(
            {"type":"object"}
        )""",
                // Passing strings
                {
                    R"""({})""",
                    R"""({"foo": "bar"})""",
                },
                // Failing strings
                {
                    "",
                    "[]",
                    "null",
                    R"""("")""",
                    "true",
                });

    test_schema(
        "exotic formats (list)",
        // Schema
        R"""({
            "items": [
                { "format": "date" },
                { "format": "uuid" },
                { "format": "time" },
                { "format": "date-time" }
            ]
        })""",
        // Passing strings
        {
            // "{}", // NOTE: This string passes for this schema on https://www.jsonschemavalidator.net/ -- should it?
            // "[]", // NOTE: This string passes for this schema on https://www.jsonschemavalidator.net/ -- should it?
            R"""(["2012-04-23", "12345678-1234-1234-1234-1234567890ab", "18:25:43.511Z", "2012-04-23T18:25:43.511Z"])""",
            //R"""(["2012-04-23","12345678-1234-1234-1234-1234567890ab"])""", // NOTE: This string passes for this schema on https://www.jsonschemavalidator.net/ -- should it?
            //R"""({"foo": "bar"})""", // NOTE: This string passes for this schema on https://www.jsonschemavalidator.net/ -- should it?
        },
        // Failing strings
        {
            R"""(["foo", "bar"])""",
            R"""(["12345678-1234-1234-1234-1234567890ab"])""",
        });

    test_schema("string",
                // Schema
                R"""({
            "type": "string"
        })""",
                // Passing strings
                {
                    R"""("foo")""",
                    R"""("bar")""",
                    R"""("")""",
                },
                // Failing strings
                {
                    R"""({})""",
                    R"""("foo": "bar")""",
                });

    test_schema("string w/ min length 1",
                // Schema
                R"""({
            "type": "string",
            "minLength": 1
        })""",
                // Passing strings
                {
                    R"""("foo")""",
                    R"""("bar")""",
                },
                // Failing strings
                {
                    R"""("")""",
                    R"""({})""",
                    R"""("foo": "bar")""",
                });

    test_schema("string w/ min length 3",
                // Schema
                R"""({
                "type": "string",
                "minLength": 3
        })""",
                // Passing strings
                {
                    R"""("foo")""",
                    R"""("bar")""",
                    R"""("foobar")""",
                },
                // Failing strings
                {
                    R"""("")""",
                    R"""("f")""",
                    R"""("fo")""",
                });

    test_schema("string w/ max length",
                // Schema
                R"""({
            "type": "string",
            "maxLength": 3
        })""",
                // Passing strings
                {
                    R"""("foo")""",
                    R"""("bar")""",
                    R"""("")""",
                    R"""("f")""",
                    R"""("fo")""",
                },
                // Failing strings
                {
                    R"""("foobar")""",
                });

    test_schema("string w/ min & max length",
                // Schema
                R"""({
            "type": "string",
            "minLength": 1,
            "maxLength": 4
        })""",
                // Passing strings
                {
                    R"""("foo")""",
                    R"""("bar")""",
                    R"""("f")""",
                    R"""("barf")""",
                },
                // Failing strings
                {
                    R"""("")""",
                    R"""("barfo")""",
                    R"""("foobar")""",
                });

    test_schema("boolean",
                // Schema
                R"""({
            "type": "boolean"
        })""",
                // Passing strings
                {
                    "true",
                    "false",
                },
                // Failing strings
                {
                    R"""("")""",
                    R"""("true")""",
                    R"""(True)""",
                    R"""(FALSE)""",
                });

    test_schema("integer",
                // Schema
                R"""({
            "type": "integer"
        })""",
                // Passing strings
                {
                    R"""(0)""",
                    R"""(12345)""",
                    R"""(1234567890123456)""",
                },
                // Failing strings
                {
                    R"""()""",
                    R"""(01)""",
                    R"""(007)""",
                    R"""(12345678901234567  )""",
                });

    test_schema("string const",
                // Schema
                R"""({
            "const": "foo"
        })""",
                // Passing strings
                {
                    R"""("foo")""",
                },
                // Failing strings
                {
                    R"""(foo)""",
                    R"""("bar")""",
                });

    test_schema("non-string const",
                // Schema
                R"""({
            "const": true
        })""",
                // Passing strings
                {
                    R"""(true)""",
                },
                // Failing strings
                {
                    R"""()""",
                    R"""(foo)""",
                    R"""("true")""",
                });

    test_schema("non-string const",
                // Schema
                R"""({
            "enum": ["red", "amber", "green", null, 42, ["foo"]]
        })""",
                // Passing strings
                {
                    R"""("red")""",
                    R"""(null)""",
                    R"""(42)""",
                    R"""(["foo"])""",
                },
                // Failing strings
                {
                    R"""()""",
                    R"""(420)""",
                    R"""(true)""",
                    R"""(foo)""",
                });

    test_schema("simple pattern",
                // Schema
                R"""({
            "pattern": "^[a-zA-Z0-9_-]*$"
        })""",
                // Passing strings
                {
                    R"""("")""",
                    R"""("He_llo-12")""",
                },
                // Failing strings
                {
                    R"""("!")""",
                    R"""("Hello World")""",
                });

    test_schema("pattern with escapes",
                // Schema
                R"""({
            "pattern": "^a\\^\\$\\.\\[\\]\\(\\)\\|\\{\\}\\*\\+\\?b$"
        })""",
                // Passing strings
                {
                    R"""("a^$.[]()|{}*+?b")""",
                },
                // Failing strings
                {
                    R"""("ab")""",
                });

    test_schema("",
                // Schema
                R"""(
            {
                "type": ["array", "null"],
                "items": { "type": "string" }
            }
        )""",
                // Passing strings
                {
                    "null",
                    "[]",
                    "[\"123\"]",
                    "[\"foo\", \"bar\"]",
                },
                // Failing strings
                {
                    "",
                    "[123]",
                    "\"foo\"",
                    "[\"foo\", 42]",
                });

    test_schema("min+max items",
                // Schema
                R"""({
            "items": {
                "type": ["number", "integer"]
            },
            "minItems": 3,
            "maxItems": 5
        })""",
                // Passing strings
                {
                    R"""([1, 2, 3])""",
                    R"""([1, 2, 3, 4])""",
                    R"""([1, 2, 3, 4, 5])""",
                    // this is in fact correct; keyword do not apply if the type is wrong
                    R"""(1)""",
                },
                // Failing strings
                {
                    R"""([1, 2])""",
                    R"""([1, 2, 3, 4, 5, 6])""",
                });

    // Properties (from: https://json-schema.org/understanding-json-schema/reference/object#properties)
    test_schema("object properties",
                // Schema
                R"""({
            "type": "object",
            "properties": {
                "number": { "type": "number" },
                "street_name": { "type": "string" },
                "street_type": { "enum": ["Street", "Avenue", "Boulevard"] }
            },
            "additionalProperties": false
        })""",
                // Passing strings
                {
                    R"""({ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue"})""",
                    // "By default, leaving out properties is valid"
                    R"""({ "street_name": "Pennsylvania" })""",
                    R"""({ "number": 1600, "street_name": "Pennsylvania" })""",
                    // "By extension, even an empty object is valid"
                    R"""({})""",
                    R"""({ "number": 1600, "street_name": "Pennsylvania", "street_type": "Avenue" })""",
                },
                // Failing strings
                {
                    // Change datatype from number to string
                    R"""({ "number": "1600", "street_name": "Pennsylvania", "street_type":"Avenue"})""",
                    // Reorder properties
                    R"""({ "street_name": "Pennsylvania", "number": 1600 })""",
                    // Reorder properties
                    R"""({ "number": "1600", "street_name": "Pennsylvania", "street_type":"Avenue"})""",
                    // Additional properties set to false
                    R"""({ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue", "direction":"NW"})""",

                });

    test_schema("additional properties can't override other properties",
                R"""({
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "additionalProperties": true
        })""",
                // Passing strings
                {
                    R"""({"a": 42})""",
                    R"""({"c": ""})""",
                    R"""({"a": 42, "c": ""})""",
                    R"""({"a_": ""})""",
                },
                // Failing strings
                {
                    R"""()""",
                    R"""({"a": ""})""",
                    R"""({"a": "", "b": ""})""",
                });

    // Properties (from: https://json-schema.org/understanding-json-schema/reference/object#properties)
    test_schema("object properties, additionalProperties: true",
                // Schema
                R"""({
            "type": "object",
            "properties": {
                "number": { "type": "number" },
                "street_name": { "type": "string" },
                "street_type": { "enum": ["Street", "Avenue", "Boulevard"] }
            },
            "additionalProperties": true
        })""",
                // Passing strings
                {
                    // "By extension, even an empty object is valid"
                    R"""({})""",
                    R"""({"number":1600,"street_name":"Pennsylvania","street_type":"Avenue"})""",
                    // "By default, leaving out properties is valid"
                    R"""({ "street_name": "Pennsylvania" })""",
                    R"""({ "number": 1600, "street_name": "Pennsylvania" })""",
                    // "By default, providing additional properties is valid"
                    R"""({ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue", "direction":"NW"})""",
                    R"""({ "number": 1600, "street_name": "Pennsylvania", "street_type": "Avenue" })""",
                },
                // Failing strings
                {
                    // Change datatype from number to string
                    R"""({ "number": "1600", "street_name": "Pennsylvania", "street_type":"Avenue"})""",
                    // Reorder properties
                    R"""({ "street_name": "Pennsylvania", "number": 1600, "street_type":"Avenue"})""",
                });

    // Additional properties: false
    test_schema(
        "required + optional props each in original order",
        // Schema
        R"""({
            "type": "object",
            "properties": {
                "number": { "type": "number" },
                "street_name": { "type": "string" },
                "street_type": { "enum": ["Street", "Avenue", "Boulevard"] }
            },
            "additionalProperties": false
        })""",
        // Passing strings
        {
            R"""({ "street_name": "Pennsylvania" })""",
            R"""({ "number": 1600, "street_type":"Avenue"})""",
            R"""({ "number": 1600, "street_name": "Pennsylvania" })""",
            R"""({ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue"})""",
            // Spaces are permitted around enum values
            R"""({ "number": 1600, "street_name": "Pennsylvania", "street_type": "Avenue" })""",
        },
        // Failing strings
        {
            // Reorder properties
            R"""({ "street_type": "Avenue", "number": 1600 })""",
            // Add "direction"
            R"""({ "number": 1600, "street_name": "Pennsylvania", "street_type": "Avenue", "direction": "NW" })""",
        });

    test_schema("required + optional props each in original order",
                // Schema
                R"""({
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "string"},
                "d": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        })""",
                // Passing strings
                {
                    R"""({"b": "foo", "a": "bar"})""",
                    R"""({"b":"foo","a":"bar","d":"qux"})""",
                    R"""({"b":"foo", "a":"bar", "d":"qux", "c":"baz"})""",
                },
                // Failing strings
                {
                    R"""({"a": "foo", "b": "bar"})""",
                    R"""({"b": "bar"})""",
                    R"""({"a": "foo", "c": "baz"})""",
                    R"""({"a":"foo", "b":"bar", "c":"baz", "d":"qux"})""",
                });

    // NOTE: Example from https://json-schema.org/learn/getting-started-step-by-step#define-required-properties
    test_schema(
        "required props",
        // Schema
        R"""({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/product.schema.json",
            "title": "Product",
            "description": "A product from Acme's catalog",
            "type": "object",
            "properties": {
                "productId": {
                "description": "The unique identifier for a product",
                "type": "integer"
                },
                "productName": {
                "description": "Name of the product",
                "type": "string"
                },
                "price": {
                "description": "The price of the product",
                "type": "number",
                "exclusiveMinimum": 0
                },
                "tags": {
                "description": "Tags for the product",
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 1,
                "DISABLED_uniqueItems": true
                },
                "dimensions": {
                "type": "object",
                "properties": {
                    "length": {
                    "type": "number"
                    },
                    "width": {
                    "type": "number"
                    },
                    "height": {
                    "type": "number"
                    }
                },
                "required": [ "length", "width", "height" ]
                }
            },
            "required": [ "productId", "productName", "price" ]
        })""",
        // Passing strings
        {
            R"""({"productId": 1, "productName": "A green door", "price": 12.50})""",
            R"""({"productId": 1, "productName": "A green door", "price": 12.50, "tags": ["home", "green"]})""",
            R"""({"productId": 1, "productName": "A green door", "price": 12.50, "tags": ["home", "green"], "dimensions": {"length": 785, "width": 250.5, "height": -0.359}})""",
        },
        // Failing strings
        {
            R"""({})""",  // Missing all required properties
            R"""({"productName": "A green door", "price": 12.50, "productId": 1})""",  // Out of order properties
            // `exclusiveMinimum` is OK for llg
            R"""({"productId": 1, "productName": "A green door", "price": -12.50})""",
            R"""({"productId": 1, "productName": "A green door"})""",  // Missing required property (price)
            R"""({"productName": "A green door", "price": 12.50})""",  // Missing required property (productId)
            R"""({"productId": 1, "productName": "A green door", "price": 12.50, "tags": []})""",  // tags is empty, but minItems is 1
            R"""({"productId": 1, "productName": "A green door", "price": 12.50, "dimensions": {"length": 785, "width": 250.5, "height": -0.359}, "tags": ["home", "green"]})""",  // Tags and dimensions are out of order
            // TODO: The following line should fail, but currently it passes. `uniqueItems` is not supported, as it would likely be too difficult to implement.
            // R"""({"productId": 1, "productName": "A green door", "price": 12.50, "tags": ["home", "green", "home"]})""",
        });
}

static void one_hot(llama_token_data_array & tok_arr, llama_token selected) {
    auto n_vocab = tok_arr.size;

    tok_arr.selected = -1;
    tok_arr.sorted   = false;
    for (llama_token token_id = 0; token_id < (llama_token) n_vocab; token_id++) {
        tok_arr.data[token_id].id    = token_id;
        tok_arr.data[token_id].logit = 0.0f;
    }

    tok_arr.data[selected].logit = 100.0f;
}

static void test_sampler_chain(void) {
    auto sparams            = llama_sampler_chain_default_params();
    sparams.no_perf         = false;
    llama_sampler * sampler = llama_sampler_chain_init(sparams);

    const auto grammar_data = R"(%llguidance {}
start: /[A-Z ]*/)";

    llama_sampler_chain_add(sampler, llama_sampler_init_llg(vocab, "lark", grammar_data));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    auto input  = "ALL YOUR BASE ARE BELONG TO US";
    auto tokens = common_tokenize(vocab, input, false, false);

    auto n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token_data> cur;
    cur.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token) n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{ token_id, 0.0f, 0.0f });
    }
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    for (const auto token : tokens) {
        one_hot(tok_arr, token);

        fprintf(stderr, "applying token: %d\n", token);
        llama_sampler_apply(sampler, &tok_arr);

        auto idx = tok_arr.selected;
        fprintf(stderr, " -> %d %f\n", cur[idx].id, cur[idx].logit);
        assert(cur[tok_arr.selected].id == token);
        llama_sampler_accept(sampler, token);
    }

    auto tok_eos = llama_vocab_eot(vocab);
    if (tok_eos == LLAMA_TOKEN_NULL) {
        tok_eos = llama_vocab_eos(vocab);
    }

    one_hot(tok_arr, tok_eos);

    llama_sampler_apply(sampler, &tok_arr);
    assert(cur[tok_arr.selected].id == tok_eos);
}

// The Gemma 4 sampling grammar, compiled through llguidance for real.
//
// This closes a gap that was documented in test-chat.cpp and left open: that
// suite skips Lark grammars, because it builds constraints through the GBNF
// path. So the constraint the server ACTUALLY samples under when built with
// LLAMA_LLGUIDANCE=ON -- the whole point of this fork -- was compiled nowhere in
// the test suite, and a grammar that llguidance rejects outright would have gone
// unnoticed while every test passed.
//
// Both entry rules are exercised, because a resumed generation enters at
// `resume_reasoning` rather than `start` and that is a different compiled
// grammar. See common_chat_grammar_set_entry.
static void test_gemma4_chat_grammar(const std::string & grammars_dir) {
    common_chat_grammar_init(grammars_dir);
    std::string base = common_chat_grammar_get("gemma4");
    if (base.empty()) {
        fprintf(stderr, "gemma4 grammar not found in %s\n", grammars_dir.c_str());
        assert(false);
    }

    // Substitute the response-schema placeholder exactly as the server does for a
    // request with no response_format (an empty schema: any JSON value).
    //
    // This MUST match production. Compiling the raw registry text instead leaves
    // `{{RESPONSE_SCHEMA}}` in place, which is not valid Lark -- and a grammar
    // that fails to compile fails OPEN, so every string matches and the whole
    // test passes vacuously while asserting nothing at all.
    // ALL occurrences, as the server does. The FIRST one is in the grammar's own
    // header comment, so replacing only the first leaves the real rule untouched
    // and the grammar still fails to compile -- which, failing open, looks like a
    // pass. That is exactly what happened while writing this test.
    auto substitute = [&base](const std::string & placeholder, const std::string & with) {
        assert(base.find(placeholder) != std::string::npos);
        for (size_t at = base.find(placeholder); at != std::string::npos;
             at = base.find(placeholder, at + with.size())) {
            base.replace(at, placeholder.size(), with);
        }
    };
    substitute("{{RESPONSE_SCHEMA}}", "%json {\"type\": \"object\"}");
    // The no-tools form, which is what a request without `tools` produces. The
    // per-tool alternation is exercised in test_gemma4_tool_schema, through the
    // production path rather than by retyping the emitter's output here.
    substitute("{{TOOL_SCHEMA}}", "tool_call_directive \":\" func_name gemma4_dict?");
    // The no-caller-grammar form. With one supplied it becomes a subgrammar
    // reference, which needs the grammar LIST encoding and so cannot be
    // compiled as a bare Lark string; test_gemma4_user_grammar covers that
    // through the production path.
    substitute("{{USER_GRAMMAR}}", "content");
    // Tools declared: the cases below include tool calls, which are only
    // representable when the request offered tools. test_gemma4_fsm_conformance
    // is where the no-tools form is walked.
    substitute("{{TOOL_CALL_ALT}}", "tool_call_request |");
    substitute("{{EXTRA_CHANNELS}}", "");

    // Wire strings are built from the tag vocabulary, not retyped. `<|` opens,
    // `<NAME|>` closes, `<|NAME|>` is self-delimiting -- the same three forms the
    // grammar names -- so a change to the delimiters does not mean editing every
    // string here by hand.
    auto open  = [](const std::string & n) { return "<|" + n + ">"; };
    auto close = [](const std::string & n) { return "<" + n + "|>"; };
    auto self  = [](const std::string & n) { return "<|" + n + "|>"; };

    const std::string thought_open  = open("channel") + "thought";
    const std::string thought_close = close("channel");
    const std::string call_open     = open("tool_call") + "call:";
    const std::string call_close    = close("tool_call");
    const std::string q             = self("\"");
    // A tool-calling turn's required closer: it does not end at <turn|>, it hands
    // over by OPENING the response block and stopping there. See tool_call_request.
    const std::string await_resp = open("tool_response");

    // Exactly what the server hands llguidance for an ordinary request: the
    // {{RESPONSE_SCHEMA}} placeholder is substituted only when the caller asked
    // for a response_format, so on every other request the grammar goes through
    // as-is. If that does not compile, nothing does.
    test("gemma4 sampling grammar (entry: turn_start)",
         common_chat_grammar_set_entry(base, "turn_start"),
         {
             "Hello, world!",
             thought_open + "\nthinking" + thought_close + "Hello, world!",
             call_open + "get_time{city:" + q + "London" + q + "}" + call_close + await_resp,
             thought_open + "\nchecking" + thought_close + call_open + "f{}" + call_close + await_resp,
             // S1's `object?`: `call:name` with the braces omitted is a
             // well-formed zero-argument call.
             call_open + "f" + call_close + await_resp,
             // A dotted MCP name. Nothing in the tag vocabulary stops it, and with
             // no tools declared FUNC_NAME is what bounds the name.
             call_open + "filesystem.read_file{}" + call_close + await_resp,
         },
         {
             // Two thought openers with no close between them. `content` may hold
             // a stray CLOSE tag (see the note on LT_TEXT -- excluding those would
             // cost the model `<html>` in ordinary prose), but it can never hold
             // an OPEN tag, so a second `<|channel>` has to be a real one and the
             // first thought must have closed before it.
             thought_open + "\n" + thought_open,
             // The closer is required, not optional: a turn cannot simply stop
             // after its calls (POLICIES.md#closing-tokens-are-required).
             call_open + "f{}" + call_close,
         });

    // MEASURED, UNRESOLVED: llguidance ACCEPTS a generation that stops inside an
    // unclosed thought -- `<|channel>thought\nstill thinking` with no
    // `<channel|>` is reported as a complete match, EOS and all. The grammar
    // plainly requires the closer (`channel_block: channel_open channel_body
    // channel_close_tag`), and the extraction parser rejects it, so this is the
    // sampler's token-level enforcement disagreeing with the character-level
    // grammar, not a grammar bug.
    //
    // It matters: "closing tokens are required" is supposed to mean the sampler
    // cannot stop mid-thought, and this says it can. Deliberately NOT asserted
    // either way here -- asserting it valid would bless behaviour that looks
    // wrong, asserting it invalid would fail on behaviour not yet understood.
    // Needs a minimal repro against llguidance before either.

    // Resuming inside a thought the caller prefilled: the opener is already in
    // the prompt, so the model continues the body and must close it.
    test("gemma4 sampling grammar (entry: resume_reasoning)",
         common_chat_grammar_set_entry(base, "resume_reasoning"),
         {
             " thinking" + thought_close + "Hello, world!",
             " thinking" + thought_close + call_open + "f{}" + call_close + await_resp,
         },
         {
             // Opening a SECOND thought while already inside one, and dropping
             // into content leaving the first unclosed. Both were reachable while
             // the sampler was pinned to the fresh-turn entry, which is exactly
             // what entry selection fixes.
             thought_open + "\nnested" + thought_close + "Hi",
             "Hello, world!",
         });
}

// The per-tool alternation, compiled from the REQUEST rather than from a
// hand-typed copy of what the emitter is believed to produce.
//
// It goes through common_chat_templates_apply, so what llguidance is handed here
// is byte-for-byte what the server hands it for the same request. Retyping the
// expected Lark instead would only assert that the test author and the emitter
// agree, which is the assertion least worth making: the emitter could be wrong
// in exactly the way the copy is.
static void test_gemma4_tool_schema() {
    common_chat_tool get_time{
        /* .name = */ "get_time",
        /* .description = */ "Get the current time in a city",
        /* .parameters = */ R"({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" },
                "utc":  { "type": "boolean", "description": "Report in UTC" }
            },
            "required": ["city"]
        })",
    };
    // No arguments at all -- the commonest shape an agent declares, and the one
    // S1 lets a call spell without braces.
    common_chat_tool ping{
        /* .name = */ "system.ping",
        /* .description = */ "Check liveness",
        /* .parameters = */ R"({ "type": "object", "properties": {} })",
    };

    common_chat_msg user;
    user.role    = "user";
    user.content = "what time is it in London?";

    common_chat_templates_inputs inputs;
    inputs.messages             = { user };
    inputs.tools                = { get_time, ping };
    inputs.add_generation_prompt = true;

    auto tmpls  = common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, /* chat_template_override= */ "",
                                   /* bos_token_override= */ "", /* eos_token_override= */ "",
                                   /* chat_format_override= */ "gemma4"));
    auto params = common_chat_templates_apply(tmpls.get(), inputs);
    assert(!params.grammar.empty());

    const std::string call_open  = "<|tool_call>call:";
    const std::string call_close = "<tool_call|>";
    const std::string await_resp = "<|tool_response>";
    const std::string q          = "<|\"|>";

    test("gemma4 tool schema (declared tools)", params.grammar,
         {
             call_open + "get_time{city:" + q + "London" + q + "}" + call_close + await_resp,
             call_open + "get_time{city:" + q + "London" + q + ",utc:true}" + call_close + await_resp,
             // A dotted name reaches the sampler as a literal, so MCP-style names
             // need nothing from FUNC_NAME.
             call_open + "system.ping{}" + call_close + await_resp,
             // Braces omitted: legal precisely because `ping` has no required
             // property. `get_time` does, so the same spelling is rejected below.
             call_open + "system.ping" + call_close + await_resp,
         },
         {
             // Undeclared tool.
             call_open + "not_a_real_tool{city:" + q + "x" + q + "}" + call_close + await_resp,
             // Undeclared argument on a declared tool.
             call_open + "get_time{bogus:" + q + "x" + q + "}" + call_close + await_resp,
             // Declared argument, wrong type.
             call_open + "get_time{city:123}" + call_close + await_resp,
             // Required argument missing, both spellings.
             call_open + "get_time{}" + call_close + await_resp,
             call_open + "get_time" + call_close + await_resp,
             // One tool's name with another tool's arguments -- the correlation
             // the per-tool alternation exists to enforce.
             call_open + "system.ping{city:" + q + "London" + q + "}" + call_close + await_resp,
         });

    // tool_choice: "required" -- the turn cannot end without a call.
    //
    // Content is not merely discouraged here, it is absent from the production:
    // `content` is an unbounded regex, so allowing it before a required call
    // says "you must call eventually" without ever requiring the model to stop
    // talking. Measured live against a 12B when the rule still permitted it: on
    // a chatty prompt the model answered in prose and then degenerated into
    // repeated emoji until max_tokens, because EOS stayed masked (the call was
    // still owed) and more content was always legal.
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto required_params = common_chat_templates_apply(tmpls.get(), inputs);

    test("gemma4 tool schema (tool_choice: required)", required_params.grammar,
         {
             call_open + "get_time{city:" + q + "London" + q + "}" + call_close + await_resp,
             call_open + "system.ping" + call_close + await_resp,
         },
         {
             // A turn that answers instead of calling.
             "I'd be happy to help with that!",
             // Content first, then a call: still rejected, because content is
             // what makes "required" unbounded.
             "Let me check." + call_open + "system.ping" + call_close + await_resp,
         });
}

// What llguidance ACTUALLY permitted, token by token.
//
// Every other test here is black-box: it asks whether a finished string is
// accepted. That cannot distinguish "the sampler steered the model through the
// format" from "the model happened to produce something valid" -- and it cannot
// see a grammar that failed to compile and is therefore constraining nothing,
// which is this project's recurring failure mode.
//
// So this walks a generation the way the sampler does: at each position, apply
// the grammar to a flat logit array, count how many of the 262k tokens survive,
// and record whether ordinary prose was among them. `n_allowed == 1` means the
// step was FORCED -- the model had no choice at all.
struct mask_step {
    llama_token              accepted;
    std::string              piece;
    size_t                   n_allowed;
    bool                     text_allowed;  // could the model have written prose instead?
    std::vector<std::string> allowed;       // the whole legal set, when it is small
    // Which of the format's markers survived the mask here. Counting tokens says
    // how MANY were legal; this says WHICH, and the official ordering is a claim
    // about which marker may follow which -- so it is the only form of the
    // question that can check conformance against the spec.
    std::set<std::string>    allowed_markers;
};

// Every control token Google's prompt-formatting page names, by the spelling the
// grammar uses. Built from the tokenizer, so a marker that is not one token in
// this vocabulary fails here rather than silently testing as prose.
static std::vector<std::pair<std::string, llama_token>> marker_probes() {
    static const char * const kMarkers[] = {
        "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>",
        "<|tool_response>", "<tool_response|>", "<|tool>", "<tool|>", "<|\"|>", "<|think|>",
    };
    std::vector<std::pair<std::string, llama_token>> out;
    for (const char * m : kMarkers) {
        auto ids = common_tokenize(vocab, m, false, true);
        assert(ids.size() == 1 && "a Gemma 4 control token must be exactly one token");
        out.emplace_back(m, ids[0]);
    }
    return out;
}

// Above this, the set is a free-text run and listing it is noise.
static constexpr size_t kSmallMaskLimit = 8;

static std::vector<mask_step> walk_mask(const std::string &              input,
                                        llama_sampler *                  grammar,
                                        const std::vector<llama_token> & text_probes,
                                        bool *                           eos_allowed_at_end,
                                        const std::vector<std::pair<std::string, llama_token>> & markers = {}) {
    llama_sampler_reset(grammar);
    // parse_special=true: this stands in for tokens a model emitted, not for a
    // spelling a user typed. See the note on match_string.
    auto      tokens  = common_tokenize(vocab, input, false, true);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token_data> cur;
    cur.reserve(n_vocab);
    for (llama_token t = 0; t < n_vocab; t++) {
        cur.emplace_back(llama_token_data{ t, 0.0f, 0.0f });
    }
    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };

    std::vector<mask_step> out;
    for (auto tok : tokens) {
        for (llama_token t = 0; t < n_vocab; t++) {
            cur[t].logit = 0.0f;
        }
        llama_sampler_apply(grammar, &arr);

        mask_step s;
        s.accepted     = tok;
        s.n_allowed    = 0;
        s.text_allowed = false;
        for (llama_token t = 0; t < n_vocab; t++) {
            if (cur[t].logit >= 0.0f) {
                s.n_allowed++;
            }
        }
        for (auto p : text_probes) {
            if (cur[p].logit >= 0.0f) {
                s.text_allowed = true;
                break;
            }
        }
        for (const auto & [name, id] : markers) {
            if (cur[id].logit >= 0.0f) {
                s.allowed_markers.insert(name);
            }
        }
        if (s.n_allowed <= kSmallMaskLimit) {
            for (llama_token t = 0; t < n_vocab; t++) {
                if (cur[t].logit < 0.0f) {
                    continue;
                }
                char    tb[256];
                int32_t tl = llama_detokenize(vocab, &t, 1, tb, sizeof(tb), false, true);
                s.allowed.emplace_back(tb, tl > 0 ? (size_t) tl : 0);
            }
            std::sort(s.allowed.begin(), s.allowed.end());
        }
        char    buf[256];
        int32_t len = llama_detokenize(vocab, &tok, 1, buf, sizeof(buf), false, true);
        s.piece.assign(buf, len > 0 ? (size_t) len : 0);
        out.push_back(s);

        llama_sampler_accept(grammar, tok);
    }

    if (eos_allowed_at_end != nullptr) {
        for (llama_token t = 0; t < n_vocab; t++) {
            cur[t].logit = 0.0f;
        }
        llama_sampler_apply(grammar, &arr);
        auto eos = llama_vocab_eot(vocab);
        if (eos == LLAMA_TOKEN_NULL) {
            eos = llama_vocab_eos(vocab);
        }
        *eos_allowed_at_end = cur[eos].logit >= 0.0f;
    }
    return out;
}

// The FSM contract, asserted at the token level.
//
// One tool is declared (`get_time`, required string `city`) with
// tool_choice=required, so the whole turn is determined except for the argument
// text. Every structural position is checked for the property that matters
// there: that prose was masked out, and that the step was forced where only one
// continuation is legal.
static void test_gemma4_mask_walk() {
    common_chat_tool get_time{
        /* .name = */ "get_time",
        /* .description = */ "Get the current time in a city",
        /* .parameters = */ R"({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        })",
    };

    common_chat_msg user;
    user.role    = "user";
    user.content = "what time is it in London?";

    common_chat_templates_inputs inputs;
    inputs.messages              = { user };
    inputs.tools                 = { get_time };
    inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking       = false;

    auto tmpls  = common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, /* chat_template_override= */ "",
                                   /* bos_token_override= */ "", /* eos_token_override= */ "",
                                   /* chat_format_override= */ "gemma4"));
    auto params = common_chat_templates_apply(tmpls.get(), inputs);
    assert(!params.grammar.empty());

    auto * grammar = llama_sampler_init_llg(vocab, "lark", params.grammar.c_str());
    assert(grammar != nullptr);

    // Probes for "could this have been ordinary prose instead?". Common English
    // pieces; if any survives the mask, the model was free to write text.
    std::vector<llama_token> text_probes;
    for (const char * w : { "The", " the", "Hello", " I", "Sure", " a" }) {
        auto ids = common_tokenize(vocab, w, false, false);
        if (!ids.empty()) {
            text_probes.push_back(ids[0]);
        }
    }
    assert(!text_probes.empty());

    const std::string generation =
        "<|tool_call>call:get_time{city:<|\"|>London<|\"|>}<tool_call|><|tool_response>";

    bool eos_at_end = false;
    auto steps      = walk_mask(generation, grammar, text_probes, &eos_at_end);
    assert(!steps.empty());

    fprintf(stderr, "\n  llguidance mask walk (tool_choice=required, 1 tool declared):\n");
    fprintf(stderr, "    %-4s %-18s %9s %7s %7s  %s\n",
            "step", "emitted", "allowed", "forced", "prose?", "legal set (when small)");
    for (size_t i = 0; i < steps.size(); i++) {
        const auto & s = steps[i];
        std::string  shown;
        for (char c : s.piece) {
            shown += (c == '\n') ? '.' : c;
        }
        std::string legal;
        for (const auto & a : s.allowed) {
            legal += (legal.empty() ? "" : " | ") + a;
        }
        fprintf(stderr, "    %-4zu %-18s %9zu %7s %7s  %s\n", i, shown.c_str(), s.n_allowed,
                s.n_allowed == 1 ? "YES" : "", s.text_allowed ? "OPEN" : "masked", legal.c_str());
    }
    fprintf(stderr, "    EOS allowed at end: %s\n", eos_at_end ? "yes" : "NO");

    // 1. The turn has exactly ONE legal first token, and the two constraints
    //    that produce that are worth reading together. tool_choice=required
    //    removed the content branch; `enable_thinking: false` put the
    //    empty-thought prefill in the prompt, which leaves the model at
    //    IN_CONTENT -- an entry whose rules carry no `channel_block?`, because
    //    the thought phase is over.
    //
    //    So opening a thought is not merely unlikely on a thinking-off call, it
    //    is unrepresentable. This used to read
    //    `{ "<|channel>", "<|tool_call>" }`: the state shared `turn_start`, and a
    //    request that asked for no thinking could open a thought channel anyway
    //    -- the "ghost channel" the prefill exists to suppress, still reachable
    //    through the grammar.
    assert(!steps[0].text_allowed);
    assert((steps[0].allowed == std::vector<std::string>{ "<|tool_call>" }));
    assert(steps[0].n_allowed == 1);

    // 2. Nowhere in a tool call is prose legal EXCEPT inside the string argument,
    //    whose body is deliberately unrestricted (a city name is arbitrary text).
    //    Locate that span by the delimiter token so the assertion does not depend
    //    on a hardcoded step index.
    const llama_token str_delim = common_tokenize(vocab, "<|\"|>", false, true).at(0);
    bool              in_string = false;
    size_t            prose_outside_string = 0;
    for (const auto & s : steps) {
        if (s.accepted == str_delim) {
            in_string = !in_string;
            continue;
        }
        if (!in_string && s.text_allowed) {
            prose_outside_string++;
        }
    }
    assert(prose_outside_string == 0);

    // 3. Closing the call is FORCED -- one legal token, no choice.
    const size_t n = steps.size();
    assert(steps[n - 2].piece == "<tool_call|>");
    assert(steps[n - 2].n_allowed == 1);

    // 4. After a call closes, the FSM offers exactly two continuations: another
    //    call, or the hand-over. Nothing else -- no prose, no `<turn|>`, no
    //    second thought.
    //
    //    This is where an assertion of `n_allowed == 1` was first written, and the
    //    walk corrected it: `tool_calls: tool_call+` means parallel calls are
    //    legal, so two is the right answer and one would have been a grammar that
    //    forbids them. Reading the mask is what told the difference; a
    //    conformance run never would have.
    assert(steps[n - 1].piece == "<|tool_response>");
    assert((steps[n - 1].allowed == std::vector<std::string>{ "<|tool_call>", "<|tool_response>" }));

    // 5. And the turn may end at the hand-over -- but only there.
    assert(eos_at_end);

    llama_sampler_free(grammar);
    fprintf(stderr, "  \xE2\x9C\x85\xEF\xB8\x8E mask walk\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Conformance: the mask at EVERY FSM state, against the documented ordering
// ─────────────────────────────────────────────────────────────────────────────
//
// test_gemma4_mask_walk above proves the sampler steers one representative
// generation. It cannot answer the question that matters -- whether what the
// mask permits is what the FORMAT permits, at every position, in every state
// the FSM can start generation from.
//
// The ordering being checked is Google's, from the Gemma 4 prompt-formatting
// page (ai.google.dev/gemma/docs/core/prompt-formatting-gemma4):
//
//   <|turn>model -> <|channel>thought -> <channel|> -> <|tool_call>call:
//                -> <tool_call|> -> <|tool_response>response: -> <tool_response|>
//                -> final text -> <turn|>
//
// Each case below states, at a named point in a generation, which control tokens
// MUST be legal and which MUST NOT. A "must not" is the half a conformance run
// can never see: a model that simply does not emit `<|channel>` inside a thought
// looks identical to a grammar that forbids it.

struct fsm_checkpoint {
    std::string               after;        // the generation prefix already accepted
    std::vector<const char *> must_allow;   // markers legal at this point
    std::vector<const char *> must_deny;    // markers the format does not permit here
    int                       prose = -1;   // 1 = free text legal, 0 = masked, -1 = don't care
};

struct fsm_case {
    const char *                entry_root;  // the FSM state's generation rule
    const char *                label;
    std::string                 generation;
    std::vector<fsm_checkpoint> checkpoints;
    // Did the request declare tools? Decides whether the tool-call alternative
    // is in the grammar at all -- which is the difference between "the model
    // chose not to call" and "the model could not".
    bool                        tools = false;
};

// Compile a grammar rooted at one entry rule, with the per-request placeholders
// filled the way production fills them.
static std::string gemma4_grammar_at(const std::string & base, const std::string & root, bool tools) {
    auto sub = [](std::string s, const std::string & ph, const std::string & with) {
        for (size_t at = s.find(ph); at != std::string::npos; at = s.find(ph, at + with.size())) {
            s.replace(at, ph.size(), with);
        }
        return s;
    };
    std::string g = base;
    g = sub(g, "{{TOOL_SCHEMA}}", "tool_call_directive \":\" func_name gemma4_dict?");
    g = sub(g, "{{RESPONSE_SCHEMA}}", "%json {\"type\": \"object\"}");
    g = sub(g, "{{USER_GRAMMAR}}", "content");
    g = sub(g, "{{TOOL_CALL_ALT}}", tools ? "tool_call_request |" : "");
    // Sampling gets one thought channel per turn; the repetition is the parser's.
    g = sub(g, "{{EXTRA_CHANNELS}}", "");
    return common_chat_grammar_set_entry(g, root);
}

static void run_fsm_case(const std::string & base, const fsm_case & tc,
                         const std::vector<std::pair<std::string, llama_token>> & markers,
                         const std::vector<llama_token> & text_probes) {
    const std::string grammar = gemma4_grammar_at(base, tc.entry_root, tc.tools);
    auto *            smpl    = llama_sampler_init_llg(vocab, "lark", grammar.c_str());
    // Null here means the grammar did not compile. That used to be survivable --
    // the sampler came back non-null and constrained nothing -- so this assert
    // is load-bearing, not defensive.
    assert(smpl != nullptr && "entry grammar failed to compile");

    const auto all = common_tokenize(vocab, tc.generation, false, true);
    auto       steps = walk_mask(tc.generation, smpl, text_probes, nullptr, markers);
    assert(steps.size() == all.size());

    fprintf(stderr, "\n  %s  (start: %s, tools %s)\n", tc.label, tc.entry_root,
            tc.tools ? "declared" : "none");
    for (const auto & cp : tc.checkpoints) {
        const std::string & prefix = cp.after;
        const auto          pre    = common_tokenize(vocab, prefix, false, true);
        // The checkpoint is located by tokenizing its prefix on its own, so it
        // must actually BE a prefix of the whole generation's tokenization --
        // otherwise the step index would silently point somewhere else.
        assert(pre.size() <= all.size());
        for (size_t i = 0; i < pre.size(); i++) {
            assert(pre[i] == all[i] && "checkpoint prefix does not tokenize as a prefix");
        }
        const auto & s = steps[pre.size()];

        std::string shown;
        for (const auto & m : s.allowed_markers) { shown += (shown.empty() ? "" : " ") + m; }
        fprintf(stderr, "    after %-46s prose=%-6s markers: %s\n",
                ("\"" + prefix + "\"").substr(0, 46).c_str(),
                s.text_allowed ? "OPEN" : "masked", shown.empty() ? "(none)" : shown.c_str());

        for (const char * m : cp.must_allow) {
            if (s.allowed_markers.count(m) == 0) {
                fprintf(stderr, "    FAIL: %s must be legal after \"%s\" but is masked\n", m, prefix.c_str());
                assert(false);
            }
        }
        for (const char * m : cp.must_deny) {
            if (s.allowed_markers.count(m) != 0) {
                fprintf(stderr, "    FAIL: %s must NOT be legal after \"%s\" but the mask allows it\n",
                        m, prefix.c_str());
                assert(false);
            }
        }
        if (cp.prose >= 0 && s.text_allowed != (cp.prose == 1)) {
            fprintf(stderr, "    FAIL: prose should be %s after \"%s\"\n",
                    cp.prose == 1 ? "legal" : "masked", prefix.c_str());
            assert(false);
        }
    }
    llama_sampler_free(smpl);
}

static void test_gemma4_fsm_conformance(const std::string & grammars_dir) {
    common_chat_grammar_init(grammars_dir);
    const std::string base = common_chat_grammar_get("gemma4");
    assert(!base.empty());

    const auto markers = marker_probes();

    std::vector<llama_token> text_probes;
    for (const char * w : { "The", " the", "Hello", " I", "Sure", " a" }) {
        auto ids = common_tokenize(vocab, w, false, false);
        if (!ids.empty()) { text_probes.push_back(ids[0]); }
    }
    assert(!text_probes.empty());

    const std::string TH_OPEN  = "<|channel>thought";
    const std::string TH_CLOSE = "<channel|>";
    const std::string CALL     = "<|tool_call>call:";
    const std::string CALL_END = "<tool_call|>";
    const std::string HANDOVER = "<|tool_response>";

    const std::vector<fsm_case> cases = {
        // ── INITIAL / IN_GENERATION_PROMPT / IN_CONTENT, no demand ────────────
        {
            "turn_start", "fresh model turn, tools declared",
            TH_OPEN + "\nthinking" + TH_CLOSE + "Hello." + "<turn|>",
            {
                // A turn may open with a thought, open with a call, answer
                // directly, or be empty -- all four are the format's, and all
                // four must be reachable.
                { "", { "<|channel>", "<|tool_call>", "<turn|>" },
                      { "<channel|>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
                        "<|turn>", "<|tool>", "<tool|>" }, 1 },
                // A channel's KIND is a production, so `<|channel>` alone leaves
                // only the word "thought" -- no marker may follow it directly.
                { TH_OPEN.substr(0, 10), { }, { "<channel|>", "<|channel>", "<|tool_call>", "<turn|>" }, 0 },
                // Inside a thought: prose, and the closer. NOT a second thought,
                // and NOT a tool call -- the ordering puts the call after
                // `<channel|>`, never inside the channel.
                { TH_OPEN, { "<channel|>" }, { "<|channel>", "<|tool_call>", "<turn|>", "<|tool_response>" }, 1 },
                // Back in content after the thought closes -- and a SECOND
                // thought is not on the menu. One channel per generated turn is
                // what the documented ordering has, and permitting more was an
                // unbounded loop the model fell into live.
                { TH_OPEN + "\nthinking" + TH_CLOSE,
                  { "<|tool_call>", "<turn|>" },
                  { "<|channel>", "<channel|>", "<tool_call|>", "<|tool_response>", "<|turn>" }, 1 },
                // And the model can end its own turn from content.
                { TH_OPEN + "\nthinking" + TH_CLOSE + "Hello.",
                  { "<turn|>", "<|tool_call>" },
                  { "<|channel>", "<channel|>", "<tool_call|>", "<|turn>" }, 1 },
            },
            /* tools = */ true,
        },
        // ── the same turn, with NO tools declared ─────────────────────────────
        //
        // The one that was wrong. `{{TOOL_SCHEMA}}` degrades to any-name/any-args
        // when nothing was declared, so leaving the alternative in let a plain
        // chat request open a call to a function that does not exist. It is not
        // enough that the model usually would not: the mask has to make it
        // unrepresentable, and every OTHER marker must stay exactly as it was.
        {
            "turn_start", "fresh model turn, NO tools declared",
            TH_OPEN + "\nthinking" + TH_CLOSE + "Hello." + "<turn|>",
            {
                { "", { "<|channel>", "<turn|>" },
                      { "<|tool_call>", "<channel|>", "<tool_call|>", "<|tool_response>" }, 1 },
                { TH_OPEN, { "<channel|>" }, { "<|tool_call>", "<|channel>", "<turn|>" }, 1 },
                { TH_OPEN + "\nthinking" + TH_CLOSE,
                  { "<turn|>" }, { "<|tool_call>", "<|channel>", "<channel|>" }, 1 },
                { TH_OPEN + "\nthinking" + TH_CLOSE + "Hello.",
                  { "<turn|>" }, { "<|tool_call>", "<|channel>", "<channel|>" }, 1 },
            },
            /* tools = */ false,
        },
        // ── the tool-call ordering, exactly as documented ─────────────────────
        {
            "turn_start", "tool call: call -> args -> close -> hand-over",
            CALL + "get_time{city:<|\"|>London<|\"|>}" + CALL_END + HANDOVER,
            {
                // Having opened a call, the directive is the only continuation:
                // no prose, and no marker at all.
                { "<|tool_call>", { }, { "<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>",
                                         "<turn|>", "<|tool_response>", "<|\"|>" }, 0 },
                // A string argument is delimited by <|"|>, and inside it the
                // body is deliberately unrestricted -- a value is arbitrary text.
                { CALL + "get_time{city:", { "<|\"|>" }, { "<tool_call|>", "<|tool_response>", "<turn|>" } },
                // The hand-over is not reachable until the call has closed.
                { CALL + "get_time{city:<|\"|>London<|\"|>}",
                  { "<tool_call|>" }, { "<|tool_response>", "<turn|>", "<|channel>" }, 0 },
                // Closed: another call, or the hand-over. `<turn|>` is NOT one of
                // them -- a calling turn hands over, it does not close.
                { CALL + "get_time{city:<|\"|>London<|\"|>}" + CALL_END,
                  { "<|tool_call>", "<|tool_response>" }, { "<turn|>", "<|channel>", "<tool_call|>" }, 0 },
            },
            /* tools = */ true,
        },
        // ── IN_REASONING: generation resumes inside a thought already opened ──
        {
            "resume_reasoning", "resume inside an open thought",
            "still thinking" + TH_CLOSE + "Answer." + "<turn|>",
            {
                // The opener is already in the prompt. Emitting another would be
                // a thought inside a thought; this entry exists precisely so it
                // is unrepresentable.
                { "", { "<channel|>" }, { "<|channel>", "<|tool_call>", "<turn|>", "<|tool_response>" }, 1 },
                { "still thinking", { "<channel|>" }, { "<|channel>", "<|tool_call>", "<turn|>" }, 1 },
                { "still thinking" + TH_CLOSE, { "<turn|>", "<|tool_call>" },
                  { "<|channel>", "<channel|>", "<|turn>" }, 1 },
            },
            /* tools = */ true,
        },
        // ── tool_choice: "required" ───────────────────────────────────────────
        {
            "turn_start_tool_call", "tool_choice=required leaves no way not to call",
            TH_OPEN + "\npicking" + TH_CLOSE + CALL + "f{}" + CALL_END + HANDOVER,
            {
                // Think first, or call. Answering is not on the menu, and
                // neither is ending the turn.
                { "", { "<|channel>", "<|tool_call>" }, { "<turn|>", "<channel|>", "<|tool_response>" }, 0 },
                // Even after the thought closes, the call is still owed.
                { TH_OPEN + "\npicking" + TH_CLOSE, { "<|tool_call>" },
                  { "<turn|>", "<|channel>", "<|tool_response>" }, 0 },
            },
            /* tools = */ true,
        },
        // ── response_format: json_schema ──────────────────────────────────────
        {
            "turn_start_response_format", "a schema demand, no tools to call instead",
            "```json\n{}\n```",
            {
                // The fence, or a thought first. Not prose, which is the whole
                // difference between a demand and a suggestion.
                { "", { "<|channel>" }, { "<turn|>", "<|tool_call>", "<channel|>" }, 0 },
            },
            /* tools = */ false,
        },
        {
            "turn_start_response_format", "a schema demand, or a tool call instead",
            CALL + "f{}" + CALL_END + HANDOVER,
            {
                // Both branches live, and prose still is not one of them. The
                // tool alternative does not weaken the demand: it is bounded, so
                // "answer in the schema" has not degraded to "eventually".
                { "", { "<|channel>", "<|tool_call>" }, { "<turn|>", "<channel|>" }, 0 },
            },
            /* tools = */ true,
        },
        // ── response_format: a caller's own grammar ───────────────────────────
        {
            "turn_start_user_grammar", "a caller grammar stands where content would",
            // {{USER_GRAMMAR}} is `content` here: with a real caller grammar the
            // production is a subgrammar reference, which needs the grammar-list
            // encoding. test_gemma4_user_grammar drives that one live.
            "anything at all<turn|>",
            {
                { "", { "<|channel>", "<turn|>" }, { "<channel|>", "<|tool_call>", "<|tool_response>" }, 1 },
            },
            /* tools = */ false,
        },
        {
            "turn_start_user_grammar", "either call a tool, or answer in the grammar",
            CALL + "f{}" + CALL_END + HANDOVER,
            {
                // THE case this alternative exists for -- an agent stage saying
                // "here are your tools, and if you are not going to use one, the
                // answer looks like THIS".
                { "", { "<|channel>", "<|tool_call>" }, { "<channel|>", "<tool_call|>" }, 1 },
                // Having opened a call, the grammar's branch is gone: it is a
                // choice, not a concatenation.
                { "<|tool_call>", { }, { "<turn|>", "<|channel>", "<tool_call|>" }, 0 },
            },
            /* tools = */ true,
        },
        // ── and every demand again from the OTHER state ───────────────────────
        // The resume entries are where getting this wrong is invisible: the
        // opener is already in the prompt, so a grammar that still expects one
        // reads the rest of the thought as content and loses the reasoning.
        {
            "resume_reasoning_tool_call", "resumed mid-thought, still owes a call",
            "deciding" + TH_CLOSE + CALL + "f{}" + CALL_END + HANDOVER,
            {
                { "", { "<channel|>" }, { "<|channel>", "<|tool_call>", "<turn|>" }, 1 },
                // Thought closed, and the only way on is the call.
                { "deciding" + TH_CLOSE, { "<|tool_call>" }, { "<turn|>", "<|channel>", "<channel|>" }, 0 },
            },
            /* tools = */ true,
        },
        {
            "resume_reasoning_response_format", "resumed mid-thought, still owes a schema",
            "deciding" + TH_CLOSE + "```json\n{}\n```",
            {
                { "", { "<channel|>" }, { "<|channel>", "<|tool_call>", "<turn|>" }, 1 },
                { "deciding" + TH_CLOSE, { }, { "<turn|>", "<|tool_call>", "<|channel>", "<channel|>" }, 0 },
            },
            /* tools = */ false,
        },
        {
            "resume_reasoning_user_grammar", "resumed mid-thought, then the caller's grammar",
            "deciding" + TH_CLOSE + "anything<turn|>",
            {
                { "", { "<channel|>" }, { "<|channel>", "<|tool_call>", "<turn|>" }, 1 },
                { "deciding" + TH_CLOSE, { "<turn|>" }, { "<channel|>", "<|tool_call>" }, 1 },
            },
            /* tools = */ false,
        },
        {
            "resume_reasoning_user_grammar", "resumed mid-thought, tool call still available",
            "deciding" + TH_CLOSE + CALL + "f{}" + CALL_END + HANDOVER,
            {
                { "", { "<channel|>" }, { "<|channel>", "<|tool_call>", "<turn|>" }, 1 },
                // Thought closed: call a tool, or answer in the grammar. Both.
                { "deciding" + TH_CLOSE, { "<turn|>", "<|tool_call>" }, { "<channel|>" }, 1 },
            },
            /* tools = */ true,
        },
        // ── IN_CONTENT: the thought phase is already over ─────────────────────
        //
        // Reached by the empty-thought prefill (every `enable_thinking: false`
        // request), by continue_final_message mid-answer, and by carrying on
        // after a tool response with thinking off. In all three the prompt has
        // passed the thought, so `<|channel>` must be gone from the very first
        // mask -- these four entries are the turn_start ones minus that opener.
        {
            "resume_content_tool_call", "thinking off + tool_choice=required: the call is FORCED",
            CALL + "f{}" + CALL_END + HANDOVER,
            {
                // One legal token, and it is the call. No thought to hide in and
                // no prose to stall with -- the strongest guarantee in the file.
                { "", { "<|tool_call>" },
                     { "<|channel>", "<channel|>", "<turn|>", "<|tool_response>" }, 0 },
            },
            /* tools = */ true,
        },
        {
            "resume_content_response_format", "thinking off + a schema: straight to the block",
            "```json\n{}\n```",
            {
                { "", { }, { "<|channel>", "<channel|>", "<turn|>", "<|tool_call>" }, 0 },
            },
            /* tools = */ false,
        },
        {
            "resume_content_user_grammar", "thinking off + a caller grammar",
            "anything<turn|>",
            {
                { "", { "<turn|>" }, { "<|channel>", "<channel|>", "<|tool_call>" }, 1 },
            },
            /* tools = */ false,
        },
    };

    for (const auto & tc : cases) {
        run_fsm_case(base, tc, markers, text_probes);
    }

    // EVERY entry root the registry can hand out has a case above.
    //
    // Without this the suite covers whichever roots someone remembered to write
    // a case for, and a fifth demand added later would be untested while the
    // test still reported "FSM conformance: N cases". The registry enumerates
    // itself (common_chat_gemma4_entry_roots_all), so the coverage check cannot
    // go stale either.
    std::set<std::string> covered;
    for (const auto & tc : cases) {
        covered.insert(tc.entry_root);
    }
    for (const auto & root : common_chat_gemma4_entry_roots_all()) {
        if (covered.count(root) == 0) {
            fprintf(stderr, "    FAIL: entry root '%s' is selectable but no conformance case walks it\n",
                    root.c_str());
            assert(false);
        }
    }
    fprintf(stderr, "\n  \xE2\x9C\x85\xEF\xB8\x8E FSM conformance: %zu cases over all %zu entry roots\n",
            cases.size(), common_chat_gemma4_entry_roots_all().size());
}

// ─────────────────────────────────────────────────────────────────────────────
// response_format: lark_grammar / gbnf_grammar
// ─────────────────────────────────────────────────────────────────────────────
//
// Through the PRODUCTION path -- common_chat_templates_apply -- because what is
// under test is the composition, not a grammar string retyped here. The result
// is an llguidance grammar LIST, so this is also the only test that exercises
// that encoding end to end.
//
// The two syntaxes are asserted to produce the SAME mask, which is what makes
// the GBNF converter checkable: it is a port of llguidance's own script, and
// "the converted grammar constrains identically" is the property that matters
// rather than the text it emits.
static void test_gemma4_user_grammar() {
    const auto markers = marker_probes();
    std::vector<llama_token> text_probes;
    for (const char * w : { "The", " the", "Hello", " I", "Sure", " a" }) {
        auto ids = common_tokenize(vocab, w, false, false);
        if (!ids.empty()) { text_probes.push_back(ids[0]); }
    }

    common_chat_msg user;
    user.role    = "user";
    user.content = "is the sky blue?";

    auto tmpls = common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, /* chat_template_override= */ "",
                                   /* bos_token_override= */ "", /* eos_token_override= */ "",
                                   /* chat_format_override= */ "gemma4"));

    // `thinking` decides which FSM state the prompt leaves the model in, and so
    // which entry the grammar is rooted at: on -> the prompt stops at
    // `<|turn>model` (IN_GENERATION_PROMPT, may open a thought); off -> the
    // empty-thought prefill lands it at IN_CONTENT, past the thought phase.
    auto compile_for = [&](const std::string & grammar, bool thinking) {
        common_chat_templates_inputs inputs;
        inputs.messages              = { user };
        inputs.grammar               = grammar;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = thinking;
        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        // The composed artefact is a grammar LIST, not a bare Lark string: the
        // caller's grammar is the second entry, referenced by name from the
        // first. One matcher comes out of it, which is the point.
        assert(params.grammar.compare(0, 12, "{\"grammars\":") == 0);
        assert(params.grammar.find("response_grammar") != std::string::npos);
        auto * smpl = llama_sampler_init_llg(vocab, "llguidance", params.grammar.c_str());
        assert(smpl != nullptr && "composed grammar list failed to compile");
        return smpl;
    };

    // The same language written both ways, chosen to cover the parts of the
    // conversion that are not a transliteration: character classes become
    // regexes, `{m,n}` repetition carries over, groups nest, `root` is renamed
    // to `start`.
    const struct {
        const char * label;
        const char * lark;
        const char * gbnf;
        const char * generation;
    } pairs[] = {
        { "literal alternation",
          "%llguidance {}\nstart: \"yes\" | \"no\"\n",
          "root ::= \"yes\" | \"no\"\n",
          "yes<turn|>" },
        { "character class + repetition",
          "%llguidance {}\nstart: /[a-z]/{2,4} \"!\"\n",
          "root ::= [a-z]{2,4} \"!\"\n",
          "abc!<turn|>" },
        { "nested group, referenced rule",
          "%llguidance {}\nstart: \"(\" item (\",\" item)* \")\"\nitem: /[0-9]/+\n",
          "root ::= \"(\" item (\",\" item)* \")\"\nitem ::= [0-9]+\n",
          "(1,22)<turn|>" },
    };

    for (const auto & p : pairs) {
        // Both thinking settings, because they root the grammar at different
        // entries and the difference is exactly what the caller asked for:
        // thinking on, the model may reason first and then answer in the
        // grammar; thinking off, it goes straight to the answer and cannot open
        // a channel at all.
        auto * lark    = compile_for(p.lark, /* thinking = */ false);
        auto * gbnf    = compile_for(p.gbnf, /* thinking = */ false);
        auto * lark_th = compile_for(p.lark, /* thinking = */ true);

        for (auto * smpl : { lark, gbnf, lark_th }) {
            const bool thinking = (smpl == lark_th);
            auto steps = walk_mask(p.generation, smpl, text_probes, nullptr, markers);
            assert(!steps.empty());

            // The caller stated the answer's shape, so prose is not an answer and
            // the mask has to say so. This is the whole difference between a
            // grammar that is enforced and one that was merely sent.
            assert(!steps[0].text_allowed);
            // The caller constrained the ANSWER, not the reasoning channel -- so
            // a thought is available exactly when the request asked for one, and
            // never otherwise. With thinking off the empty-thought prefill has
            // already opened and closed one, and a second is unrepresentable.
            assert(steps[0].allowed_markers.count("<|channel>") == (thinking ? 1u : 0u));
            // ...and either way the turn cannot end before an answer is given.
            assert(steps[0].allowed_markers.count("<turn|>") == 0);

            // Having answered, closing the turn is all that is left.
            const auto & last = steps[steps.size() - 1];
            assert(last.piece == "<turn|>");
            assert(!last.text_allowed);
            assert(last.allowed_markers.count("<turn|>") == 1);
        }

        // Identical masks, step for step. This is the converter's real
        // assertion: not that it emits particular text, but that the grammar it
        // produces constrains the model the same way the Lark original does.
        auto a = walk_mask(p.generation, lark, text_probes, nullptr, markers);
        auto b = walk_mask(p.generation, gbnf, text_probes, nullptr, markers);
        assert(a.size() == b.size());
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i].n_allowed != b[i].n_allowed || a[i].allowed_markers != b[i].allowed_markers) {
                fprintf(stderr, "    FAIL: %s diverges at step %zu (lark %zu allowed, gbnf %zu)\n",
                        p.label, i, a[i].n_allowed, b[i].n_allowed);
                assert(false);
            }
        }
        fprintf(stderr, "    %-32s lark == gbnf across %zu steps\n", p.label, a.size());

        llama_sampler_free(lark);
        llama_sampler_free(gbnf);
        llama_sampler_free(lark_th);
    }
    fprintf(stderr, "  \xE2\x9C\x85\xEF\xB8\x8E response_format lark_grammar/gbnf_grammar: same mask, prose masked\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// The entry rule is chosen BY the FSM state, not by the request alone
// ─────────────────────────────────────────────────────────────────────────────
//
// The conformance walk above shows each entry rule masks correctly. That is only
// half the claim: the other half is that the rule generation actually starts at
// is the one the state the RENDERED PROMPT left the model in calls for.
//
// It is worth pinning because it has already been wrong in the way that does not
// show. Generation resuming inside a prefilled thought was compiled at
// `turn_start`, which expects a `<|channel>thought` opener the delta does not
// have -- so the model was free to open a second thought inside the first, or to
// drop into content leaving the first unclosed, and only extraction was any the
// wiser.
static std::string entry_root_of(const std::string & grammar) {
    // With a caller grammar composed in, this is an llguidance grammar LIST and
    // the format grammar is its first entry. Decoding it rather than scanning
    // the raw string matters: the caller's grammar has a `start:` rule of its
    // own, and a scan finds that one.
    std::string lark = grammar;
    if (grammar.compare(0, 12, "{\"grammars\":") == 0) {
        lark = nlohmann::json::parse(grammar).at("grammars").at(0).at("lark_grammar").get<std::string>();
    }
    // `start: X` is appended last by common_chat_grammar_set_entry.
    const size_t at = lark.rfind("\nstart: ");
    if (at == std::string::npos) {
        return {};
    }
    size_t      i = at + strlen("\nstart: ");
    std::string name;
    while (i < lark.size() && (isalnum((unsigned char) lark[i]) || lark[i] == '_')) {
        name += lark[i++];
    }
    return name;
}

// The whitespace the response-schema grammar allows between JSON tokens.
//
// This guards a SILENT corruption, which is why it is asserted structurally
// rather than left to the live suite. Forbidding whitespace (`whitespace_flexible:
// false`, which stood here once) makes the model's own top candidates after
// `"colour":` -- ` "`, ` "#`, ` ["`, all leading with a space -- illegal, so the
// mask falls through to a legal MERGED token (`">`, `":`, `"]`) and the extra
// character lands inside the string where every byte is legal. The result is
// `{"colour":">Blue"}`: schema-valid, corrupt, and invisible to every conformance
// check there is. Measured 16/20 on a free-string schema.
//
// Allowing unbounded whitespace is the opposite failure -- the model emits
// newlines forever and never reaches the comma -- so the pattern has to be
// non-empty AND newline-free, and each clause below pins one of those.
static void test_gemma4_response_schema_whitespace() {
    auto tmpls = common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, /* chat_template_override= */ "",
                                   /* bos_token_override= */ "", /* eos_token_override= */ "",
                                   /* chat_format_override= */ "gemma4"));

    common_chat_msg user;
    user.role    = "user";
    user.content = "hello";

    const std::string schema = R"({"type":"object","properties":{"colour":{"type":"string"}},"required":["colour"]})";

    auto grammar_for = [&](const std::string & json_schema) {
        common_chat_templates_inputs inputs;
        inputs.messages             = { user };
        inputs.json_schema          = json_schema;
        inputs.add_generation_prompt = true;
        return common_chat_templates_apply(tmpls.get(), inputs).grammar;
    };

    // The %json RULE the composer emits, so the assertions below are about the
    // grammar actually handed to llguidance and not about this file's idea of it.
    //
    // Anchored on the rule name rather than on "%json ", because the placeholder
    // is substituted everywhere it appears -- including in gemma4.lark's own
    // comment describing it, which therefore also ends up holding a %json line
    // and is what an unanchored search finds first.
    auto json_line = [](const std::string & grammar) {
        const std::string rule = "response_content: %json ";
        const auto        at   = grammar.find(rule);
        if (at == std::string::npos) {
            fprintf(stderr, "    FAIL: composed grammar carries no response_content rule\n");
            assert(false);
        }
        return grammar.substr(at, grammar.find('\n', at) - at);
    };

    {
        const auto line = json_line(grammar_for(schema));

        // NO x-guidance IS ADDED. This assertion is inverted from what it was,
        // because the setting it used to require was measured to be the worst
        // of the three options rather than the best.
        //
        // The reasoning that produced it: llguidance's default whitespace can
        // repeat, so pin it to a single space instead. The flaw: the pattern
        // becomes the grammar's SKIP, and a skip re-applies between every pair
        // of tokens, so " " means "one space, arbitrarily often" -- pinning it
        // bounds nothing. Worse, removing the newline makes a model trained on
        // pretty-printed JSON pile probability onto the one whitespace
        // character still legal. Measured on a nested-array schema, 12B, N=25:
        //   default [\x20\x0A\x0D\x09]+   0/25 runaway
        //   whitespace_pattern " "         4/25 runaway (~36 kB of spaces)
        //   whitespace_flexible false      0/25 runaway, but 6/20 clean strings
        // The default is the only setting clean in BOTH regimes.
        if (line.find("x-guidance") != std::string::npos) {
            fprintf(stderr, "    FAIL: an x-guidance override was injected; the default is what works\n    %s\n",
                    line.c_str());
            assert(false);
        }
        fprintf(stderr, "\n    no whitespace override injected (llguidance default)\n");
    }

    {
        // The documented escape hatch: a caller who states their own x-guidance
        // keeps it. Claimed in a comment for a while before anything checked it.
        auto caller = json::parse(schema);
        caller["x-guidance"] = json{ { "whitespace_pattern", "[ \\t]" } };
        const auto line = json_line(grammar_for(caller.dump()));
        // The needle is the pattern as it appears in the DUMPED grammar, where the
        // backslash is JSON-escaped -- searching for the C++ string finds nothing
        // and reads as an overwrite that never happened.
        if (line.find("[ \\\\t]") == std::string::npos) {
            fprintf(stderr, "    FAIL: caller's own x-guidance was overwritten\n    %s\n", line.c_str());
            assert(false);
        }
        fprintf(stderr, "    caller-supplied x-guidance preserved\n");
    }

    fprintf(stderr, "  \xE2\x9C\x85\xEF\xB8\x8E response-schema whitespace: llguidance default, caller override wins\n");
}

static void test_gemma4_entry_selection() {
    auto tmpls = common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, /* chat_template_override= */ "",
                                   /* bos_token_override= */ "", /* eos_token_override= */ "",
                                   /* chat_format_override= */ "gemma4"));

    common_chat_msg user;
    user.role    = "user";
    user.content = "hello";

    common_chat_tool get_time{
        /* .name = */ "get_time",
        /* .description = */ "Get the time",
        /* .parameters = */ R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
    };

    // A prefilled, still-open thought. The renderer leaves the model mid-
    // `reasoning`, which is the one state whose entry rules differ.
    common_chat_msg half_thought;
    half_thought.role              = "assistant";
    half_thought.reasoning_content = "let me think";

    // A completed round trip: the model called a tool, the runtime answered, and
    // the turn is still open for the model to continue in.
    common_chat_msg tool_caller;
    tool_caller.role = "assistant";
    tool_caller.tool_calls.push_back({ "get_time", R"({"city":"Oslo"})", "call_1" });

    common_chat_msg tool_result;
    tool_result.role         = "tool";
    tool_result.content      = "13:45";
    tool_result.tool_call_id = "call_1";

    struct sel_case {
        const char *                 label;
        std::vector<common_chat_msg> messages;
        common_chat_continuation     continuation;
        bool                         generation_prompt;
        std::vector<common_chat_tool> tools;
        common_chat_tool_choice      tool_choice;
        std::string                  json_schema;
        std::string                  grammar;
        const char *                 expect_root;
        // Thinking off makes the renderer emit the empty-thought prefill, which
        // is what puts the model at IN_CONTENT rather than IN_GENERATION_PROMPT.
        bool                         thinking = true;
    };

    const std::vector<sel_case> cases = {
        { "fresh turn, no demand",
          { user }, COMMON_CHAT_CONTINUATION_NONE, true, {}, COMMON_CHAT_TOOL_CHOICE_AUTO, "", "",
          "turn_start" },
        { "fresh turn, tool_choice=required",
          { user }, COMMON_CHAT_CONTINUATION_NONE, true, { get_time }, COMMON_CHAT_TOOL_CHOICE_REQUIRED, "", "",
          "turn_start_tool_call" },
        { "fresh turn, response_format json_schema",
          { user }, COMMON_CHAT_CONTINUATION_NONE, true, {}, COMMON_CHAT_TOOL_CHOICE_AUTO,
          R"({"type":"object"})", "",
          "turn_start_response_format" },
        { "fresh turn, response_format lark_grammar",
          { user }, COMMON_CHAT_CONTINUATION_NONE, true, {}, COMMON_CHAT_TOOL_CHOICE_AUTO, "",
          "%llguidance {}\nstart: \"yes\"\n",
          "turn_start_user_grammar" },
        // The state changes and every root changes with it -- same four demands,
        // four different rules, decided by where the prompt stopped.
        { "resumed mid-thought, no demand",
          { user, half_thought }, COMMON_CHAT_CONTINUATION_REASONING, false, {},
          COMMON_CHAT_TOOL_CHOICE_AUTO, "", "",
          "resume_reasoning" },
        { "resumed mid-thought, tool_choice=required",
          { user, half_thought }, COMMON_CHAT_CONTINUATION_REASONING, false, { get_time },
          COMMON_CHAT_TOOL_CHOICE_REQUIRED, "", "",
          "resume_reasoning_tool_call" },
        { "resumed mid-thought, response_format json_schema",
          { user, half_thought }, COMMON_CHAT_CONTINUATION_REASONING, false, {},
          COMMON_CHAT_TOOL_CHOICE_AUTO, R"({"type":"object"})", "",
          "resume_reasoning_response_format" },
        { "resumed mid-thought, response_format lark_grammar",
          { user, half_thought }, COMMON_CHAT_CONTINUATION_REASONING, false, {},
          COMMON_CHAT_TOOL_CHOICE_AUTO, "", "%llguidance {}\nstart: \"yes\"\n",
          "resume_reasoning_user_grammar" },
        // And a third state. `enable_thinking: false` makes the renderer emit
        // the empty-thought prefill, which opens AND closes a thought in the
        // prompt -- so the model is at IN_CONTENT, past the thought phase, and
        // gets entries that cannot open another.
        // No demand: `turn_start`, thought opener and all. Dropping it here
        // protects nothing and cornered the model -- see gemma4.lark.
        { "thinking off (prefill), no demand",
          { user }, COMMON_CHAT_CONTINUATION_NONE, true, {}, COMMON_CHAT_TOOL_CHOICE_AUTO, "", "",
          "turn_start", /* thinking = */ false },
        { "thinking off (prefill), tool_choice=required",
          { user }, COMMON_CHAT_CONTINUATION_NONE, true, { get_time }, COMMON_CHAT_TOOL_CHOICE_REQUIRED, "", "",
          "resume_content_tool_call", /* thinking = */ false },
        { "thinking off (prefill), response_format json_schema",
          { user }, COMMON_CHAT_CONTINUATION_NONE, true, {}, COMMON_CHAT_TOOL_CHOICE_AUTO,
          R"({"type":"object"})", "",
          "resume_content_response_format", /* thinking = */ false },
        { "thinking off (prefill), response_format lark_grammar",
          { user }, COMMON_CHAT_CONTINUATION_NONE, true, {}, COMMON_CHAT_TOOL_CHOICE_AUTO, "",
          "%llguidance {}\nstart: \"yes\"\n",
          "resume_content_user_grammar", /* thinking = */ false },
        // After a TOOL RESPONSE the turn is still open and the model is about to
        // speak in it for the first time -- so it may think or answer, and the
        // entry has to be the one that permits both.
        //
        // Thinking on, the renderer emits the re-opener and generation resumes
        // INSIDE it. Thinking off it emits nothing, and this used to fall
        // through to IN_CONTENT's default -- which was harmless only while
        // IN_CONTENT still allowed a leading thought. It does not any more, and
        // the 12B degenerated into emitting the word "thought" as prose.
        { "after a tool response, thinking on",
          { user, tool_caller, tool_result }, COMMON_CHAT_CONTINUATION_NONE, true, { get_time },
          COMMON_CHAT_TOOL_CHOICE_AUTO, "", "",
          "resume_reasoning", /* thinking = */ true },
        { "after a tool response, thinking off",
          { user, tool_caller, tool_result }, COMMON_CHAT_CONTINUATION_NONE, true, { get_time },
          COMMON_CHAT_TOOL_CHOICE_AUTO, "", "",
          "turn_start", /* thinking = */ false },
    };

    fprintf(stderr, "\n  entry rule selected from the FSM state:\n");
    for (const auto & tc : cases) {
        common_chat_templates_inputs inputs;
        inputs.messages              = tc.messages;
        inputs.tools                 = tc.tools;
        inputs.tool_choice           = tc.tool_choice;
        inputs.json_schema           = tc.json_schema;
        inputs.grammar               = tc.grammar;
        inputs.continue_final_message = tc.continuation;
        inputs.add_generation_prompt  = tc.generation_prompt;
        inputs.enable_thinking        = tc.thinking;

        auto        params = common_chat_templates_apply(tmpls.get(), inputs);
        const auto  root   = entry_root_of(params.grammar);
        fprintf(stderr, "    %-46s -> %s\n", tc.label, root.c_str());
        if (root != tc.expect_root) {
            fprintf(stderr, "    FAIL: expected %s\n", tc.expect_root);
            assert(false);
        }
    }
    fprintf(stderr, "  \xE2\x9C\x85\xEF\xB8\x8E entry selection: %zu cases\n", cases.size());
}

// The startup contract check must actually FAIL when the contract is broken.
//
// A check that has only ever been observed passing is indistinguishable from one
// that always passes -- and this one guards a failure mode (an entry rule that
// does not exist, so llguidance constrains nothing) whose whole character is
// that it looks fine.
static void test_gemma4_contract_check_catches_drift(const std::string & grammars_dir) {
    common_chat_grammar_init(grammars_dir);
    std::string base = common_chat_grammar_get("gemma4");
    assert(!base.empty());

    auto sub = [](std::string s, const std::string & ph, const std::string & with) {
        for (size_t at = s.find(ph); at != std::string::npos; at = s.find(ph, at + with.size())) {
            s.replace(at, ph.size(), with);
        }
        return s;
    };
    // The PEG spellings, which are what chat_grammar_to_peg substitutes -- the
    // transpiler has no `%json`, and the schema was already enforced at sampling
    // time anyway.
    base = sub(base, "{{TOOL_SCHEMA}}", "tool_call_directive \":\" func_name gemma4_dict?");
    base = sub(base, "{{RESPONSE_SCHEMA}}", "__JSON_VALUE__");
    base = sub(base, "{{USER_GRAMMAR}}", "content");
    base = sub(base, "{{TOOL_CALL_ALT}}", "tool_call_request |");
    base = sub(base, "{{EXTRA_CHANNELS}}", "(channel_block content)*");

    // Intact: the contract holds. (common_chat_grammar_init already asserted
    // this at startup; repeating it here is what makes the negative case below
    // meaningful rather than a check of nothing.)
    common_chat_gemma4_check_state_rule_contract(common_lark_to_peg(base, "conversation"));

    // Now remove one selectable entry rule, exactly the way it would go missing:
    // somebody renames or deletes a rule and the table still names it.
    const std::string victim = "turn_start_user_grammar:";
    const size_t      at     = base.find("\n" + victim);
    assert(at != std::string::npos);
    const size_t eol = base.find('\n', at + 1);
    std::string  broken = base.substr(0, at) + base.substr(eol);

    bool threw = false;
    try {
        common_chat_gemma4_check_state_rule_contract(common_lark_to_peg(broken, "conversation"));
    } catch (const std::exception & e) {
        threw = true;
        // And it must SAY which rule, or the message is not actionable.
        assert(std::string(e.what()).find("turn_start_user_grammar") != std::string::npos);
    }
    assert(threw && "the state<->rule contract check did not notice a missing entry rule");
    fprintf(stderr, "  \xE2\x9C\x85\xEF\xB8\x8E state<->rule contract: holds, and fails when broken\n");
}

// Isolation harness: compile ONE grammar file and report. Driven by an env var
// so a grammar can be bisected from the shell without recompiling C++.
//   LLG_GRAMMAR_FILE=/path/to.lark ./test-grammar-llguidance <vocab>
static int compile_only(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    std::string src;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) { src.append(buf, n); }
    fclose(f);

    auto * smpl = llama_sampler_init_llg(vocab, "lark", src.c_str());
    if (smpl == nullptr) {
        fprintf(stdout, "COMPILE: sampler is null\n");
        return 1;
    }
    // A compile failure surfaces at first apply, not at init (it fails OPEN).
    const char * probe = getenv("LLG_MATCH");
    const std::string input = probe ? probe : "";
    const bool ok = match_string(input, smpl);
    fprintf(stdout, "COMPILE: init ok, match(%s)=%d\n", input.c_str(), (int) ok);
    llama_sampler_free(smpl);
    return 0;
}

// The renderer must not emit a BOS the tokenizer is going to add.
//
// This has to be asserted with a REAL bos token, which is why it did not exist
// before: every other test in the tree constructs templates with
// `bos_token_override = ""`, so none of them has ever had a BOS to duplicate. The
// defect was therefore invisible offline and showed up as one
// `check_double_bos_eos` warning per request in a server log -- two BOS tokens at
// the front of every prompt, shifting every position after them.
//
// `add_bos` mirrors `llama_vocab_get_add_bos()`, and the server tokenizes the
// rendered prompt with add_special=true. So the rule is: the renderer emits BOS
// only when the tokenizer will not.
static void test_gemma4_bos_not_doubled() {
    const std::string bos = "<bos>";

    // Driven at the RENDERER, not through common_chat_templates_apply: that
    // function derives add_bos from the templates object, which in turn reads it
    // off a vocab. Going through it would mean loading two models that differ
    // only in `add_bos`, to test four lines. The renderer is where the decision
    // is made and where its inputs are directly expressible.
    auto rendered_with = [&](bool add_bos) {
        common_chat_render_params params;
        params.messages = json::array({
            json{ { "role", "user" }, { "content", "hi" } },
        });
        params.add_generation_prompt = true;
        params.add_bos               = add_bos;
        return common_chat_gemma4_render(params, bos).prompt;
    };

    const auto tokenizer_adds = rendered_with(true);
    if (tokenizer_adds.compare(0, bos.size(), bos) == 0) {
        fprintf(stderr, "    FAIL: renderer emitted a BOS the tokenizer will also add\n      %s\n",
                tokenizer_adds.substr(0, 40).c_str());
        assert(false);
    }

    // The other half matters too: a vocab that does NOT add BOS must still get
    // one, or the prompt is missing it entirely.
    const auto renderer_adds = rendered_with(false);
    if (renderer_adds.compare(0, bos.size(), bos) != 0) {
        fprintf(stderr, "    FAIL: no BOS from either side\n      %s\n",
                renderer_adds.substr(0, 40).c_str());
        assert(false);
    }

    fprintf(stderr, "\n    add_bos=true  -> '%s...'\n", tokenizer_adds.substr(0, 16).c_str());
    fprintf(stderr, "    add_bos=false -> '%s...'\n", renderer_adds.substr(0, 16).c_str());
    fprintf(stderr, "  \xE2\x9C\x85\xEF\xB8\x8E BOS emitted exactly once, whoever emits it\n");
}

// Format resolution: request > config > declared metadata, and nothing else.
//
// This was verified by hand -- start a server four ways and read the log line --
// which is exactly the kind of check that stops being run. It is in-process here
// because `common_chat_format_resolve` takes the model and reports its own source
// string, so every precedence claim is an assertion rather than a log grep.
//
// The two ctest registrations of this binary supply the two model shapes for
// free: llama-bpe declares an architecture with no plugin, the Gemma 4 GGUF
// declares one with a plugin. So `registered_arch` selects which half of the
// metadata behaviour to assert, and both halves run on every CI pass.
//
// The request tier is NOT here: it lives in the server's parameter parsing, and
// is covered end-to-end in tools/server/tests/unit/test_chat_format.py.
static void test_chat_format_resolution(const llama_model * model, bool registered_arch) {
    fprintf(stderr, "\n  chat-format resolution (%s architecture):\n",
            registered_arch ? "registered" : "unregistered");

    // ── config tier ──────────────────────────────────────────────────────────
    {
        std::string source;
        const auto  name = common_chat_format_resolve(model, "gemma4", &source);
        assert(name == "gemma4");
        assert(source == "--chat-format");
        fprintf(stderr, "    --chat-format gemma4              -> '%s' from %s\n", name.c_str(), source.c_str());
    }

    // Config BEATS metadata, and this is the assertion that proves it rather than
    // merely being consistent with it: the model on hand declares an
    // architecture, so if metadata were consulted first this would resolve
    // happily. It must instead throw, and blame --chat-format.
    {
        bool threw = false;
        try {
            common_chat_format_resolve(model, "definitely-not-a-format", nullptr);
        } catch (const std::exception & e) {
            threw = true;
            const std::string msg = e.what();
            assert(msg.find("definitely-not-a-format") != std::string::npos);
            assert(msg.find("--chat-format") != std::string::npos);
            // Naming what IS available is half the value of the error.
            assert(msg.find("gemma4") != std::string::npos);
            fprintf(stderr, "    --chat-format <unknown>           -> throws, blames --chat-format\n");
        }
        assert(threw);
    }

    // ── metadata tier ────────────────────────────────────────────────────────
    if (registered_arch) {
        std::string source;
        const auto  name = common_chat_format_resolve(model, "", &source);
        assert(name == "gemma4");
        assert(source == "GGUF general.architecture");
        fprintf(stderr, "    no --chat-format                  -> '%s' from %s\n", name.c_str(), source.c_str());
    } else {
        // A model this build cannot serve fails NAMING the architecture it
        // declared -- the difference between "unsupported model" and "half-works".
        bool threw = false;
        try {
            common_chat_format_resolve(model, "", nullptr);
        } catch (const std::exception & e) {
            threw = true;
            const std::string msg = e.what();
            assert(msg.find("GGUF general.architecture") != std::string::npos);
            fprintf(stderr, "    no --chat-format, no plugin       -> throws, blames the GGUF\n");
        }
        assert(threw);
    }

    // ── nothing to resolve from ──────────────────────────────────────────────
    // Not an error: common_chat_verify_template legitimately reaches here without
    // a model. The error belongs where a format is actually needed.
    {
        std::string source;
        const auto  name = common_chat_format_resolve(nullptr, "", &source);
        assert(name.empty());
        assert(source == "unresolved");
        fprintf(stderr, "    no model, no --chat-format        -> unresolved, no throw\n");
    }

    fprintf(stderr, "  \xE2\x9C\x85\xEF\xB8\x8E chat-format resolution: config > metadata, both attributed\n");
}

int main(int argc, const char ** argv) {
    fprintf(stdout, "Running llguidance integration tests...\n");

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <vocab-file> [chat-grammars-dir]\n", argv[0]);
        return 1;
    }

    const char * vocab_file = argv[1];

    fprintf(stderr, "reading vocab from: '%s'\n", vocab_file);

    llama_model *   model;
    llama_context * ctx;

    llama_backend_init();

    // load the vocab
    {
        auto mparams = llama_model_default_params();

        mparams.vocab_only = true;

        model = llama_model_load_from_file(vocab_file, mparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, vocab_file);
            return 1;
        }

        // needed?
        auto cparams = llama_context_default_params();

        ctx = llama_init_from_model(model, cparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, vocab_file);
            llama_model_free(model);
            return 1;
        }
    }

    vocab = llama_model_get_vocab(model);

    // LLG_DUMP_USER_DEFINED=1 lists every token the vocabulary declares
    // USER_DEFINED, with its text. common/llguidance.cpp hands those to
    // llguidance as special (unforgeable by a regex, addressable by name in a
    // grammar), which is only safe if none of them is ordinary text a model would
    // use in prose -- otherwise the sampler would quietly lose the ability to say
    // it. There is no API to enumerate attributes, so this walks the vocabulary.
    if (getenv("LLG_DUMP_USER_DEFINED")) {
        const int n = llama_vocab_n_tokens(vocab);
        int       count = 0;
        for (llama_token t = 0; t < n; t++) {
            if ((llama_vocab_get_attr(vocab, t) & LLAMA_TOKEN_ATTR_USER_DEFINED) == 0) {
                continue;
            }
            char   buf[512];
            int32_t len = llama_detokenize(vocab, &t, 1, buf, sizeof(buf), false, true);
            fprintf(stdout, "%d\t%.*s\n", t, len > 0 ? len : 0, buf);
            count++;
        }
        fprintf(stderr, "USER_DEFINED tokens: %d of %d\n", count, n);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    if (const char * only = getenv("LLG_GRAMMAR_FILE")) {
        int rc = compile_only(only);
        llama_free(ctx);
        llama_model_free(model);
        return rc;
    }

    // Given a grammars dir, run ONLY the Gemma 4 case.
    //
    // The two need different vocabularies and cannot share a run. llguidance
    // builds a token mask, so what a grammar accepts is a property of the
    // tokenizer: `<|channel>` is a single special token in Gemma 4 (id 100) and
    // an arbitrary run of ordinary tokens in llama-bpe. Testing the Gemma 4
    // grammar against llama-bpe measures a tokenization the model will never
    // produce.
    //
    // The reverse is also true: the generic cases above were written for
    // llama-bpe and do not survive Gemma 4's 262k-token vocabulary -- the `+`
    // quantifier case dies with "Too many items (limit 50000; mask); try
    // avoiding single-byte/short lexemes". That is a property of those test
    // grammars, not of this fork, so they stay on the vocab they were written
    // for.
    if (argc == 3) {
        test_gemma4_chat_grammar(argv[2]);
        test_gemma4_tool_schema();
        test_gemma4_mask_walk();
        test_gemma4_fsm_conformance(argv[2]);
        test_gemma4_user_grammar();
        test_gemma4_response_schema_whitespace();
        test_gemma4_bos_not_doubled();
        test_gemma4_entry_selection();
        test_chat_format_resolution(model, /* registered_arch = */ true);
        test_gemma4_contract_check_catches_drift(argv[2]);
        llama_free(ctx);
        llama_model_free(model);
        fprintf(stdout, "All tests passed.\n");
        return 0;
    }

    test_simple_grammar();
    test_complex_grammar();
    test_special_chars();
    test_quantifiers();
    test_json_schema();

    // llama-bpe declares `llama`, which has no plugin -- so this run is what
    // covers the unsupported-model half of resolution.
    test_chat_format_resolution(model, /* registered_arch = */ false);

    test_sampler_chain();

    llama_free(ctx);
    llama_model_free(model);

    fprintf(stdout, "All tests passed.\n");
    return 0;
}
