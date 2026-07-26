#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "chat.h"
#include "sampling.h"

#include <cassert>
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

    auto tmpls  = common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, "gemma4"));
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
};

// Above this, the set is a free-text run and listing it is noise.
static constexpr size_t kSmallMaskLimit = 8;

static std::vector<mask_step> walk_mask(const std::string &              input,
                                        llama_sampler *                  grammar,
                                        const std::vector<llama_token> & text_probes,
                                        bool *                           eos_allowed_at_end) {
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

    auto tmpls  = common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, "gemma4"));
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

    // 1. The turn cannot begin with prose. tool_choice=required removed the
    //    content branch, so the very first token is already constrained -- to a
    //    call, or to opening a thought first (`channel_block?`).
    assert(!steps[0].text_allowed);
    assert((steps[0].allowed == std::vector<std::string>{ "<|channel>", "<|tool_call>" }));

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

    test_sampler_chain();

    llama_free(ctx);
    llama_model_free(model);

    fprintf(stdout, "All tests passed.\n");
    return 0;
}
