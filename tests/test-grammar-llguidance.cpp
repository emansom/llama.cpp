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
    auto tokens = common_tokenize(vocab, input, false, false);

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
    const std::string placeholder = "{{RESPONSE_SCHEMA}}";
    const std::string schema      = "%json {\"type\": \"object\"}";
    assert(base.find(placeholder) != std::string::npos);
    for (size_t at = base.find(placeholder); at != std::string::npos;
         at = base.find(placeholder, at + schema.size())) {
        base.replace(at, placeholder.size(), schema);
    }

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

    // Exactly what the server hands llguidance for an ordinary request: the
    // {{RESPONSE_SCHEMA}} placeholder is substituted only when the caller asked
    // for a response_format, so on every other request the grammar goes through
    // as-is. If that does not compile, nothing does.
    test("gemma4 sampling grammar (entry: turn_start)",
         common_chat_grammar_set_entry(base, "turn_start"),
         {
             "Hello, world!",
             thought_open + "\nthinking" + thought_close + "Hello, world!",
             call_open + "get_time{city:" + q + "London" + q + "}" + call_close,
             thought_open + "\nchecking" + thought_close + call_open + "f{}" + call_close,
         },
         {
             // Two thought openers with no close between them. `content` may hold
             // a stray CLOSE tag (see the note on LT_TEXT -- excluding those would
             // cost the model `<html>` in ordinary prose), but it can never hold
             // an OPEN tag, so a second `<|channel>` has to be a real one and the
             // first thought must have closed before it.
             thought_open + "\n" + thought_open,
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
             " thinking" + thought_close + call_open + "f{}" + call_close,
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
