//  Tests chat handling, including grammar genration and parsing for tool calling, for various templates.
//
//  Also acts as a CLI to generate a Markdown summary of the formats of Jinja templates,
//  e.g. given Minja (http://github.com/google/minja) checked out in parent dir:
//
//    cmake -B build && cmake --build build --parallel && ./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
//
#include "../src/llama-grammar.h"
#include "../src/unicode.h"
#include "../tools/server/server-chat.h"
#include "chat-auto-parser.h"
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

static std::ostream & operator<<(std::ostream & os, const common_chat_msg_diff & diff) {
    os << "{ content_delta: " << diff.content_delta << "; ";
    os << "reasoning_content_delta: " << diff.reasoning_content_delta << "; ";
    if (diff.tool_call_index != std::string::npos) {
        os << "tool_call_index: " << diff.tool_call_index << "; ";
        os << "tool_call_delta.name: " << diff.tool_call_delta.name << "; ";
        os << "tool_call_delta.id: " << diff.tool_call_delta.id << "; ";
        os << "tool_call_delta.arguments: " << diff.tool_call_delta.arguments << "; ";
    }
    os << "}";
    return os;
}

// operator<< for vector<common_chat_msg_diff>:
static std::ostream & operator<<(std::ostream & os, const std::vector<common_chat_msg_diff> & diffs) {
    os << "[\n";
    for (const auto & diff : diffs) {
        os << "  " << diff << ",\n";
    }
    os << "]";
    return os;
}

static std::ostream & operator<<(std::ostream & os, const common_chat_msg & msg) {
    os << "{ role: " << msg.role << "; ";
    os << "content: " << msg.content << "; ";
    os << "content_parts: [\n";
    for (const auto & part : msg.content_parts) {
        os << "  { type: " << part.type << "; text: " << part.text << " },\n";
    }
    os << "]; ";
    os << "reasoning_content: " << msg.reasoning_content << "; ";
    os << "tool_calls: [\n";
    for (const auto & tool_call : msg.tool_calls) {
        os << "  { name: " << tool_call.name << "; arguments: " << tool_call.arguments << "; id: " << tool_call.id
           << " },\n";
    }
    os << "]";
    os << "}";
    return os;
}

template <class T> static bool equals(const T & expected, const T & actual) {
    return expected == actual;
}

static common_chat_msg normalize(const common_chat_msg & msg) {
    common_chat_msg normalized = msg;
    for (auto & tool_call : normalized.tool_calls) {
        try {
            tool_call.arguments = json::parse(tool_call.arguments).dump();
        } catch (const std::exception &) {
        }
    }
    return normalized;
}

template <> bool equals(const common_chat_msg & expected, const common_chat_msg & actual) {
    return normalize(expected) == normalize(actual);
}

template <class T> static void assert_equals(const T & expected, const T & actual) {
    if (!equals(expected, actual)) {
        std::ostringstream oss_expected;
        oss_expected << expected;
        std::ostringstream oss_actual;
        oss_actual << actual;
        LOG_ERR("Expected: %s\n", oss_expected.str().c_str());
        LOG_ERR("Actual: %s\n", oss_actual.str().c_str());
        common_log_flush(common_log_main());
        throw std::runtime_error("Test failed");
    }
}

static void assert_contains(const std::string & haystack, const std::string & needle) {
    if (haystack.find(needle) == std::string::npos) {
        LOG_ERR("Expected to contain: %s\n", needle.c_str());
        LOG_ERR("Actual: %s\n", haystack.c_str());
        common_log_flush(common_log_main());
        throw std::runtime_error("Test failed");
    }
}

static void assert_not_contains(const std::string & haystack, const std::string & needle) {
    if (haystack.find(needle) != std::string::npos) {
        LOG_ERR("Expected NOT to contain: %s\n", needle.c_str());
        LOG_ERR("Actual: %s\n", haystack.c_str());
        common_log_flush(common_log_main());
        throw std::runtime_error("Test failed");
    }
}

static void assert_ends_with(const std::string & str, const std::string & suffix) {
    if (str.size() < suffix.size() ||
        str.compare(str.size() - suffix.size(), suffix.size(), suffix) != 0) {
        LOG_ERR("Expected to end with: %s\n", suffix.c_str());
        LOG_ERR("Actual: %s\n", str.c_str());
        common_log_flush(common_log_main());
        throw std::runtime_error("Test failed");
    }
}

static std::string read_file(const std::string & path) {
    std::ifstream fs(path, std::ios_base::binary);
    if (!fs.is_open()) {
        fs = std::ifstream("../" + path, std::ios_base::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }
    }
    fs.seekg(0, std::ios_base::end);
    auto size = fs.tellg();
    fs.seekg(0);
    std::string out;
    out.resize(static_cast<size_t>(size));
    fs.read(out.data(), static_cast<std::streamsize>(size));
    return out;
}

// Build the chat-format handle a test operates on.
//
// There is no template to load. The format plugin -- renderer, validator,
// tracker, grammar -- IS the spec (docs/fork/ARCHITECTURE.md#normative-sources),
// so a test declares WHICH FORMAT it exercises, and asserts explicit expected
// bytes rather than parity against a template file.
//
// The string handed to common_chat_templates_init is retained only for /props
// reporting; nothing branches on it since format selection is by name.
static common_chat_templates_ptr gemma4_templates() {
    return common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, "gemma4"));
}

static std::unique_ptr<llama_grammar> build_grammar(const std::string & grammar_str) {
    return std::unique_ptr<llama_grammar>(
        llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0));
}

// Helper to format a code point as a readable string
static std::string format_codepoint(uint32_t cp) {
    if (cp >= 32 && cp < 127) {
        return std::string("'") + static_cast<char>(cp) + "'";
    } else if (cp == '\n') {
        return "'\\n'";
    } else if (cp == '\r') {
        return "'\\r'";
    } else if (cp == '\t') {
        return "'\\t'";
    } else {
        return "U+" + std::to_string(cp);
    }
}

// Helper to format expected element from grammar stack
static std::string format_expected_element(const llama_grammar_rules & /* rules*/, const llama_grammar_element * elem) {
    if (!elem) {
        return "<end>";
    }

    switch (elem->type) {
        case LLAMA_GRETYPE_END:
            return "<end of rule>";
        case LLAMA_GRETYPE_ALT:
            return "<alternative>";
        case LLAMA_GRETYPE_RULE_REF:
            {
                // Find rule name - just show rule ID for now
                return "<rule-" + std::to_string(elem->value) + ">";
            }
        case LLAMA_GRETYPE_CHAR:
            {
                std::string                   result;
                const llama_grammar_element * pos   = elem;
                bool                          first = true;

                do {
                    if (!first) {
                        result += " | ";
                    }
                    first = false;

                    if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                        // Range like [a-z]
                        result += "[" + format_codepoint(pos->value) + "-" + format_codepoint(pos[1].value) + "]";
                        pos += 2;
                    } else {
                        result += format_codepoint(pos->value);
                        pos += 1;
                    }
                } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

                return result;
            }
        case LLAMA_GRETYPE_CHAR_NOT:
            {
                std::string                   result = "[^";
                const llama_grammar_element * pos    = elem;
                bool                          first  = true;

                do {
                    if (!first) {
                        result += " ";
                    }
                    first = false;

                    if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                        result += format_codepoint(pos->value) + "-" + format_codepoint(pos[1].value);
                        pos += 2;
                    } else {
                        result += format_codepoint(pos->value);
                        pos += 1;
                    }
                } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

                return result + "]";
            }
        case LLAMA_GRETYPE_CHAR_ANY:
            return "<any char>";
        case LLAMA_GRETYPE_TOKEN:
            return "<token-" + std::to_string(elem->value) + ">";
        case LLAMA_GRETYPE_TOKEN_NOT:
            return "<not-token-" + std::to_string(elem->value) + ">";
        default:
            return "<unknown>";
    }
}

// Get description of what the grammar expects at current position
static std::string get_expected_description(const llama_grammar_rules & rules, const llama_grammar_stacks & stacks) {
    if (stacks.empty()) {
        return "<no valid continuations>";
    }

    std::string           result;
    std::set<std::string> seen;

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            if (seen.insert("<end>").second) {
                if (!result.empty()) {
                    result += " OR ";
                }
                result += "<end>";
            }
            continue;
        }

        const llama_grammar_element * elem = stack.back();
        std::string                   desc = format_expected_element(rules, elem);
        if (seen.insert(desc).second) {
            if (!result.empty()) {
                result += " OR ";
            }
            result += desc;
        }
    }

    return result;
}

// Result of a detailed grammar match attempt
struct grammar_match_result {
    bool        success            = false;  // Did the string fully match the grammar?
    size_t      matched_bytes      = 0;      // Bytes successfully matched before failure
    size_t      matched_codepoints = 0;      // Codepoints successfully matched before failure
    size_t      total_bytes        = 0;      // Total bytes in input
    size_t      total_codepoints   = 0;      // Total codepoints in input
    std::string matched_prefix;              // The portion that was successfully matched
    std::string failing_char;                // The character that caused failure (if any)
    std::string expected_description;        // What the grammar expected at failure point
    bool        incomplete = false;          // True if matched all input but grammar expects more
};

// Detailed version of match_string that returns failure information
static grammar_match_result match_string_detailed(const std::string & input, llama_grammar * grammar) {
    grammar_match_result result;
    result.total_bytes = input.size();

    const auto cpts         = unicode_cpts_from_utf8(input);
    result.total_codepoints = cpts.size();

    auto &       stacks_cur = llama_grammar_get_stacks(grammar);
    const auto & rules      = llama_grammar_get_rules(grammar);

    size_t byte_pos = 0;

    for (size_t i = 0; i < cpts.size(); i++) {
        const auto & cpt = cpts[i];

        // Get expected before accepting (for error reporting)
        std::string expected_before = get_expected_description(rules, stacks_cur);

        llama_grammar_accept(grammar, cpt);

        // Calculate byte position for this codepoint
        size_t cpt_bytes = 0;
        if (cpt < 0x80) {
            cpt_bytes = 1;
        } else if (cpt < 0x800) {
            cpt_bytes = 2;
        } else if (cpt < 0x10000) {
            cpt_bytes = 3;
        } else {
            cpt_bytes = 4;
        }

        if (stacks_cur.empty()) {
            // Grammar failed to match at this point
            result.matched_bytes        = byte_pos;
            result.matched_codepoints   = i;
            result.matched_prefix       = input.substr(0, byte_pos);
            result.failing_char         = format_codepoint(cpt);
            result.expected_description = expected_before;
            result.incomplete           = false;
            return result;
        }

        byte_pos += cpt_bytes;
    }

    // All input matched - check if grammar is complete
    result.matched_bytes      = input.size();
    result.matched_codepoints = cpts.size();
    result.matched_prefix     = input;

    if (std::any_of(stacks_cur.begin(), stacks_cur.end(), [](const auto & stack) { return stack.empty(); })) {
        // An empty stack means that the grammar has been completed
        result.success    = true;
        result.incomplete = false;
    } else {
        // Grammar expects more input
        result.success              = false;
        result.incomplete           = true;
        result.expected_description = get_expected_description(rules, stacks_cur);
    }

    return result;
}

// TODO: extract to common helper (copied from test-grammar-integration.cpp)
static bool match_string(const std::string & input, llama_grammar * grammar) {
    const auto cpts = unicode_cpts_from_utf8(input);

    auto & stacks_cur = llama_grammar_get_stacks(grammar);

    for (const auto & cpt : cpts) {
        llama_grammar_accept(grammar, cpt);

        if (stacks_cur.empty()) {
            // no stacks means that the grammar failed to match at this point
            return false;
        }
    }

    if (std::any_of(stacks_cur.begin(), stacks_cur.end(), [](const auto & stack) { return stack.empty(); })) {
        // An empty stack means that the grammar has been completed
        return true;
    }

    return false;
}

static std::string renormalize_json(const std::string & json_str) {
    try {
        auto json_obj = json::parse(json_str);
        return json_obj.dump();
    } catch (const std::exception & e) {
        return "";  // ignore parial JSON contents for comparison purposes
    }
}

static void assert_msg_equals(const common_chat_msg & expected,
                              const common_chat_msg & actual,
                              bool                    ignore_whitespace_differences = false) {
    assert_equals(expected.role, actual.role);
    if (ignore_whitespace_differences) {
        assert_equals(string_strip(expected.content), string_strip(actual.content));
    } else {
        assert_equals(expected.content, actual.content);
    }
    assert_equals(expected.content_parts.size(), actual.content_parts.size());
    for (size_t i = 0; i < expected.content_parts.size(); i++) {
        const auto & expected_part = expected.content_parts[i];
        const auto & actual_part   = actual.content_parts[i];
        assert_equals(expected_part.type, actual_part.type);
        if (ignore_whitespace_differences) {
            assert_equals(string_strip(expected_part.text), string_strip(actual_part.text));
        } else {
            assert_equals(expected_part.text, actual_part.text);
        }
    }
    if (ignore_whitespace_differences) {
        assert_equals(string_strip(expected.reasoning_content), string_strip(actual.reasoning_content));
    } else {
        assert_equals(expected.reasoning_content, actual.reasoning_content);
    }
    assert_equals(expected.tool_calls.size(), actual.tool_calls.size());
    for (size_t i = 0; i < expected.tool_calls.size(); i++) {
        const auto & expected_tool_call = expected.tool_calls[i];
        const auto & actual_tool_call   = actual.tool_calls[i];
        assert_equals(expected_tool_call.name, actual_tool_call.name);
        assert_equals(renormalize_json(expected_tool_call.arguments), renormalize_json(actual_tool_call.arguments));
        assert_equals(expected_tool_call.id, actual_tool_call.id);
    }
}

static common_chat_tool special_function_tool{
    /* .name = */ "special_function",
    /* .description = */ "I'm special",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "arg1": {
                "type": "integer",
                "description": "The arg."
            }
        },
        "required": ["arg1"]
    })",
};
static common_chat_tool special_function_tool_with_optional_param{
    /* .name = */ "special_function_with_opt",
    /* .description = */ "I'm special but have optional stuff",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "arg1": {
                "type": "integer",
                "description": "The arg."
            },
            "arg2": {
                "type": "integer",
                "description": "The optional arg."
            }
        },
        "required": ["arg1"]
    })",
};

static common_chat_tool empty_args_tool{
    /* .name = */ "empty_args",
    /* .description = */ "A tool that takes no arguments",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {}
    })",
};

static common_chat_tool empty_args_tool_no_properties{
    /* .name = */ "empty_args_no_props",
    /* .description = */ "A tool that takes no arguments and has no properties",
    /* .parameters = */ R"({
        "type": "object"
    })",
};

static common_chat_tool python_tool{
    /* .name = */ "python",
    /* .description = */ "an ipython interpreter",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute."
            }
        },
        "required": ["code"]
    })",
};

static common_chat_tool html_tool{
    /* .name = */ "html",
    /* .description = */ "an html validator",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "markup": {
                "type": "string",
                "description": "HTML markup to validate."
            }
        },
        "required": ["markup"]
    })",
};

static common_chat_tool get_time_tool{
    /* .name = */ "get_time",
    /* .description = */ "Get the current time in a city",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    })",
};

static common_chat_tool get_weather_tool{
    /* .name = */ "get_weather",
    /* .description = */ "Get the current weather in a city",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    })",
};

static common_chat_tool todo_list{
    /* .name = */ "todo_list",
    /* .description = */ "Create or update the todo list",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of TODO list items"
            }
        },
        "required": ["todos"]
    })",
};

static common_chat_tool edit_tool{
    /* .name = */ "edit",
    /* .description = */ "Edit file",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Path of file to edit"
            },
            "oldString": {
                "type": "string",
                "description": "String to replace"
            },
            "newString": {
                "type": "string",
                "description": "New (replacement) value"
            }
        },
        "required": ["filename", "oldString", "newString"]
    })",
};

static common_chat_tool manage_todo_list_tool{
    /* .name = */ "manage_todo_list",
    /* .description = */ "Create or update the todo list",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of TODO list items"
            }
        },
        "required": ["todos"]
    })",
};

static common_chat_tool run_in_terminal_tool{
    /* .name = */ "run_in_terminal",
    /* .description = */ "Run a shell command.",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run"
            }
        },
        "required": ["command"]
    })",
};

static common_chat_tool magic_tool{
    /* .name = */ "magic",
    /* .description = */ "Magic tool that takes a hash",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "ref": {
                "type": "string"
            }
        },
        "required": ["name", "ref"]
    })",
};

static common_chat_tool magic_int_tool{
    /* .name = */ "magic_int",
    /* .description = */ "Magic tool that takes a hash",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "ref": {
                "type": "integer"
            },
            "name": {
                "type": "string"
            }
        },
        "required": ["ref"]
    })",
};

static common_chat_tool amount_tool{
    /* .name = */ "amount",
    /* .description = */ "Amount converter",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "orig": {
                "type": "number"
            }
        },
        "required": ["orig"]
    })",
};

static common_chat_tool toggle_tool{
    /* .name = */ "toggle",
    /* .description = */ "Toggle a feature",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "Whether to enable the feature"
            }
        },
        "required": ["enabled"]
    })",
};

static common_chat_tool nullable_tool{
    /* .name = */ "set_nullable",
    /* .description = */ "Set a nullable value",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "value": {
                "type": "null",
                "description": "A null value"
            }
        },
        "required": ["value"]
    })",
};

static common_chat_tool config_tool{
    /* .name = */ "set_config",
    /* .description = */ "Set configuration",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "description": "Configuration dict"
            }
        },
        "required": ["config"]
    })",
};

static common_chat_tool calendar_create_event_tool{
    /* .name = */ "Calendar.create_event",
    /* .description = */ "Create a calendar event",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "participants": { "type": "array", "items": { "type": "string" } },
            "metadata": { "type": "object" }
        },
        "required": ["title", "participants", "metadata"]
    })",
};

static common_chat_tool imaginary_number_tool{
    /* .name = */ "imaginary_number",
    /* .description = */ "Imaginary number converter",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "number": {
                "type": "object",
                "properties": {
                    "real": {
                        "type": "number"
                    },
                    "imaginary": {
                        "type": "number"
                    }
                },
                "required": ["real", "imaginary"]
            }
        },
        "required": ["number"]
    })",
};

static common_chat_tool nullable_string_tool{
    /* .name = */ "set_nullable_str",
    /* .description = */ "Set a nullable string value",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "name": {
                "type": ["string", "null"],
                "description": "A nullable string"
            }
        },
        "required": ["name"]
    })",
};

static common_chat_tool nullable_string_null_first_tool{
    /* .name = */ "set_nullable_str_nf",
    /* .description = */ "Set a nullable string value with null first in type array",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "name": {
                "type": ["null", "string"],
                "description": "A nullable string with null first"
            }
        },
        "required": ["name"]
    })",
};

static common_chat_tool nullable_int_tool{
    /* .name = */ "set_nullable_int",
    /* .description = */ "Set a nullable integer value",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "count": {
                "type": ["integer", "null"],
                "description": "A nullable integer"
            }
        },
        "required": ["count"]
    })",
};

static common_chat_tool enum_no_type_tool{
    /* .name = */ "set_unit",
    /* .description = */ "Set a temperature unit",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "unit": {
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["unit"]
    })",
};

static common_chat_tool string_param_tool{
    /* .name = */ "string_param",
    /* .description = */ "Tool with string parameter for testing",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "A text parameter"
            }
        },
        "required": []
    })",
};

static common_chat_tool quoted_unquoted_tool{
    /* .name = */ "quoted_unquoted",
    /* .description = */ "Tool with two string parameters, one for quoted string, one for unquoted",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "quoted": {
                "type": "string",
                "description": "Quoted value"
            },
            "unquoted": {
                "type": "string",
                "description": "Unquoted value"
            }
        },
        "required": ["quoted", "unquoted"]
    })",
};


static common_chat_tool tool_2req_4opt{
    /* .name = */ "tool_2req_4opt",
    /* .description = */ "Tool with 2 required and 4 optional params",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "req1": { "type": "string", "description": "Required string" },
            "req2": { "type": "integer", "description": "Required int" },
            "opt1": { "type": "string", "description": "Optional string 1" },
            "opt2": { "type": "integer", "description": "Optional int 1" },
            "opt3": { "type": "string", "description": "Optional string 2" },
            "opt4": { "type": "integer", "description": "Optional int 2" }
        },
        "required": ["req1", "req2"]
    })",
};

static common_chat_tool tool_2req_5opt{
    /* .name = */ "tool_2req_5opt",
    /* .description = */ "Tool with 2 required and 5 optional params",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "req1": { "type": "string", "description": "Required string" },
            "req2": { "type": "integer", "description": "Required int" },
            "opt1": { "type": "string", "description": "Optional string 1" },
            "opt2": { "type": "integer", "description": "Optional int 1" },
            "opt3": { "type": "string", "description": "Optional string 2" },
            "opt4": { "type": "integer", "description": "Optional int 2" },
            "opt5": { "type": "string", "description": "Optional string 3" }
        },
        "required": ["req1", "req2"]
    })",
};

static std::vector<common_chat_tool> tools{ special_function_tool, special_function_tool_with_optional_param,
                                            python_tool, html_tool, todo_list };

const common_chat_msg message_user{
    "user",
    "Hey there!",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

const common_chat_msg message_user_parts{
    "user",
    /* .content = */ "",
    /* .content_parts = */
    {
     { "text", "Hey" },
     { "text", "there" },
     },
    /* .tool_calls = */
    {                 },
    /* .reasoning_content = */
    "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

static common_chat_msg simple_assist_msg(const std::string & content,
                                         const std::string & reasoning_content = "",
                                         const std::string & tool_name         = "",
                                         const std::string & arguments         = "",
                                         const std::string & id                = "") {
    common_chat_msg msg;
    msg.role              = "assistant";
    msg.content           = content;
    msg.reasoning_content = reasoning_content;
    if (!tool_name.empty() || !id.empty()) {
        msg.tool_calls.push_back({ tool_name, arguments, id });
    }
    return msg;
}

static common_chat_msg message_with_tool_calls(const std::string & tool_name, const std::string & arguments) {
    return simple_assist_msg("", "", tool_name, arguments);
}

static common_chat_msg message_with_tool_calls_and_reasoning(const std::string & tool_name,
                                                             const std::string & arguments,
                                                             const std::string & reasoning) {
    return simple_assist_msg("", reasoning, tool_name, arguments);
}

static common_chat_msg message_with_reasoning_content_and_multiple_tool_calls(
    const std::string &                                      reasoning,
    const std::string &                                      content,
    const std::vector<std::pair<std::string, std::string>> & tool_calls) {
    common_chat_msg msg;
    msg.role              = "assistant";
    msg.content           = content;
    msg.reasoning_content = reasoning;
    for (const auto & [name, args] : tool_calls) {
        msg.tool_calls.push_back({ name, args, "" });
    }
    return msg;
}

static common_chat_msg message_with_content_and_tool_call(const std::string & content,
                                                          const std::string & tool_name,
                                                          const std::string & arguments) {
    return simple_assist_msg(content, "", tool_name, arguments);
}

static common_chat_msg message_with_reasoning_and_tool_call(const std::string & reasoning,
                                                            const std::string & tool_name,
                                                            const std::string & arguments) {
    return simple_assist_msg("", reasoning, tool_name, arguments);
}

const common_chat_msg message_assist       = simple_assist_msg("Hello, world!\nWhat's up?");
const common_chat_msg message_assist_empty = simple_assist_msg("");
const common_chat_msg message_assist_thoughts_unparsed_deepseek =
    simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_unparsed_md =
    simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```");
const common_chat_msg message_assist_thoughts_unparsed_md_partial =
    simple_assist_msg("<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}");

const common_chat_msg message_assist_thoughts_unparsed_r7b =
    simple_assist_msg("<|START_THINKING|>I'm\nthinking<|END_THINKING|>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_unparsed_magistral =
    simple_assist_msg("[THINK]raisonnement[/THINK]Réponse");
const common_chat_msg message_assist_thoughts = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking");
const common_chat_msg message_assist_thoughts_unopened_unparsed =
    simple_assist_msg("I'm\nthinking</think>Hello, world!\nWhat's up?");
const common_chat_msg message_assist_thoughts_no_content = simple_assist_msg("", "I'm\nthinking");
const common_chat_msg message_assist_call = simple_assist_msg("", "", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_noopt =
    simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_withopt =
    simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1, \"arg2\": 2}");
const common_chat_msg message_assist_call_content =
    simple_assist_msg("Hello, world!\nWhat's up?", "", "special_function", "{\"arg1\":1}");
const common_chat_msg message_assist_call_empty_args  = simple_assist_msg("", "", "special_function");
const common_chat_msg message_assist_call_cutoff_args = simple_assist_msg("", "", "special_function", "{\"arg");
const common_chat_msg message_assist_call_thoughts =
    simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\":1}");
const common_chat_msg message_assist_call_thoughts_unparsed =
    simple_assist_msg("<think>I'm\nthinking</think>\n\n", "", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_thoughts_content =
    simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_id =
    simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "123456789");
const common_chat_msg message_assist_call_idx =
    simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "0");
const common_chat_msg message_assist_thoughts_call_idx =
    simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}", /* id = */ "0");
const common_chat_msg message_assist_thoughts_partial_call =
    simple_assist_msg("", "I'm\nthinking", "special_function", "", /* id = */ "0");
const common_chat_msg message_assist_call_python = simple_assist_msg("", "", "python", "{\"code\":\"print('hey')\"}");
const common_chat_msg message_assist_call_python_lines =
    simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')\"}");
const common_chat_msg message_assist_call_python_lines_unclosed =
    simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')");
const common_chat_msg message_assist_json_content =
    simple_assist_msg("{\n  \"response\": \"Hello, world!\\nWhat's up?\"\n}");
const common_chat_msg message_assist_prefill_content   = simple_assist_msg("Hello, ", "I'm thinking");
const common_chat_msg message_assist_prefill_reasoning = simple_assist_msg("", "I'm");

// Use for PEG parser implementations
struct peg_test_case {
    common_chat_templates_inputs params;
    std::string                  input;
    common_chat_msg              expect;
    bool                         is_partial            = false;
    bool                         expect_reconstruction = false;
};

struct make_peg_parser {
    common_chat_params params_;
    common_peg_arena   arena_;
    bool               detailed_debug_;

    make_peg_parser(common_chat_templates *              tmpls,
                    const common_chat_templates_inputs & inputs,
                    bool                                 detailed_debug = false) {
        detailed_debug_ = detailed_debug;
        params_         = common_chat_templates_apply(tmpls, inputs);
        arena_.load(params_.parser);
    }

    common_chat_msg parse(const std::string & msg, bool is_partial) const {
        common_chat_parser_params parser_params(params_);
        parser_params.debug = detailed_debug_;
        return common_chat_peg_parse(arena_, msg, is_partial, parser_params);
    }
};

// Global template filter for --template flag
static std::string g_template_filter;

// When true, run reconstruction test on every non-partial test and report results
static bool g_force_reconstruction_test = false;

static void test_peg_parser(common_chat_templates *                      tmpls,
                            const std::function<void(peg_test_case &)> & init,
                            bool                                         detailed_debug) {
    // UTF-8-safe truncation helper (same as in test_parser_with_streaming)
    constexpr auto utf8_truncate_safe_len = [](const std::string_view s) -> size_t {
        auto len = s.size();
        if (len == 0) {
            return 0;
        }
        auto i = len;
        for (size_t back = 0; back < 4 && i > 0; ++back) {
            --i;
            unsigned char c = s[i];
            if ((c & 0x80) == 0) {
                return len;
            }
            if ((c & 0xC0) == 0xC0) {
                size_t expected_len = 0;
                if ((c & 0xE0) == 0xC0) {
                    expected_len = 2;
                } else if ((c & 0xF0) == 0xE0) {
                    expected_len = 3;
                } else if ((c & 0xF8) == 0xF0) {
                    expected_len = 4;
                } else {
                    return i;
                }
                if (len - i >= expected_len) {
                    return len;
                }
                return i;
            }
        }
        return len - std::min(len, size_t(3));
    };

    peg_test_case tc;
    init(tc);
    if (tc.params.messages.empty()) {
        tc.params.messages = { message_user };
    }
    if (tc.expect.role.empty()) {
        tc.expect.role = "assistant";
    }

    auto parser = make_peg_parser(tmpls, tc.params, detailed_debug);
    if (detailed_debug) {
        LOG_DBG("Using parser: \n%s\n", parser.arena_.dump(parser.arena_.root()).c_str());
        LOG_DBG("Generation prompt: '%s'\n", parser.params_.generation_prompt.c_str());
    }

    common_chat_msg msg_accum;
    common_chat_msg msg_prev;
    msg_accum.role = msg_prev.role = "assistant";

    for (size_t i = 1; i <= tc.input.size(); ++i) {
        auto            is_partial  = i < tc.input.size() || tc.is_partial;
        // Use UTF-8 safe truncation to avoid corrupting multi-byte characters
        size_t          safe_len    = utf8_truncate_safe_len(std::string_view(tc.input).substr(0, i));
        std::string     prefix      = tc.input.substr(0, safe_len);
        common_chat_msg msg_current = parser.parse(prefix, is_partial);

        for (const auto & diff : common_chat_msg_diff::compute_diffs(msg_prev, msg_current)) {
            if (!diff.reasoning_content_delta.empty()) {
                msg_accum.reasoning_content += diff.reasoning_content_delta;
            }
            if (!diff.content_delta.empty()) {
                msg_accum.content += diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                // During partial parsing, a new tool call may appear with empty name initially
                // The name gets filled in as more input is parsed
                while (msg_accum.tool_calls.size() <= diff.tool_call_index) {
                    msg_accum.tool_calls.push_back({ "", "", "" });
                }
                // Always update name and id from diff (may change during incremental parsing), but only if the delta
                // actually contains them
                if (!diff.tool_call_delta.name.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].name = diff.tool_call_delta.name;
                }
                if (!diff.tool_call_delta.id.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].id = diff.tool_call_delta.id;
                }
                if (!diff.tool_call_delta.arguments.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].arguments += diff.tool_call_delta.arguments;
                }
            }
        }
        try {
            assert_msg_equals(msg_current, msg_accum, true);
        } catch (std::exception & e) {
            throw std::runtime_error((std::string("Error comparing accumulated message to current: ") + e.what()).c_str());
        }

        msg_prev = msg_current;
    }

    if (!tc.is_partial) {
        assert_msg_equals(tc.expect, parser.parse(tc.input, false), true);
    }
    assert_msg_equals(tc.expect, msg_accum, true);

    // Test grammar if present in params.
    //
    // Lark grammars are skipped: they are compiled by llguidance, not by
    // build_grammar()'s GBNF path, so feeding one in fails with
    // "Failed to build grammar: %llguidance {}".
    //
    // KNOWN COVERAGE GAP: this means the Lark sampling constraint -- the one the
    // server actually uses when built with LLAMA_LLGUIDANCE=ON -- is validated
    // nowhere in test-chat. Only the extraction half (the PEG parse asserted
    // above) is covered. Closing it needs a test that compiles the Lark grammar
    // through llguidance and asserts the wire shape is accepted or rejected as
    // expected; see docs/fork/WORKFLOWS.md#verifying-a-conformance-claim.
    const bool is_lark = parser.params_.grammar.rfind("%llguidance", 0) == 0;
    if (!parser.params_.grammar.empty() && !is_lark) {
        auto grammar = build_grammar(parser.params_.grammar);
        if (!grammar) {
            throw std::runtime_error("Failed to build grammar: " + parser.params_.grammar);
        }

        // In production, grammar triggers match against the full generated text
        // including the generation prompt. All positions are in full_input coordinates.
        const auto & gen_prompt = parser.params_.generation_prompt;
        std::string full_input = gen_prompt + tc.input;

        // Determine whether the reasoning-budget sampler path applies: tool-call grammar
        // with all WORD triggers and thinking tags present. In production, the reasoning
        // budget sampler inhibits grammar application while inside thinking blocks —
        // triggers inside <think>...</think> are suppressed.
        bool use_reasoning_budget_path = false;
        if (parser.params_.grammar_lazy && !parser.params_.thinking_end_tag.empty()) {
            use_reasoning_budget_path = true;
            for (const auto & trigger : parser.params_.grammar_triggers) {
                if (trigger.type != COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    use_reasoning_budget_path = false;
                    break;
                }
            }
        }

        // Find the earliest trigger position to determine the constrained portion
        auto earliest_trigger_pos = std::string::npos;

        if (use_reasoning_budget_path) {
            // Reasoning-budget path: simulate thinking-aware trigger detection.
            // Walk through full_input tracking thinking state; only match triggers
            // when outside thinking blocks.
            const auto & think_start = parser.params_.thinking_start_tag;
            const auto & think_end   = parser.params_.thinking_end_tag;

            bool in_thinking = false;
            for (size_t i = 0; i < full_input.size(); ++i) {
                if (!in_thinking && !think_start.empty()
                        && full_input.compare(i, think_start.size(), think_start) == 0) {
                    in_thinking = true;
                    i += think_start.size() - 1;
                    continue;
                }
                if (in_thinking && full_input.compare(i, think_end.size(), think_end) == 0) {
                    in_thinking = false;
                    i += think_end.size() - 1;
                    continue;
                }
                if (in_thinking) {
                    continue;
                }
                // Outside thinking — check if any trigger word starts here
                for (const auto & trigger : parser.params_.grammar_triggers) {
                    if (full_input.compare(i, trigger.value.size(), trigger.value) == 0) {
                        if (earliest_trigger_pos == std::string::npos || i < earliest_trigger_pos) {
                            earliest_trigger_pos = i;
                        }
                    }
                }
                if (earliest_trigger_pos != std::string::npos) {
                    break;  // found the earliest
                }
            }

            // If the reasoning-budget path found no trigger outside thinking but the test
            // expects tool calls, this template nests tool calls inside thinking
            // blocks (e.g. Kimi). Fall back to the legacy path for this case.
            if (earliest_trigger_pos == std::string::npos && !tc.expect.tool_calls.empty()) {
                use_reasoning_budget_path = false;
            }
        }

        if (!use_reasoning_budget_path) {
            // Legacy path: find triggers without thinking-awareness
            for (const auto & trigger : parser.params_.grammar_triggers) {
                size_t      pos = std::string::npos;
                std::smatch match;
                switch (trigger.type) {
                    case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                        {
                            const auto & word = trigger.value;
                            pos               = full_input.find(word);
                            break;
                        }
                    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                        {
                            const auto & compiled = std::regex(trigger.value);
                            if (std::regex_search(full_input, match, compiled)) {
                                pos = match.position(compiled.mark_count());
                            }
                            break;
                        }
                    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                        {
                            // In production, PATTERN_FULL triggers are checked against
                            // the text generated so far, growing token by token. Simulate
                            // by trying every prefix of full_input.
                            const auto & compiled = std::regex(trigger.value);
                            for (size_t end = gen_prompt.size(); end <= full_input.size(); ++end) {
                                std::string prefix = full_input.substr(0, end);
                                if (std::regex_match(prefix, match, compiled)) {
                                    pos = std::string::npos;
                                    for (size_t gi = 1; gi < match.size(); ++gi) {
                                        if (match[gi].length() > 0) {
                                            pos = match.position(gi);
                                            break;
                                        }
                                    }
                                    if (pos == std::string::npos) {
                                        pos = match.position(0);
                                    }
                                    break;
                                }
                            }
                            break;
                        }
                    default:
                        throw std::runtime_error("Unknown trigger type");
                }
                if (pos != std::string::npos) {
                    if (earliest_trigger_pos == std::string::npos || pos < earliest_trigger_pos) {
                        earliest_trigger_pos = pos;
                    }
                }
            }
        }

        // If the test expects tool calls and the grammar is lazy, the trigger must fire.
        // Otherwise the grammar would never activate in production and tool calls wouldn't
        // be constrained. A silent skip here would hide broken triggers.
        if (parser.params_.grammar_lazy && !tc.expect.tool_calls.empty() && !tc.is_partial
                && earliest_trigger_pos == std::string::npos) {
            std::string trigger_desc;
            for (const auto & trigger : parser.params_.grammar_triggers) {
                trigger_desc += "\n  [type=" + std::to_string(trigger.type) + "] " + trigger.value;
            }
            throw std::runtime_error(
                "Grammar trigger did not fire, but test expects tool calls (lazy grammar).\n"
                ">>> Input: " + full_input + "\n"
                ">>> Triggers (" + std::to_string(parser.params_.grammar_triggers.size()) + "):" + trigger_desc);
        }

        // Determine the constrained portion of input to test against grammar.
        // If the trigger position falls inside the generation prompt, the grammar
        // sampler was already active before model output began — constrain from the
        // start of the model output (i.e. tc.input).
        std::string constrained = full_input;
        bool grammar_triggered = false;
        if (earliest_trigger_pos != std::string::npos) {
            auto constrain_from = std::max(earliest_trigger_pos, gen_prompt.size());
            constrained = full_input.substr(constrain_from);
            grammar_triggered = true;
        } else if (!parser.params_.grammar_lazy) {
            // For non-lazy grammars, the entire input should match
            grammar_triggered = true;
        }

        // Test the constrained portion against the grammar
        if (grammar_triggered && !tc.is_partial) {
            auto result = match_string_detailed(constrained, grammar.get());
            if (!result.success) {
                std::string error_msg;
                if (result.incomplete) {
                    error_msg =
                        "Grammar matched all input but expects more:\n\n"
                        ">>> Input: " + tc.input +
                        "\n\n>>> Constrained: " + constrained +
                        "\n\n>>> Matched prefix (" + std::to_string(result.matched_bytes) + " bytes, " +
                        std::to_string(result.matched_codepoints) + " codepoints): " +
                        (result.matched_prefix.size() > 100 ? result.matched_prefix.substr(0, 100) + "..." : result.matched_prefix) +
                        "\n\n>>> Expected next: " + result.expected_description +
                        "\n\n>>> Grammar: " + parser.params_.grammar;
                } else {
                    error_msg =
                        "Grammar match failed:\n\n"
                        ">>> Input: " + tc.input +
                        "\n\n>>> Constrained: " + constrained +
                        "\n\n>>> Matched prefix (" + std::to_string(result.matched_bytes) + " bytes, " +
                        std::to_string(result.matched_codepoints) + " codepoints): " +
                        (result.matched_prefix.size() > 100 ? result.matched_prefix.substr(0, 100) + "..." : result.matched_prefix) +
                        "\n\n>>> Failing character: " + result.failing_char +
                        "\n\n>>> Expected: " + result.expected_description +
                        "\n\n>>> Grammar: " + parser.params_.grammar;
                }
                throw std::runtime_error(error_msg);
            }
        }
    }

    // Reconstruction test: verify that appending the parsed message to the original
    // messages and re-rendering the template (without generation prompt) reproduces
    // the original prompt + input exactly, or as a proper prefix (the template may
    // append end-of-turn tokens after the assistant message).
    if ((tc.expect_reconstruction || g_force_reconstruction_test) && !tc.is_partial) {
        // Start from tc.expect but copy tool call arguments from the actual parser
        // output, which preserves original JSON formatting (e.g. {"arg1":1} vs {"arg1": 1}).
        auto reconstruction_msg = tc.expect;
        auto parsed_msg         = parser.parse(tc.input, false);
        for (size_t i = 0; i < reconstruction_msg.tool_calls.size() && i < parsed_msg.tool_calls.size(); i++) {
            reconstruction_msg.tool_calls[i].arguments = parsed_msg.tool_calls[i].arguments;
        }
        common_chat_templates_inputs reconstruction_inputs = tc.params;
        reconstruction_inputs.messages.push_back(reconstruction_msg);
        reconstruction_inputs.add_generation_prompt = false;

        auto reconstruction_params = common_chat_templates_apply(tmpls, reconstruction_inputs);
        std::string expected_text  = parser.params_.prompt + tc.input;
        bool match = reconstruction_params.prompt == expected_text ||
            (reconstruction_params.prompt.size() > expected_text.size() &&
             reconstruction_params.prompt.compare(0, expected_text.size(), expected_text) == 0);
        if (!match && g_force_reconstruction_test && !tc.expect_reconstruction) {
            // In forced mode, report mismatch but don't fail
            // Find the first difference position
            size_t diff_pos = 0;
            size_t min_len  = std::min(expected_text.size(), reconstruction_params.prompt.size());
            while (diff_pos < min_len && expected_text[diff_pos] == reconstruction_params.prompt[diff_pos]) {
                diff_pos++;
            }
            size_t ctx_start = diff_pos > 60 ? diff_pos - 60 : 0;
            size_t ctx_end_e = std::min(expected_text.size(), diff_pos + 40);
            size_t ctx_end_r = std::min(reconstruction_params.prompt.size(), diff_pos + 40);
            LOG_ERR("\x1b[31m[RECONSTRUCTION FAIL]\x1b[0m "
                    "first diff at byte %zu (expected len=%zu, reconstructed len=%zu)\n"
                    "  expected:      ...%s...\n"
                    "  reconstructed: ...%s...\n",
                    diff_pos, expected_text.size(), reconstruction_params.prompt.size(),
                    expected_text.substr(ctx_start, ctx_end_e - ctx_start).c_str(),
                    reconstruction_params.prompt.substr(ctx_start, ctx_end_r - ctx_start).c_str());
        } else if (!match) {
            std::string error_msg =
                "Reconstruction mismatch:\n\n"
                ">>> Expected (prompt + input):\n" + expected_text +
                "\n\n>>> Reconstructed:\n" + reconstruction_params.prompt;
            throw std::runtime_error(error_msg);
        } else if (g_force_reconstruction_test) {
            LOG_INF("\x1b[32m[RECONSTRUCTION OK]\x1b[0m\n");
        }
    }
}

// Fluent builder for PEG parser tests
class peg_test_builder;

class peg_tester {
    common_chat_templates_ptr tmpls_;
    bool                      detailed_debug_;
    friend class peg_test_builder;

  public:
    explicit peg_tester(const bool detailed_debug = false) :
        tmpls_(gemma4_templates()),
        detailed_debug_(detailed_debug) {}

    // The format this tester exercises. Used by the --template filter and as the
// log label; there is no template file behind it.
    const char * format_name() const { return "gemma4"; }

    peg_test_builder test(const std::string & input);
};

class peg_test_builder {
    peg_tester &  tester_;
    peg_test_case tc_;

  public:
    peg_test_builder(peg_tester & tester, const std::string & input) : tester_(tester) {
        tc_.input = input;
        tc_.params.add_generation_prompt = true;
    }

    // Parameter setters
    peg_test_builder & reasoning_format(common_reasoning_format fmt) {
        tc_.params.reasoning_format = fmt;
        return *this;
    }

    peg_test_builder & tools(std::vector<common_chat_tool> tools) {
        tc_.params.tools = std::move(tools);
        return *this;
    }

    peg_test_builder & enable_thinking(bool val) {
        tc_.params.enable_thinking = val;
        return *this;
    }

    peg_test_builder & parallel_tool_calls(bool val) {
        tc_.params.parallel_tool_calls = val;
        return *this;
    }

    peg_test_builder & add_generation_prompt(bool val) {
        tc_.params.add_generation_prompt = val;
        return *this;
    }

    peg_test_builder & continue_final_message(common_chat_continuation cont) {
        tc_.params.continue_final_message = cont;
        return *this;
    }

    peg_test_builder & json_schema(const std::string & schema) {
        tc_.params.json_schema = schema;
        return *this;
    }

    peg_test_builder & is_partial(bool val) {
        tc_.is_partial = val;
        return *this;
    }

    peg_test_builder & expect_reconstruction(bool val = true) {
        tc_.expect_reconstruction = val;
        return *this;
    }

    // Expect setters
    peg_test_builder & expect(const common_chat_msg & msg) {
        tc_.expect = msg;
        return *this;
    }

    peg_test_builder & expect_content(const std::string & content) {
        tc_.expect.content = content;
        return *this;
    }

    peg_test_builder & expect_reasoning(const std::string & reasoning) {
        tc_.expect.reasoning_content = reasoning;
        return *this;
    }

    peg_test_builder & expect_tool_calls(std::vector<common_chat_tool_call> calls) {
        tc_.expect.tool_calls = std::move(calls);
        return *this;
    }

    peg_test_builder & tool_choice(common_chat_tool_choice choice) {
        tc_.params.tool_choice = choice;
        return *this;
    }

    peg_test_builder & messages(std::vector<common_chat_msg> messages) {
        tc_.params.messages = std::move(messages);
        return *this;
    }

    // Execute the test
    void run() {
        // Check template filter
        if (!g_template_filter.empty()) {
            // Case-insensitive substring match
            std::string template_path_lower = tester_.format_name();
            std::string filter_lower        = g_template_filter;
            std::transform(template_path_lower.begin(), template_path_lower.end(), template_path_lower.begin(),
                           ::tolower);
            std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
            if (template_path_lower.find(filter_lower) == std::string::npos) {
                // Skip this test
                return;
            }
        }
        LOG_INF("\n\x1b[38;5;126m[%s]\x1b[0m\n%s\n\n", tester_.format_name(), tc_.input.c_str());
        test_peg_parser(tester_.tmpls_.get(), [this](peg_test_case & t) { t = tc_; }, tester_.detailed_debug_);
    }
};

peg_test_builder peg_tester::test(const std::string & input) {
    return peg_test_builder(*this, input);
}

static void test_msgs_oaicompat_json_conversion() {
    LOG_DBG("%s\n", __func__);
    std::vector<common_chat_msg> msgs{
        message_user,
        message_user_parts,
        message_assist_call,
        message_assist_call_thoughts,
        message_assist_call_thoughts_unparsed,
        message_assist_call_thoughts_content,
        message_assist_call_id,
        message_assist_call_idx,
        message_assist_call_python,
    };
    for (const auto & msg : msgs) {
        auto oai_json = common_chat_msgs_to_json_oaicompat({ msg });
        auto msgs2    = common_chat_msgs_parse_oaicompat(oai_json);
        assert_equals((size_t) 1, msgs2.size());
        const auto & msg2 = msgs2[0];
        assert_msg_equals(msg, msg2);
    }
    assert_equals(std::string("[\n"
                              "  {\n"
                              "    \"role\": \"user\",\n"
                              "    \"content\": [\n"
                              "      {\n"
                              "        \"type\": \"text\",\n"
                              "        \"text\": \"Hey\"\n"
                              "      },\n"
                              "      {\n"
                              "        \"type\": \"text\",\n"
                              "        \"text\": \"there\"\n"
                              "      }\n"
                              "    ]\n"
                              "  }\n"
                              "]"),
                  common_chat_msgs_to_json_oaicompat({ message_user_parts }).dump(2));

    // Note: content is "" instead of null due to workaround for templates that render null as "None"
    assert_equals(std::string("[\n"
                              "  {\n"
                              "    \"role\": \"assistant\",\n"
                              "    \"content\": \"\",\n"
                              "    \"tool_calls\": [\n"
                              "      {\n"
                              "        \"type\": \"function\",\n"
                              "        \"function\": {\n"
                              "          \"name\": \"python\",\n"
                              "          \"arguments\": \"{\\\"code\\\":\\\"print('hey')\\\"}\"\n"
                              "        }\n"
                              "      }\n"
                              "    ]\n"
                              "  }\n"
                              "]"),
                  common_chat_msgs_to_json_oaicompat({ message_assist_call_python }).dump(2));

    auto res = common_chat_msgs_parse_oaicompat(json::parse("[{\"role\": \"assistant\", \"tool_calls\": []}]"));
    assert_equals<size_t>(1, res.size());
    assert_equals<std::string>(res[0].role, "assistant");
    assert_equals(true, res[0].content.empty());
    assert_equals(true, res[0].tool_calls.empty());

    try {
        common_chat_msgs_parse_oaicompat(json::parse("[{\"role\": \"assistant\"}]"));
        throw std::runtime_error("Expected exception");
    } catch (const std::exception & e) {
        if (std::string(e.what()).find("'content'") == std::string::npos) {
            throw std::runtime_error("Expected exception about missing 'content'");
        }
    }
}

static void test_msg_token_delimiters_split() {
    LOG_DBG("%s\n", __func__);

    // Delimiters that share a leading token, distinguished by the second token,
    // to exercise the per-position token matching.
    const common_chat_msg_delimiters delims = {
        { { COMMON_CHAT_ROLE_USER,      "", { 10, 11 } },
          { COMMON_CHAT_ROLE_ASSISTANT, "", { 10, 12 } } }
    };

    // Empty inputs
    assert_equals<size_t>(0, common_chat_msg_delimiters{}.split({}).spans.size());
    assert_equals<size_t>(0, common_chat_msg_delimiters{}.split({ 10, 11 }).spans.size());
    assert_equals<size_t>(0, delims.split({}).spans.size());

    // No delimiters match -> no spans
    assert_equals<size_t>(0, delims.split({ 100, 101, 102 }).spans.size());

    // Multi-role conversation: <user>Hi<assistant>Hello<user>Bye
    {
        const llama_tokens tokens = {
            10, 11,            // <user>
            100, 101,          // Hi
            10, 12,            // <assistant>
            200, 201, 202,     // Hello
            10, 11,            // <user>
            300, 301,          // Bye
        };

        const auto result = delims.split(tokens);
        const auto & spans = result.spans;
        assert_equals<size_t>(3, spans.size());

        assert_equals(COMMON_CHAT_ROLE_USER, spans[0].role);
        assert_equals<size_t>(0, spans[0].pos);
        assert_equals<size_t>(4, spans[0].len);

        assert_equals(COMMON_CHAT_ROLE_ASSISTANT, spans[1].role);
        assert_equals<size_t>(4, spans[1].pos);
        assert_equals<size_t>(5, spans[1].len);

        assert_equals(COMMON_CHAT_ROLE_USER, spans[2].role);
        assert_equals<size_t>(9, spans[2].pos);
        assert_equals<size_t>(4, spans[2].len);

        // is_user_start() is true at the token position where a user span begins
        assert_equals(true,  result.is_user_start(0));
        assert_equals(false, result.is_user_start(4));  // assistant span
        assert_equals(true,  result.is_user_start(9));
    }

    // Content before the first delimiter is not captured as a span
    {
        const llama_tokens tokens = {
            500, 501,    // leading content (dropped)
            10, 11,      // <user>
            100,         // Hi
        };

        const auto spans = delims.split(tokens).spans;
        assert_equals<size_t>(1, spans.size());
        assert_equals(COMMON_CHAT_ROLE_USER, spans[0].role);
        assert_equals<size_t>(2, spans[0].pos);
        assert_equals<size_t>(3, spans[0].len);
    }

    // Skipped regions (media chunks) are jumped over but still count as span content
    {
        const llama_tokens tokens = {
            10, 11,             // <user>
            LLAMA_TOKEN_NULL,   // media chunk (3 tokens)
            LLAMA_TOKEN_NULL,
            LLAMA_TOKEN_NULL,
            100,                // Hi
            10, 12,             // <assistant>
        };

        const std::map<size_t, size_t> skips = { { 2, 3 } };

        const auto spans = delims.split(tokens, skips).spans;
        assert_equals<size_t>(2, spans.size());

        assert_equals(COMMON_CHAT_ROLE_USER, spans[0].role);
        assert_equals<size_t>(0, spans[0].pos);
        assert_equals<size_t>(6, spans[0].len);

        assert_equals(COMMON_CHAT_ROLE_ASSISTANT, spans[1].role);
        assert_equals<size_t>(6, spans[1].pos);
        assert_equals<size_t>(2, spans[1].len);
    }

    // A delimiter sequence inside a skipped region is not matched
    {
        const llama_tokens tokens = {
            10, 11,      // <user>
            10, 12,      // skipped region that happens to contain delimiter tokens
            100,         // Hi
        };

        const std::map<size_t, size_t> skips = { { 2, 2 } };

        const auto spans = delims.split(tokens, skips).spans;
        assert_equals<size_t>(1, spans.size());
        assert_equals(COMMON_CHAT_ROLE_USER, spans[0].role);
        assert_equals<size_t>(0, spans[0].pos);
        assert_equals<size_t>(5, spans[0].len);
    }
}

static void test_tools_oaicompat_json_conversion() {
    LOG_DBG("%s\n", __func__);
    std::vector<common_chat_tool> tools{
        special_function_tool,
        python_tool,
    };

    for (const auto & tool : tools) {
        auto oai_json = common_chat_tools_to_json_oaicompat({ tool });
        auto tools2   = common_chat_tools_parse_oaicompat(oai_json);
        assert_equals((size_t) 1, tools2.size());
        auto tool2 = tools2[0];
        assert_equals(tool.name, tool2.name);
        assert_equals(tool.description, tool2.description);
        assert_equals(json::parse(tool.parameters).dump(2), json::parse(tool2.parameters).dump(2));
    }

    assert_equals(std::string("[\n"
                              "  {\n"
                              "    \"type\": \"function\",\n"
                              "    \"function\": {\n"
                              "      \"name\": \"special_function\",\n"
                              "      \"description\": \"I'm special\",\n"
                              "      \"parameters\": {\n"
                              "        \"type\": \"object\",\n"
                              "        \"properties\": {\n"
                              "          \"arg1\": {\n"
                              "            \"type\": \"integer\",\n"
                              "            \"description\": \"The arg.\"\n"
                              "          }\n"
                              "        },\n"
                              "        \"required\": [\n"
                              "          \"arg1\"\n"
                              "        ]\n"
                              "      }\n"
                              "    }\n"
                              "  }\n"
                              "]"),
                  common_chat_tools_to_json_oaicompat({ special_function_tool }).dump(2));
}

static void test_convert_responses_to_chatcmpl() {
    LOG_DBG("%s\n", __func__);

    // Test basic conversion with input messages (user/assistant alternating)
    {
        json input = json::parse(R"({
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "hi wassup"
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Hey! 👋 Not much, just here ready to chat. What's up with you? Anything I can help you with today?"
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "hi"
                }
            ],
            "model": "gpt-5-mini",
            "stream": false,
            "text": {},
            "reasoning": {
                "effort": "medium"
            }
        })");

        json result = server_chat_convert_responses_to_chatcmpl(input);

        // Verify messages were converted correctly
        assert_equals(true, result.contains("messages"));
        assert_equals(true, result.at("messages").is_array());
        assert_equals((size_t)3, result.at("messages").size());

        // Check first message (user)
        const auto & msg0 = result.at("messages")[0];
        assert_equals(std::string("user"), msg0.at("role").get<std::string>());
        assert_equals(true, msg0.at("content").is_array());
        assert_equals(std::string("text"), msg0.at("content")[0].at("type").get<std::string>());
        assert_equals(std::string("hi wassup"), msg0.at("content")[0].at("text").get<std::string>());

        // Check second message (assistant)
        const auto & msg1 = result.at("messages")[1];
        assert_equals(std::string("assistant"), msg1.at("role").get<std::string>());
        assert_equals(true, msg1.at("content").is_array());
        assert_equals(std::string("text"), msg1.at("content")[0].at("type").get<std::string>());
        assert_equals(std::string("Hey! 👋 Not much, just here ready to chat. What's up with you? Anything I can help you with today?"), msg1.at("content")[0].at("text").get<std::string>());

        // Check third message (user)
        const auto & msg2 = result.at("messages")[2];
        assert_equals(std::string("user"), msg2.at("role").get<std::string>());
        assert_equals(true, msg2.at("content").is_array());
        assert_equals(std::string("text"), msg2.at("content")[0].at("type").get<std::string>());
        assert_equals(std::string("hi"), msg2.at("content")[0].at("text").get<std::string>());

        // Verify other fields preserved
        assert_equals(std::string("gpt-5-mini"), result.at("model").get<std::string>());
        assert_equals(false, result.at("stream").get<bool>());
    }

    // Test string input
    {
        json input = json::parse(R"({
            "input": "Hello, world!",
            "model": "test-model"
        })");

        json result = server_chat_convert_responses_to_chatcmpl(input);

        assert_equals((size_t)1, result.at("messages").size());
        const auto & msg = result.at("messages")[0];
        assert_equals(std::string("user"), msg.at("role").get<std::string>());
        assert_equals(std::string("Hello, world!"), msg.at("content").get<std::string>());
    }

    // Test with instructions (system message)
    {
        json input = json::parse(R"({
            "input": "Hello",
            "instructions": "You are a helpful assistant.",
            "model": "test-model"
        })");

        json result = server_chat_convert_responses_to_chatcmpl(input);

        assert_equals((size_t)2, result.at("messages").size());
        const auto & sys_msg = result.at("messages")[0];
        assert_equals(std::string("system"), sys_msg.at("role").get<std::string>());
        assert_equals(std::string("You are a helpful assistant."), sys_msg.at("content").get<std::string>());
    }

    // Test with max_output_tokens conversion
    {
        json input = json::parse(R"({
            "input": "Hello",
            "model": "test-model",
            "max_output_tokens": 100
        })");

        json result = server_chat_convert_responses_to_chatcmpl(input);

        assert_equals(true, result.contains("max_tokens"));
        assert_equals(false, result.contains("max_output_tokens"));
        assert_equals(100, result.at("max_tokens").get<int>());
    }

    // Test mixed Responses tools: convert only function tools
    {
        json input = json::parse(R"({
            "input": "Hello",
            "model": "test-model",
            "tools": [
                {
                    "type": "web_search"
                },
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            }
                        },
                        "required": ["location"]
                    }
                },
                {
                    "type": "image_generation"
                },
                {
                    "type": "mcp",
                    "server_label": "test-server"
                },
                {
                    "type": "namespace",
                    "name": "browser"
                }
            ]
        })");

        json result = server_chat_convert_responses_to_chatcmpl(input);

        assert_equals(true, result.contains("tools"));
        assert_equals(true, result.at("tools").is_array());
        assert_equals((size_t)1, result.at("tools").size());

        const auto & tool = result.at("tools")[0];
        assert_equals(std::string("function"), tool.at("type").get<std::string>());
        assert_equals(std::string("get_weather"), tool.at("function").at("name").get<std::string>());
        assert_equals(true, tool.at("function").at("strict").get<bool>());
    }

    // Test non-function Responses tools are ignored
    {
        json input = json::parse(R"({
            "input": "Hello",
            "model": "test-model",
            "tools": [
                {
                    "type": "web_search"
                },
                {
                    "type": "image_generation"
                },
                {
                    "type": "mcp",
                    "server_label": "test-server"
                },
                {
                    "type": "namespace",
                    "name": "browser"
                }
            ]
        })");

        json result = server_chat_convert_responses_to_chatcmpl(input);

        assert_equals(false, result.contains("tools"));
    }
}

// Shared LFM2 parser cases - all variants use one output format and parser
static void test_lfm2_parser(bool detailed_debug) {
    auto tst = peg_tester(detailed_debug);

    // Basic content only
    tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

    // Single tool call without reasoning
    tst.test("<|tool_call_start|>[special_function(arg1=1)]<|tool_call_end|>")
        .tools({ special_function_tool })
        .expect(message_assist_call)
        .run();

    // Tool call with string argument
    tst.test("<|tool_call_start|>[get_time(city=\"XYZCITY\")]<|tool_call_end|>")
        .tools({ get_time_tool })
        .expect(message_with_tool_calls("get_time", "{\"city\":\"XYZCITY\"}"))
        .run();

    // Python literals become JSON
    tst.test("<|tool_call_start|>[toggle(enabled=True)]<|tool_call_end|>")
        .tools({ toggle_tool })
        .expect(message_with_tool_calls("toggle", R"({"enabled": true})"))
        .run();

    tst.test("<|tool_call_start|>[set_nullable(value=None)]<|tool_call_end|>")
        .tools({ nullable_tool })
        .expect(message_with_tool_calls("set_nullable", R"({"value": null})"))
        .run();

    // Nested Python literal
    tst.test("<|tool_call_start|>[set_config(config={\"enabled\": True, \"count\": 3})]<|tool_call_end|>")
        .tools({ config_tool })
        .expect(message_with_tool_calls("set_config", R"({"config": {"enabled": true, "count": 3}})"))
        .run();

    // JSON literals are accepted too
    tst.test("<|tool_call_start|>[set_config(config={\"enabled\": true, \"note\": null})]<|tool_call_end|>")
        .tools({ config_tool })
        .expect(message_with_tool_calls("set_config", R"({"config": {"enabled": true, "note": null}})"))
        .run();

    // Dotted function name with structured args
    tst.test("<|tool_call_start|>[Calendar.create_event(title=\"demo\", participants=[\"Alice\", \"Bob\"], "
             "metadata={\"priority\": \"high\", \"reminder\": true})]<|tool_call_end|>")
        .tools({ calendar_create_event_tool })
        .expect(message_with_tool_calls(
            "Calendar.create_event",
            R"({"title": "demo", "participants": ["Alice", "Bob"], "metadata": {"priority": "high", "reminder": true}})"))
        .run();

    // Markdown links stay content
    tst.test("Use this format: [link text](url). Example: [Wikipedia](https://www.wikipedia.org).")
        .tools({ get_time_tool })
        .expect(simple_assist_msg("Use this format: [link text](url). Example: [Wikipedia](https://www.wikipedia.org)."))
        .run();

    // Python tool with multiline code in string: the \n in the literal decodes to a real
    // newline, emitted as a JSON \n escape (not a doubled backslash).
    tst.test("<|tool_call_start|>[python(code=\"def hello():\\n    print('hey')\")]<|tool_call_end|>")
        .tools({ python_tool })
        .expect_tool_calls({
            { "python", R"#({"code": "def hello():\n    print('hey')"})#", "" }
        })
        .run();

    // String escape sequences decode to their actual characters (newline + tab here),
    // so a "write a two line file" style call produces real line breaks, not literal "\n".
    tst.test("<|tool_call_start|>[python(code=\"First line\\nSecond line\\tindented\")]<|tool_call_end|>")
        .tools({ python_tool })
        .expect_tool_calls({
            { "python", R"#({"code": "First line\nSecond line\tindented"})#", "" }
        })
        .run();

    // Escaped quotes inside a string argument survive the round-trip.
    tst.test("<|tool_call_start|>[python(code=\"print(\\\"hi\\\")\")]<|tool_call_end|>")
        .tools({ python_tool })
        .expect_tool_calls({
            { "python", R"#({"code": "print(\"hi\")"})#", "" }
        })
        .run();

    // Content before tool call (no reasoning)
    tst.test("Let me check the time.<|tool_call_start|>[get_time(city=\"Paris\")]<|tool_call_end|>")
        .tools({ get_time_tool })
        .expect(message_with_reasoning_content_and_multiple_tool_calls(
            "", "Let me check the time.", { { "get_time", "{\"city\":\"Paris\"}" } }
        ))
        .run();

    // Multiple tool calls (parallel)
    tst.test("<|tool_call_start|>[special_function(arg1=1), special_function_with_opt(arg1=1, arg2=2)]<|tool_call_end|>")
        .parallel_tool_calls(true)
        .tools({ special_function_tool, special_function_tool_with_optional_param })
        .expect_tool_calls({
            { "special_function", R"({"arg1": 1})", {} },
            { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
        })
        .run();

    // Partial tool call (streaming)
    tst.test("<|tool_call_start|>[special_function(arg1=")
        .tools({ special_function_tool })
        .is_partial(true)
        .expect(simple_assist_msg("", "", "special_function", "{\"arg1\": "))
        .run();

    // Tool call with empty arguments
    tst.test("<|tool_call_start|>[empty_args()]<|tool_call_end|>")
        .tools({ empty_args_tool })
        .expect(simple_assist_msg("", "", "empty_args", "{}"))
        .run();

}

static void test_template_output_peg_parsers(bool detailed_debug) {
    LOG_DBG("%s\n", __func__);

    // JSON schemas
    const char * invoice_schema = R"({
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "date": {"type": "string"}
        }
    })";

    const char * const_schema = R"({
        "const": "42"
    })";







    {
        // Google Gemma 4 (tool calling with Gemma4 dict format)
        auto tst = peg_tester();

        tst.test("Hello, world!").expect(simple_assist_msg("Hello, world!")).run();

        // Reasoning and content
        tst.test(
                "<|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(message_assist_thoughts)
            .run();

        // Empty reasoning (budget=0: sampler forces end tag before newline)
        tst.test(
                "<|channel>thought<channel|>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(simple_assist_msg("Hello, world!\nWhat's up?", ""))
            .run();

        // Reasoning and content with reasoning_format = none
        tst.test(
                "<|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_NONE)
            .expect_content("<|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?")
            .run();

        // Simple tool call with string argument
        tst.test(
                "<|tool_call>call:get_time{city:<|\"|>London<|\"|>}<tool_call|>")
            .tools({ get_time_tool })
            .expect(message_with_tool_calls("get_time", R"({"city": "London"})"))
            .run();

        // Tool call with string argument containing special chars
        tst.test(
                "<|tool_call>call:get_time{city:<|\"|>San Francisco<|\"|>}<tool_call|>")
            .tools({ get_time_tool })
            .expect(message_with_tool_calls("get_time", R"({"city": "San Francisco"})"))
            .run();

        // Tool call with empty args
        tst.test(
                "<|tool_call>call:empty_args{}<tool_call|>")
            .tools({ empty_args_tool })
            .expect(message_with_tool_calls("empty_args", "{}"))
            .run();

        // Tool call with string and content
        tst.test(
                "Hello, world!\nWhat's up?<|tool_call>call:get_time{city:<|\"|>Paris<|\"|>}<tool_call|>")
            .tools({ get_time_tool })
            .expect(message_with_content_and_tool_call("Hello, world!\nWhat's up?", "get_time", R"({"city": "Paris"})"))
            .run();

        // Parallel tool calls
        tst.test(
                "<|tool_call>call:get_time{city:<|\"|>London<|\"|>}<tool_call|>"
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>")
            .tools({ get_time_tool, get_weather_tool })
            .parallel_tool_calls(true)
            .expect_tool_calls({
                { "get_time", R"({"city": "London"})", "" },
                { "get_weather", R"({"city": "Paris"})", "" },
            })
            .run();

        // Tool call with integer argument (number type)
        tst.test(
                "<|tool_call>call:special_function{arg1:42}<tool_call|>")
            .tools({ special_function_tool })
            .expect(message_with_tool_calls("special_function", R"({"arg1": 42})"))
            .run();

        // Tool call with negative number argument
        tst.test(
                "<|tool_call>call:special_function{arg1:-7}<tool_call|>")
            .tools({ special_function_tool })
            .expect(message_with_tool_calls("special_function", R"({"arg1": -7})"))
            .run();

        // Tool call with decimal number argument
        tst.test(
                "<|tool_call>call:amount{orig:3.14}<tool_call|>")
            .tools({ amount_tool })
            .expect(message_with_tool_calls("amount", R"({"orig": 3.14})"))
            .run();

        // Tool call with boolean argument (true)
        tst.test(
                "<|tool_call>call:toggle{enabled:true}<tool_call|>")
            .tools({ toggle_tool })
            .expect(message_with_tool_calls("toggle", R"({"enabled": true})"))
            .run();

        // Tool call with boolean argument (false)
        tst.test(
                "<|tool_call>call:toggle{enabled:false}<tool_call|>")
            .tools({ toggle_tool })
            .expect(message_with_tool_calls("toggle", R"({"enabled": false})"))
            .run();

        // Tool call with null argument
        tst.test(
                "<|tool_call>call:set_nullable{value:null}<tool_call|>")
            .tools({ nullable_tool })
            .expect(message_with_tool_calls("set_nullable", R"({"value": null})"))
            .run();

        // Tool call with array argument (todo list)
        tst.test(
                "<|tool_call>call:todo_list{todos:[<|\"|>buy milk<|\"|>,<|\"|>walk dog<|\"|>]}<tool_call|>")
            .tools({ todo_list })
            .expect(message_with_tool_calls("todo_list", R"({"todos":["buy milk","walk dog"]})"))
            .run();

        // Tool call with object/dict argument
        tst.test(
                "<|tool_call>call:set_config{config:{theme:<|\"|>dark<|\"|>,count:3}}<tool_call|>")
            .tools({ config_tool })
            .expect(message_with_tool_calls("set_config", R"({"config":{"theme":"dark","count":3}})"))
            .run();

        // Tool call with empty array
        tst.test(
                "<|tool_call>call:todo_list{todos:[]}<tool_call|>")
            .tools({ todo_list })
            .expect(message_with_tool_calls("todo_list", R"({"todos":[]})"))
            .run();

        // Tool call with empty dict
        tst.test(
                "<|tool_call>call:set_config{config:{}}<tool_call|>")
            .tools({ config_tool })
            .expect(message_with_tool_calls("set_config", R"({"config":{}})"))
            .run();

        // Tool call with scientific notation number
        tst.test(
                "<|tool_call>call:amount{orig:1.5e10}<tool_call|>")
            .tools({ amount_tool })
            .expect(message_with_tool_calls("amount", R"({"orig": 1.5e10})"))
            .run();

        // A trailing empty thought: the grammar's `(channel_block content)*`
        // with an empty final content. Valid output, so it is parsed.
        //
        // The three cases that used to sit here -- a dangling `<channel|>`, a
        // doubled `<channel|>`, and a leading kindless `<|channel>` -- were
        // removed with the productions that tolerated them. llguidance enforces
        // this same grammar while sampling, so the model cannot emit any of
        // them; a parser that accepted them would be describing a language the
        // format does not have.
        tst.test(
                "<|channel>thought\n<channel|>Hello, world!\nWhat's up?<|channel>thought\n<channel|>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .expect(message_assist)
            .run();

        // Continuation tests
        tst.test("world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .messages({ message_user, message_assist_prefill_content })
            .add_generation_prompt(false)
            .continue_final_message(COMMON_CHAT_CONTINUATION_CONTENT)
            .expect_reasoning("I'm thinking")
            .expect_content("Hello, world!\nWhat's up?")
            .run();

        tst.test(" thinking<channel|>Hello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .messages({ message_user, message_assist_prefill_reasoning })
            .add_generation_prompt(false)
            .continue_final_message(COMMON_CHAT_CONTINUATION_REASONING)
            .expect_reasoning("I'm thinking")
            .expect_content("Hello, world!\nWhat's up?")
            .run();

        {
            // additional tests for https://github.com/ggml-org/llama.cpp/pull/21760
            auto tmpls = gemma4_templates();

            common_chat_msg tool_call_msg = simple_assist_msg(
                "Let me check.", "", "special_function", "{\"arg1\": 1}","c0");

            common_chat_msg tool_msg;
            tool_msg.role         = "tool";
            tool_msg.tool_name    = "special_function";
            tool_msg.tool_call_id = "c0";
            tool_msg.content      = "{\"r\":\"ok\"}";

            {
                common_chat_templates_inputs inputs;
                inputs.messages              = { message_user, tool_call_msg, tool_msg };
                inputs.tools                 = { special_function_tool };
                inputs.add_generation_prompt = true;

                auto params = common_chat_templates_apply(tmpls.get(), inputs);

                // Tool responses render inside the model turn, so that turn is
                // still open and generation resumes in it with an unclosed
                // thought re-opener -- NOT a fresh "<|turn>model", which would
                // split one model turn in two. enable_thinking defaults true.
                if (!string_ends_with(params.prompt, "<|channel>thought\n")) {
                    throw std::runtime_error(
                        "Gemma 4: expected a thought re-opener after a tool response, got: ..." +
                        params.prompt.substr(params.prompt.size() > 60 ? params.prompt.size() - 60 : 0));
                }
            }

            {
                common_chat_templates_inputs inputs;
                inputs.messages              = { message_user, tool_call_msg, tool_msg };
                inputs.tools                 = { special_function_tool };
                inputs.add_generation_prompt = false;

                auto params = common_chat_templates_apply(tmpls.get(), inputs);

                if (string_ends_with(params.prompt, "<|turn>model\n")) {
                    throw std::runtime_error("Gemma 4: generation prompt was modified despite add_generation_prompt=false");
                }
            }
        }


    }










    // DeepSeek V3.2 tests - format uses DSML markup:
    //   <｜DSML｜function_calls>
    //   <｜DSML｜invoke name="foo">
    //   <｜DSML｜parameter name="bar" string="true|false">value</｜DSML｜parameter>
    //   </｜DSML｜invoke>
    //   </｜DSML｜function_calls>
    // Reasoning uses <think>...</think>. The generation prompt ends in <think> (thinking mode)
    // or <think></think> (non-thinking mode).

    // DeepSeek V4 tests - same DSML markup as V3.2, but the tool call block is named
    // "tool_calls" and the non-thinking generation prompt ends in a bare </think>
    // instead of an empty <think></think> pair.

    // GLM-4.6 tests - format: <tool_call>function_name\n<arg_key>...</arg_key>\n<arg_value>...</arg_value>\n</tool_call>

    // GLM-4.7-Flash tests - format: <tool_call>function_name<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>
    // Note: Template uses forced-open thinking mode (prompt ends with <think>)

    // Verify the throw path produces a readable error message, not std::out_of_range.
    // #20424 introduced effective_input = generation_prompt + input, but the throw
    // uses input.substr(result.end) where result.end is in effective_input space.

    // Reka Edge

    // Apriel 1.5

    // Apriel 1.6 Thinker (reasoning-only support)

    // Mistral Small 3.2 - FUNC_BRACKET_TAG format: [TOOL_CALLS]func_name[CALL_ID]id[ARGS]{...}
    // Devstral




    // GPT-OSS format tests


    // GigaChat V3

    // GigaChat V3.1

    // MiniCPM5 - XML tool calls with <function name="..."><param name="...">...</param></function>
}

static void test_template_generation_prompt() {
    common_chat_msg system_msg;
    system_msg.role = "system";
    system_msg.content ="You are a helpful assistant.";

    common_chat_msg tool_call_msg = simple_assist_msg("", "", "special_function", "{\"arg1\": 1}");

    common_chat_msg tool_msg;
    tool_msg.role = "tool";
    tool_msg.tool_name = "special_function";
    tool_msg.tool_call_id = "call0";
    tool_msg.content = "Sunny";

    struct test_case_options {
        std::vector<common_chat_msg> messages;
        bool                         add_generation_prompt  = true;
        common_chat_continuation     continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    };

    auto basic = [&]() {
        test_case_options opts;
        opts.messages = { system_msg, message_user };
        return opts;
    };

    auto continuation_content = [&]() {
        test_case_options opts;
        opts.messages               = { system_msg, message_user, message_assist_prefill_content };
        opts.add_generation_prompt  = false;
        opts.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
        return opts;
    };

    auto continuation_reasoning = [&]() {
        test_case_options opts;
        opts.messages               = { system_msg, message_user, message_assist_prefill_reasoning };
        opts.add_generation_prompt  = false;
        opts.continue_final_message = COMMON_CHAT_CONTINUATION_REASONING;
        return opts;
    };

    // There is no "generation prompt" to assert any more. That field existed so a
    // caller could recover where generation begins and re-prepend it to the model's
    // output as a text prefix; the format plugin now reports the state the prompt
    // leaves the model in, and the parser picks its entry rule from that.
    //
    // So the two things worth checking are what the prompt ENDS with, and which
    // state that tail implies. Asserting the tail alone would not catch a wrong
    // entry state, and a wrong entry state silently parses the model's reply with
    // the wrong root.
    auto check = [&](const common_chat_templates_ptr & tmpls,
                     const test_case_options & opts,
                     const std::string & expected_tail,
                     common_chat_format_state expected_entry) {
        common_chat_templates_inputs inputs;
        inputs.messages               = opts.messages;
        inputs.add_generation_prompt  = opts.add_generation_prompt;
        inputs.continue_final_message = opts.continue_final_message;

        auto params = common_chat_templates_apply(tmpls.get(), inputs);

        assert_contains(params.prompt, system_msg.content);
        assert_contains(params.prompt, message_user.content);
        assert_ends_with(params.prompt, expected_tail);
        assert_equals(static_cast<int>(expected_entry), static_cast<int>(params.entry_state));
    };




    {
        using S = common_chat_format_state;
        auto tmpls = gemma4_templates();

        // A fresh turn opener: the model has not spoken, and may open with a thought.
        check(tmpls, basic(), "<|turn>model\n", S::IN_GENERATION_PROMPT);

        // Prefilled content: the thought is COMPLETE (closed with "\n<channel|>", as
        // the writer spec emits it) and the model resumes in content.
        check(tmpls, continuation_content(),
              "<|turn>model\n<|channel>thought\nI'm thinking\n<channel|>Hello, ", S::IN_CONTENT);

        // Prefilled reasoning: the thought is left UNCLOSED, so the model resumes
        // inside it and the parser enters at `resume_reasoning`.
        check(tmpls, continuation_reasoning(),
              "<|turn>model\n<|channel>thought\nI'm", S::IN_REASONING);

        // Last message is a tool response: the model turn is still open, so there is
        // no new "<|turn>model" -- generation resumes in the same turn, inside the
        // prefilled thought.
        test_case_options after_tool_call = continuation_reasoning();
        after_tool_call.messages          = { system_msg, message_user, tool_call_msg, tool_msg, message_assist_prefill_reasoning };
        check(tmpls, after_tool_call, "<|channel>thought\nI'm", S::IN_REASONING);
    }








}

// Test the developer role to system workaround with a simple mock template

// Verify reasoning-trace retention rules in the DeepSeek-V4 template:
// all traces are retained unless drop_thinking is true AND the conversation
// has no tool calls, in which case only the last (after-final-user) trace is
// kept and earlier ones are dropped.

// Verify that consecutive tool results are rendered in the tool call order of the
// preceding assistant message (matched by tool call id), as required by the reference
// DeepSeek-V4 implementation.



static void test_msg_diffs_compute() {
    LOG_DBG("%s\n", __func__);
    {
        common_chat_msg msg1;

        common_chat_msg msg2;
        msg2.content = "Hello, world!";

        common_chat_msg_diff diff;
        diff.content_delta = "Hello, world!";

        assert_equals({ diff }, common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg1;
        msg1.content = "Hello,";

        common_chat_msg msg2;
        msg2.content = "Hello, world!";

        common_chat_msg_diff diff;
        diff.content_delta = " world!";

        assert_equals({ diff }, common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg0;

        common_chat_msg msg1;
        msg1.tool_calls = {
            { "special_function", "{\"ar", /* .id = */ "123" }
        };

        common_chat_msg msg2;
        msg2.tool_calls = {
            { "special_function", "{\"arg1\": 1}", /* .id = */ "123" }
        };

        common_chat_msg_diff diff01;
        diff01.tool_call_index           = 0;
        diff01.tool_call_delta.name      = "special_function";
        diff01.tool_call_delta.id        = "123";
        diff01.tool_call_delta.arguments = "{\"ar";

        assert_equals({ diff01 }, common_chat_msg_diff::compute_diffs(msg0, msg1));

        common_chat_msg_diff diff12;
        diff12.tool_call_index           = 0;
        // Note: neither id nor name change here.
        diff12.tool_call_delta.arguments = "g1\": 1}";

        assert_equals({ diff12 }, common_chat_msg_diff::compute_diffs(msg1, msg2));
    }
    {
        common_chat_msg msg0;

        common_chat_msg msg2;
        msg2.tool_calls = {
            { "f1", "{\"arg1\": 1}", /* .id = */ "123" },
            { "f2", "{\"arg2\": 2}", /* .id = */ "222" },
        };

        common_chat_msg_diff diff1;
        diff1.tool_call_index           = 0;
        diff1.tool_call_delta.name      = "f1";
        diff1.tool_call_delta.id        = "123";
        diff1.tool_call_delta.arguments = "{\"arg1\": 1}";

        common_chat_msg_diff diff2;
        diff2.tool_call_index           = 1;
        diff2.tool_call_delta.name      = "f2";
        diff2.tool_call_delta.id        = "222";
        diff2.tool_call_delta.arguments = "{\"arg2\": 2}";

        assert_equals({ diff1, diff2 }, common_chat_msg_diff::compute_diffs(msg0, msg2));
    }
}

// Grammar-file-driven formats need the registry populated before any template is
// applied, or common_chat_grammar_require() throws. The server does this from
// --chat-grammars-dir; tests read straight out of the source tree.
//
// Tries the repo root first (how test-chat already finds models/templates/*.jinja),
// then one and two levels up so the binary also works when run from build/ or
// build/bin/. Throwing here beats an empty registry, which would mean generating
// with no sampling constraint at all.
static void init_chat_grammars() {
    for (const char * dir : { "grammars/chat", "../grammars/chat", "../../grammars/chat" }) {
        std::error_code ec;
        if (std::filesystem::is_directory(dir, ec)) {
            common_chat_grammar_init(dir);
            return;
        }
    }
    throw std::runtime_error(
        "test-chat: could not locate grammars/chat -- run from the llama.cpp repo root");
}

int main(int argc, char ** argv) {
    // Must run before ANY test: several tests apply a Gemma 4 template, and a
    // grammar-file-driven format throws from common_chat_grammar_require() when
    // the registry is empty. Initialising next to the PEG tests is too late --
    // test_template_generation_prompt() gets there first.
    init_chat_grammars();

    bool detailed_debug    = false;
    bool only_run_filtered = false;

    // Check for --template and --detailed flags
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--template" && i + 1 < argc) {
            g_template_filter = argv[++i];
            // Only run PEG parser tests with the filter
            only_run_filtered = true;
        }
        if (arg == "--detailed") {
            detailed_debug = true;
            common_log_set_verbosity_thold(999);
        }
        if (arg == "--force-reconstruction-test") {
            g_force_reconstruction_test = true;
            only_run_filtered          = true;
        }
    }

    if (only_run_filtered) {
        test_template_output_peg_parsers(detailed_debug);
        std::cout << "\n[chat] All template tests passed!" << '\n';
        return 0;
    }

#ifndef _WIN32
    // Check if any argument is a .jinja file (for template format detection mode)
    bool has_jinja_files = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--detailed") {
            continue;
        }
        if (arg.size() >= 6 && arg.rfind(".jinja") == arg.size() - 6) {
            has_jinja_files = true;
            break;
        }
    }

    if (has_jinja_files) {
        common_chat_templates_inputs inputs;
        common_chat_msg              msg;
        msg.role        = "user";
        msg.content     = "Hey";
        inputs.messages = { msg };
        inputs.tools    = { special_function_tool };

        std::cout << "| Template | Format |\n";
        std::cout << "|----------|--------|\n";

        for (int i = 1; i < argc; i++) {
            try {
                std::string path = argv[i];
                if (path.rfind(".jinja") != path.size() - 6) {
                    std::cerr << "Skipping non-jinja file: " << path << '\n';
                    continue;
                }
                auto tmpls = gemma4_templates();
                auto         parts  = string_split(path, "/");
                const auto & name   = parts[parts.size() - 1];
                const auto * format = common_chat_format_name(common_chat_templates_apply(tmpls.get(), inputs).format);
                std::cout << "| " << name << " | " << format << " |\n";
            } catch (const std::exception & e) {
                std::cerr << "Failed to process " << argv[i] << ": " << e.what() << '\n';
            }
        }
    } else
#endif
    {
        test_msg_diffs_compute();
        test_msgs_oaicompat_json_conversion();
        test_msg_token_delimiters_split();
        test_tools_oaicompat_json_conversion();
        test_convert_responses_to_chatcmpl();
        test_template_generation_prompt();
        test_template_output_peg_parsers(detailed_debug);
        std::cout << "\n[chat] All tests passed!" << '\n';
    }
    return 0;
}
