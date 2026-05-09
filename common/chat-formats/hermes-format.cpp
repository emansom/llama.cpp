#include "chat-formats/hermes-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

// Hermes FSM-state-to-grammar-rule registry. Same structure as the json-
// tagged formats (Kimi-K2, etc.) — content + reasoning + tool-call envelope
// + name + args. Hermes has no separate tool-id rule.
const common_chat_format_state_rules hermes_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Hermes prompt writer. Mirrors models/templates/NousResearch-Hermes-3-
// Llama-3.1-8B-tool_use.jinja. The system message is non-trivial: it
// synthesises a `name(arg: type, ...) -> RetT - description` python-style
// docstring per tool, plus an Args block. Each tool object is emitted as an
// inline JSON-like blob (string formatted, NOT via tojson).
// ──────────────────────────────────────────────────────────────────────────────

namespace {
using ordered_json = nlohmann::ordered_json;

// JSON serializer matching Jinja `tojson` defaults: items `, ` and keys `: `
// separators, no ASCII escaping. Used for `tool.parameters | tojson` and
// for tool_call.arguments.
std::string to_jinja_json(const ordered_json & v) {
    if (v.is_null())   return "null";
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_string() || v.is_number()) return v.dump();
    if (v.is_array()) {
        std::string out = "[";
        bool first = true;
        for (const auto & item : v) {
            if (!first) out += ", ";
            first = false;
            out += to_jinja_json(item);
        }
        out += "]";
        return out;
    }
    if (v.is_object()) {
        std::string out = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            out += ordered_json(it.key()).dump();
            out += ": ";
            out += to_jinja_json(it.value());
        }
        out += "}";
        return out;
    }
    return "";
}

// Mirror of the Jinja `json_to_python_type(json_spec)` macro (lines 1-31).
// Recursively converts JSON-schema types to Python type names.
std::string json_to_python_type(const ordered_json & spec) {
    if (!spec.is_object()) return "Any";
    if (!spec.contains("type")) return "Any";

    const auto & t = spec["type"];
    if (t.is_string()) {
        const std::string s = t.get<std::string>();
        if (s == "string")  return "str";
        if (s == "number")  return "float";
        if (s == "integer") return "int";
        if (s == "boolean") return "bool";
        if (s == "array") {
            // Jinja: json_spec | items — calls a Python `items` (loose). The
            // template author probably meant `json_spec.items` (the JSON
            // schema field). We match that interpretation.
            ordered_json items = spec.value("items", ordered_json{});
            return "list[" + json_to_python_type(items) + "]";
        }
        if (s == "object") {
            if (spec.contains("additionalProperties")) {
                return "dict[str, " + json_to_python_type(spec["additionalProperties"]) + "]";
            }
            return "dict";
        }
    } else if (t.is_array()) {
        // Iterable type → Union[T1, T2, ...].
        std::string out = "Union[";
        for (size_t i = 0; i < t.size(); ++i) {
            ordered_json sub = ordered_json::object();
            sub["type"] = t[i];
            out += json_to_python_type(sub);
            if (i + 1 < t.size()) {
                out += ",";
            }
        }
        out += "]";
        return out;
    }
    return "Any";
}

// Render one tool's system-message blob.
//
// Mirrors Jinja lines 38-82. The output for a single tool is:
//   {"type": "function", "function": {"name": "X", "description": "X(args) -> RetT - desc\n\n    Args:\n        ...\n    Returns:\n        ...", "parameters": {...JSON...}}}
//
// Notable pitfalls reproduced verbatim:
//   * Trailing whitespace on the synthesised description's first line is
//     preserved (the template emits a literal newline after the description).
//   * Args block has 4-space indentation per param; description is `| trim`-ed.
//   * If parameters has no properties, emits `"parameters": {}` (even though
//     `tool.parameters | tojson` would produce a different shape).
std::string render_tool_block(const ordered_json & tool_in) {
    ordered_json tool = tool_in;
    if (tool.contains("function") && tool["function"].is_object()) {
        tool = tool["function"];
    }

    std::string out = "{\"type\": \"function\", \"function\": ";
    out += "{\"name\": \"" + tool.value("name", std::string{}) + "\", ";
    out += "\"description\": \"" + tool.value("name", std::string{}) + "(";

    const ordered_json parameters = tool.value("parameters", ordered_json::object());
    const ordered_json properties = parameters.value("properties", ordered_json::object());

    // Param list: "name: type, name: type, ..."
    bool first = true;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (!first) out += ", ";
        first = false;
        out += it.key() + ": " + json_to_python_type(it.value());
    }
    out += ")";

    // Optional return type.
    if (tool.contains("return") && tool["return"].is_object()) {
        out += " -> " + json_to_python_type(tool["return"]);
    }

    out += " - " + tool.value("description", std::string{}) + "\n\n";

    // Args block.
    if (!properties.empty()) {
        out += "    Args:\n";
        for (auto it = properties.begin(); it != properties.end(); ++it) {
            const ordered_json & param = it.value();
            const std::string desc = param.value("description", std::string{});
            // | trim
            size_t s = desc.find_first_not_of(" \t\n\r");
            size_t e = desc.find_last_not_of(" \t\n\r");
            const std::string trimmed = (s == std::string::npos)
                ? std::string{} : desc.substr(s, e - s + 1);
            out += "        " + it.key() + "(" + json_to_python_type(param) + "): " + trimmed;
        }
    }

    if (tool.contains("return") && tool["return"].is_object() &&
        tool["return"].contains("description")) {
        out += "\n    Returns:\n        " + tool["return"]["description"].get<std::string>();
    }

    out += "\"";
    out += ", \"parameters\": ";
    if (properties.empty()) {
        out += "{}";
    } else {
        out += to_jinja_json(parameters);
    }
    // Two closing braces: one for the `function` object, one for the
    // outer `{"type": "function", "function": ...}` wrapper. The upstream
    // Hermes Jinja templates were missing the second close --- fixed in
    // models/templates/NousResearch-Hermes-{2-Pro,3}-Llama-3*-tool_use.jinja
    // alongside this writer.
    out += "}}";
    return out;
}

}  // namespace

std::string common_chat_hermes_render(const autoparser::generation_params & inputs,
                                      const std::string & bos_token) {
    std::ostringstream out;

    // 1. BOS.
    out << bos_token;

    // 2. System message: canonical tool-calling instructions + per-tool blobs.
    out << "<|im_start|>system\n";
    out << "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> ";

    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        for (size_t i = 0; i < inputs.tools.size(); ++i) {
            out << render_tool_block(inputs.tools[i]);
            if (i + 1 < inputs.tools.size()) {
                out << "\n";
            }
        }
    }

    out << " </tools>";
    out << "Use the following pydantic model json schema for each tool call you will make: "
           "{\"properties\": {\"name\": {\"title\": \"Name\", \"type\": \"string\"}, "
           "\"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}}, "
           "\"required\": [\"name\", \"arguments\"], \"title\": \"FunctionCall\", \"type\": \"object\"}}\n";
    out << "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n";
    out << "<tool_call>\n";
    out << "{\"name\": <function-name>, \"arguments\": <args-dict>}\n";
    out << "</tool_call><|im_end|>\n";

    // 3. Per-message rendering.
    if (!inputs.messages.is_array()) {
        return out.str();
    }
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});

        const bool has_tool_calls = role == "assistant" && message.contains("tool_calls") &&
            message["tool_calls"].is_array() && !message["tool_calls"].empty();

        if (role == "user" || role == "system" ||
            (role == "assistant" && !has_tool_calls)) {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        } else if (role == "assistant") {
            // Assistant with tool_calls.
            out << "<|im_start|>assistant";
            for (const auto & tool_call_in : message["tool_calls"]) {
                ordered_json tool_call = tool_call_in;
                if (tool_call.contains("function") && tool_call["function"].is_object()) {
                    tool_call = tool_call["function"];
                }
                out << "\n<tool_call>\n";
                out << "{";
                out << "\"name\": \"" << tool_call.value("name", std::string{}) << "\"";
                if (tool_call.contains("arguments")) {
                    out << ", \"arguments\": ";
                    const auto & args = tool_call["arguments"];
                    if (args.is_string()) {
                        out << args.get<std::string>();
                    } else {
                        out << to_jinja_json(args);
                    }
                }
                out << "}";
                out << "\n</tool_call>";
            }
            out << "<|im_end|>\n";
        } else if (role == "tool") {
            // Determine prev/next role for the surrounding `<|im_start|>tool` /
            // `<|im_end|>` framing (matches Jinja's loop.previtem / nextitem).
            const std::string prev_role = i > 0
                ? inputs.messages[i - 1].value("role", std::string{}) : std::string{};
            const std::string next_role = i + 1 < inputs.messages.size()
                ? inputs.messages[i + 1].value("role", std::string{}) : std::string{};
            const bool is_last = i + 1 == inputs.messages.size();

            if (i > 0 && prev_role != "tool") {
                out << "<|im_start|>tool\n";
            }
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "<tool_response>\n" << content;
            if (!is_last) {
                out << "\n</tool_response>\n";
            } else {
                out << "\n</tool_response>";
            }
            if ((!is_last && next_role != "tool") || is_last) {
                out << "<|im_end|>";
            }
        }
    }

    // 4. Generation prompt.
    if (inputs.add_generation_prompt) {
        out << "<|im_start|>assistant\n";
    }

    return out.str();
}
