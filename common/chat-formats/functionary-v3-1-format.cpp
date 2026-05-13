#include "chat-formats/functionary-v3-1-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

// Mirror Jinja's default `tojson()` (no indent, `, ` item separator,
// `: ` key separator).
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

// Mirror the Jinja `tool` rendering: descriptions wrap an OpenAI-style
// `{"type": "function", "function": {...}}` envelope OR a bare tool.
const ordered_json & inner_tool(const ordered_json & tool) {
    if (tool.is_object() && tool.contains("type") && tool.contains("function")) {
        return tool["function"];
    }
    return tool;
}

}  // namespace

std::string common_chat_functionary_v3_1_render(const autoparser::generation_params & inputs,
                                                const std::string & bos_token) {
    std::ostringstream out;

    ordered_json tools = inputs.tools;
    if (!tools.is_array()) {
        tools = ordered_json::array();
    }

    // Strip code_interpreter tools (Functionary handles them via a separate
    // system prelude); test fixtures never include them.
    bool has_code_interpreter = false;
    {
        ordered_json filtered = ordered_json::array();
        for (const auto & t : tools) {
            if (t.is_object() && t.value("type", std::string{}) == "code_interpreter") {
                has_code_interpreter = true;
            } else {
                filtered.push_back(t);
            }
        }
        tools = std::move(filtered);
    }

    out << bos_token << "<|start_header_id|>system<|end_header_id|>\n\n";
    if (has_code_interpreter) {
        out << "Environment: ipython\n\n";
    } else {
        // Jinja `{%- else -%}\n    {{ "\n"}}\n{%- endif %}` emits the body
        // of the else branch verbatim, which is `    \n` (four-space lstrip
        // indent + literal `\n`).
        out << "    \n";
    }
    out << "Cutting Knowledge Date: December 2023\n\n";

    if (!tools.empty()) {
        out << "\nYou have access to the following functions:\n\n";
        for (const auto & t_in : tools) {
            const ordered_json & t = inner_tool(t_in);
            const std::string name = t.value("name", std::string{});
            const std::string description = t.value("description", std::string{});
            out << "Use the function '" << name << "' to '" << description << "'\n"
                << to_jinja_json(t) << "\n\n";
        }
        out << "\n"
            << "Think very carefully before calling functions.\n"
            << "If a you choose to call a function ONLY reply in the following format:\n"
            << "<{start_tag}={function_name}>{parameters}{end_tag}\n"
            << "where\n\n"
            << "start_tag => `<function`\n"
            << "parameters => a JSON dict with the function argument name as key and function argument value as value.\n"
            << "end_tag => `</function>`\n\n"
            << "Here is an example,\n"
            << "<function=example_function_name>{\"example_name\": \"example_value\"}</function>\n\n"
            << "Reminder:\n"
            << "- If looking for real time information use relevant functions before falling back to brave_search\n"
            << "- Function calls MUST follow the specified format, start with <function= and end with </function>\n"
            << "- Required parameters MUST be specified\n"
            << "- Only call one function at a time\n"
            << "- Put the entire function call reply on one line\n\n";
    }
    out << "<|eot_id|>";

    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            const std::string role = m.value("role", std::string{});
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (role == "user" || role == "system") {
                out << "<|start_header_id|>" << role << "<|end_header_id|>\n\n"
                    << content << "<|eot_id|>";
            } else if (role == "tool") {
                out << "<|start_header_id|>ipython<|end_header_id|>\n\n"
                    << content << "<|eot_id|>";
            } else {  // assistant
                out << "<|start_header_id|>" << role << "<|end_header_id|>\n\n"
                    << content;
                const bool has_tool_calls =
                    m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty();
                if (has_tool_calls) {
                    for (const auto & tc_in : m["tool_calls"]) {
                        ordered_json tc = tc_in;
                        if (tc.contains("function") && tc["function"].is_object()) {
                            tc = tc["function"];
                        }
                        const std::string fn = tc.value("name", std::string{});
                        const auto & args   = tc.contains("arguments")
                            ? tc["arguments"] : ordered_json::object();
                        if (fn == "python") {
                            out << "<|python_tag|>";
                            if (args.is_string()) {
                                out << args.get<std::string>();
                            } else {
                                out << to_jinja_json(args);
                            }
                        } else {
                            out << "<function=" << fn << ">";
                            if (args.is_string()) {
                                out << args.get<std::string>();
                            } else {
                                out << to_jinja_json(args);
                            }
                            out << "</function>";
                        }
                    }
                    out << "<|eom_id|>";
                } else {
                    out << "<|eot_id|>";
                }
            }
        }
    }

    if (inputs.add_generation_prompt) {
        out << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    return out.str();
}
