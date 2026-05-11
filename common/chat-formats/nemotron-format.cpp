#include "chat-formats/nemotron-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {
using ordered_json = nlohmann::ordered_json;

// JSON serializer matching Jinja `tojson` defaults: items `, ` and keys `: `.
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

std::string python_str(const ordered_json & v) {
    if (v.is_null())    return "None";
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_string())  return v.get<std::string>();
    return v.dump();
}

std::string trim_str(const std::string & s) {
    size_t s_pos = s.find_first_not_of(" \t\n\r");
    if (s_pos == std::string::npos) return {};
    size_t e_pos = s.find_last_not_of(" \t\n\r");
    return s.substr(s_pos, e_pos - s_pos + 1);
}

// Mirror of the Jinja `render_extra_keys(json_dict, handled_keys)` macro.
// Recursively emits `<key>value</key>` blocks for any keys in `json_dict`
// not in `handled_keys`. Mapping/sequence values are tojson-serialised,
// otherwise `python_str`.
void render_extra_keys(std::string & out, const ordered_json & dict,
                       const std::unordered_set<std::string> & handled) {
    if (!dict.is_object()) return;
    for (auto it = dict.begin(); it != dict.end(); ++it) {
        if (handled.count(it.key())) continue;
        const auto & v = it.value();
        out += "\n<" + it.key() + ">";
        if (v.is_object() || (v.is_array() && !v.is_string())) {
            out += to_jinja_json(v);
        } else {
            out += python_str(v);
        }
        out += "</" + it.key() + ">";
    }
}

// Build the system+tools block payload (without the `<|im_start|>system\n`
// prefix or `<|im_end|>\n` suffix; the caller adds those).
std::string build_tools_block(const ordered_json & tools) {
    std::string out;
    out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto & tool_in : tools) {
        ordered_json tool = tool_in;
        if (tool.contains("function") && tool["function"].is_object()) {
            tool = tool["function"];
        }
        out += "\n<function>\n<name>" + tool.value("name", std::string{}) + "</name>";
        if (tool.contains("description") && tool["description"].is_string()) {
            out += "\n<description>" + trim_str(tool["description"].get<std::string>()) + "</description>";
        }
        out += "\n<parameters>";

        const ordered_json parameters = tool.contains("parameters") ? tool["parameters"] : ordered_json::object();
        if (parameters.is_object() && parameters.contains("properties") &&
            parameters["properties"].is_object()) {
            for (auto it = parameters["properties"].begin(); it != parameters["properties"].end(); ++it) {
                const auto & param_name = it.key();
                const auto & param_fields = it.value();
                out += "\n<parameter>";
                out += "\n<name>" + param_name + "</name>";
                if (param_fields.contains("type")) {
                    out += "\n<type>" + python_str(param_fields["type"]) + "</type>";
                }
                if (param_fields.contains("description") && param_fields["description"].is_string()) {
                    out += "\n<description>" + trim_str(param_fields["description"].get<std::string>()) + "</description>";
                }
                if (param_fields.contains("enum")) {
                    out += "\n<enum>" + to_jinja_json(param_fields["enum"]) + "</enum>";
                }
                render_extra_keys(out, param_fields, {"name", "type", "description", "enum"});
                out += "\n</parameter>";
            }
        }
        // Note: Jinja macro `render_extra_keys` emits a leading space before
        // each block when invoked at the top of a line via `{% set %}`. The
        // template includes a literal space in the macro indentation that
        // surfaces as ` ` here. Reproducing the exact byte sequence requires
        // a leading space before the parameters-extra block (matching
        // Jinja's `{% set handled_keys = ... %}` whitespace).
        out += " ";
        render_extra_keys(out, parameters, {"type", "properties", "required"});
        if (parameters.is_object() && parameters.contains("required")) {
            out += "\n<required>" + to_jinja_json(parameters["required"]) + "</required>";
        }
        out += "\n</parameters>";
        render_extra_keys(out, tool, {"type", "name", "description", "parameters"});
        out += "\n</function>";
    }
    out += "\n</tools>";
    out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
           "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
           "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
           "This is the value for the second parameter\nthat can span\nmultiple lines\n"
           "</parameter>\n</function>\n</tool_call>\n\n"
           "<IMPORTANT>\nReminder:\n"
           "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
           "- Required parameters MUST be specified\n"
           "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
           "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
           "</IMPORTANT>";
    return out;
}

void render_tool_call(std::ostringstream & out, const ordered_json & tool_call_in) {
    ordered_json tool_call = tool_call_in;
    if (tool_call.contains("function") && tool_call["function"].is_object()) {
        tool_call = tool_call["function"];
    }
    out << "<tool_call>\n<function=" << tool_call.value("name", std::string{}) << ">\n";
    if (tool_call.contains("arguments")) {
        ordered_json args = tool_call["arguments"];
        if (args.is_string()) {
            try {
                args = ordered_json::parse(args.get<std::string>());
            } catch (...) {
                args = ordered_json::object();
            }
        }
        if (args.is_object()) {
            for (auto it = args.begin(); it != args.end(); ++it) {
                out << "<parameter=" << it.key() << ">\n";
                const auto & v = it.value();
                if (v.is_object() || v.is_array()) {
                    out << to_jinja_json(v);
                } else {
                    out << python_str(v);
                }
                out << "\n</parameter>\n";
            }
        }
    }
    out << "</function>\n</tool_call>\n";
}

}  // namespace

std::string common_chat_nemotron_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    const bool has_messages = inputs.messages.is_array() && !inputs.messages.empty();
    const bool first_is_system = has_messages &&
        inputs.messages[0].is_object() &&
        inputs.messages[0].value("role", std::string{}) == "system";

    // 1. Determine system_message + loop_messages split.
    std::string system_message;
    bool        system_message_set = false;
    size_t      loop_start = 0;
    if (first_is_system) {
        const auto & c = inputs.messages[0].value("content", ordered_json{});
        if (c.is_string()) system_message = c.get<std::string>();
        system_message_set = true;
        loop_start = 1;
    }

    // 2. Find last_user_idx within loop_messages.
    int last_user_idx = -1;
    for (size_t i = loop_start; i < inputs.messages.size(); ++i) {
        if (inputs.messages[i].value("role", std::string{}) == "user") {
            last_user_idx = static_cast<int>(i - loop_start);
        }
    }

    // 3. System block.
    if (system_message_set) {
        out << "<|im_start|>system\n" << system_message;
    } else if (has_tools) {
        out << "<|im_start|>system\n";
    }
    if (has_tools) {
        if (system_message_set && !system_message.empty()) {
            out << "\n\n";
        }
        out << build_tools_block(inputs.tools);
    }
    if (system_message_set || has_tools) {
        out << "<|im_end|>\n";
    }

    // 4. Per-message loop (over loop_messages).
    const bool truncate_history_thinking = true;
    for (size_t i = loop_start; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});
        const int loop_idx = static_cast<int>(i - loop_start);

        if (role == "assistant") {
            std::string content;
            const bool has_reasoning = message.contains("reasoning_content") &&
                message["reasoning_content"].is_string() &&
                !trim_str(message["reasoning_content"].get<std::string>()).empty();
            if (has_reasoning) {
                std::string raw_content;
                if (message.contains("content") && message["content"].is_string()) {
                    raw_content = message["content"].get<std::string>();
                }
                content = "<think>\n" + message["reasoning_content"].get<std::string>() +
                          "\n</think>\n" + raw_content;
            } else {
                if (message.contains("content") && message["content"].is_string()) {
                    content = message["content"].get<std::string>();
                }
                if (content.find("<think>") == std::string::npos &&
                    content.find("</think>") == std::string::npos) {
                    content = "<think></think>" + content;
                }
            }

            const bool has_tool_calls = message.contains("tool_calls") &&
                message["tool_calls"].is_array() && !message["tool_calls"].empty();
            if (has_tool_calls) {
                out << "<|im_start|>assistant\n";
                const bool include_content = !(truncate_history_thinking && loop_idx < last_user_idx);
                std::string trimmed = trim_str(content);
                if (!trimmed.empty()) {
                    if (include_content) {
                        out << trimmed << "\n";
                    } else {
                        std::string c = content;
                        if (c.find("</think>") != std::string::npos) {
                            const std::string close = "</think>";
                            const size_t pos = c.rfind(close);
                            c = c.substr(pos + close.size());
                        } else if (c.find("<think>") != std::string::npos) {
                            const std::string open = "<think>";
                            const size_t pos = c.find(open);
                            c = c.substr(0, pos);
                        }
                        c = "<think></think>" + trim_str(c);
                        if (!c.empty()) {
                            out << c << "\n";
                        }
                    }
                } else {
                    out << "<think></think>";
                }
                for (const auto & tc : message["tool_calls"]) {
                    render_tool_call(out, tc);
                }
                out << "<|im_end|>\n";
            } else {
                if (!(truncate_history_thinking && loop_idx < last_user_idx)) {
                    out << "<|im_start|>assistant\n" << trim_str(content) << "<|im_end|>\n";
                } else {
                    std::string c = content;
                    if (c.find("<think>") != std::string::npos &&
                        c.find("</think>") != std::string::npos) {
                        const std::string close = "</think>";
                        const size_t pos = c.rfind(close);
                        c = "<think></think>" + c.substr(pos + close.size());
                    }
                    c = trim_str(c);
                    if (!c.empty()) {
                        out << "<|im_start|>assistant\n" << c << "<|im_end|>\n";
                    } else {
                        out << "<|im_start|>assistant\n<|im_end|>\n";
                    }
                }
            }
        } else if (role == "user" || role == "system") {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        } else if (role == "tool") {
            const std::string prev_role = i > 0
                ? inputs.messages[i - 1].value("role", std::string{}) : std::string{};
            if (i > 0 && prev_role != "tool") {
                out << "<|im_start|>user\n";
            }
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "<tool_response>\n" << content << "\n</tool_response>\n";
            const bool is_last = (i + 1 == inputs.messages.size());
            const std::string next_role = is_last ? std::string{}
                : inputs.messages[i + 1].value("role", std::string{});
            if ((!is_last && next_role != "tool") || is_last) {
                out << "<|im_end|>\n";
            }
        } else {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        }
    }

    // 5. Generation prompt.
    if (inputs.add_generation_prompt) {
        if (inputs.enable_thinking) {
            out << "<|im_start|>assistant\n<think>\n";
        } else {
            out << "<|im_start|>assistant\n<think></think>";
        }
    }
    return out.str();
}
