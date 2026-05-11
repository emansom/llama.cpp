#include "chat-formats/qwen3-coder-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>
#include <unordered_set>

namespace {
using ordered_json = nlohmann::ordered_json;

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

// Mirror of Jinja `render_extra_keys`: emit `<key>value</key>` blocks for any
// keys not in `handled`. Mapping/sequence values are tojson-serialised,
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

std::string build_tools_block(const ordered_json & tools) {
    std::string out;
    out += "\n\n# Tools\n\nYou have access to the following tools:\n\n<tools>";
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
                render_extra_keys(out, param_fields, {"name", "type", "description"});
                out += "\n</parameter>";
            }
        }
        out += " ";
        render_extra_keys(out, parameters, {"type", "properties"});
        out += "\n</parameters>";
        render_extra_keys(out, tool, {"type", "name", "description", "parameters"});
        out += "\n</function>";
    }
    out += "\n</tools>";
    out += "\n\nIf you choose to call a tool ONLY reply in the following format with NO suffix:\n\n"
           "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
           "value_1\n</parameter>\n<parameter=example_parameter_2>\nvalue_2\n</parameter>\n"
           "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
           "- Function calls MUST follow the specified format: the tool calling block MUST begin with an opening <tool_call> tag and end with a closing </tool_call> tag.\n"
           "- Required parameters MUST be specified\n"
           "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
           "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
           "</IMPORTANT>";
    return out;
}

void render_tool_call(std::ostringstream & out, const ordered_json & tool_call_in) {
    ordered_json tc = tool_call_in;
    if (tc.contains("function") && tc["function"].is_object()) {
        tc = tc["function"];
    }
    out << "\n<tool_call>\n<function=" << tc.value("name", std::string{}) << ">\n";
    if (tc.contains("arguments")) {
        ordered_json args = tc["arguments"];
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
    out << "</function>\n</tool_call>";
}

}  // namespace

std::string common_chat_qwen3_coder_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    const bool has_messages = inputs.messages.is_array() && !inputs.messages.empty();
    const bool first_is_system = has_messages &&
        inputs.messages[0].is_object() &&
        inputs.messages[0].value("role", std::string{}) == "system";
    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    size_t loop_start = 0;
    std::string system_message;
    bool system_message_set = false;
    if (first_is_system) {
        const auto & c = inputs.messages[0].value("content", ordered_json{});
        if (c.is_string()) system_message = c.get<std::string>();
        system_message_set = true;
        loop_start = 1;
    }

    if (system_message_set) {
        out << "<|im_start|>system\n" << system_message;
    } else if (has_tools) {
        out << "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.";
    }
    if (has_tools) {
        out << build_tools_block(inputs.tools);
    }
    if (system_message_set || has_tools) {
        out << "<|im_end|>\n";
    }

    for (size_t i = loop_start; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});
        std::string content;
        if (message.contains("content") && message["content"].is_string()) {
            content = message["content"].get<std::string>();
        }
        const bool has_tool_calls = role == "assistant" && message.contains("tool_calls") &&
            message["tool_calls"].is_array() && !message["tool_calls"].empty();

        if (has_tool_calls) {
            out << "<|im_start|>assistant";
            std::string trimmed = trim_str(content);
            if (!trimmed.empty()) {
                out << "\n" << trimmed << "\n";
            }
            for (const auto & tc : message["tool_calls"]) {
                render_tool_call(out, tc);
            }
            out << "<|im_end|>\n";
        } else if (role == "user" || role == "system" || role == "assistant") {
            out << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        } else if (role == "tool") {
            const std::string prev_role = i > 0
                ? inputs.messages[i - 1].value("role", std::string{}) : std::string{};
            if (i > 0 && prev_role != "tool") {
                out << "<|im_start|>user";
            }
            out << "\n<tool_response>\n" << content << "\n</tool_response>";
            const bool is_last = (i + 1 == inputs.messages.size());
            const std::string next_role = is_last ? std::string{}
                : inputs.messages[i + 1].value("role", std::string{});
            if ((!is_last && next_role != "tool") || is_last) {
                out << "<|im_end|>\n";
            }
        }
    }

    if (inputs.add_generation_prompt) {
        out << "<|im_start|>assistant\n";
    }
    return out.str();
}
