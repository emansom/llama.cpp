#include "chat-formats/nemotron-nano-v2-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

std::string to_jinja_json(const ordered_json & v) {
    if (v.is_null()) {
        return "null";
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
    }
    if (v.is_string() || v.is_number()) {
        return v.dump();
    }
    if (v.is_array()) {
        std::string out = "[";
        bool first = true;
        for (const auto & item : v) {
            if (!first) {
                out += ", ";
            }
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
            if (!first) {
                out += ", ";
            }
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

}  // namespace

std::string common_chat_nemotron_nano_v2_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    // System block: emit either the explicit system message or a placeholder.
    std::string system_msg;
    bool first_is_system = false;
    if (inputs.messages.is_array() && !inputs.messages.empty() &&
        inputs.messages[0].is_object() &&
        inputs.messages[0].value("role", std::string{}) == "system" &&
        inputs.messages[0].contains("content") &&
        inputs.messages[0]["content"].is_string()) {
        system_msg = inputs.messages[0]["content"].get<std::string>();
        first_is_system = true;
    }
    out << "<SPECIAL_10>System\n" << system_msg << "<SPECIAL_12>\n";

    if (inputs.messages.is_array()) {
        for (size_t i = 0; i < inputs.messages.size(); ++i) {
            if (i == 0 && first_is_system) {
                continue;
            }
            const auto & m = inputs.messages[i];
            const std::string role = m.value("role", std::string{});
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (role == "user") {
                out << "<SPECIAL_11>User\n" << content << "<SPECIAL_12>\n";
            } else if (role == "assistant") {
                out << "<SPECIAL_11>Assistant\n" << content;
                if (m.contains("tool_calls") && m["tool_calls"].is_array() &&
                    !m["tool_calls"].empty()) {
                    out << "<TOOLCALL>[";
                    bool first = true;
                    for (const auto & tc_in : m["tool_calls"]) {
                        ordered_json tc = tc_in;
                        if (tc.contains("function") && tc["function"].is_object()) {
                            tc = tc["function"];
                        }
                        if (!first) {
                            out << ", ";
                        }
                        first = false;
                        out << "{\"name\": \"" << tc.value("name", std::string{}) << "\", \"arguments\": ";
                        const auto & args = tc.contains("arguments") ? tc["arguments"]
                            : ordered_json::object();
                        if (args.is_string()) {
                            out << args.get<std::string>();
                        } else {
                            out << to_jinja_json(args);
                        }
                        out << "}";
                    }
                    out << "]</TOOLCALL>";
                }
                out << "<SPECIAL_12>\n";
            } else if (role == "tool") {
                out << "<SPECIAL_11>Tool\n" << content << "<SPECIAL_12>\n";
            }
        }
    }

    if (inputs.add_generation_prompt) {
        out << "<SPECIAL_11>Assistant\n";
    }
    return out.str();
}
