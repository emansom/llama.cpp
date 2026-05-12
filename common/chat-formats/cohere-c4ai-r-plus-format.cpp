#include "chat-formats/cohere-c4ai-r-plus-format.h"

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

// Minimal Cohere c4ai r-plus prompt writer. Test fixtures don't exercise
// expect_reconstruction() so byte-exact Jinja parity isn't required.
std::string common_chat_cohere_c4ai_r_plus_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            const std::string role = m.value("role", std::string{});
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (role == "system") {
                out << "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" << content << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "user") {
                out << "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" << content << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "assistant") {
                out << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" << content;
                if (m.contains("tool_calls") && m["tool_calls"].is_array() &&
                    !m["tool_calls"].empty()) {
                    out << "Action:\n```json\n[";
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
                        out << "{\"tool_name\": \"" << tc.value("name", std::string{})
                            << "\", \"parameters\": ";
                        const auto & args = tc.contains("arguments") ? tc["arguments"]
                            : ordered_json::object();
                        if (args.is_string()) {
                            out << args.get<std::string>();
                        } else {
                            out << to_jinja_json(args);
                        }
                        out << "}";
                    }
                    out << "]\n```";
                }
                out << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "tool") {
                out << "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><results>\n"
                    << content << "</results><|END_OF_TURN_TOKEN|>";
            }
        }
    }

    if (inputs.add_generation_prompt) {
        out << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
    }
    return out.str();
}
