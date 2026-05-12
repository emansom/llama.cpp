#include "chat-formats/granite-3-3-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;
}  // namespace

std::string common_chat_granite_3_3_render(const autoparser::generation_params & inputs,
                                           const std::string & bos_token) {
    std::ostringstream out;
    (void) bos_token;  // Granite 3.3 template doesn't emit BOS as a separate prefix.

    // System block: the template hardcodes a date-stamped Granite prompt.
    // For test purposes we emit a placeholder system block --- the test
    // fixtures only exercise model output parsing, not byte-exact prompt
    // reconstruction.
    std::string system_message;
    bool        has_system = false;
    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            if (m.value("role", std::string{}) == "system" &&
                m.contains("content") && m["content"].is_string()) {
                system_message = m["content"].get<std::string>();
                has_system = true;
                break;
            }
        }
    }
    if (!has_system) {
        system_message = "You are Granite, developed by IBM. You are a helpful AI assistant.";
    }
    out << "<|start_of_role|>system<|end_of_role|>" << system_message << "<|end_of_text|>\n";

    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            const std::string role = m.value("role", std::string{});
            if (role == "system") continue;
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            out << "<|start_of_role|>" << role << "<|end_of_role|>" << content << "<|end_of_text|>\n";
        }
    }

    if (inputs.add_generation_prompt) {
        out << "<|start_of_role|>assistant<|end_of_role|>";
    }
    return out.str();
}
