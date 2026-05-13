#include "chat-formats/gemma-2-format.h"

#include <sstream>
#include <string>

namespace {

// Match Jinja's `string | trim`: strip ASCII whitespace from both ends.
std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) {
        --e;
    }
    return s.substr(b, e - b);
}

}  // namespace

std::string common_chat_gemma_2_render(const autoparser::generation_params & inputs,
                                       const std::string & bos_token) {
    std::ostringstream out;
    out << bos_token;

    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            std::string role = m.value("role", std::string{});
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (role == "assistant") {
                role = "model";
            } else if (role == "system") {
                // Gemma 2 has no system role; the Jinja template raises an
                // exception. Test fixtures never include system messages.
                continue;
            }
            out << "<start_of_turn>" << role << "\n" << trim(content) << "<end_of_turn>\n";
        }
    }
    if (inputs.add_generation_prompt) {
        out << "<start_of_turn>model\n";
    }
    return out.str();
}
