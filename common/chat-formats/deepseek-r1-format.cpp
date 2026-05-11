#include "chat-formats/deepseek-r1-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

constexpr const char * USER_OPEN = "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>";       // <｜User｜>
constexpr const char * ASST_OPEN = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";   // <｜Assistant｜>
constexpr const char * EOS_MARK  = "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>";  // <｜end▁of▁sentence｜>

}  // namespace

std::string common_chat_deepseek_r1_render(const autoparser::generation_params & inputs,
                                           const std::string & bos_token) {
    std::ostringstream out;

    // 1. Aggregate system messages into a single string prefix (DeepSeek-R1
    //    embeds the system prompt before the BOS-relative content stream).
    std::string system_prompt;
    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            if (m.value("role", std::string{}) == "system" &&
                m.contains("content") && m["content"].is_string()) {
                system_prompt += m["content"].get<std::string>();
            }
        }
    }

    out << bos_token << system_prompt;

    // 2. Per-message rendering.
    if (!inputs.messages.is_array()) {
        if (inputs.add_generation_prompt) {
            out << ASST_OPEN << "<think>\n";
        }
        return out.str();
    }
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});
        if (role == "user") {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << USER_OPEN << content;
        } else if (role == "assistant") {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            // Past-assistant content: strip thinking block (split on
            // `</think>`, take the last segment).
            const std::string close = "</think>";
            size_t close_pos = content.find(close);
            if (close_pos != std::string::npos) {
                content = content.substr(close_pos + close.size());
            }
            out << ASST_OPEN << content << EOS_MARK;
        }
    }

    // 3. Generation prompt: `<｜Assistant｜><think>\n`.
    if (inputs.add_generation_prompt) {
        out << ASST_OPEN << "<think>\n";
    }
    return out.str();
}
