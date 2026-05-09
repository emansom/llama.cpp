#include "chat-formats/lfm2-5-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

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

std::string content_to_string(const ordered_json & content) {
    if (content.is_string()) return content.get<std::string>();
    if (content.is_null())   return std::string{};
    return to_jinja_json(content);
}

// Mirror of Jinja `| trim` filter: strip leading/trailing whitespace.
std::string trim_ws(const std::string & s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return std::string{};
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

}  // namespace

std::string common_chat_lfm2_5_render(const autoparser::generation_params & inputs,
                                      const std::string & bos_token) {
    std::ostringstream out;

    // 1. BOS.
    out << bos_token;

    // 2. Build system_prompt: pull from first system message + append tools list.
    std::string system_prompt;
    ordered_json messages = inputs.messages;
    if (messages.is_array() && !messages.empty() &&
        messages[0].is_object() && messages[0].value("role", std::string{}) == "system") {
        system_prompt = content_to_string(messages[0].value("content", ordered_json{}));
        messages.erase(0);
    }

    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        if (!system_prompt.empty()) {
            system_prompt += "\n";
        }
        // No <|tool_list_start|>/<|tool_list_end|> wrappers in LFM2.5.
        system_prompt += "List of tools: [";
        for (size_t i = 0; i < inputs.tools.size(); ++i) {
            const auto & tool = inputs.tools[i];
            if (tool.is_string()) {
                system_prompt += tool.get<std::string>();
            } else {
                system_prompt += to_jinja_json(tool);
            }
            if (i + 1 < inputs.tools.size()) {
                system_prompt += ", ";
            }
        }
        system_prompt += "]";
    }

    // 3. Emit system block if non-empty.
    if (!system_prompt.empty()) {
        out << "<|im_start|>system\n" << system_prompt << "<|im_end|>\n";
    }

    // 4. Find last assistant index for past-thinking-strip decision.
    int last_assistant_index = -1;
    if (messages.is_array()) {
        for (size_t i = 0; i < messages.size(); ++i) {
            if (messages[i].is_object() &&
                messages[i].value("role", std::string{}) == "assistant") {
                last_assistant_index = static_cast<int>(i);
            }
        }
    }

    // 5. Per-message rendering.
    if (messages.is_array()) {
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto & msg = messages[i];
            const std::string role = msg.value("role", std::string{});
            out << "<|im_start|>" << role << "\n";

            std::string content = content_to_string(msg.value("content", ordered_json{}));

            // Past-but-not-last assistant messages: strip `<think>...</think>` prefix.
            // (Jinja: `keep_past_thinking` defaults to false; we don't expose it here.)
            const bool is_past_assistant =
                role == "assistant" && static_cast<int>(i) != last_assistant_index;
            if (is_past_assistant) {
                size_t close_pos = content.rfind("</think>");
                if (close_pos != std::string::npos) {
                    content = trim_ws(content.substr(close_pos + sizeof("</think>") - 1));
                }
            }

            out << content << "<|im_end|>\n";
        }
    }

    // 6. Generation prompt.
    if (inputs.add_generation_prompt) {
        out << "<|im_start|>assistant\n";
    }

    return out.str();
}

// LFM2.5 FSM-state-to-grammar-rule registry. Same Python-style call shape
// as LFM2 (and same FSM states); only the optional outer wrapper tokens
// (<|tool_call_start|>...<|tool_call_end|>) differ in the grammar.
const common_chat_format_state_rules lfm2_5_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,      "content" },
        { common_chat_format_state::IN_REASONING,    "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL,    "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME,    "func-name" },
        { common_chat_format_state::IN_TOOL_ARG_KEY, "arg-name" },
        { common_chat_format_state::IN_TOOL_ARG_VAL, "arg-value" },
    }
};
