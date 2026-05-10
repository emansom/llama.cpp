#include "chat-formats/qwq-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

// JSON serializer matching Jinja `tojson` defaults: items `, ` and keys `: `
// separators, no ASCII escaping.
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

// Mirror of QwQ's past-assistant content stripping: when the assistant
// message isn't the last one in the conversation, split on `</think>` and
// take the last segment, lstripped of `\n`. Used to drop reasoning from
// past assistant turns (matches the template's `not loop.last` branch).
std::string strip_past_thinking(const std::string & content) {
    size_t pos = content.rfind("</think>");
    if (pos == std::string::npos) {
        return content;
    }
    std::string after = content.substr(pos + sizeof("</think>") - 1);
    size_t lstrip_start = after.find_first_not_of('\n');
    return lstrip_start == std::string::npos ? std::string{} : after.substr(lstrip_start);
}

}  // namespace

std::string common_chat_qwq_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    // 1. System message: emitted with tools list when tools present, otherwise
    //    just from the first system message (or omitted entirely if none).
    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    const bool first_is_system = inputs.messages.is_array() && !inputs.messages.empty() &&
        inputs.messages[0].is_object() &&
        inputs.messages[0].value("role", std::string{}) == "system";

    if (has_tools) {
        out << "<|im_start|>system\n";
        if (first_is_system) {
            const auto & content = inputs.messages[0].value("content", ordered_json{});
            if (content.is_string()) {
                out << content.get<std::string>();
            }
        }
        // Note: QwQ emits no default "You are Qwen..." text when no system
        // message is present (unlike Qwen2.5).
        out << "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            << "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto & tool : inputs.tools) {
            out << "\n" << to_jinja_json(tool);
        }
        out << "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            << "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
    } else if (first_is_system) {
        std::string content;
        const auto & c = inputs.messages[0].value("content", ordered_json{});
        if (c.is_string()) content = c.get<std::string>();
        out << "<|im_start|>system\n" << content << "<|im_end|>\n";
    }

    // 2. Per-message rendering. Note: when has_tools=true we already consumed
    //    the first system message into the system block above; otherwise it
    //    was consumed into `<|im_start|>system...<|im_end|>`. In both cases
    //    the loop's `loop.first` only triggers special handling for the
    //    first message, but the FIRST iteration matches the system-skipping
    //    logic by checking `not loop.first`.
    if (!inputs.messages.is_array()) {
        return out.str();
    }
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});
        const bool is_first = (i == 0);
        const bool is_last  = (i + 1 == inputs.messages.size());
        const bool has_tool_calls = role == "assistant" && message.contains("tool_calls") &&
            message["tool_calls"].is_array() && !message["tool_calls"].empty();

        if (role == "user" || (role == "system" && !is_first)) {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        } else if (role == "assistant" && !has_tool_calls) {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            if (!is_last) {
                content = strip_past_thinking(content);
            }
            out << "<|im_start|>assistant\n" << content << "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            if (!is_last) {
                content = strip_past_thinking(content);
            }
            out << "<|im_start|>assistant";
            if (!content.empty()) {
                out << "\n" << content;
            }
            for (const auto & tool_call_in : message["tool_calls"]) {
                ordered_json tool_call = tool_call_in;
                if (tool_call.contains("function") && tool_call["function"].is_object()) {
                    tool_call = tool_call["function"];
                }
                out << "\n<tool_call>\n{\"name\": \"" << tool_call.value("name", std::string{}) << "\"";
                if (tool_call.contains("arguments")) {
                    out << ", \"arguments\": ";
                    const auto & args = tool_call["arguments"];
                    if (args.is_string()) {
                        out << args.get<std::string>();
                    } else {
                        out << to_jinja_json(args);
                    }
                }
                out << "}\n</tool_call>";
            }
            out << "<|im_end|>\n";
        } else if (role == "tool") {
            // Reuse the previous `<|im_start|>user` opener when consecutive.
            const std::string prev_role = i > 0
                ? inputs.messages[i - 1].value("role", std::string{}) : std::string{};
            if (i == 0 || prev_role != "tool") {
                out << "<|im_start|>user";
            }
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "\n<tool_response>\n" << content << "\n</tool_response>";
            const std::string next_role = i + 1 < inputs.messages.size()
                ? inputs.messages[i + 1].value("role", std::string{}) : std::string{};
            if (is_last || next_role != "tool") {
                out << "<|im_end|>\n";
            }
        }
    }

    // 3. Generation prompt: `<|im_start|>assistant\n<think>\n` plus optional
    //    `</think>` when enable_thinking is false.
    if (inputs.add_generation_prompt) {
        out << "<|im_start|>assistant\n<think>\n";
        if (!inputs.enable_thinking) {
            out << "</think>";
        }
    }

    return out.str();
}
