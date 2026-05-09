#include "chat-formats/kimi-k2-format.h"

#include <nlohmann/json.hpp>

#include <sstream>

namespace {
using ordered_json = nlohmann::ordered_json;

// JSON serializer matching Jinja's `| tojson` filter default formatting:
// item separator `, ` and key separator `: ` (Python-like compact form with
// spaces). nlohmann::ordered_json::dump() emits `,` / `:` with no spaces, so
// it cannot be used directly when the rendered prompt must match Jinja's
// output byte-for-byte.
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

// Render assistant message body content (string or content_parts array).
// Mirrors moonshotai-Kimi-K2.jinja lines 28-37.
std::string render_message_body(const ordered_json & content) {
    if (content.is_null()) {
        return "";
    }
    if (content.is_string()) {
        return content.get<std::string>();
    }
    std::string out;
    if (content.is_array()) {
        for (const auto & part : content) {
            if (!part.is_object()) {
                continue;
            }
            const bool is_image =
                (part.contains("type") && part["type"].is_string() && part["type"].get<std::string>() == "image") ||
                part.contains("image") ||
                part.contains("image_url");
            if (is_image) {
                out += "<|media_start|>image<|media_content|><|media_pad|><|media_end|>";
            } else if (part.contains("text") && part["text"].is_string()) {
                out += part["text"].get<std::string>();
            }
        }
    }
    return out;
}
}  // namespace

std::string common_chat_kimi_k2_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    // 1. Tools declaration (Jinja: {%- if tools -%} ... {%- endif -%}).
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        out << "<|im_system|>tool_declare<|im_middle|>" << to_jinja_json(inputs.tools) << "<|im_end|>";
    }

    // 2. Messages (Jinja: {%- for message in messages -%}).
    bool first_iteration = true;
    for (const auto & msg : inputs.messages) {
        const std::string role = msg.value("role", "");

        // First-message default system prompt (Jinja lines 5-7).
        if (first_iteration && role != "system") {
            out << "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>";
        }
        first_iteration = false;

        // Role-marker open (Jinja lines 8-16).
        if (role == "system") {
            out << "<|im_system|>system<|im_middle|>";
        } else if (role == "user") {
            out << "<|im_user|>user<|im_middle|>";
        } else if (role == "assistant") {
            out << "<|im_assistant|>assistant<|im_middle|>";
        } else if (role == "tool") {
            out << "<|im_system|>tool<|im_middle|>";
        }

        // Body.
        const bool has_tool_calls = role == "assistant"
            && msg.contains("tool_calls")
            && msg["tool_calls"].is_array()
            && !msg["tool_calls"].empty();
        if (has_tool_calls) {
            // Jinja lines 17-25.
            if (msg.contains("content") && !msg["content"].is_null()) {
                out << render_message_body(msg["content"]);
            }
            out << "<|tool_calls_section_begin|>";
            int idx = 0;
            for (const auto & tc : msg["tool_calls"]) {
                const std::string func_name = tc["function"]["name"].get<std::string>();
                const std::string formatted_id = "functions." + func_name + ":" + std::to_string(idx);
                out << "<|tool_call_begin|>" << formatted_id
                    << "<|tool_call_argument_begin|>"
                    << to_jinja_json(tc["function"]["arguments"])
                    << "<|tool_call_end|>";
                ++idx;
            }
            out << "<|tool_calls_section_end|>";
        } else if (role == "tool") {
            // Jinja line 27 emits literal backslash-n (two chars), NOT a real
            // newline — the template's raw-text path passes \n through verbatim.
            const std::string tool_call_id = msg.value("tool_call_id", "");
            const ordered_json content     = msg.value("content", ordered_json{});
            out << "## Return of " << tool_call_id << "\\n" << render_message_body(content);
        } else if (msg.contains("content")) {
            // Jinja lines 28-37.
            out << render_message_body(msg["content"]);
        }

        out << "<|im_end|>";
    }

    // 3. Generation prompt (Jinja lines 41-43).
    if (inputs.add_generation_prompt) {
        out << "<|im_assistant|>assistant<|im_middle|>";
    }

    return out.str();
}

// Kimi-K2 FSM-state-to-grammar-rule registry.
//
// Maps each `common_chat_format_state` reached by the json-tagged tracker
// (see common/chat-formats/json-tagged-format.cpp) to the corresponding
// rule name in `grammars/chat/kimi-k2.lark`. Rule names use hyphens to match
// lark-to-peg's underscore-to-hyphen normalization (lark-to-peg.cpp:633-635).
//
// INITIAL and DONE are inter-rule (entry / terminal) states with no
// associated rule — they are intentionally absent from the registry.
const common_chat_format_state_rules kimi_k2_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_ID,   "tool-id" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};
