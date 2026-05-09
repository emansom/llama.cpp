#include "chat-formats/ministral-3-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

// Mirrors Jinja `tojson` (default, ensure_ascii=False per Mistral convention):
// `, ` and `: ` separators, no ASCII escaping.
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

// Default system message embedded in the template (line 2). Inserted when
// the conversation has no explicit system role.
constexpr const char * DEFAULT_SYSTEM_MESSAGE =
    "# HOW YOU SHOULD THINK AND ANSWER\n\n"
    "First draft your thinking process (inner monologue) until you arrive "
    "at a response. Format your response using Markdown, and use LaTeX for "
    "any mathematical equations. Write both your thoughts and the response "
    "in the same language as the input.\n\n"
    "Your thinking process must follow the template below:[THINK]Your "
    "thoughts or/and draft, like working through an exercise on scratch "
    "paper. Be as casual and as long as you want until you are confident to "
    "generate the response to the user.[/THINK]Here, provide a "
    "self-contained response.";

// Preprocess messages: for system/assistant messages, convert
// `reasoning_content` into a `thinking`-typed content block prepended to
// any existing content (which is wrapped as a `text`-typed block when it
// was a string). Mirrors the logic that previously lived in
// common_chat_params_init_ministral_3.
ordered_json preprocess_messages(const ordered_json & messages) {
    ordered_json out = ordered_json::array();
    if (!messages.is_array()) {
        return out;
    }
    for (const auto & msg : messages) {
        const std::string role = msg.value("role", "");
        if (role != "system" && role != "assistant") {
            out.push_back(msg);
            continue;
        }

        ordered_json content = ordered_json::array();

        if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string()) {
            content.push_back({
                { "type",     "thinking" },
                { "thinking", msg["reasoning_content"].get<std::string>() },
            });
        }

        if (msg.contains("content")) {
            const auto & c = msg["content"];
            if (c.is_string()) {
                content.push_back({
                    { "type", "text" },
                    { "text", c.get<std::string>() },
                });
            } else if (c.is_array()) {
                for (const auto & block : c) {
                    content.push_back(block);
                }
            }
        }

        ordered_json adjusted = msg;
        adjusted["content"] = content;
        adjusted.erase("reasoning_content");
        out.push_back(adjusted);
    }
    return out;
}

// Render the body of a system or assistant message --- iterates the typed
// content blocks (text / thinking).
std::string render_typed_content(const ordered_json & content) {
    std::string out;
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        return out;
    }
    for (const auto & block : content) {
        if (!block.is_object()) {
            continue;
        }
        const std::string type = block.value("type", std::string{});
        if (type == "text") {
            out += block.value("text", std::string{});
        } else if (type == "thinking") {
            out += "[THINK]" + block.value("thinking", std::string{}) + "[/THINK]";
        }
    }
    return out;
}

// Render the body of a user message --- supports text and image chunks.
std::string render_user_content(const ordered_json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        return std::string{};
    }
    // The Jinja template sorts blocks alphabetically by `type` when there
    // are exactly 2 blocks (puts `image` before `text`); otherwise iterates
    // in original order.
    ordered_json blocks = content;
    if (blocks.size() == 2) {
        std::sort(blocks.begin(), blocks.end(), [](const ordered_json & a, const ordered_json & b) {
            return a.value("type", std::string{}) < b.value("type", std::string{});
        });
    }
    std::string out;
    for (const auto & block : blocks) {
        if (!block.is_object()) {
            continue;
        }
        const std::string type = block.value("type", std::string{});
        if (type == "text") {
            out += block.value("text", std::string{});
        } else if (type == "image" || type == "image_url") {
            out += "[IMG]";
        }
    }
    return out;
}

}  // namespace

std::string common_chat_ministral_3_render(const autoparser::generation_params & inputs,
                                           const std::string & bos_token,
                                           const std::string & eos_token) {
    std::ostringstream out;

    // 1. BOS.
    out << bos_token;

    // 2. Preprocess: convert reasoning_content for system/assistant messages.
    const ordered_json adjusted_messages = preprocess_messages(inputs.messages);

    // 3. System prompt: from first message if role=system, otherwise default.
    ordered_json loop_messages = adjusted_messages;
    if (loop_messages.is_array() && !loop_messages.empty() &&
        loop_messages[0].is_object() &&
        loop_messages[0].value("role", std::string{}) == "system") {
        out << "[SYSTEM_PROMPT]" << render_typed_content(loop_messages[0].value("content", ordered_json{}))
            << "[/SYSTEM_PROMPT]";
        loop_messages.erase(0);
    } else {
        out << "[SYSTEM_PROMPT]" << DEFAULT_SYSTEM_MESSAGE << "[/SYSTEM_PROMPT]";
    }

    // 4. Tools block.
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        out << "[AVAILABLE_TOOLS]" << to_jinja_json(inputs.tools) << "[/AVAILABLE_TOOLS]";
    }

    // 5. Per-message rendering.
    if (loop_messages.is_array()) {
        for (const auto & msg : loop_messages) {
            const std::string role = msg.value("role", "");
            if (role == "user") {
                out << "[INST]" << render_user_content(msg.value("content", ordered_json{}))
                    << "[/INST]";
            } else if (role == "assistant") {
                // Body: typed content (text/thinking blocks).
                out << render_typed_content(msg.value("content", ordered_json{}));
                // Tool calls.
                if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
                    for (const auto & tool : msg["tool_calls"]) {
                        out << "[TOOL_CALLS]";
                        const std::string name = tool["function"].value("name", std::string{});
                        const auto & args_val  = tool["function"]["arguments"];
                        std::string arguments;
                        if (args_val.is_string()) {
                            arguments = args_val.get<std::string>();
                            if (arguments.empty()) {
                                arguments = "{}";
                            }
                        } else {
                            arguments = to_jinja_json(args_val);
                        }
                        out << name << "[ARGS]" << arguments;
                    }
                }
                // EOS.
                out << eos_token;
            } else if (role == "tool") {
                std::string content;
                const auto & c = msg.value("content", ordered_json{});
                if (c.is_string()) {
                    content = c.get<std::string>();
                } else if (!c.is_null()) {
                    content = to_jinja_json(c);
                }
                out << "[TOOL_RESULTS]" << content << "[/TOOL_RESULTS]";
            }
        }
    }

    return out.str();
}

// Ministral-3 FSM-state-to-grammar-rule registry. Includes a reasoning
// channel ([THINK]/[/THINK]) but no tool-id rule.
const common_chat_format_state_rules ministral_3_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};
