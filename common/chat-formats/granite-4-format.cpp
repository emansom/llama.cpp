#include "chat-formats/granite-4-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <stdexcept>
#include <string>

const common_chat_format_state_rules granite_4_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Granite 4.0 prompt writer
// ──────────────────────────────────────────────────────────────────────────────

namespace {
using ordered_json = nlohmann::ordered_json;

// JSON serializer matching Jinja `tojson` defaults: items `, ` and keys `: `.
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

// Concatenate the text segments of a content array (`{type: text, text: ...}`).
// Multiple text segments are joined with `\n`. Returns an empty string for
// null/undefined.
std::string concat_text_content(const ordered_json & content) {
    if (content.is_null()) {
        return {};
    }
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string out;
        for (const auto & entry : content) {
            if (!entry.is_object()) continue;
            if (entry.value("type", std::string{}) != "text") continue;
            if (!out.empty()) out += "\n";
            if (entry.contains("text") && entry["text"].is_string()) {
                out += entry["text"].get<std::string>();
            }
        }
        return out;
    }
    return {};
}

constexpr const char * TOOLS_PREFIX =
    "You are a helpful assistant with access to the following tools. "
    "You may call one or more tools to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";

constexpr const char * TOOLS_SUFFIX =
    "\n</tools>\n\n"
    "For each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>. "
    "If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.";

constexpr const char * DEFAULT_SYSTEM =
    "You are a helpful assistant. Please ensure responses are professional, accurate, and safe.";

std::string build_tools_block(const ordered_json & tools) {
    std::string out = TOOLS_PREFIX;
    for (const auto & tool : tools) {
        out += "\n";
        out += to_jinja_json(tool);
    }
    out += TOOLS_SUFFIX;
    return out;
}

}  // namespace

std::string common_chat_granite_4_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    const bool has_messages = inputs.messages.is_array() && !inputs.messages.empty();
    const bool first_is_system = has_messages &&
        inputs.messages[0].is_object() &&
        inputs.messages[0].value("role", std::string{}) == "system";
    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    // 1. Build the system message text. The template concatenates: the first
    //    system message (if any) + tools block + documents block (RAG, not
    //    surfaced here). When everything is empty, fall back to the default
    //    "You are a helpful assistant..." text.
    std::string system_msg;
    if (first_is_system) {
        system_msg = concat_text_content(inputs.messages[0].value("content", ordered_json{}));
    }
    if (has_tools) {
        const std::string tools_block = build_tools_block(inputs.tools);
        if (!system_msg.empty()) {
            system_msg += "\n\n" + tools_block;
        } else {
            system_msg = tools_block;
        }
    }
    if (system_msg.empty()) {
        system_msg = DEFAULT_SYSTEM;
    }
    out << "<|start_of_role|>system<|end_of_role|>" << system_msg << "<|end_of_text|>\n";

    // 2. Per-message rendering.
    if (!has_messages) {
        if (inputs.add_generation_prompt) {
            out << "<|start_of_role|>assistant<|end_of_role|>";
        }
        return out.str();
    }
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});
        const std::string content = concat_text_content(message.value("content", ordered_json{}));
        const bool is_first = (i == 0);

        if (role == "user" || (role == "system" && !is_first)) {
            out << "<|start_of_role|>" << role << "<|end_of_role|>" << content << "<|end_of_text|>\n";
        } else if (role == "system") {
            // Already consumed into the system block.
        } else if (role == "assistant") {
            out << "<|start_of_role|>assistant<|end_of_role|>" << content;
            const bool has_tool_calls = message.contains("tool_calls") &&
                message["tool_calls"].is_array() && !message["tool_calls"].empty();
            if (has_tool_calls) {
                const auto & tcs = message["tool_calls"];
                for (size_t k = 0; k < tcs.size(); ++k) {
                    // Newline separator between content and the first tool
                    // call (only if content is non-empty), or between
                    // consecutive tool calls.
                    if ((k == 0 && !content.empty()) || k > 0) {
                        out << "\n";
                    }
                    ordered_json tc = tcs[k];
                    if (tc.contains("function") && tc["function"].is_object()) {
                        tc = tc["function"];
                    }
                    out << "<tool_call>\n{\"name\": \"" << tc.value("name", std::string{})
                        << "\", \"arguments\": ";
                    const auto & args = tc.contains("arguments") ? tc["arguments"] : ordered_json::object();
                    if (args.is_string()) {
                        out << args.get<std::string>();
                    } else {
                        out << to_jinja_json(args);
                    }
                    out << "}\n</tool_call>";
                }
            }
            out << "<|end_of_text|>\n";
        } else if (role == "tool") {
            const std::string prev_role = i > 0
                ? inputs.messages[i - 1].value("role", std::string{}) : std::string{};
            if (i == 0 || prev_role != "tool") {
                out << "<|start_of_role|>user<|end_of_role|>";
            }
            out << "\n<tool_response>\n" << content << "\n</tool_response>";
            const bool is_last = (i + 1 == inputs.messages.size());
            const std::string next_role = is_last ? std::string{}
                : inputs.messages[i + 1].value("role", std::string{});
            if (is_last || next_role != "tool") {
                out << "<|end_of_text|>\n";
            }
        } else {
            throw std::runtime_error("Granite 4.0 writer: unexpected message role: " + role);
        }
    }

    // 3. Generation prompt.
    if (inputs.add_generation_prompt) {
        out << "<|start_of_role|>assistant<|end_of_role|>";
    }
    return out.str();
}
