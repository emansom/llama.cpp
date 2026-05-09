#include "chat-formats/gpt-oss-format.h"

// GPT-OSS (Harmony) FSM-state-to-grammar-rule registry. The grammar has
// reasoning (analysis turn), content turn (final/commentary), and function
// tool calls; no tool-id rule.
const common_chat_format_state_rules gpt_oss_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// GPT-OSS (Harmony) prompt writer
// Mirrors models/templates/openai-gpt-oss-120b.jinja.
// ──────────────────────────────────────────────────────────────────────────────

#include <nlohmann/json.hpp>

#include <ctime>
#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

// JSON serializer matching Jinja `tojson` defaults (ensure_ascii=true would
// escape non-ASCII; ASCII-only fixtures don't trigger that path). Items use
// `, ` and keys use `: ` separators.
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

// Format a time_point as YYYY-MM-DD using the local timezone (matching
// Jinja's `strftime_now("%Y-%m-%d")` which uses `localtime`).
std::string format_date(std::chrono::system_clock::time_point tp) {
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm * lt  = std::localtime(&t);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", lt);
    return std::string(buf);
}

// Mirror of `render_typescript_type` macro (lines 10-106). Recurses for
// nested array/object types.
std::string render_typescript_type(const ordered_json & param_spec, const ordered_json & required_params) {
    if (!param_spec.is_object()) return "any";

    auto type_eq = [&](const char * t) {
        return param_spec.contains("type") && param_spec["type"].is_string() &&
               param_spec["type"].get<std::string>() == t;
    };
    auto nullable = [&]() {
        return param_spec.value("nullable", false);
    };

    // Array of types (e.g., ["object", "object"]).
    if (param_spec.contains("type") && param_spec["type"].is_array() &&
        !param_spec["type"].empty()) {
        const auto & arr = param_spec["type"];
        if (arr.size() > 1) {
            std::string out;
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) out += " | ";
                out += arr[i].is_string() ? arr[i].get<std::string>() : arr[i].dump();
            }
            return out;
        }
        return arr[0].is_string() ? arr[0].get<std::string>() : arr[0].dump();
    }

    if (type_eq("array")) {
        if (param_spec.contains("items") && param_spec["items"].is_object()) {
            const auto & items = param_spec["items"];
            std::string out;
            if (items.value("type", std::string{}) == "string") {
                out = "string[]";
            } else if (items.value("type", std::string{}) == "number") {
                out = "number[]";
            } else if (items.value("type", std::string{}) == "integer") {
                out = "number[]";
            } else if (items.value("type", std::string{}) == "boolean") {
                out = "boolean[]";
            } else {
                std::string inner_type = render_typescript_type(items, required_params);
                if (inner_type == "object | object" || inner_type.size() > 50) {
                    out = "any[]";
                } else {
                    out = inner_type + "[]";
                }
            }
            if (nullable()) out += " | null";
            return out;
        }
        std::string out = "any[]";
        if (nullable()) out += " | null";
        return out;
    }

    if (param_spec.contains("oneOf") && param_spec["oneOf"].is_array()) {
        const auto & one_of = param_spec["oneOf"];
        bool has_object_variants = false;
        for (const auto & v : one_of) {
            if (v.is_object() && v.value("type", std::string{}) == "object") {
                has_object_variants = true;
            }
        }
        if (has_object_variants && one_of.size() > 1) {
            return "any";
        }
        std::string out;
        for (size_t i = 0; i < one_of.size(); ++i) {
            const auto & variant = one_of[i];
            out += render_typescript_type(variant, required_params);
            if (variant.contains("description") && variant["description"].is_string()) {
                out += "// " + variant["description"].get<std::string>();
            }
            if (variant.contains("default")) {
                out += "// default: " + to_jinja_json(variant["default"]);
            }
            if (i + 1 < one_of.size()) {
                out += " | ";
            }
        }
        return out;
    }

    if (type_eq("string")) {
        if (param_spec.contains("enum") && param_spec["enum"].is_array()) {
            std::string out = "\"";
            const auto & enums = param_spec["enum"];
            for (size_t i = 0; i < enums.size(); ++i) {
                if (i > 0) out += "\" | \"";
                out += enums[i].get<std::string>();
            }
            out += "\"";
            return out;
        }
        std::string out = "string";
        if (nullable()) out += " | null";
        return out;
    }
    if (type_eq("number") || type_eq("integer")) return "number";
    if (type_eq("boolean")) return "boolean";

    if (type_eq("object")) {
        if (param_spec.contains("properties") && param_spec["properties"].is_object()) {
            std::string out = "{\n";
            const auto & props = param_spec["properties"];
            const auto & req   = param_spec.value("required", ordered_json::array());
            bool first = true;
            for (auto it = props.begin(); it != props.end(); ++it) {
                if (!first) out += ", ";
                first = false;
                const std::string & prop_name = it.key();
                bool is_required = false;
                if (req.is_array()) {
                    for (const auto & r : req) {
                        if (r.is_string() && r.get<std::string>() == prop_name) {
                            is_required = true;
                            break;
                        }
                    }
                }
                out += prop_name;
                if (!is_required) out += "?";
                out += ": ";
                out += render_typescript_type(it.value(), req);
            }
            out += "}";
            return out;
        }
        return "object";
    }
    return "any";
}

// Mirror of `render_tool_namespace` macro (lines 108-148).
std::string render_tool_namespace(const std::string & namespace_name, const ordered_json & tools) {
    std::string out;
    out += "## " + namespace_name + "\n\n";
    out += "namespace " + namespace_name + " {\n\n";

    for (const auto & tool_in : tools) {
        ordered_json tool = tool_in;
        if (tool.contains("function") && tool["function"].is_object()) {
            tool = tool["function"];
        }
        out += "// " + tool.value("description", std::string{}) + "\n";
        out += "type " + tool.value("name", std::string{}) + " = ";
        if (tool.contains("parameters") && tool["parameters"].is_object() &&
            tool["parameters"].contains("properties") && tool["parameters"]["properties"].is_object() &&
            !tool["parameters"]["properties"].empty()) {
            out += "(_: {\n";
            const auto & props = tool["parameters"]["properties"];
            const auto & required = tool["parameters"].value("required", ordered_json::array());
            for (auto it = props.begin(); it != props.end(); ++it) {
                const std::string & param_name = it.key();
                const ordered_json & param_spec = it.value();
                if (param_spec.contains("description") && param_spec["description"].is_string()) {
                    out += "// " + param_spec["description"].get<std::string>() + "\n";
                }
                out += param_name;
                bool is_required = false;
                if (required.is_array()) {
                    for (const auto & r : required) {
                        if (r.is_string() && r.get<std::string>() == param_name) {
                            is_required = true;
                            break;
                        }
                    }
                }
                if (!is_required) out += "?";
                out += ": ";
                out += render_typescript_type(param_spec, required);
                if (param_spec.contains("default")) {
                    if (param_spec.contains("enum")) {
                        out += ", // default: " + (param_spec["default"].is_string()
                            ? param_spec["default"].get<std::string>()
                            : param_spec["default"].dump());
                    } else if (param_spec.contains("oneOf")) {
                        out += "// default: " + (param_spec["default"].is_string()
                            ? param_spec["default"].get<std::string>()
                            : param_spec["default"].dump());
                    } else {
                        out += ", // default: " + to_jinja_json(param_spec["default"]);
                    }
                }
                out += ",\n";
            }
            out += "}) => any;\n\n";
        } else {
            out += "() => any;\n\n";
        }
    }

    out += "} // namespace " + namespace_name;
    return out;
}

// Mirror of `build_system_message` macro (lines 196-223).
std::string build_system_message(const autoparser::generation_params & inputs) {
    std::string out;
    // Default identity (template line 197-199). model_identity defaults are
    // not exposed via generation_params; we use the canonical default.
    out += "You are ChatGPT, a large language model trained by OpenAI.\n";
    out += "Knowledge cutoff: 2024-06\n";
    out += "Current date: " + format_date(inputs.now) + "\n\n";
    out += "Reasoning: medium\n\n";

    // builtin_tools support is not plumbed through generation_params; if
    // future use requires it, add a `builtin_tools` field there. For now,
    // skip the `# Tools` block from `build_system_message`.

    out += "# Valid channels: analysis, commentary, final. Channel must be included for every message.";
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        out += "\nCalls to these tools must go to the commentary channel: 'functions'.";
    }
    return out;
}

}  // namespace

std::string common_chat_gpt_oss_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    // 1. Always-emit system message.
    out << "<|start|>system<|message|>" << build_system_message(inputs) << "<|end|>";

    // 2. Extract developer message (first system/developer message body).
    std::string developer_message;
    ordered_json loop_messages = inputs.messages;
    if (loop_messages.is_array() && !loop_messages.empty() && loop_messages[0].is_object()) {
        const std::string r0 = loop_messages[0].value("role", std::string{});
        if (r0 == "developer" || r0 == "system") {
            const auto & content = loop_messages[0].value("content", ordered_json{});
            if (content.is_string()) {
                developer_message = content.get<std::string>();
            }
            loop_messages.erase(0);
        }
    }

    // 3. Render developer message + tool namespace.
    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    if (!developer_message.empty() || has_tools) {
        out << "<|start|>developer<|message|>";
        if (!developer_message.empty()) {
            out << "# Instructions\n\n" << developer_message << "\n\n";
        }
        if (has_tools) {
            out << "# Tools\n\n" << render_tool_namespace("functions", inputs.tools);
        }
        out << "<|end|>";
    }

    // 4. Per-message rendering. Preprocess: copy `reasoning_content` to
    //    `thinking` (the GPT-OSS template expects the latter) and drop
    //    `content` when the message also has `tool_calls` (the template
    //    raises if both are present).
    std::string last_tool_call_name;
    if (!loop_messages.is_array()) {
        loop_messages = ordered_json::array();
    }
    for (auto & msg : loop_messages) {
        if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string()) {
            msg["thinking"] = msg["reasoning_content"];
            if (msg.contains("tool_calls") && msg["tool_calls"].is_array() &&
                !msg["tool_calls"].empty()) {
                msg.erase("content");
            }
        }
    }
    for (size_t i = 0; i < loop_messages.size(); ++i) {
        const auto & message = loop_messages[i];
        const std::string role = message.value("role", std::string{});

        if (role == "assistant") {
            if (message.contains("tool_calls") && message["tool_calls"].is_array() &&
                !message["tool_calls"].empty()) {
                // Look for a future final-message assistant (no tool_calls).
                bool future_final_message = false;
                for (size_t j = i + 1; j < loop_messages.size(); ++j) {
                    if (loop_messages[j].value("role", std::string{}) == "assistant" &&
                        !(loop_messages[j].contains("tool_calls") &&
                          loop_messages[j]["tool_calls"].is_array() &&
                          !loop_messages[j]["tool_calls"].empty())) {
                        future_final_message = true;
                    }
                }

                ordered_json tool_call = message["tool_calls"][0];
                if (tool_call.contains("function") && tool_call["function"].is_object()) {
                    tool_call = tool_call["function"];
                }

                const bool has_content = message.contains("content") &&
                    message["content"].is_string() &&
                    !message["content"].get<std::string>().empty();
                const bool has_thinking = message.contains("thinking") &&
                    message["thinking"].is_string() &&
                    !message["thinking"].get<std::string>().empty();
                if (has_content && !future_final_message) {
                    out << "<|start|>assistant<|channel|>analysis<|message|>"
                        << message["content"].get<std::string>() << "<|end|>";
                } else if (has_thinking && !future_final_message) {
                    out << "<|start|>assistant<|channel|>analysis<|message|>"
                        << message["thinking"].get<std::string>() << "<|end|>";
                }

                std::string content_type = "json";
                if (tool_call.contains("content_type") && tool_call["content_type"].is_string()) {
                    content_type = tool_call["content_type"].get<std::string>();
                }
                out << "<|start|>assistant to=functions." << tool_call.value("name", std::string{})
                    << "<|channel|>commentary " << content_type << "<|message|>"
                    << to_jinja_json(tool_call.value("arguments", ordered_json::object()))
                    << "<|call|>";
                last_tool_call_name = tool_call.value("name", std::string{});
            } else if (i + 1 == loop_messages.size() && !inputs.add_generation_prompt) {
                // Training shape: emit thinking analysis (if present) then final with <|return|>.
                if (message.contains("thinking") && message["thinking"].is_string()) {
                    out << "<|start|>assistant<|channel|>analysis<|message|>"
                        << message["thinking"].get<std::string>() << "<|end|>";
                }
                std::string content;
                if (message.contains("content") && message["content"].is_string()) {
                    content = message["content"].get<std::string>();
                }
                out << "<|start|>assistant<|channel|>final<|message|>" << content << "<|return|>";
            } else {
                std::string content;
                if (message.contains("content") && message["content"].is_string()) {
                    content = message["content"].get<std::string>();
                }
                out << "<|start|>assistant<|channel|>final<|message|>" << content << "<|end|>";
                last_tool_call_name.clear();
            }
        } else if (role == "tool") {
            std::string content;
            if (message.contains("content")) {
                if (message["content"].is_string()) {
                    content = to_jinja_json(message["content"]);
                } else {
                    content = to_jinja_json(message["content"]);
                }
            }
            out << "<|start|>functions." << last_tool_call_name
                << " to=assistant<|channel|>commentary<|message|>" << content << "<|end|>";
        } else if (role == "user") {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "<|start|>user<|message|>" << content << "<|end|>";
        }
    }

    // 5. Generation prompt: `<|start|>assistant` (the trailing newline in
    //    the template is stripped by the surrounding `{%- endif -%}`).
    if (inputs.add_generation_prompt) {
        out << "<|start|>assistant";
    }

    return out.str();
}
