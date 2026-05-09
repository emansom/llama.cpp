#include "chat-formats/gigachat-v3-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

// GigaChat-v3 FSM-state-to-grammar-rule registry. No reasoning channel and
// no tool-id; the grammar is content + a single function-call turn.
const common_chat_format_state_rules gigachat_v3_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};

namespace {
using ordered_json = nlohmann::ordered_json;

// Token sequences (Jinja: `add_tokens.role_sep`, `add_tokens.message_sep`).
constexpr const char * ROLE_SEP    = "<|role_sep|>\n";
constexpr const char * MESSAGE_SEP = "<|message_sep|>\n\n";

// Verbatim DEVSYSTEM text from GigaChat3-10B-A1.8B.jinja (line 220-290). The
// Jinja template captures it via `{%- set DEVSYSTEM -%}...{%- endset -%}` and
// renders it as the developer-system message body.
constexpr const char * DEVSYSTEM = R"(<role_description>
Description of the roles available in the dialog.

`developer system`
A message added by Sber before the main dialog. It has the highest priority and sets global, non-overridable conditions (for example, conversation rules, the safety policy, the assistant's overall response style, etc.).

`system`
A system instruction added by developers or by the user, but with a lower priority than `developer system`. It usually describes the assistant's instructions, a specific response style, and other conditions for this particular dialog.

`user`
A message or request from the user. The assistant follows it if it does not conflict with higher-priority instructions (see <instruction_priority>).

`user memory`
A sequence of the most up-to-date long-term facts about the user at the time of their request, presented as a JSON list of strings. Facts are listed in chronological order, meaning newer facts are appended to the end of the sequence. When facts are changed or deleted, records of previous facts remain in the sequence. The assistant saves facts using a function and uses them in accordance with the <memory_guidelines> block below.

`added files`
Metadata about files available for use in the dialog, presented in JSON format. It contains the following keys: id (a unique file identifier), name (file name), type (file type).

`assistant`
The assistant's reply to the user's request. If the system instruction or the user does not set additional rules for `assistant`, this reply must comply with the instructions in the <assistant_guidelines> block below. The list of functions available to call is contained in `function descriptions`. The name of the required function and its arguments will be generated next by the `function call` role. In its replies, the assistant follows the instructions in accordance with <instruction_priority>.

`function descriptions`
Function descriptions in TypeScript format. A function is a special tool (or a set of instructions) that the assistant can call to perform specific actions, computations, or obtain data needed to solve the user's task. Each function description contains blocks with the name, description, and arguments. Sometimes the description contains separate blocks with return parameters and usage examples that illustrate the correct call and arguments.

`function call`
The function that `assistant` calls based on the dialog context, and its arguments. The function is invoked in strict accordance with the instructions in the <function_usage> block.

`function result`
The result of the last function call.
</role_description>

<available_modalities>
The assistant can work with the following modalities: text, available functions.
</available_modalities>

<instruction_priority>
If instructions from different roles conflict within the dialog context, observe the following priorities:  
`developer system` > `system` > `user` > `function descriptions` > `function result` > `user memory`
</instruction_priority>

<function_usage>
Basic instructions for working with functions.

Only call those functions that are described in `function descriptions`.

Call available functions when, according to their description, such a call will help provide a more complete and/or accurate answer to the user's request. Fill in function arguments using information from the dialog context. If a function could help answer the request but a required argument is missing from the context, ask the user for the missing data before calling the function. If a necessary function is unavailable or an error occurs, briefly inform the user and, if possible, suggest an alternative.
</function_usage>

<memory_guidelines>
Rules for using facts in long-term memory:

If there is no message under the `user memory` role in the dialog, this is equivalent to the absence of long-term facts about the user in memory. In that case, information about the user is limited to the current dialog, and no new facts should be saved.
</memory_guidelines>

<assistant_guidelines>
You are a helpful assistant.

# Instructions
- Strictly follow the instruction priority.
- Maintain a logical chain of reasoning when answering the user's question.
- For complex questions (for example, STEM), try to answer in detail unless the system message or dialog context limits the response length.
- Be helpful, truthful, and avoid unsafe or prohibited content in your responses.
- Try to reply in the language in which the user asked their question.
</assistant_guidelines>

A dialog will follow below.
The dialog may include various roles described in the <role_description> block.
Each turn begins with the role name and a special token that marks the end of the role's full name, and ends with a special end-of-turn token.
Your task is to continue the dialog from the last specified role in accordance with the dialog context.)";

// JSON serializer matching `tojson(ensure_ascii=False)` defaults: items with
// `, ` and keys with `: `, no escape of non-ASCII. Same as the Functionary
// helper but kept local to GigaChat to avoid introducing a coupling.
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

// Mirror of GigaChat3's `json_schema_to_typescript` macro (lines 6-117).
std::string json_schema_to_typescript(const ordered_json & schema, const std::string & indent) {
    static const std::vector<std::string> ADDITIONAL_JSON_KEYS = {
        "format", "maxItems", "maximum", "minItems", "minimum", "pattern",
    };

    std::string ty;
    if (schema.is_object() && schema.contains("type")) {
        if (schema["type"].is_string()) {
            ty = schema["type"].get<std::string>();
        }
    }

    if (ty == "object") {
        std::string out = "{\n";
        const ordered_json props    = schema.value("properties", ordered_json::object());
        const ordered_json required = schema.value("required", ordered_json::array());

        const bool has_additional_props = schema.contains("additionalProperties");
        ordered_json additional_props_type;
        bool         additional_props_present = false;
        if (has_additional_props) {
            const auto & ap = schema["additionalProperties"];
            if (ap.is_boolean() && ap.get<bool>()) {
                additional_props_type = { {"type", "any"} };
                additional_props_present = true;
            } else if (ap.is_object()) {
                additional_props_type = ap;
                additional_props_present = true;
            }
        }

        for (auto it = props.begin(); it != props.end(); ++it) {
            const std::string & key = it.key();
            const ordered_json & val = it.value();

            // Description comments (split on \n, skip empty lines).
            if (val.contains("description") && val["description"].is_string()) {
                const std::string desc = val["description"].get<std::string>();
                std::string line;
                std::istringstream ss(desc);
                while (std::getline(ss, line)) {
                    // strip() check: if line trimmed of whitespace is non-empty.
                    bool any_non_ws = false;
                    for (char c : line) {
                        if (c != ' ' && c != '\t' && c != '\r') {
                            any_non_ws = true;
                            break;
                        }
                    }
                    if (any_non_ws) {
                        out += indent + "// " + line + "\n";
                    }
                }
            }

            // Additional JSON keys (format, maxItems, etc.).
            if (val.is_object()) {
                for (auto ait = val.begin(); ait != val.end(); ++ait) {
                    const std::string & add_key = ait.key();
                    if (std::find(ADDITIONAL_JSON_KEYS.begin(), ADDITIONAL_JSON_KEYS.end(), add_key) ==
                        ADDITIONAL_JSON_KEYS.end()) {
                        continue;
                    }
                    const ordered_json & add_val = ait.value();
                    if (add_val.is_string()) {
                        out += indent + "// " + add_key + ": \"" + add_val.get<std::string>() + "\"\n";
                    } else {
                        out += indent + "// " + add_key + ": " + to_jinja_json(add_val) + "\n";
                    }
                }
            }

            // Property definition.
            const std::string type_str = json_schema_to_typescript(val, indent + "  ");

            bool is_required = false;
            if (required.is_array()) {
                for (const auto & r : required) {
                    if (r.is_string() && r.get<std::string>() == key) {
                        is_required = true;
                        break;
                    }
                }
            }
            out += indent + key + (is_required ? "" : "?") + ": " + type_str + ",";

            // Default-value comment.
            if (val.is_object() && (val.contains("default") || val.contains("defalut_value"))) {
                ordered_json def = val.contains("default") ? val["default"] : val["defalut_value"];
                if (def.is_string()) {
                    out += " // default: \"" + def.get<std::string>() + "\"";
                } else {
                    out += " // default: " + to_jinja_json(def);
                }
            }

            out += "\n";
        }

        if (has_additional_props && additional_props_present) {
            const std::string additional_type_str = json_schema_to_typescript(additional_props_type, indent + "  ");
            out += indent + "[key: string]: " + additional_type_str + "\n";
        }

        // Closing brace at one-level-less indent (Jinja: `indent[: indent_len - 2]`).
        std::string closing_indent = indent.size() >= 2 ? indent.substr(0, indent.size() - 2) : std::string{};
        out += closing_indent + "}";
        return out;
    }

    if (ty == "string") {
        if (schema.contains("enum") && schema["enum"].is_array()) {
            std::string out;
            const auto & enums = schema["enum"];
            for (size_t i = 0; i < enums.size(); ++i) {
                if (i > 0) {
                    out += " | ";
                }
                out += "\"" + enums[i].get<std::string>() + "\"";
            }
            return out;
        }
        if (schema.contains("format") && schema["format"].is_string()) {
            const std::string fmt = schema["format"].get<std::string>();
            if (fmt == "date-time" || fmt == "date") {
                return "Date";
            }
        }
        return "string";
    }

    if (ty == "number" || ty == "integer") {
        if (schema.contains("enum") && schema["enum"].is_array()) {
            std::string out;
            const auto & enums = schema["enum"];
            for (size_t i = 0; i < enums.size(); ++i) {
                if (i > 0) {
                    out += " | ";
                }
                out += to_jinja_json(enums[i]);
            }
            return out;
        }
        return "number";
    }

    if (ty == "boolean") {
        return "boolean";
    }

    if (ty == "array") {
        if (schema.contains("items")) {
            return json_schema_to_typescript(schema["items"], indent) + "[]";
        }
        return "Array<any>";
    }

    return "any";
}

// Mirror of `render_tool_namespace` macro (lines 123-172).
std::string render_tool_namespace(const std::string & namespace_name, const ordered_json & tools) {
    std::vector<std::string> sections;
    sections.push_back("namespace " + namespace_name + " {");

    for (const auto & tool_in : tools) {
        ordered_json tool = tool_in;
        if (tool.contains("function") && tool["function"].is_object()) {
            tool = tool["function"];
        }

        std::vector<std::string> content_lines;

        // Tool description (split on \n, skip empty lines).
        if (tool.contains("description") && tool["description"].is_string()) {
            const std::string desc = tool["description"].get<std::string>();
            std::string line;
            std::istringstream ss(desc);
            while (std::getline(ss, line)) {
                bool any_non_ws = false;
                for (char c : line) {
                    if (c != ' ' && c != '\t' && c != '\r') {
                        any_non_ws = true;
                        break;
                    }
                }
                if (any_non_ws) {
                    content_lines.push_back("// " + line);
                }
            }
        }

        // Tool signature.
        std::string main_body;
        const std::string name = tool.value("name", std::string{});
        if (tool.contains("parameters") && tool["parameters"].is_object() &&
            tool["parameters"].contains("properties")) {
            const std::string param_type = json_schema_to_typescript(tool["parameters"], "  ");
            main_body = "type " + name + " = (_: " + param_type + ") => ";
        } else {
            main_body = "type " + name + " = () => ";
        }

        // Return type.
        if (tool.contains("return_parameters") && tool["return_parameters"].is_object() &&
            tool["return_parameters"].contains("properties")) {
            const std::string return_type = json_schema_to_typescript(tool["return_parameters"], "  ");
            main_body += return_type;
        } else {
            main_body += "any";
        }

        main_body += ";\n";
        content_lines.push_back(main_body);

        // Join content_lines with \n.
        std::string joined;
        for (size_t i = 0; i < content_lines.size(); ++i) {
            if (i > 0) {
                joined += "\n";
            }
            joined += content_lines[i];
        }
        sections.push_back(std::move(joined));
    }

    sections.push_back("} // namespace " + namespace_name);

    std::string out;
    for (size_t i = 0; i < sections.size(); ++i) {
        if (i > 0) {
            out += "\n";
        }
        out += sections[i];
    }
    return out;
}

// Mirror of `render_role_message` macro (lines 177-189).
// Emits: `<role><|role_sep|>\n<content><|message_sep|>\n\n`
std::string render_role_message(const std::string & role, const ordered_json & content) {
    std::string body;
    if (content.is_string()) {
        body = content.get<std::string>();
    } else if (content.is_null()) {
        body = "";
    } else {
        body = to_jinja_json(content);
    }
    return role + ROLE_SEP + body + MESSAGE_SEP;
}

// Mirror of `render_function_call` macro (lines 192-209).
// Builds JSON string `{"name": "...", "arguments": ...}` and renders it as
// the body of a `function call` role-message.
std::string render_function_call(const ordered_json & tool_call) {
    ordered_json call = tool_call;
    if (call.contains("function") && call["function"].is_object()) {
        call = call["function"];
    }

    const std::string name      = call.value("name", std::string{});
    std::string       arguments;
    const auto & args_val = call["arguments"];
    if (args_val.is_string()) {
        arguments = args_val.get<std::string>();
    } else {
        arguments = to_jinja_json(args_val);
    }

    const std::string body = "{\"name\": \"" + name + "\", \"arguments\": " + arguments + "}";
    // Wrap as a `function call` role-message.
    return std::string("function call") + ROLE_SEP + body + MESSAGE_SEP;
}

}  // namespace

std::string common_chat_gigachat_v3_render(const autoparser::generation_params & inputs,
                                           const std::string & bos_token) {
    std::ostringstream out;

    // 1. BOS.
    out << bos_token;

    // 2. DEVSYSTEM (always).
    out << render_role_message("developer system", ordered_json(DEVSYSTEM));

    // 3. Optional system message (only if first message is system; consume it).
    ordered_json messages = inputs.messages;
    if (messages.is_array() && !messages.empty() &&
        messages[0].is_object() && messages[0].value("role", std::string{}) == "system") {
        out << render_role_message("system", messages[0].value("content", ordered_json{}));
        messages.erase(0);
    }

    // 4. Optional tools description.
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        const std::string tools_content = render_tool_namespace("functions", inputs.tools) + "\n\n";
        out << render_role_message("function descriptions", ordered_json(tools_content));
    }

    // 5. Main message loop.
    if (messages.is_array()) {
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto & msg = messages[i];
            const std::string role = msg.value("role", std::string{});

            if (role == "tool") {
                out << render_role_message("function result", msg.value("content", ordered_json{}));
            } else if (role == "assistant") {
                out << render_role_message("assistant", msg.value("content", ordered_json{}));
                if (msg.contains("tool_calls") && msg["tool_calls"].is_array() && !msg["tool_calls"].empty()) {
                    out << render_function_call(msg["tool_calls"][0]);
                }
            } else {
                out << render_role_message(role, msg.value("content", ordered_json{}));
            }

            // Generation prompt: only after the LAST iteration AND if last role isn't assistant.
            if (i + 1 == messages.size() && inputs.add_generation_prompt && role != "assistant") {
                out << "assistant" << ROLE_SEP;
            }
        }
    }

    return out.str();
}
