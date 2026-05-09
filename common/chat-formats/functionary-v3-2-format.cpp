#include "chat-formats/functionary-v3-2-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

// Functionary-v3.2 FSM-state-to-grammar-rule registry. The grammar has no
// reasoning channel and no tool-id rule, so those states are absent.
// Rule names use hyphens (lark-to-peg normalization).
const common_chat_format_state_rules functionary_v3_2_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Schema generator: mirrors the Jinja macros in meetkai-functionary-medium-v3.2.jinja
// (`json_schema_to_typescript`, `get_parameter_typescript`, `get_param_info`,
// `convert_data_type`, `get_format_param`, `get_enum_option_str`,
// `get_array_typescript`, `append_new_param_info`, `generate_schema_from_functions`)
// in C++. Output must be byte-identical to the Jinja path so the prompt the
// model sees is unchanged.
// ──────────────────────────────────────────────────────────────────────────────

namespace {
using ordered_json = nlohmann::ordered_json;

// Mirrors Jinja `convert_data_type`.
std::string convert_data_type(const std::string & param_type) {
    if (param_type == "integer" || param_type == "float") {
        return "number";
    }
    return param_type;
}

// Mirrors Jinja `get_format_param`. Returns "<|NONE|>" if no format is set.
std::string get_format_param(const ordered_json & param) {
    if (param.contains("format")) {
        return param["format"].get<std::string>();
    }
    if (param.contains("oneOf") && param["oneOf"].is_array()) {
        std::string out;
        const auto & one_of = param["oneOf"];
        for (size_t i = 0; i < one_of.size(); ++i) {
            const auto & item = one_of[i];
            if (item.contains("format")) {
                if (i + 1 == one_of.size()) {
                    out += item["format"].get<std::string>();
                } else {
                    out += item["format"].get<std::string>() + " or ";
                }
            }
        }
        if (!out.empty()) {
            return out;
        }
    }
    return "<|NONE|>";
}

// Mirrors Jinja `get_param_info`: builds the `// description...` comment line
// or returns "<|NONE|>" when there's nothing to comment.
std::string get_param_info(const ordered_json & param) {
    const std::string format_param = get_format_param(param);
    const bool has_description = param.contains("description");
    const bool has_default     = param.contains("default");
    const bool has_format      = format_param != "<|NONE|>";
    const bool has_max         = param.contains("maximum");
    const bool has_min         = param.contains("minimum");
    const bool has_max_len     = param.contains("maxLength");
    const bool has_min_len     = param.contains("minLength");

    if (!has_description && !has_default && !has_format && !has_max && !has_min && !has_max_len && !has_min_len) {
        return "<|NONE|>";
    }

    std::string out = "//";

    if (has_description) {
        std::string desc = param["description"].get<std::string>();
        if (desc.empty() || desc.back() != '.') {
            desc += ".";
        }
        out += " " + desc;
    }

    if (has_default) {
        const auto & def = param["default"];
        const std::string param_type = param.value("type", std::string{"any"});
        std::string default_value;
        if (def.is_string() && param_type == "string") {
            default_value = "\"" + def.get<std::string>() + "\"";
        } else {
            default_value = def.dump();
        }
        out += " Default=" + default_value + ".";
    }

    if (has_format) {
        out += " Format=" + format_param;
    }

    static constexpr const char * fields[][2] = {
        { "maximum",   "Maximum" },
        { "minimum",   "Minimum" },
        { "maxLength", "Maximum length" },
        { "minLength", "Minimum length" },
    };
    for (const auto & f : fields) {
        if (param.contains(f[0])) {
            out += std::string(" ") + f[1] + "=" + param[f[0]].dump();
        }
    }

    return out;
}

// Mirrors Jinja `get_enum_option_str`: joins enum options with ` | `, wrapping
// strings in double-quotes.
std::string get_enum_option_str(const ordered_json & enum_options) {
    std::string out;
    for (size_t i = 0; i < enum_options.size(); ++i) {
        const auto & v = enum_options[i];
        if (v.is_string()) {
            out += "\"" + v.get<std::string>() + "\"";
        } else {
            out += v.dump();
        }
        if (i + 1 < enum_options.size()) {
            out += " | ";
        }
    }
    return out;
}

// Mirrors Jinja `get_param_type`. Returns the typescript type for a value
// (with `integer`/`float` collapsed to `number`).
std::string get_param_type(const ordered_json & param) {
    if (param.contains("type")) {
        const auto & raw = param["type"];
        std::string param_type;
        if (raw.is_array()) {
            std::string joined;
            for (size_t i = 0; i < raw.size(); ++i) {
                if (i > 0) {
                    joined += " | ";
                }
                joined += raw[i].is_string() ? raw[i].get<std::string>() : raw[i].dump();
            }
            param_type = joined;
        } else if (raw.is_string()) {
            param_type = raw.get<std::string>();
        }
        return convert_data_type(param_type);
    }
    if (param.contains("oneOf") && param["oneOf"].is_array()) {
        std::vector<std::string> types;
        for (const auto & item : param["oneOf"]) {
            if (item.contains("type") && item["type"].is_string()) {
                std::string t = item["type"].get<std::string>();
                if (std::find(types.begin(), types.end(), t) == types.end()) {
                    types.push_back(std::move(t));
                }
            }
        }
        std::string joined;
        for (size_t i = 0; i < types.size(); ++i) {
            if (i > 0) {
                joined += " | ";
            }
            joined += types[i];
        }
        return convert_data_type(joined);
    }
    return std::string{};
}

std::string append_new_param_info(const std::string & param_declaration,
                                  const std::string & comment_info,
                                  const ordered_json & examples_info,
                                  int                  depth) {
    std::string offset = depth >= 1 ? std::string(4 * depth, ' ') : std::string{};
    std::string out;
    if (comment_info != "<|NONE|>") {
        out += "\n" + offset + comment_info;
        if (!examples_info.empty()) {
            for (const auto & example : examples_info) {
                std::string ex = example.is_string() ? example.get<std::string>() : example.dump();
                std::string replaced;
                for (char c : ex) {
                    if (c == '\'') {
                        replaced += '"';
                    } else {
                        replaced += c;
                    }
                }
                out += "\n" + offset + "// " + replaced;
            }
        }
    }
    out += "\n" + offset + param_declaration;
    return out;
}

std::string get_parameter_typescript(const ordered_json & properties,
                                     const ordered_json & required_params,
                                     int                  depth);

// Mirrors Jinja's `format_argument` from the GigaChat-style helper... wait,
// Functionary doesn't use that. The macros only handle scalars/arrays/objects
// via `get_parameter_typescript`. No standalone `format_argument` here.

// Mirrors Jinja `get_array_typescript`. `param_name` of empty string represents
// the macro's `None` (used for the recursive call from inside an array).
std::string get_array_typescript(const std::string & param_name,
                                 const ordered_json & param_dic,
                                 int                  depth) {
    const std::string offset    = depth >= 1 ? std::string(4 * depth, ' ') : std::string{};
    const ordered_json items_info = param_dic.value("items", ordered_json::object());

    if (items_info.empty()) {
        if (!param_name.empty()) {
            return "\n" + offset + param_name + ": []";
        }
        return "\n" + offset + "[]";
    }

    const std::string array_type = get_param_type(items_info);
    if (array_type == "object") {
        std::string out;
        if (!param_name.empty()) {
            out += "\n" + offset + param_name + ": {";
        } else {
            out += "\n" + offset + "{";
        }
        const ordered_json props    = items_info.value("properties", ordered_json::object());
        const ordered_json required = items_info.value("required", ordered_json::array());
        out += get_parameter_typescript(props, required, depth + 1);
        out += "\n" + offset + "}[]";
        return out;
    }
    if (array_type == "array") {
        std::string item_info = get_array_typescript(std::string{}, items_info, depth + 1);
        if (param_name.empty()) {
            return "\n" + item_info + "[]";
        }
        // Trim leading whitespace+newline of item_info (mirroring `|trim` in Jinja).
        size_t first_non_ws = item_info.find_first_not_of(" \t\n\r");
        std::string trimmed = first_non_ws == std::string::npos ? std::string{} : item_info.substr(first_non_ws);
        return "\n" + offset + param_name + ": " + trimmed + "[]";
    }
    if (items_info.contains("enum") && items_info["enum"].is_array()) {
        const std::string item_type = get_enum_option_str(items_info["enum"]);
        if (param_name.empty()) {
            return "(" + item_type + ")[]";
        }
        return "\n" + offset + param_name + ": (" + item_type + ")[]";
    }
    if (param_name.empty()) {
        return "\n" + array_type + "[]";
    }
    return "\n" + offset + param_name + ": " + array_type + "[],";
}

std::string get_parameter_typescript(const ordered_json & properties,
                                     const ordered_json & required_params,
                                     int                  depth) {
    std::string out;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        const std::string & param_name = it.key();
        const ordered_json & param      = it.value();
        if (!param.is_object()) {
            continue;
        }

        const std::string  comment_info  = get_param_info(param);
        ordered_json examples_info       = ordered_json::array();
        if (param.contains("examples") && param["examples"].is_array()) {
            examples_info.push_back("Example " + param_name + ":");
            for (const auto & ex : param["examples"]) {
                examples_info.push_back(ex);
            }
        }

        bool is_required = false;
        if (required_params.is_array()) {
            for (const auto & rp : required_params) {
                if (rp.is_string() && rp.get<std::string>() == param_name) {
                    is_required = true;
                    break;
                }
            }
        }
        std::string param_declaration = param_name;
        if (!is_required) {
            param_declaration += "?";
        }

        std::string param_type = get_param_type(param);

        const std::string offset = depth >= 1 ? std::string(4 * depth, ' ') : std::string{};

        if (param_type == "object") {
            if (comment_info != "<|NONE|>") {
                out += "\n" + offset + comment_info;
            }
            if (!examples_info.empty()) {
                for (const auto & example : examples_info) {
                    std::string ex = example.is_string() ? example.get<std::string>() : example.dump();
                    std::string replaced;
                    for (char c : ex) {
                        replaced += (c == '\'') ? '"' : c;
                    }
                    out += "\n" + offset + "// " + replaced;
                }
            }
            param_declaration += ": {";
            out += "\n" + offset + param_declaration;
            ordered_json sub_props    = param.value("properties", ordered_json::object());
            ordered_json sub_required = param.value("required", ordered_json::array());
            out += get_parameter_typescript(sub_props, sub_required, depth + 1);
            out += "\n" + offset + "},";
        } else if (param_type == "array") {
            const ordered_json item_info = param.value("items", ordered_json::object());
            if (!item_info.contains("type")) {
                param_declaration += ": [],";
                out += append_new_param_info(param_declaration, comment_info, examples_info, depth);
            } else {
                if (comment_info != "<|NONE|>") {
                    out += "\n" + offset + comment_info;
                }
                if (!examples_info.empty()) {
                    for (const auto & example : examples_info) {
                        std::string ex = example.is_string() ? example.get<std::string>() : example.dump();
                        std::string replaced;
                        for (char c : ex) {
                            replaced += (c == '\'') ? '"' : c;
                        }
                        out += "\n" + offset + "// " + replaced;
                    }
                }
                std::string array_decl = get_array_typescript(param_declaration, param, depth);
                if (array_decl.empty() || array_decl.back() != ',') {
                    array_decl += ",";
                }
                out += array_decl;
            }
        } else {
            if (param.contains("enum")) {
                param_type = get_enum_option_str(param["enum"]);
            }
            if (param.value("nullable", false)) {
                param_type += " | null";
            }
            param_declaration += ": " + param_type + ",";
            out += append_new_param_info(param_declaration, comment_info, examples_info, depth);
        }
    }
    return out;
}

std::string generate_schema_from_functions(const ordered_json & functions,
                                            const std::string & namespace_name = "functions") {
    std::string out;
    out += "// Supported function definitions that should be called when necessary.\n";
    out += "namespace " + namespace_name + " {\n\n";

    for (const auto & function_in : functions) {
        ordered_json function = function_in;
        if (function.contains("function")) {
            function = function["function"];
        }
        const std::string function_name = function.value("name", std::string{});
        if (function_name.empty()) {
            continue;
        }
        const std::string description = function.value("description", std::string{});
        const ordered_json parameters  = function.value("parameters", ordered_json::object());

        out += "// " + description + "\n";
        out += "type " + function_name;
        if (parameters.is_object() && parameters.contains("properties")) {
            out += " = (_: {";
            ordered_json required_params = parameters.value("required", ordered_json::array());
            out += get_parameter_typescript(parameters["properties"], required_params, 0);
            out += "\n}) => ";
        } else {
            out += " = () => ";
        }

        // Functionary supports `response_parameters` as a return type.
        const ordered_json return_params = function.value("return_parameters", ordered_json{});
        if (return_params.is_object() && return_params.contains("properties")) {
            out += get_parameter_typescript(return_params["properties"],
                                             return_params.value("required", ordered_json::array()),
                                             0);
        } else {
            out += "any";
        }
        out += ";\n\n";
    }

    out += "} // namespace " + namespace_name;
    return out;
}

// Render a per-message body for user/system/tool roles. These formats embed
// content directly (the Jinja template uses `message['content']` as a string).
std::string render_simple_body(const ordered_json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_null()) {
        return std::string{};
    }
    return content.dump();
}

}  // namespace

std::string common_chat_functionary_v3_2_render(const autoparser::generation_params & inputs,
                                                const std::string & bos_token) {
    std::ostringstream out;

    // 1. BOS + initial system header with tools schema.
    ordered_json tools = inputs.tools;
    if (!tools.is_array()) {
        tools = ordered_json::array();
    }

    out << bos_token
        << "<|start_header_id|>system<|end_header_id|>\n\n"
        << "You are capable of executing available function(s) if required.\n"
        << "Only execute function(s) when absolutely necessary.\n"
        << "Ask for the required input to:recipient==all\n"
        << "Use JSON for function arguments.\n"
        << "Respond in this format:\n"
        << ">>>${recipient}\n"
        << "${content}\n"
        << "Available functions:\n"
        << generate_schema_from_functions(tools)
        << "<|eot_id|>";

    // 2. Optional code_interpreter system block.
    bool has_code_interpreter = false;
    for (const auto & tool : tools) {
        if (tool.is_object() && tool.value("type", std::string{}) == "code_interpreter") {
            has_code_interpreter = true;
            break;
        }
    }
    if (has_code_interpreter) {
        out << "<|start_header_id|>system<|end_header_id|>\n\n"
            << "When you send a message containing Python code to python, "
            << "it will be executed in a stateful Jupyter notebook environment. "
            << "python will respond with the output of the execution or time out after 60.0 seconds. "
            << "The drive at '/mnt/data' can be used to save and persist user files.<|eot_id|>";
    }

    // 3. Per-message rendering.
    for (const auto & msg : inputs.messages) {
        const std::string role = msg.value("role", "");
        if (role == "user" || role == "system" || role == "tool") {
            out << "<|start_header_id|>" << role << "<|end_header_id|>\n\n"
                << render_simple_body(msg.value("content", ordered_json{}))
                << "<|eot_id|>";
        } else {
            // assistant
            out << "<|start_header_id|>" << role << "<|end_header_id|>\n\n";
            const ordered_json content = msg.value("content", ordered_json{});
            const bool has_content = content.is_string()
                ? !content.get<std::string>().empty()
                : (!content.is_null() && !content.empty());
            if (has_content) {
                out << ">>>all\n" << render_simple_body(content);
            }
            if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
                for (const auto & tc : msg["tool_calls"]) {
                    const std::string fn_name = tc["function"]["name"].get<std::string>();
                    const ordered_json args   = tc["function"]["arguments"];
                    out << ">>>" << fn_name << "\n";
                    if (args.is_string()) {
                        out << args.get<std::string>();
                    } else {
                        out << args.dump();
                    }
                }
            }
            out << "<|eot_id|>";
        }
    }

    // 4. Generation prompt.
    if (inputs.add_generation_prompt) {
        out << "<|start_header_id|>assistant<|end_header_id|>\n\n>>>";
    }

    return out.str();
}
