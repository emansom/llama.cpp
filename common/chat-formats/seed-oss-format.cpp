#include "chat-formats/seed-oss-format.h"

#include "chat-peg-parser.h"
#include "peg-parser.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

std::string normalize_rule(const std::string & rule) {
    std::string out = rule;
    for (char & c : out) {
        if (c == '_') c = '-';
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

bool tag_is(const common_peg_ast_node & node, const char * tag) {
    return node.tag == tag;
}

std::string escape_json_inner(const std::string & s) {
    auto dumped = ordered_json(s).dump();
    if (dumped.size() >= 2 && dumped.front() == '"' && dumped.back() == '"') {
        return dumped.substr(1, dumped.size() - 2);
    }
    return dumped;
}

std::string trim_str(const std::string & s) {
    size_t s_pos = s.find_first_not_of(" \t\n\r");
    if (s_pos == std::string::npos) return {};
    size_t e_pos = s.find_last_not_of(" \t\n\r");
    return s.substr(s_pos, e_pos - s_pos + 1);
}

std::string py_type(const std::string & t) {
    if (t == "string")  return "str";
    if (t == "number" || t == "integer") return "int";
    if (t == "boolean") return "bool";
    if (t == "array")   return "list";
    return "Any";
}

}  // namespace

void common_chat_seed_oss_tracker::advance(const common_peg_ast_node & node) {
    if (tag_is(node, "tool")) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::DONE;
        } else {
            state_ = common_chat_format_state::IN_TOOL_CALL;
        }
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_NAME) && !node.is_partial) {
        state_ = common_chat_format_state::IN_TOOL_NAME;
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARG_NAME) && !node.is_partial) {
        state_ = common_chat_format_state::IN_TOOL_ARG_KEY;
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARG_VALUE) && !node.is_partial) {
        state_ = common_chat_format_state::IN_TOOL_ARG_VAL;
        return;
    }
    if (tag_is(node, common_chat_peg_builder::REASONING)) {
        state_ = common_chat_format_state::IN_REASONING;
        return;
    }
    if (tag_is(node, common_chat_peg_builder::CONTENT)) {
        state_ = common_chat_format_state::IN_CONTENT;
        return;
    }
}

std::vector<std::string> common_chat_seed_oss_tracker::expected_productions() const {
    switch (state_) {
        case common_chat_format_state::INITIAL:
        case common_chat_format_state::IN_REASONING:
            return {"reasoning", "content", "tool-open"};
        case common_chat_format_state::DONE:
        case common_chat_format_state::IN_CONTENT:
            return {"content", "tool-open"};
        case common_chat_format_state::IN_TOOL_CALL:
            return {"tool-name"};
        case common_chat_format_state::IN_TOOL_NAME:
            return {"parameter", "tool-close"};
        case common_chat_format_state::IN_TOOL_PARAM:
            return {"parameter-name", "parameter-value"};
        default:
            return {};
    }
}

std::vector<common_chat_decoded_event> common_chat_seed_oss_decoder::decode(
    const common_peg_ast_node & node) {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;

    if (tag_is(node, common_chat_peg_builder::REASONING)) {
        events.push_back({K::REASONING_TEXT, std::string(node.text), {}, {}, false, node.is_partial});
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::CONTENT)) {
        events.push_back({K::CONTENT_TEXT, std::string(node.text), {}, {}, false, node.is_partial});
        return events;
    }
    if (tag_is(node, "tool")) {
        if (in_tool_ && last_tool_complete_) {
            events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        }
        events.push_back({K::TOOL_OPEN, {}, {}, {}, false, node.is_partial});
        in_tool_ = true;
        last_tool_complete_ = !node.is_partial;
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_NAME)) {
        if (!node.is_partial) {
            events.push_back({K::TOOL_NAME, std::string(node.text), {}, {}, false, false});
        }
        return events;
    }
    const std::string r = normalize_rule(node.rule);
    if (r == "parameter") {
        if (!node.is_partial && arena_) {
            constexpr int kArgDepth = 4;
            auto name_id  = arena_->find_by_tag(node, common_chat_peg_builder::TOOL_ARG_NAME, kArgDepth);
            auto value_id = arena_->find_by_tag(node, common_chat_peg_builder::TOOL_ARG_VALUE, kArgDepth);
            std::string key;
            if (name_id != COMMON_PEG_INVALID_AST_ID) {
                const auto & name_node = arena_->get(name_id);
                key = std::string(name_node.text);
                handled_ids_.insert(name_id);
            }
            std::string value;
            if (value_id != COMMON_PEG_INVALID_AST_ID) {
                const auto & value_node = arena_->get(value_id);
                value = std::string(value_node.text);
                handled_ids_.insert(value_id);
            }
            // Seed-OSS values are written inline (`<parameter=K>V</parameter>`)
            // but multi-line values get a single leading newline after `>` and
            // a single trailing newline before `</parameter>` as formatting.
            // Strip exactly one leading and one trailing newline if present;
            // preserve interior newlines and any additional trailing newlines
            // that belong to the value (e.g. multi-line code blocks that end
            // with their own `\n`).
            if (!value.empty() && value.front() == '\n') {
                value.erase(0, 1);
            }
            if (!value.empty() && value.back() == '\n') {
                value.pop_back();
            }
            if (!key.empty()) {
                events.push_back({K::TOOL_ARG_KV, {}, std::move(key), std::move(value), true, false});
            }
        }
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARG_NAME) ||
        tag_is(node, common_chat_peg_builder::TOOL_ARG_VALUE)) {
        return events;
    }

    return events;
}

std::vector<common_chat_decoded_event> common_chat_seed_oss_decoder::on_finalize() {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;
    if (in_tool_ && last_tool_complete_) {
        events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        in_tool_ = false;
        last_tool_complete_ = false;
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_seed_oss_transformer::shape(
    const common_chat_decoded_event & event) {
    using DK = common_chat_decoded_event::kind;
    using SK = common_chat_shaped_event::kind;
    std::vector<common_chat_shaped_event> out;

    switch (event.k) {
        case DK::REASONING_TEXT:
            out.push_back({SK::REASONING_TEXT, event.text, event.is_partial});
            break;
        case DK::CONTENT_TEXT:
            out.push_back({SK::CONTENT_TEXT, event.text, event.is_partial});
            break;
        case DK::TOOL_OPEN:
            out.push_back({SK::TOOL_OPEN, {}, event.is_partial});
            in_tool_         = true;
            first_arg_       = true;
            args_json_buffer_ = "{";
            break;
        case DK::TOOL_NAME:
            out.push_back({SK::TOOL_NAME, event.text, false});
            break;
        case DK::TOOL_ID:
            out.push_back({SK::TOOL_ID, event.text, false});
            break;
        case DK::TOOL_ARGS_RAW:
            out.push_back({SK::TOOL_ARGS_JSON, event.text, false});
            break;
        case DK::TOOL_ARG_KV:
            if (in_tool_) {
                if (!first_arg_) args_json_buffer_ += ',';
                first_arg_ = false;
                args_json_buffer_ += '"';
                args_json_buffer_ += escape_json_inner(event.key);
                args_json_buffer_ += "\":";
                bool embedded_raw = false;
                try {
                    auto parsed = ordered_json::parse(event.value);
                    if (!parsed.is_string()) {
                        args_json_buffer_ += parsed.dump();
                        embedded_raw = true;
                    }
                } catch (...) {
                    // fall through to string
                }
                if (!embedded_raw) {
                    args_json_buffer_ += '"';
                    args_json_buffer_ += escape_json_inner(event.value);
                    args_json_buffer_ += '"';
                }
            }
            break;
        case DK::TOOL_CLOSE:
            if (in_tool_) {
                args_json_buffer_ += '}';
                out.push_back({SK::TOOL_ARGS_JSON, args_json_buffer_, false});
                args_json_buffer_.clear();
                in_tool_   = false;
                first_arg_ = true;
            }
            out.push_back({SK::TOOL_CLOSE, {}, event.is_partial});
            break;
    }
    return out;
}

std::vector<common_chat_shaped_event> common_chat_seed_oss_transformer::on_finalize() {
    std::vector<common_chat_shaped_event> out;
    if (in_tool_ && !args_json_buffer_.empty()) {
        out.push_back({common_chat_shaped_event::kind::TOOL_ARGS_JSON, args_json_buffer_, true});
    }
    return out;
}

const common_chat_format_state_rules seed_oss_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,    "content" },
        { common_chat_format_state::IN_REASONING,  "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL,  "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME,  "func-name" },
        { common_chat_format_state::IN_TOOL_PARAM, "parameter" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Seed-OSS prompt writer
// ──────────────────────────────────────────────────────────────────────────────

namespace {

constexpr const char * BOS = "<seed:bos>";
constexpr const char * EOS = "<seed:eos>";
constexpr const char * THINK_OPEN  = "<seed:think>";
constexpr const char * THINK_CLOSE = "</seed:think>";
constexpr const char * TOOL_OPEN   = "<seed:tool_call>";
constexpr const char * TOOL_CLOSE  = "</seed:tool_call>";

std::string py_docstring_for_tool(const ordered_json & item) {
    if (!item.is_object() || item.value("type", std::string{}) != "function") return {};
    const auto & function = item["function"];
    const auto & parameters = function.value("parameters", ordered_json::object());
    const auto & properties = parameters.value("properties", ordered_json::object());
    const auto & required = parameters.value("required", ordered_json::array());

    std::ostringstream out;
    out << "\n\n\nFunction:\ndef " << function.value("name", std::string{}) << "(";
    bool first = true;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (!first) out << ",";
        first = false;
        out << "\n        " << it.key() << ": "
            << py_type(it.value().value("type", std::string{}));
    }
    out << "):\n    \"\"\"\n    " << trim_str(function.value("description", std::string{}));

    if (!properties.empty()) {
        out << "\n\n    Args:";
        for (auto it = properties.begin(); it != properties.end(); ++it) {
            out << "\n\n    - " << it.key() << " ("
                << py_type(it.value().value("type", std::string{})) << ")";
            bool is_required = false;
            for (const auto & req : required) {
                if (req.is_string() && req.get<std::string>() == it.key()) {
                    is_required = true;
                    break;
                }
            }
            out << (is_required ? " [必填]" : " [选填]") << ": ";
            out << it.value().value("description", std::string{});
        }
    }

    if (function.contains("returns") && function["returns"].is_object() &&
        function["returns"].contains("properties") &&
        function["returns"]["properties"].is_object() &&
        !function["returns"]["properties"].empty()) {
        out << "\n\n    Returns:";
        for (auto it = function["returns"]["properties"].begin();
             it != function["returns"]["properties"].end(); ++it) {
            out << "\n\n    - " << it.key() << " ("
                << py_type(it.value().value("type", std::string{})) << "): "
                << it.value().value("description", std::string{});
        }
    }

    out << "\n\n    \"\"\"";
    return out.str();
}

}  // namespace

std::string common_chat_seed_oss_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    const bool has_messages = inputs.messages.is_array() && !inputs.messages.empty();
    const bool first_is_system = has_messages &&
        inputs.messages[0].is_object() &&
        inputs.messages[0].value("role", std::string{}) == "system";
    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    size_t loop_start = 0;
    bool system_message_set = false;
    std::string system_message;
    if (first_is_system) {
        const auto & c = inputs.messages[0].value("content", ordered_json{});
        if (c.is_string()) system_message = c.get<std::string>();
        system_message_set = true;
        loop_start = 1;
    }

    // System block.
    if (system_message_set) {
        out << BOS << "system\n" << system_message;
    } else if (has_tools) {
        out << BOS << "system\nYou are Doubao, a helpful AI assistant. "
                     "You may call one or more functions to assist with the user query.";
    }
    if (has_tools) {
        for (const auto & item : inputs.tools) {
            out << py_docstring_for_tool(item);
        }
        out << "\n\n工具调用请遵循如下格式:\n"
               "<seed:tool_call>\n<function=example_function_name>\n"
               "<parameter=example_parameter_1>value_1</parameter>\n"
               "<parameter=example_parameter_2>This is the value for the second parameter\n"
               "that can span\nmultiple lines</parameter>\n"
               "</function>\n</seed:tool_call>\n";
    }
    if (system_message_set || has_tools) {
        out << EOS;
    }

    // Per-message loop.
    for (size_t i = loop_start; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});
        std::string content;
        if (message.contains("content") && message["content"].is_string()) {
            content = message["content"].get<std::string>();
        }
        const bool has_tool_calls = role == "assistant" && message.contains("tool_calls") &&
            message["tool_calls"].is_array() && !message["tool_calls"].empty();

        if (has_tool_calls) {
            out << BOS << role;
            std::string reasoning;
            if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
                reasoning = trim_str(message["reasoning_content"].get<std::string>());
            }
            if (!reasoning.empty()) {
                out << "\n" << THINK_OPEN << reasoning << THINK_CLOSE;
            }
            std::string trimmed_content = trim_str(content);
            if (!trimmed_content.empty()) {
                out << "\n" << trimmed_content << "\n";
            }
            for (const auto & tc_in : message["tool_calls"]) {
                ordered_json tc = tc_in;
                if (tc.contains("function") && tc["function"].is_object()) {
                    tc = tc["function"];
                }
                out << "\n" << TOOL_OPEN << "\n<function=" << tc.value("name", std::string{}) << ">\n";
                if (tc.contains("arguments")) {
                    ordered_json args = tc["arguments"];
                    if (args.is_string()) {
                        try {
                            args = ordered_json::parse(args.get<std::string>());
                        } catch (...) {
                            args = ordered_json::object();
                        }
                    }
                    if (args.is_object()) {
                        for (auto it = args.begin(); it != args.end(); ++it) {
                            out << "<parameter=" << it.key() << ">";
                            const auto & v = it.value();
                            if (v.is_string()) {
                                out << v.get<std::string>();
                            } else {
                                out << v.dump();
                            }
                            out << "</parameter>\n";
                        }
                    }
                }
                out << "</function>\n" << TOOL_CLOSE;
            }
            out << EOS;
        } else if (role == "user" || role == "system") {
            out << BOS << role << "\n" << content << EOS;
        } else if (role == "assistant") {
            out << BOS << "assistant";
            std::string reasoning;
            if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
                reasoning = trim_str(message["reasoning_content"].get<std::string>());
            }
            if (!reasoning.empty()) {
                out << "\n" << THINK_OPEN << reasoning << THINK_CLOSE;
            }
            std::string trimmed_content = trim_str(content);
            if (!trimmed_content.empty()) {
                out << "\n" << trimmed_content << EOS;
            }
        } else {
            out << BOS << role << "\n" << content << EOS;
        }
    }

    if (inputs.add_generation_prompt) {
        out << BOS << "assistant\n";
    }
    return out.str();
}
