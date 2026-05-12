#include "chat-formats/minimax-m2-format.h"

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
        if (c == '_') {
            c = '-';
        }
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

// Strip a leading and a trailing `\n` from `s` (the grammar's reasoning
// span includes the `\n` after `<think>` and the `\n` before `</think>`).
std::string strip_outer_newlines(std::string s) {
    if (!s.empty() && s.front() == '\n') {
        s.erase(0, 1);
    }
    if (!s.empty() && s.back() == '\n') {
        s.pop_back();
    }
    return s;
}

}  // namespace

void common_chat_minimax_m2_tracker::advance(const common_peg_ast_node & node) {
    if (tag_is(node, "tool")) {
        state_ = node.is_partial ? common_chat_format_state::IN_TOOL_CALL
                                  : common_chat_format_state::DONE;
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

std::vector<std::string> common_chat_minimax_m2_tracker::expected_productions() const {
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

std::vector<common_chat_decoded_event> common_chat_minimax_m2_decoder::decode(
    const common_peg_ast_node & node) {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;

    if (tag_is(node, common_chat_peg_builder::REASONING)) {
        // Reasoning span includes the `\n` immediately after `<think>` and
        // the `\n` before `</think>` (per grammar). Strip outer newlines
        // so the surfaced reasoning_content matches the Jinja expectation.
        std::string text = strip_outer_newlines(std::string(node.text));
        events.push_back({K::REASONING_TEXT, std::move(text), {}, {}, false, node.is_partial});
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

std::vector<common_chat_decoded_event> common_chat_minimax_m2_decoder::on_finalize() {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;
    if (in_tool_ && last_tool_complete_) {
        events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        in_tool_ = false;
        last_tool_complete_ = false;
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_minimax_m2_transformer::shape(
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
                if (!first_arg_) {
                    args_json_buffer_ += ',';
                }
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
                } catch (const std::exception &) {
                    // fall through to JSON string
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

std::vector<common_chat_shaped_event> common_chat_minimax_m2_transformer::on_finalize() {
    std::vector<common_chat_shaped_event> out;
    if (in_tool_ && !args_json_buffer_.empty()) {
        out.push_back({common_chat_shaped_event::kind::TOOL_ARGS_JSON, args_json_buffer_, true});
    }
    return out;
}

const common_chat_format_state_rules minimax_m2_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,    "content" },
        { common_chat_format_state::IN_REASONING,  "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL,  "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME,  "func-name" },
        { common_chat_format_state::IN_TOOL_PARAM, "parameter" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Minimal MiniMax-M2 prompt writer.
// ──────────────────────────────────────────────────────────────────────────────

std::string common_chat_minimax_m2_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    out << "]~!b[" << "]~b]system\n";
    bool has_explicit_system = false;
    if (inputs.messages.is_array() && !inputs.messages.empty() &&
        inputs.messages[0].is_object() &&
        inputs.messages[0].value("role", std::string{}) == "system" &&
        inputs.messages[0].contains("content") &&
        inputs.messages[0]["content"].is_string()) {
        out << inputs.messages[0]["content"].get<std::string>();
        has_explicit_system = true;
    } else {
        out << "You are a helpful assistant.";
    }
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        out << "\n\n# Tools\nYou may call one or more tools to assist with the user query.\n"
               "Here are the tools available in JSONSchema format:\n\n<tools>\n";
        for (const auto & tool : inputs.tools) {
            out << "<tool>" << to_jinja_json(tool) << "</tool>\n";
        }
        out << "</tools>\n\n"
               "When making tool calls, use XML format to invoke tools and pass parameters:\n"
               "\n<minimax:tool_call>\n<invoke name=\"tool-name-1\">\n"
               "<parameter name=\"param-key-1\">param-value-1</parameter>\n"
               "...\n</invoke>\n</minimax:tool_call>";
    }
    out << "[e~[\n";

    if (inputs.messages.is_array()) {
        for (size_t i = 0; i < inputs.messages.size(); ++i) {
            if (i == 0 && has_explicit_system) {
                continue;
            }
            const auto & m = inputs.messages[i];
            const std::string role = m.value("role", std::string{});
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (role == "user") {
                out << "]~b]user\n" << content << "[e~[\n";
            } else if (role == "assistant") {
                out << "]~b]ai\n" << content << "[e~[\n";
            } else if (role == "tool") {
                const std::string prev_role = i > 0
                    ? inputs.messages[i - 1].value("role", std::string{}) : std::string{};
                if (i == 0 || prev_role != "tool") {
                    out << "]~b]tool";
                }
                out << "\n<response>" << content << "</response>";
                const bool is_last = (i + 1 == inputs.messages.size());
                const std::string next_role = is_last ? std::string{}
                    : inputs.messages[i + 1].value("role", std::string{});
                if (is_last || next_role != "tool") {
                    out << "[e~[\n";
                }
            }
        }
    }

    if (inputs.add_generation_prompt) {
        out << "]~b]ai\n<think>\n";
    }
    return out.str();
}
