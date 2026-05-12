#include "chat-formats/deepseek-v3-1-format.h"

#include "chat-peg-parser.h"
#include "peg-parser.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

bool tag_is(const common_peg_ast_node & node, const char * tag) {
    return node.tag == tag;
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

}  // namespace

void common_chat_deepseek_v3_1_tracker::advance(const common_peg_ast_node & node) {
    if (tag_is(node, "tool")) {
        state_ = node.is_partial ? common_chat_format_state::IN_TOOL_CALL
                                  : common_chat_format_state::DONE;
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_NAME) && !node.is_partial) {
        state_ = common_chat_format_state::IN_TOOL_NAME;
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARGS) && !node.is_partial) {
        state_ = common_chat_format_state::IN_TOOL_ARGS;
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

std::vector<std::string> common_chat_deepseek_v3_1_tracker::expected_productions() const {
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
            return {"tool-args"};
        default:
            return {};
    }
}

std::vector<common_chat_decoded_event> common_chat_deepseek_v3_1_decoder::decode(
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
    if (tag_is(node, common_chat_peg_builder::TOOL_ARGS)) {
        events.push_back({K::TOOL_ARGS_RAW, std::string(node.text), {}, {}, false, node.is_partial});
        return events;
    }
    return events;
}

std::vector<common_chat_decoded_event> common_chat_deepseek_v3_1_decoder::on_finalize() {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;
    if (in_tool_ && last_tool_complete_) {
        events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        in_tool_ = false;
        last_tool_complete_ = false;
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_deepseek_v3_1_transformer::shape(
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
            break;
        case DK::TOOL_NAME:
            out.push_back({SK::TOOL_NAME, event.text, false});
            break;
        case DK::TOOL_ARGS_RAW:
            out.push_back({SK::TOOL_ARGS_JSON, event.text, event.is_partial});
            break;
        case DK::TOOL_CLOSE:
            out.push_back({SK::TOOL_CLOSE, {}, event.is_partial});
            break;
        default:
            break;
    }
    return out;
}

const common_chat_format_state_rules deepseek_v3_1_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Writer (minimal --- no reconstruction tests for V3.1)
// ──────────────────────────────────────────────────────────────────────────────

namespace {

constexpr const char * USER_OPEN = "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>";
constexpr const char * ASST_OPEN = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
constexpr const char * EOS_MARK  = "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>";
constexpr const char * TC_BEGIN  = "<\xef\xbd\x9c" "tool" "\xe2\x96\x81" "calls" "\xe2\x96\x81" "begin" "\xef\xbd\x9c>";
constexpr const char * TC_END    = "<\xef\xbd\x9c" "tool" "\xe2\x96\x81" "calls" "\xe2\x96\x81" "end" "\xef\xbd\x9c>";
constexpr const char * T_BEGIN   = "<\xef\xbd\x9c" "tool" "\xe2\x96\x81" "call" "\xe2\x96\x81" "begin" "\xef\xbd\x9c>";
constexpr const char * T_END     = "<\xef\xbd\x9c" "tool" "\xe2\x96\x81" "call" "\xe2\x96\x81" "end" "\xef\xbd\x9c>";
constexpr const char * T_SEP     = "<\xef\xbd\x9c" "tool" "\xe2\x96\x81" "sep" "\xef\xbd\x9c>";
constexpr const char * TO_BEGIN  = "<\xef\xbd\x9c" "tool" "\xe2\x96\x81" "output" "\xe2\x96\x81" "begin" "\xef\xbd\x9c>";
constexpr const char * TO_END    = "<\xef\xbd\x9c" "tool" "\xe2\x96\x81" "output" "\xe2\x96\x81" "end" "\xef\xbd\x9c>";

}  // namespace

std::string common_chat_deepseek_v3_1_render(const autoparser::generation_params & inputs,
                                             const std::string & bos_token) {
    std::ostringstream out;
    out << bos_token;

    // Aggregate system messages first (DeepSeek-V3.1 emits them as a prefix
    // before per-turn role markers).
    std::string system_prompt;
    bool first_sp = true;
    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            if (m.value("role", std::string{}) == "system" &&
                m.contains("content") && m["content"].is_string()) {
                if (first_sp) {
                    system_prompt += m["content"].get<std::string>();
                    first_sp = false;
                } else {
                    system_prompt += "\n\n" + m["content"].get<std::string>();
                }
            }
        }
    }
    out << system_prompt;

    bool is_last_user = false;
    bool is_tool      = false;

    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            const std::string role = m.value("role", std::string{});
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (role == "user") {
                is_tool = false;
                is_last_user = true;
                out << USER_OPEN << content;
            } else if (role == "assistant") {
                const bool has_tool_calls = m.contains("tool_calls") &&
                    m["tool_calls"].is_array() && !m["tool_calls"].empty();
                if (has_tool_calls) {
                    if (is_last_user) {
                        out << ASST_OPEN << "<think></think>";
                    }
                    is_last_user = false;
                    is_tool      = false;
                    if (!content.empty()) {
                        out << content;
                    }
                    out << TC_BEGIN;
                    for (const auto & tc_in : m["tool_calls"]) {
                        ordered_json tc = tc_in;
                        if (tc.contains("function") && tc["function"].is_object()) {
                            tc = tc["function"];
                        }
                        out << T_BEGIN << tc.value("name", std::string{}) << T_SEP;
                        const auto & args = tc.contains("arguments") ? tc["arguments"]
                            : ordered_json::object();
                        if (args.is_string()) {
                            out << args.get<std::string>();
                        } else {
                            out << to_jinja_json(args);
                        }
                        out << T_END;
                    }
                    out << TC_END << EOS_MARK;
                } else {
                    if (is_last_user) {
                        out << ASST_OPEN << "<think></think>";
                    }
                    is_last_user = false;
                    // Strip `<think>...</think>` from past content.
                    const std::string close = "</think>";
                    size_t pos = content.find(close);
                    if (pos != std::string::npos) {
                        content = content.substr(pos + close.size());
                    }
                    out << content << EOS_MARK;
                    is_tool = false;
                }
            } else if (role == "tool") {
                is_last_user = false;
                is_tool      = true;
                out << TO_BEGIN << content << TO_END;
            }
        }
    }

    if (inputs.add_generation_prompt && is_last_user && !is_tool) {
        out << ASST_OPEN;
        if (inputs.enable_thinking) {
            out << "<think>";
        } else {
            out << "<think></think>";
        }
    }
    return out.str();
}
