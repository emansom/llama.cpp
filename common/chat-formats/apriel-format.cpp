#include "chat-formats/apriel-format.h"

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

}  // namespace

void common_chat_apriel_tracker::advance(const common_peg_ast_node & node) {
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

std::vector<std::string> common_chat_apriel_tracker::expected_productions() const {
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

std::vector<common_chat_decoded_event> common_chat_apriel_decoder::decode(
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

std::vector<common_chat_decoded_event> common_chat_apriel_decoder::on_finalize() {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;
    if (in_tool_ && last_tool_complete_) {
        events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        in_tool_ = false;
        last_tool_complete_ = false;
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_apriel_transformer::shape(
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

const common_chat_format_state_rules apriel_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Writers
// ──────────────────────────────────────────────────────────────────────────────

namespace {

void render_tool_call(std::ostringstream & out, const ordered_json & tc_in) {
    ordered_json tc = tc_in;
    if (tc.contains("function") && tc["function"].is_object()) {
        tc = tc["function"];
    }
    out << "{\"name\": \"" << tc.value("name", std::string{}) << "\", \"arguments\": ";
    const auto & args = tc.contains("arguments") ? tc["arguments"] : ordered_json::object();
    if (args.is_string()) {
        out << args.get<std::string>();
    } else {
        out << to_jinja_json(args);
    }
    out << "}";
}

}  // namespace

std::string common_chat_apriel_render(const autoparser::generation_params & inputs,
                                      const std::string & bos_token) {
    std::ostringstream out;
    out << bos_token;

    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            const std::string role = m.value("role", std::string{});
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (role == "system") {
                out << "<|begin_system|>\n" << content << "\n";
            } else if (role == "user") {
                out << "<|begin_user|>\n" << content;
            } else if (role == "assistant") {
                out << "\n<|begin_assistant|>\n" << content;
                if (m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty()) {
                    out << "\n<tool_calls>[";
                    bool first = true;
                    for (const auto & tc : m["tool_calls"]) {
                        if (!first) out << ", ";
                        first = false;
                        render_tool_call(out, tc);
                    }
                    out << "]</tool_calls>";
                }
                out << "\n<|end|>\n";
            } else if (role == "tool") {
                out << "<|begin_tool_result|>\n" << content << "\n";
            }
        }
    }
    if (inputs.add_generation_prompt) {
        out << "\n<|begin_assistant|>\n";
    }
    return out.str();
}

std::string common_chat_apriel_thinker_render(const autoparser::generation_params & inputs,
                                              const std::string & bos_token) {
    std::ostringstream out;
    out << bos_token;

    // The Apriel-1.6 template emits a system block with `reasoning_prompt`
    // before the first non-system message. For test fixtures (typically
    // user+assistant) this means the system block always comes first.
    constexpr const char * REASONING_PROMPT =
        "You are a thoughtful, systematic AI assistant from ServiceNow Language Models (SLAM) lab.\n"
        "    Analyze each question carefully, present your reasoning step-by-step, then provide the final\n"
        "    response after the marker [BEGIN FINAL RESPONSE].";

    const bool has_messages = inputs.messages.is_array() && !inputs.messages.empty();
    if (has_messages && inputs.messages[0].value("role", std::string{}) != "system") {
        out << "<|begin_system|>\n" << REASONING_PROMPT << "\n";
    }

    if (inputs.messages.is_array()) {
        for (const auto & m : inputs.messages) {
            const std::string role = m.value("role", std::string{});
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (role == "system") {
                out << "<|begin_system|>\n" << REASONING_PROMPT << "\n" << content << "\n";
            } else if (role == "user") {
                out << "<|begin_user|>\n" << content;
            } else if (role == "assistant") {
                out << "\n<|begin_assistant|>\n" << content;
                if (m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty()) {
                    out << "\n<tool_calls>[";
                    bool first = true;
                    for (const auto & tc : m["tool_calls"]) {
                        if (!first) out << ", ";
                        first = false;
                        render_tool_call(out, tc);
                    }
                    out << "]</tool_calls>";
                }
            } else if (role == "tool") {
                out << "<|begin_tool_result|>\n" << content << "\n";
            }
        }
    }
    if (inputs.add_generation_prompt) {
        out << "\n<|begin_assistant|>\nHere are my reasoning steps:\n";
    }
    return out.str();
}
