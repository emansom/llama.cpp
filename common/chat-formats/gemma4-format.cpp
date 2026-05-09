#include "chat-formats/gemma4-format.h"

#include "chat-peg-parser.h"
#include "peg-parser.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <string>

namespace {

using ordered_json = nlohmann::ordered_json;

bool tag_is(const common_peg_ast_node & node, const char * tag) {
    return node.tag == tag;
}

std::string normalize_rule(const std::string & rule) {
    std::string out = rule;
    for (char & c : out) {
        if (c == '_') {
            c = '-';
        } else {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return out;
}

std::string escape_json_string_inner(const std::string & s) {
    auto dumped = ordered_json(s).dump();
    if (dumped.size() >= 2 && dumped.front() == '"' && dumped.back() == '"') {
        return dumped.substr(1, dumped.size() - 2);
    }
    return dumped;
}

}  // namespace

void common_chat_gemma4_tracker::advance(const common_peg_ast_node & node) {
    if (tag_is(node, "tool")) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::DONE;
        } else {
            state_ = common_chat_format_state::IN_TOOL_CALL;
        }
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_NAME)) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::IN_TOOL_NAME;
        }
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_CLOSE)) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::DONE;
        }
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

std::vector<std::string> common_chat_gemma4_tracker::expected_productions() const {
    switch (state_) {
        case common_chat_format_state::INITIAL:
        case common_chat_format_state::DONE:
        case common_chat_format_state::IN_CONTENT:
        case common_chat_format_state::IN_REASONING:
            return {"content", "reasoning", "tool-open"};
        case common_chat_format_state::IN_TOOL_CALL:
            return {"tool-name"};
        case common_chat_format_state::IN_TOOL_NAME:
            return {"tool-args", "tool-close"};
        case common_chat_format_state::IN_TOOL_ARGS:
            return {"tool-close"};
        default:
            return {};
    }
}

std::string common_chat_gemma4_decoder::gemma4_to_json(common_peg_ast_id id) {
    if (arena_ == nullptr) {
        return "";
    }
    const auto & node = arena_->get(id);
    const std::string rule = normalize_rule(node.rule);

    if (node.text.empty()) {
        return "";
    }

    if (rule == "gemma4-number" || rule == "gemma4-bool" || rule == "gemma4-null") {
        return std::string(node.text);
    }

    if (rule == "gemma4-string") {
        std::string result = "\"";
        for (auto child_id : node.children) {
            const auto & child = arena_->get(child_id);
            const std::string child_rule = normalize_rule(child.rule);
            if (child_rule == "gemma4-str-content") {
                result += escape_json_string_inner(std::string(child.text));
            }
        }
        // Append the JSON closing quote when the rule body has fully matched
        // (the closing '<|"|>' literal succeeded). For partial parses the
        // string is still in flight and we keep it open.
        if (!node.is_partial) {
            result += "\"";
        }
        return result;
    }

    if (rule == "gemma4-array") {
        std::string result = "[";
        bool add_comma = false;
        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(child_id);
        }
        if (!node.is_partial) {
            result += ']';
        }
        return result;
    }

    if (rule == "gemma4-dict-key") {
        std::string result = "\"";
        result += escape_json_string_inner(std::string(node.text));
        if (!node.is_partial) {
            result += "\":";
        }
        return result;
    }

    if (rule == "gemma4-kv") {
        // During streaming, suppress kv pairs that have only the key — the
        // value child appears later. This keeps args strings monotonic.
        bool has_value = false;
        for (auto child_id : node.children) {
            const auto & child = arena_->get(child_id);
            if (normalize_rule(child.rule) == "gemma4-value") {
                has_value = true;
                break;
            }
        }
        if (node.is_partial && !has_value) {
            return "";
        }
        std::string result;
        for (auto child_id : node.children) {
            result += gemma4_to_json(child_id);
        }
        return result;
    }

    if (rule == "gemma4-dict") {
        std::string result = "{";
        bool add_comma = false;
        for (auto child_id : node.children) {
            std::string child_text = gemma4_to_json(child_id);
            if (child_text.empty()) {
                continue;  // skip suppressed partial kv pairs
            }
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += child_text;
        }
        if (!node.is_partial) {
            result += '}';
        }
        return result;
    }

    if (rule == "gemma4-value") {
        if (!node.children.empty()) {
            return gemma4_to_json(node.children[0]);
        }
        return std::string(node.text);
    }

    return "";
}

std::vector<common_chat_decoded_event> common_chat_gemma4_decoder::decode(
    const common_peg_ast_node & node) {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;

    if (tag_is(node, common_chat_peg_builder::REASONING)) {
        events.push_back({K::REASONING_TEXT, std::string(node.text), {}, {}, false, node.is_partial});
        return events;
    }
    // 'analysis-content' rule wraps the body of a no-reasoning think block;
    // synthesize the wire-shape markers around the body so the surfaced
    // content matches what the model emitted byte-for-byte.
    if (node.rule == "analysis-content") {
        if (!node.is_partial) {
            events.push_back({K::CONTENT_TEXT, "<|channel>thought", {}, {}, false, false});
            events.push_back({K::CONTENT_TEXT, std::string(node.text), {}, {}, false, false});
            events.push_back({K::CONTENT_TEXT, "<channel|>", {}, {}, false, false});
        }
        if (arena_) {
            constexpr int kAnalysisDepth = 4;
            auto inner_id = arena_->find_by_tag(node, common_chat_peg_builder::CONTENT, kAnalysisDepth);
            if (inner_id != COMMON_PEG_INVALID_AST_ID) {
                handled_ids_.insert(inner_id);
            }
        }
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::CONTENT)) {
        if (handled_ids_.count(node.id)) {
            return events;
        }
        fprintf(stderr, "[DBG-GEMMA4 content] text=[%s] partial=%d\n", std::string(node.text).c_str(), node.is_partial);
        events.push_back({K::CONTENT_TEXT, std::string(node.text), {}, {}, false, node.is_partial});
        return events;
    }

    if (tag_is(node, "tool")) {
        // The lark-to-peg transpiler tags 'tool_call' rules with the "tool"
        // tag — its subtree contains the name and the gemma4_dict. Walk the
        // subtree here: emit TOOL_OPEN, TOOL_NAME, TOOL_ARGS_RAW (canonical
        // JSON of the dict subtree), TOOL_CLOSE — all in one decode call.
        // Subsequent visits to nested children produce no additional events
        // because we don't tag the dict / value / kv subtree.
        if (arena_ == nullptr) {
            return events;
        }
        constexpr int kMaxTagSearchDepth = 8;
        auto name_id = arena_->find_by_tag(node, common_chat_peg_builder::TOOL_NAME, kMaxTagSearchDepth);
        auto args_id = arena_->find_by_rule(node, "gemma4-dict", kMaxTagSearchDepth);
        events.push_back({K::TOOL_OPEN, {}, {}, {}, false, node.is_partial});
        if (name_id != COMMON_PEG_INVALID_AST_ID) {
            const auto & name_node = arena_->get(name_id);
            if (!name_node.is_partial) {
                events.push_back({K::TOOL_NAME, std::string(name_node.text), {}, {}, false, false});
            }
        }
        std::string args_json = (args_id != COMMON_PEG_INVALID_AST_ID)
                                    ? gemma4_to_json(args_id)
                                    : std::string("{}");
        events.push_back({K::TOOL_ARGS_RAW, std::move(args_json), {}, {}, false, node.is_partial});
        if (!node.is_partial) {
            events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        }
        return events;
    }

    return events;
}

std::vector<common_chat_shaped_event> common_chat_gemma4_transformer::shape(
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
        case DK::TOOL_ID:
            out.push_back({SK::TOOL_ID, event.text, false});
            break;
        case DK::TOOL_ARGS_RAW:
            out.push_back({SK::TOOL_ARGS_JSON, event.text, event.is_partial});
            break;
        case DK::TOOL_ARG_KV:
            // Gemma 4's decoder doesn't emit per-arg KV — it emits a single
            // pre-assembled TOOL_ARGS_RAW from gemma4_to_json(). Defensively
            // ignore unexpected per-arg events.
            break;
        case DK::TOOL_CLOSE:
            out.push_back({SK::TOOL_CLOSE, {}, event.is_partial});
            break;
    }
    return out;
}

// Gemma 4 FSM-state-to-grammar-rule registry. The args region is the
// `gemma4-dict` rule (a custom dict subtree the decoder walks via
// `gemma4_to_json()` rather than emitting per-arg KVs).
const common_chat_format_state_rules gemma4_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "gemma4-dict" },
    }
};
