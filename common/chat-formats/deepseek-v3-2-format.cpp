#include "chat-formats/deepseek-v3-2-format.h"

#include "chat-peg-parser.h"
#include "peg-parser.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <string>

namespace {

using ordered_json = nlohmann::ordered_json;

// Lowercase + underscore-to-hyphen so both grammar-file rule names ("param-name",
// "BOOL_STR") and C++ builder conventions reach the same handler.
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

}  // namespace

void common_chat_deepseek_v3_2_tracker::advance(const common_peg_ast_node & node) {
    // The DSML grammar wraps each tool call with the "tool" tag (lark-to-peg
    // auto-tag for 'tool_call' rules). Treat it like a TOOL_OPEN/TOOL_CLOSE
    // pair: enter IN_TOOL_CALL on partial, leave for DONE on completion.
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
    if (tag_is(node, common_chat_peg_builder::TOOL_ARG_NAME)) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::IN_TOOL_ARG_KEY;
        }
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARG_VALUE)) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::IN_TOOL_ARG_VAL;
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

std::vector<std::string> common_chat_deepseek_v3_2_tracker::expected_productions() const {
    switch (state_) {
        case common_chat_format_state::INITIAL:
        case common_chat_format_state::DONE:
        case common_chat_format_state::IN_CONTENT:
        case common_chat_format_state::IN_REASONING:
            return {"content", "reasoning", "tool-open"};
        case common_chat_format_state::IN_TOOL_CALL:
            return {"tool-name"};
        case common_chat_format_state::IN_TOOL_NAME:
            return {"param", "tool-close"};
        case common_chat_format_state::IN_TOOL_PARAM:
            return {"param-name", "BOOL_STR", "param-value"};
        default:
            return {};
    }
}

std::vector<common_chat_decoded_event> common_chat_deepseek_v3_2_decoder::decode(
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
        // A new tool call (DSML 'invoke' block) begins. Close the previous
        // one (if any) so the presenter commits it before starting the next.
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
    // The 'arg' rule wraps each <param ...>VALUE</param> block. Only emit
    // the KV event once per arg, when the whole arg rule has matched (its
    // closing literal '</...|DSML|...parameter>' succeeded). This avoids
    // emitting a fresh KV for each streaming-prefix length where the
    // 'arg-value' regex matches a strictly-growing prefix (which would
    // produce non-monotonic args text since the transformer appends each
    // KV as a separate "key":"value" pair).
    const std::string r = normalize_rule(node.rule);
    if (r == "arg") {
        if (!node.is_partial && arena_) {
            constexpr int kArgDepth = 4;
            auto name_id      = arena_->find_by_tag(node, common_chat_peg_builder::TOOL_ARG_NAME, kArgDepth);
            auto value_id     = arena_->find_by_tag(node, common_chat_peg_builder::TOOL_ARG_VALUE, kArgDepth);
            auto bool_str_id  = arena_->find_by_rule(node, "BOOL-STR", kArgDepth);
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
            bool is_str = false;
            if (bool_str_id != COMMON_PEG_INVALID_AST_ID) {
                const auto & bs_node = arena_->get(bool_str_id);
                is_str = (bs_node.text == "true");
            }
            if (!key.empty()) {
                events.push_back({K::TOOL_ARG_KV, {}, std::move(key), std::move(value), is_str, false});
            }
        }
        return events;
    }
    // Suppress per-tag emissions for arg-name / arg-value when their parent
    // 'arg' rule already collected them (handled_ids_).
    if (tag_is(node, common_chat_peg_builder::TOOL_ARG_NAME) ||
        tag_is(node, common_chat_peg_builder::TOOL_ARG_VALUE)) {
        return events;
    }

    return events;
}

std::vector<common_chat_decoded_event> common_chat_deepseek_v3_2_decoder::on_finalize() {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;
    // Only close the LAST tool when its 'tool' visit was non-partial. A
    // partial close would force the transformer to seal an in-flight args
    // buffer with '}' and break streaming monotonicity once a later prefix
    // extends the args past the prematurely-closed '}'.
    if (in_tool_ && last_tool_complete_) {
        events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        in_tool_ = false;
        last_tool_complete_ = false;
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_deepseek_v3_2_transformer::shape(
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
            // Begin assembling the JSON args object — DSML doesn't have an
            // envelope, so we synthesize the wrapping `{`/`}`. This is the
            // deliberate scoped exception to no-synthesis: DSML's wire shape
            // isn't JSON, so converting it requires structural characters.
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
            // DSML decoder doesn't emit this — args come as TOOL_ARG_KV.
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
                if (event.is_string) {
                    args_json_buffer_ += '"';
                    args_json_buffer_ += escape_json_inner(event.value);
                    args_json_buffer_ += '"';
                } else {
                    // Non-string: emit verbatim (number/bool/null/JSON literal).
                    args_json_buffer_ += event.value;
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

std::vector<common_chat_shaped_event> common_chat_deepseek_v3_2_transformer::on_finalize() {
    std::vector<common_chat_shaped_event> out;
    if (in_tool_ && !args_json_buffer_.empty()) {
        // Emit whatever JSON we've assembled so far, marked partial. Each
        // prefix extends this buffer as more args arrive, so the
        // resulting TOOL_ARGS_JSON sequence is prefix-monotonic.
        out.push_back({common_chat_shaped_event::kind::TOOL_ARGS_JSON, args_json_buffer_, true});
    }
    return out;
}
