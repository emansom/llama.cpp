#include "chat-formats/lfm2-format.h"

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

std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string escape_json_inner(const std::string & s) {
    auto dumped = ordered_json(s).dump();
    if (dumped.size() >= 2 && dumped.front() == '"' && dumped.back() == '"') {
        return dumped.substr(1, dumped.size() - 2);
    }
    return dumped;
}

// LFM2 arg_value can be a quoted string ("..."), number, bool (True/False),
// or "None". The transformer produces JSON for each. Handle each form here.
//
// Strings: the model emits literal characters between the quotes. A `\n` in
// the model output is a two-byte sequence (backslash + n); to preserve that
// faithfully in JSON output, we escape the backslash as `\\` and the `n`
// stays as `n`, giving `\\n`. This is "transparent" passthrough — every byte
// the model emitted ends up in the JSON arguments unchanged in meaning.
std::string value_to_json(const std::string & raw) {
    std::string v = trim(raw);
    if (v == "True") {
        return "true";
    }
    if (v == "False") {
        return "false";
    }
    if (v == "None") {
        return "null";
    }
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
        // Quoted string literal from the model. Re-encode the inner bytes as
        // a JSON string so backslashes (`\n` literal etc.) survive verbatim.
        std::string inner = v.substr(1, v.size() - 2);
        return ordered_json(inner).dump();
    }
    // Otherwise it's a number; emit verbatim.
    return v;
}

}  // namespace

void common_chat_lfm2_tracker::advance(const common_peg_ast_node & node) {
    if (tag_is(node, common_chat_peg_builder::TOOL_OPEN)) {
        if (!node.is_partial && state_ != common_chat_format_state::IN_TOOL_CALL &&
            state_ != common_chat_format_state::IN_TOOL_NAME &&
            state_ != common_chat_format_state::IN_TOOL_ARGS &&
            state_ != common_chat_format_state::IN_TOOL_ARG_KEY &&
            state_ != common_chat_format_state::IN_TOOL_ARG_VAL) {
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

std::vector<std::string> common_chat_lfm2_tracker::expected_productions() const {
    switch (state_) {
        case common_chat_format_state::INITIAL:
        case common_chat_format_state::DONE:
        case common_chat_format_state::IN_CONTENT:
        case common_chat_format_state::IN_REASONING:
            return {"content", "reasoning", "tool-open"};
        case common_chat_format_state::IN_TOOL_CALL:
            return {"tool-name"};
        case common_chat_format_state::IN_TOOL_NAME:
            return {"tool-arg-name", "tool-close"};
        case common_chat_format_state::IN_TOOL_ARG_KEY:
            return {"tool-arg-value"};
        case common_chat_format_state::IN_TOOL_ARG_VAL:
            return {"tool-arg-name", "tool-close"};
        default:
            return {};
    }
}

std::vector<common_chat_decoded_event> common_chat_lfm2_decoder::decode(
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
    // The `tool_call_block` rule contains an outer `tool-open` (the
    // '<|tool_call_start|>[') and a matching `tool-close' (']<|tool_call_end|>')
    // wrapping one or more comma-separated `tool` nodes. Each `tool` node is
    // its own call: emit OPEN+NAME+ARG_KV(s)+CLOSE per tool.
    if (tag_is(node, common_chat_peg_builder::TOOL_OPEN)) {
        // Wrapper tool-open: ignore (the per-tool OPEN comes from the `tool`
        // tag visit below).
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_CLOSE)) {
        // Wrapper tool-close: closes the LAST tool in the call block.
        if (in_tool_ && !node.is_partial) {
            events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
            in_tool_ = false;
        }
        return events;
    }
    if (tag_is(node, "tool")) {
        // A new tool call begins. Close the previous one (if any) so the
        // presenter commits it before starting the next.
        if (in_tool_) {
            events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        }
        events.push_back({K::TOOL_OPEN, {}, {}, {}, false, node.is_partial});
        in_tool_ = true;
        pending_key_.clear();
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_NAME)) {
        if (!node.is_partial) {
            events.push_back({K::TOOL_NAME, trim(std::string(node.text)), {}, {}, false, false});
        }
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARG_NAME)) {
        if (!node.is_partial) {
            pending_key_ = trim(std::string(node.text));
        }
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARG_VALUE)) {
        if (!node.is_partial && !pending_key_.empty()) {
            std::string raw_value = std::string(node.text);
            std::string trimmed   = trim(raw_value);
            bool        is_str    = (!trimmed.empty() && trimmed.front() == '"' && trimmed.back() == '"');
            events.push_back({K::TOOL_ARG_KV, {}, pending_key_, std::move(trimmed), is_str, false});
            pending_key_.clear();
        }
        return events;
    }
    return events;
}

std::vector<common_chat_decoded_event> common_chat_lfm2_decoder::on_finalize() {
    // The presenter's on_finalize commits a still-pending tool call (with
    // whatever args have been buffered so far). Don't synthesize a TOOL_CLOSE
    // here — that would force the transformer to seal an incomplete args
    // buffer with '}' and break streaming monotonicity (subsequent prefixes
    // would extend the args past the prematurely-closed '}').
    return {};
}

std::vector<common_chat_shaped_event> common_chat_lfm2_transformer::shape(
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
            // Begin assembling JSON args. Deliberate scoped synthesis: Python
            // `name(k=v)` has no JSON envelope, so we synthesize `{` here and
            // `}` on TOOL_CLOSE.
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
            // LFM2 decoder doesn't emit raw JSON args.
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
                args_json_buffer_ += value_to_json(event.value);
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

std::vector<common_chat_shaped_event> common_chat_lfm2_transformer::on_finalize() {
    std::vector<common_chat_shaped_event> out;
    if (in_tool_ && !args_json_buffer_.empty()) {
        // Emit whatever JSON we've assembled so far, marked partial. Each
        // prefix extends this buffer as more args/value arrive, so the
        // resulting TOOL_ARGS_JSON sequence is prefix-monotonic.
        out.push_back({common_chat_shaped_event::kind::TOOL_ARGS_JSON, args_json_buffer_, true});
    }
    return out;
}

// LFM2 FSM-state-to-grammar-rule registry. Per-arg KV format (Python-style
// `name(k=v, ...)`) — uses IN_TOOL_ARG_KEY / IN_TOOL_ARG_VAL states for the
// `arg-name` and `arg-value` rules respectively. No tool-id; reasoning via
// the standard <think>/</think> block.
const common_chat_format_state_rules lfm2_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,      "content" },
        { common_chat_format_state::IN_REASONING,    "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL,    "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME,    "func-name" },
        { common_chat_format_state::IN_TOOL_ARG_KEY, "arg-name" },
        { common_chat_format_state::IN_TOOL_ARG_VAL, "arg-value" },
    }
};
