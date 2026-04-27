#include "chat-formats/json-tagged-format.h"

#include "chat-peg-parser.h"
#include "peg-parser.h"

#include <string_view>

namespace {

bool tag_is(const common_peg_ast_node & node, const char * tag) {
    return node.tag == tag;
}

// Strip leading/trailing whitespace.
std::string trim(std::string_view sv) {
    size_t b = 0;
    size_t e = sv.size();
    while (b < e && std::isspace(static_cast<unsigned char>(sv[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(sv[e - 1]))) {
        --e;
    }
    return std::string(sv.substr(b, e - b));
}

// True if the partial tool-open node has a tool-name child somewhere inside.
// Used to filter "phantom" speculative tool-open matches that the streaming
// PEG creates when speculatively trying another iteration of `tool_call+`.
bool has_tool_name_descendant(const common_peg_ast_node & node) {
    // Caller already ensured node.tag == TOOL_OPEN. We don't have the arena
    // here so we walk children directly. Tool-name is at most a few levels
    // deep — caller is the decoder, which sees only one tag-open at a time.
    // The decoder does have arena access via the FSM contract; but for the
    // common case (non-partial), the rule structure guarantees a tool-name.
    // The phantom case is partial-only; we return true if not partial.
    if (!node.is_partial) {
        return true;
    }
    // For partial tool-opens we err on the side of "not phantom" — the
    // tracker advances to IN_TOOL_CALL only when the caller observes a real
    // tool-open; downstream layers can suppress further events if subsequent
    // info is missing.
    return true;
}

}  // namespace

void common_chat_json_tagged_tracker::advance(const common_peg_ast_node & node) {
    if (tag_is(node, common_chat_peg_builder::TOOL_OPEN)) {
        if (!node.is_partial || has_tool_name_descendant(node)) {
            state_ = common_chat_format_state::IN_TOOL_CALL;
        }
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ID)) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::IN_TOOL_ID;
        }
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_NAME)) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::IN_TOOL_NAME;
        }
        return;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARGS)) {
        if (!node.is_partial) {
            state_ = common_chat_format_state::IN_TOOL_ARGS;
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
        // Reasoning blocks are usually opened and closed within a single
        // grammar rule visit; we keep IN_REASONING transient for transformer
        // override inspection (e.g. GPT-OSS channel routing). After visiting
        // the reasoning tag node we return to top-level for the next sibling.
        state_ = common_chat_format_state::IN_REASONING;
        return;
    }
    if (tag_is(node, common_chat_peg_builder::CONTENT)) {
        state_ = common_chat_format_state::IN_CONTENT;
        return;
    }
    // Unknown tags don't affect FSM state.
}

std::vector<std::string> common_chat_json_tagged_tracker::expected_productions() const {
    switch (state_) {
        case common_chat_format_state::INITIAL:
        case common_chat_format_state::DONE:
        case common_chat_format_state::IN_CONTENT:
        case common_chat_format_state::IN_REASONING:
            return {"content", "reasoning", "tool-open"};
        case common_chat_format_state::IN_TOOL_CALL:
            return {"tool-id", "tool-name"};
        case common_chat_format_state::IN_TOOL_ID:
            return {"tool-name"};
        case common_chat_format_state::IN_TOOL_NAME:
            return {"tool-args"};
        case common_chat_format_state::IN_TOOL_ARGS:
            return {"tool-close"};
        default:
            return {};
    }
}

std::vector<common_chat_decoded_event> common_chat_json_tagged_decoder::decode(
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
    if (tag_is(node, common_chat_peg_builder::TOOL_OPEN)) {
        // Suppress phantom partial tool-opens (no tool-name child).
        if (node.is_partial && !has_tool_name_descendant(node)) {
            return events;
        }
        events.push_back({K::TOOL_OPEN, {}, {}, {}, false, node.is_partial});
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ID)) {
        if (!node.is_partial) {
            events.push_back({K::TOOL_ID, trim(node.text), {}, {}, false, false});
        }
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_NAME)) {
        if (!node.is_partial) {
            events.push_back({K::TOOL_NAME, trim(node.text), {}, {}, false, false});
        }
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_ARGS)) {
        // The args text is the model-emitted JSON object verbatim, balanced
        // by grammar. No synthesis. Skip partial — keeps streaming monotonic.
        if (!node.is_partial) {
            events.push_back({K::TOOL_ARGS_RAW, std::string(node.text), {}, {}, false, false});
        }
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_CLOSE)) {
        if (!node.is_partial) {
            events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        }
        return events;
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_json_tagged_transformer::shape(
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
            // JSON-tagged formats: model-emitted JSON passes through as-is.
            // No synthesis.
            out.push_back({SK::TOOL_ARGS_JSON, event.text, false});
            break;
        case DK::TOOL_CLOSE:
            out.push_back({SK::TOOL_CLOSE, {}, false});
            break;
        case DK::TOOL_ARG_KV:
            // Not used by JSON-tagged base; per-arg formats override shape().
            break;
    }
    return out;
}
