#include "chat-formats/simple-reasoning-format.h"

#include "chat-peg-parser.h"
#include "peg-parser.h"

#include <string>

namespace {
bool tag_is(const common_peg_ast_node & node, const char * tag) {
    return node.tag == tag;
}
}  // namespace

void common_chat_simple_reasoning_tracker::advance(const common_peg_ast_node & node) {
    if (tag_is(node, common_chat_peg_builder::REASONING)) {
        state_ = common_chat_format_state::IN_REASONING;
    } else if (tag_is(node, common_chat_peg_builder::CONTENT)) {
        state_ = common_chat_format_state::IN_CONTENT;
    }
}

std::vector<std::string> common_chat_simple_reasoning_tracker::expected_productions() const {
    switch (state_) {
        case common_chat_format_state::INITIAL:
        case common_chat_format_state::IN_REASONING:
            return {"reasoning", "content"};
        case common_chat_format_state::IN_CONTENT:
            return {"content"};
        default:
            return {};
    }
}

std::vector<common_chat_decoded_event> common_chat_simple_reasoning_decoder::decode(
    const common_peg_ast_node & node) {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;
    if (tag_is(node, common_chat_peg_builder::REASONING)) {
        events.push_back({K::REASONING_TEXT, std::string(node.text), {}, {}, false, node.is_partial});
    } else if (tag_is(node, common_chat_peg_builder::CONTENT)) {
        events.push_back({K::CONTENT_TEXT, std::string(node.text), {}, {}, false, node.is_partial});
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_simple_reasoning_transformer::shape(
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
        default:
            break;
    }
    return out;
}

const common_chat_format_state_rules simple_reasoning_state_rules = {
    {
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_CONTENT,   "content" },
    }
};
