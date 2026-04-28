#include "chat-formats/glm-4-7-flash-format.h"

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

// GLM-4.7-Flash arg_value encoding: per the Jinja template, strings pass
// through verbatim while non-strings are tojson'd. Detect which by trying to
// parse the value as JSON — on success, emit the parsed-then-redumped form
// (canonicalized JSON). On failure, the value is a raw string and we
// JSON-encode it.
std::string value_to_json(const std::string & raw) {
    try {
        auto parsed = ordered_json::parse(raw);
        return parsed.dump();
    } catch (const ordered_json::exception &) {
        return ordered_json(raw).dump();
    }
}

}  // namespace

void common_chat_glm_4_7_flash_tracker::advance(const common_peg_ast_node & node) {
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

std::vector<std::string> common_chat_glm_4_7_flash_tracker::expected_productions() const {
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

std::vector<common_chat_decoded_event> common_chat_glm_4_7_flash_decoder::decode(
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
        // Both the outer `tool_call` rule and the inner `tool_open` (`<tool_call>`
        // literal) carry the tool-open tag. Emit a single TOOL_OPEN per call —
        // only on the depth 0 -> 1 transition.
        if (tool_open_depth_ == 0) {
            events.push_back({K::TOOL_OPEN, {}, {}, {}, false, node.is_partial});
            pending_key_.clear();
        }
        ++tool_open_depth_;
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
            // is_string flag is informational only — the transformer decides
            // how to encode (try JSON, fall back to string).
            events.push_back({K::TOOL_ARG_KV, {}, pending_key_, std::move(raw_value), true, false});
            pending_key_.clear();
        }
        return events;
    }
    if (tag_is(node, common_chat_peg_builder::TOOL_CLOSE)) {
        // Only emit TOOL_CLOSE on a complete close marker. A partial
        // '</tool_call>' (e.g. just '</' as the streaming parser advances)
        // would otherwise let the transformer seal an incomplete args
        // buffer with '}' and break streaming monotonicity.
        if (tool_open_depth_ > 0 && !node.is_partial) {
            events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
            tool_open_depth_ = 0;
            pending_key_.clear();
        }
        return events;
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_glm_4_7_flash_transformer::shape(
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
            // Begin assembling JSON args. Deliberate scoped synthesis: the
            // GLM-4.7-Flash wire format has no JSON envelope around the args,
            // so we synthesize `{` here and `}` on TOOL_CLOSE, plus `,` / `:`
            // between KV pairs. Confined to this transformer.
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
                    args_json_buffer_ += ", ";
                }
                first_arg_ = false;
                args_json_buffer_ += '"';
                args_json_buffer_ += escape_json_inner(event.key);
                args_json_buffer_ += "\": ";
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

std::vector<common_chat_shaped_event> common_chat_glm_4_7_flash_transformer::on_finalize() {
    std::vector<common_chat_shaped_event> out;
    if (in_tool_ && !args_json_buffer_.empty()) {
        // Emit whatever JSON we've assembled so far, marked partial. Each
        // prefix extends this buffer as more KV pairs arrive, so the
        // resulting TOOL_ARGS_JSON sequence is prefix-monotonic.
        out.push_back({common_chat_shaped_event::kind::TOOL_ARGS_JSON, args_json_buffer_, true});
    }
    return out;
}
