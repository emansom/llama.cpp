#include "chat-formats/qwen3-5-format.h"

#include "chat-peg-parser.h"
#include "peg-parser.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <sstream>
#include <stdexcept>
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

// JSON serializer matching Jinja `tojson` defaults: items `, ` and keys `: `
// separators.
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

// Python-style `str(value)` conversion used by Jinja's `value | string`. For
// strings: returns the string itself (no quoting). For booleans: "True" /
// "False" (capitalised, like Python). For null: "None". Numbers: their JSON
// representation. Used by the writer for tool-call argument values that
// aren't mappings/sequences --- matches the Jinja template's `else` branch:
//   args_value | string
std::string python_str(const ordered_json & v) {
    if (v.is_null())    return "None";
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_string())  return v.get<std::string>();
    if (v.is_number())  return v.dump();
    return v.dump();
}

// `content | trim`: strip ASCII whitespace from both ends.
std::string trim_str(const std::string & s) {
    size_t s_pos = s.find_first_not_of(" \t\n\r");
    if (s_pos == std::string::npos) return {};
    size_t e_pos = s.find_last_not_of(" \t\n\r");
    return s.substr(s_pos, e_pos - s_pos + 1);
}

// `s.lstrip('\n')` (Python).
std::string lstrip_newlines(const std::string & s) {
    size_t pos = s.find_first_not_of('\n');
    return pos == std::string::npos ? std::string{} : s.substr(pos);
}

// `s.rstrip('\n')` (Python).
std::string rstrip_newlines(const std::string & s) {
    size_t pos = s.find_last_not_of('\n');
    return pos == std::string::npos ? std::string{} : s.substr(0, pos + 1);
}

// Resolve `message.content` to a single string. Vision content (list of
// items containing image/video markers) is rejected to match the template's
// `raise_exception('System message cannot contain images.')` paths --- this
// codec is non-vision.
std::string render_content(const ordered_json & content) {
    if (content.is_null()) {
        return {};
    }
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string out;
        for (const auto & item : content) {
            if (!item.is_object()) {
                throw std::runtime_error("Qwen3.5 writer: unexpected non-object content item");
            }
            if (item.contains("image") || item.contains("image_url") ||
                item.value("type", std::string{}) == "image" ||
                item.contains("video") ||
                item.value("type", std::string{}) == "video") {
                throw std::runtime_error("Qwen3.5 writer: vision content not supported");
            }
            if (item.contains("text") && item["text"].is_string()) {
                out += item["text"].get<std::string>();
            } else {
                throw std::runtime_error("Qwen3.5 writer: unexpected content item shape");
            }
        }
        return out;
    }
    throw std::runtime_error("Qwen3.5 writer: unexpected content type");
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────────
// Output parser: tracker + decoder + transformer
// ──────────────────────────────────────────────────────────────────────────────

void common_chat_qwen3_5_tracker::advance(const common_peg_ast_node & node) {
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

std::vector<std::string> common_chat_qwen3_5_tracker::expected_productions() const {
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

std::vector<common_chat_decoded_event> common_chat_qwen3_5_decoder::decode(
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
            if (!key.empty()) {
                // Qwen3.5 has no `string="true|false"` flag --- treat all
                // values as strings on the output side.
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

std::vector<common_chat_decoded_event> common_chat_qwen3_5_decoder::on_finalize() {
    using K = common_chat_decoded_event::kind;
    std::vector<common_chat_decoded_event> events;
    if (in_tool_ && last_tool_complete_) {
        events.push_back({K::TOOL_CLOSE, {}, {}, {}, false, false});
        in_tool_ = false;
        last_tool_complete_ = false;
    }
    return events;
}

std::vector<common_chat_shaped_event> common_chat_qwen3_5_transformer::shape(
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
                // Qwen3.5 has no string-vs-non-string flag. Try to parse the
                // raw value as JSON (number, boolean, null, array, object);
                // if that succeeds, embed the parsed form verbatim. Otherwise
                // fall back to a JSON string. This matches the autoparser's
                // behaviour and avoids stringifying numeric args.
                bool embedded_raw = false;
                try {
                    auto parsed = ordered_json::parse(event.value);
                    if (!parsed.is_string()) {
                        args_json_buffer_ += parsed.dump();
                        embedded_raw = true;
                    }
                } catch (...) {
                    // not valid JSON — fall through to string quoting
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

std::vector<common_chat_shaped_event> common_chat_qwen3_5_transformer::on_finalize() {
    std::vector<common_chat_shaped_event> out;
    if (in_tool_ && !args_json_buffer_.empty()) {
        out.push_back({common_chat_shaped_event::kind::TOOL_ARGS_JSON, args_json_buffer_, true});
    }
    return out;
}

// Qwen3.5 FSM-state-to-grammar-rule registry. Per-arg XML format: tool
// envelope is `tool-call`, each param wrapper is `parameter`.
const common_chat_format_state_rules qwen3_5_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,    "content" },
        { common_chat_format_state::IN_REASONING,  "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL,  "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME,  "func-name" },
        { common_chat_format_state::IN_TOOL_PARAM, "parameter" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Qwen3.5 prompt writer. Mirrors models/templates/Qwen3.5-4B.jinja.
// ──────────────────────────────────────────────────────────────────────────────

namespace {

// Render one tool call's per-arg XML body. Used inside the assistant-with-
// tool-calls branch.
void render_tool_call(std::ostringstream & out, const ordered_json & tool_call_in) {
    ordered_json tool_call = tool_call_in;
    if (tool_call.contains("function") && tool_call["function"].is_object()) {
        tool_call = tool_call["function"];
    }
    out << "<function=" << tool_call.value("name", std::string{}) << ">\n";
    if (tool_call.contains("arguments")) {
        const auto & args_in = tool_call["arguments"];
        // Arguments may arrive as a JSON object (preferred) or, defensively,
        // as a string. The template's `tojson | items` only works on
        // mappings.
        ordered_json args = args_in;
        if (args.is_string()) {
            try {
                args = ordered_json::parse(args.get<std::string>());
            } catch (...) {
                args = ordered_json::object();
            }
        }
        if (args.is_object()) {
            for (auto it = args.begin(); it != args.end(); ++it) {
                out << "<parameter=" << it.key() << ">\n";
                const auto & v = it.value();
                if (v.is_object() || (v.is_array())) {
                    out << to_jinja_json(v);
                } else {
                    out << python_str(v);
                }
                out << "\n</parameter>\n";
            }
        }
    }
    out << "</function>\n</tool_call>";
}

}  // namespace

std::string common_chat_qwen3_5_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    if (!inputs.messages.is_array() || inputs.messages.empty()) {
        throw std::runtime_error("Qwen3.5 writer: no messages provided");
    }

    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    const bool first_is_system = inputs.messages[0].is_object() &&
        inputs.messages[0].value("role", std::string{}) == "system";

    // 1. System block. With tools: emit the canonical instructions text +
    //    tools JSON list + optional appended first-system content. Without
    //    tools: emit just the first system message (if any), trimmed.
    if (has_tools) {
        out << "<|im_start|>system\n";
        out << "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto & tool : inputs.tools) {
            out << "\n" << to_jinja_json(tool);
        }
        out << "\n</tools>";
        out << "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
               "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
               "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
               "This is the value for the second parameter\nthat can span\nmultiple lines\n"
               "</parameter>\n</function>\n</tool_call>\n\n"
               "<IMPORTANT>\nReminder:\n"
               "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
               "- Required parameters MUST be specified\n"
               "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
               "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
               "</IMPORTANT>";
        if (first_is_system) {
            const auto & c = inputs.messages[0].value("content", ordered_json{});
            std::string content = trim_str(render_content(c));
            if (!content.empty()) {
                out << "\n\n" << content;
            }
        }
        out << "<|im_end|>\n";
    } else if (first_is_system) {
        const auto & c = inputs.messages[0].value("content", ordered_json{});
        std::string content = trim_str(render_content(c));
        out << "<|im_start|>system\n" << content << "<|im_end|>\n";
    }

    // 2. Compute last_query_index: backwards scan to find the most recent
    //    `user` message whose content isn't a tool-response envelope. If no
    //    such user exists in the conversation, the template raises ---
    //    surface as exception here.
    size_t last_query_index = inputs.messages.size() - 1;
    bool   multi_step_tool  = true;
    for (size_t r = 0; r < inputs.messages.size(); ++r) {
        const size_t i = inputs.messages.size() - 1 - r;
        const auto & m = inputs.messages[i];
        if (multi_step_tool && m.value("role", std::string{}) == "user") {
            std::string content = trim_str(render_content(m.value("content", ordered_json{})));
            const bool is_tool_resp =
                content.size() >= std::string("<tool_response>").size() + std::string("</tool_response>").size() &&
                content.rfind("<tool_response>", 0) == 0 &&
                content.size() >= std::string("</tool_response>").size() &&
                content.compare(content.size() - std::string("</tool_response>").size(),
                                std::string("</tool_response>").size(),
                                "</tool_response>") == 0;
            if (!is_tool_resp) {
                multi_step_tool = false;
                last_query_index = i;
            }
        }
    }
    if (multi_step_tool) {
        throw std::runtime_error("Qwen3.5 writer: no user query found in messages");
    }

    // 3. Per-message rendering.
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});
        const std::string raw_content = render_content(message.value("content", ordered_json{}));
        std::string content = trim_str(raw_content);

        if (role == "system") {
            if (i != 0) {
                throw std::runtime_error("Qwen3.5 writer: system message must be at the beginning");
            }
            // Already consumed into the system block above.
        } else if (role == "user") {
            out << "<|im_start|>user\n" << content << "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string reasoning_content;
            if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
                reasoning_content = message["reasoning_content"].get<std::string>();
            } else {
                // Inline `<think>...</think>` in content: split on `</think>`
                // (rsplit-1), the first segment's last `<think>...` chunk is
                // reasoning, the trailing part is content.
                const std::string close = "</think>";
                size_t close_pos = content.find(close);
                if (close_pos != std::string::npos) {
                    std::string before = content.substr(0, close_pos);
                    std::string after  = content.substr(close_pos + close.size());
                    // before.rstrip('\n').split('<think>')[-1].lstrip('\n')
                    before = rstrip_newlines(before);
                    size_t open_pos = before.rfind("<think>");
                    if (open_pos != std::string::npos) {
                        before = before.substr(open_pos + std::string("<think>").size());
                    }
                    reasoning_content = lstrip_newlines(before);
                    content = lstrip_newlines(after);
                }
            }
            reasoning_content = trim_str(reasoning_content);

            if (i > last_query_index) {
                out << "<|im_start|>assistant\n<think>\n" << reasoning_content
                    << "\n</think>\n\n" << content;
            } else {
                out << "<|im_start|>assistant\n" << content;
            }

            const bool has_tool_calls = message.contains("tool_calls") &&
                message["tool_calls"].is_array() && !message["tool_calls"].empty();
            if (has_tool_calls) {
                const auto & tcs = message["tool_calls"];
                for (size_t k = 0; k < tcs.size(); ++k) {
                    if (k == 0) {
                        if (!trim_str(content).empty()) {
                            out << "\n\n<tool_call>\n";
                        } else {
                            out << "<tool_call>\n";
                        }
                    } else {
                        out << "\n<tool_call>\n";
                    }
                    render_tool_call(out, tcs[k]);
                }
            }
            out << "<|im_end|>\n";
        } else if (role == "tool") {
            const std::string prev_role = i > 0
                ? inputs.messages[i - 1].value("role", std::string{}) : std::string{};
            if (i > 0 && prev_role != "tool") {
                out << "<|im_start|>user";
            }
            out << "\n<tool_response>\n" << content << "\n</tool_response>";
            const bool is_last = (i + 1 == inputs.messages.size());
            const std::string next_role = is_last ? std::string{}
                : inputs.messages[i + 1].value("role", std::string{});
            if ((!is_last && next_role != "tool") || is_last) {
                out << "<|im_end|>\n";
            }
        } else {
            throw std::runtime_error("Qwen3.5 writer: unexpected message role: " + role);
        }
    }

    // 4. Generation prompt.
    if (inputs.add_generation_prompt) {
        out << "<|im_start|>assistant\n";
        if (!inputs.enable_thinking) {
            out << "<think>\n\n</think>\n\n";
        } else {
            out << "<think>\n";
        }
    }

    return out.str();
}
