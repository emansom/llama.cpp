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

namespace {
using ordered_json = nlohmann::ordered_json;

// JSON serializer matching `tojson(ensure_ascii=False)` — items with `, ` and
// keys with `: `, no escape of non-ASCII.
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

// Mirror of Jinja `visible_text(content)` macro: returns the visible text
// from a content value (string, list of {type:text, text:...} parts, or
// fallback to dump).
std::string visible_text(const ordered_json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string out;
        for (const auto & item : content) {
            if (item.is_object() && item.value("type", std::string{}) == "text" && item.contains("text")) {
                out += item["text"].get<std::string>();
            } else if (item.is_string()) {
                out += item.get<std::string>();
            }
        }
        return out;
    }
    if (content.is_null()) {
        return std::string{};
    }
    return content.dump();
}

// Strips left, right, or both ends of whitespace (matching Python's str.strip).
std::string lstrip_chars(const std::string & s, const std::string & chars) {
    size_t pos = s.find_first_not_of(chars);
    return pos == std::string::npos ? std::string{} : s.substr(pos);
}
std::string rstrip_chars(const std::string & s, const std::string & chars) {
    size_t pos = s.find_last_not_of(chars);
    return pos == std::string::npos ? std::string{} : s.substr(0, pos + 1);
}
std::string strip_ws(const std::string & s) {
    return lstrip_chars(rstrip_chars(s, " \t\n\r"), " \t\n\r");
}

// Find last occurrence of needle in haystack; returns positions for
// split-style operations.
std::vector<std::string> split_str(const std::string & s, const std::string & sep) {
    std::vector<std::string> out;
    if (sep.empty()) {
        out.push_back(s);
        return out;
    }
    size_t start = 0;
    while (true) {
        size_t pos = s.find(sep, start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            return out;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + sep.size();
    }
}

}  // namespace

// GLM-4.7-Flash FSM-state-to-grammar-rule registry. Per-arg XML format
// (<arg_key>K</arg_key><arg_value>V</arg_value>) — uses IN_TOOL_ARG_KEY /
// IN_TOOL_ARG_VAL for the `arg-name` and `arg-value` rules respectively.
const common_chat_format_state_rules glm_4_7_flash_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,      "content" },
        { common_chat_format_state::IN_REASONING,    "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL,    "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME,    "func-name" },
        { common_chat_format_state::IN_TOOL_ARG_KEY, "arg-name" },
        { common_chat_format_state::IN_TOOL_ARG_VAL, "arg-value" },
    }
};

std::string common_chat_glm_4_7_flash_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    // 1. Constant prefix.
    out << "[gMASK]<sop>";

    // 2. Tools system block.
    const ordered_json tools = inputs.tools.is_array() ? inputs.tools : ordered_json::array();
    if (!tools.empty()) {
        out << "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            << "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
        for (const auto & tool : tools) {
            out << to_jinja_json(tool) << "\n";
        }
        out << "</tools>\n\n"
            << "For each function call, output the function name and arguments within the following XML format:\n"
            << "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
            << "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
    }

    // 3. Find last_user_index for reasoning-emit decision.
    int last_user_index = -1;
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        if (inputs.messages[i].is_object() &&
            inputs.messages[i].value("role", std::string{}) == "user") {
            last_user_index = static_cast<int>(i);
        }
    }

    // 4. Per-message rendering.
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        const auto & m   = inputs.messages[i];
        const std::string role = m.value("role", "");

        if (role == "user") {
            // Jinja: <|user|>{{ visible_text(m.content) }} — trailing \n stripped by next {%-.
            out << "<|user|>" << visible_text(m.value("content", ordered_json{}));
        } else if (role == "assistant") {
            // Jinja: <|assistant|>\n{%- set ... — \n stripped by {%-.
            out << "<|assistant|>";

            // Extract reasoning_content from m or by parsing content for </think>.
            std::string reasoning_content;
            std::string content = visible_text(m.value("content", ordered_json{}));
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                reasoning_content = m["reasoning_content"].get<std::string>();
            } else if (content.find("</think>") != std::string::npos) {
                // reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                auto first = split_str(content, "</think>");
                std::string before_close = rstrip_chars(first[0], "\n");
                auto think_parts = split_str(before_close, "<think>");
                reasoning_content = lstrip_chars(think_parts.back(), "\n");
                // content = content.split('</think>')[-1].lstrip('\n')
                content = lstrip_chars(first.back(), "\n");
            }

            // Jinja: ((clear_thinking is defined and not clear_thinking) or loop.index0 > ns.last_user_index) and reasoning_content
            // clear_thinking is undefined → first condition false; rely on loop.index0 > last_user_index.
            const bool emit_think_block =
                (static_cast<int>(i) > last_user_index) && !reasoning_content.empty();
            if (emit_think_block) {
                out << "<think>" << strip_ws(reasoning_content) << "</think>";
            } else {
                out << "</think>";
            }

            // Content (stripped) if non-empty.
            const std::string stripped_content = strip_ws(content);
            if (!stripped_content.empty()) {
                out << stripped_content;
            }

            // Tool calls. Jinja: `{%- endif -%}` after the content block strips trailing
            // whitespace, so `{% if m.tool_calls %}` sits flush against the previous emission.
            // Each `{{- '<tool_call>' + tc.name -}}` strips around, and consecutive tool calls
            // concatenate without separators.
            if (m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty()) {
                for (size_t tc_idx = 0; tc_idx < m["tool_calls"].size(); ++tc_idx) {
                    ordered_json tc = m["tool_calls"][tc_idx];
                    if (tc.contains("function") && tc["function"].is_object()) {
                        tc = tc["function"];
                    }
                    out << "<tool_call>" << tc.value("name", std::string{});
                    const ordered_json args = tc.value("arguments", ordered_json::object());
                    if (args.is_object()) {
                        for (auto it = args.begin(); it != args.end(); ++it) {
                            const std::string & k = it.key();
                            const ordered_json & v = it.value();
                            out << "<arg_key>" << k << "</arg_key><arg_value>";
                            if (v.is_string()) {
                                out << v.get<std::string>();
                            } else {
                                out << to_jinja_json(v);
                            }
                            out << "</arg_value>";
                        }
                    }
                    out << "</tool_call>";
                }
            }
        } else if (role == "tool") {
            const ordered_json content = m.value("content", ordered_json{});
            if (content.is_string()) {
                // Jinja: emit <|observation|> only if first tool or prev role wasn't tool.
                // Each {{- ... }} / {{- ... -}} sequence strips surrounding whitespace, so
                // the entire emission is one concatenated string with no separators.
                bool prev_was_tool = i > 0 &&
                    inputs.messages[i - 1].value("role", std::string{}) == "tool";
                if (!prev_was_tool) {
                    out << "<|observation|>";
                }
                out << "<tool_response>" << content.get<std::string>() << "</tool_response>";
            } else if (content.is_array()) {
                out << "<|observation|>";
                for (const auto & tr : content) {
                    // Jinja: \n<tool_response>{{ ... }}</tool_response>
                    out << "\n<tool_response>";
                    if (tr.is_object() && tr.contains("output")) {
                        if (tr["output"].is_string()) {
                            out << tr["output"].get<std::string>();
                        } else {
                            out << tr["output"].dump();
                        }
                    } else if (tr.is_string()) {
                        out << tr.get<std::string>();
                    } else {
                        out << tr.dump();
                    }
                    out << "</tool_response>";
                }
            }
        } else if (role == "system") {
            // Jinja: <|system|>{{ visible_text(m.content) }} — \n stripped by next {%-.
            out << "<|system|>" << visible_text(m.value("content", ordered_json{}));
        }
    }

    // 5. Generation prompt.
    if (inputs.add_generation_prompt) {
        out << "<|assistant|>";
        out << (inputs.enable_thinking ? "<think>" : "</think>");
    }

    return out.str();
}
