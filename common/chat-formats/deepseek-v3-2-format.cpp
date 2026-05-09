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

// DeepSeek-V3.2 FSM-state-to-grammar-rule registry. DSML XML param format:
// each `<parameter>...</parameter>` is matched by the `arg` rule and visited
// in IN_TOOL_PARAM. Tool calls live inside a `tool-call` envelope nested
// within a `tool-calls-block`.
const common_chat_format_state_rules deepseek_v3_2_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,    "content" },
        { common_chat_format_state::IN_REASONING,  "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL,  "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME,  "func-name" },
        { common_chat_format_state::IN_TOOL_PARAM, "arg" },
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// DeepSeek-V3.2 prompt writer
// ──────────────────────────────────────────────────────────────────────────────

namespace {
using ordered_json = nlohmann::ordered_json;

constexpr const char * DSML       = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";  // U+FF5C "fullwidth bar"
constexpr const char * USER_OPEN  = "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>";
constexpr const char * ASST_OPEN  = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";
constexpr const char * EOS_MARKER = "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>";

// JSON serializer matching Jinja `tojson` defaults (ensure_ascii=true would
// escape non-ASCII, which we don't reproduce; for the ASCII-only fixtures
// the output is identical). Items use `, ` and keys `: ` separators.
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

// The fixed tools header text (from line 14 of the Jinja template; many
// embedded `<｜DSML｜...>` literal markers).
std::string tools_header() {
    const std::string d = DSML;
    std::string h;
    h += "## Tools\n\nYou have access to a set of tools you can use to answer the user's question.\n";
    h += "You can invoke functions by writing a \"<" + d + "function_calls>\" block like the following as part of your reply to the user:\n";
    h += "<" + d + "function_calls>\n";
    h += "<" + d + "invoke name=\"$FUNCTION_NAME\">\n";
    h += "<" + d + "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + d + "parameter>\n";
    h += "...\n";
    h += "</" + d + "invoke>\n";
    h += "<" + d + "invoke name=\"$FUNCTION_NAME2\">\n";
    h += "...\n";
    h += "</" + d + "invoke>\n";
    h += "</" + d + "function_calls>\n\n";
    h += "String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The \"string\" attribute should be set to \"true\" for string type parameters and \"false\" for other types (numbers, booleans, arrays, objects).\n\n";
    h += "If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:\n\n";
    h += "<" + d + "function_calls>\n...\n</" + d + "function_calls>\n\n";
    h += "<function_results>\n...\n</function_results>\n\n";
    h += "<think>...thinking about results</think>\n\n";
    h += "Here are the functions available in JSONSchema format:\n<functions>\n";
    return h;
}

}  // namespace

std::string common_chat_deepseek_v3_2_render(const autoparser::generation_params & inputs,
                                             const std::string & bos_token) {
    std::ostringstream out;

    const bool thinking = inputs.enable_thinking;

    // 1. Build system_prompt: concatenate all `system` role messages.
    std::string system_prompt;
    bool        is_first_sp = true;
    if (inputs.messages.is_array()) {
        for (const auto & msg : inputs.messages) {
            if (msg.value("role", std::string{}) != "system") {
                continue;
            }
            std::string content;
            if (msg.contains("content") && msg["content"].is_string()) {
                content = msg["content"].get<std::string>();
            }
            if (is_first_sp) {
                system_prompt += content;
                is_first_sp = false;
            } else {
                system_prompt += "\n\n" + content;
            }
        }
    }

    // 2. Tools schema appended to system_prompt.
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        std::string schemas;
        for (const auto & tool : inputs.tools) {
            if (tool.is_object() && tool.value("type", std::string{}) == "function" &&
                tool.contains("function")) {
                schemas += to_jinja_json(tool["function"]) + "\n";
            }
        }
        const std::string tools_block = tools_header() + schemas + "</functions>\n";
        if (!system_prompt.empty()) {
            system_prompt += "\n\n" + tools_block;
        } else {
            system_prompt = tools_block;
        }
    }

    // 3. BOS + system_prompt.
    out << bos_token << system_prompt;

    // 4. Find last_user_idx (last user/developer message).
    int last_user_idx = -1;
    if (inputs.messages.is_array()) {
        for (size_t i = 0; i < inputs.messages.size(); ++i) {
            const std::string r = inputs.messages[i].value("role", std::string{});
            if (r == "user" || r == "developer") {
                last_user_idx = static_cast<int>(i);
            }
        }
    }

    // 5. State-machine main loop.
    bool pending_asst_marker = false;
    bool pending_tool_marker = false;

    if (!inputs.messages.is_array()) {
        return out.str();
    }
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        const auto & message = inputs.messages[i];
        const std::string role = message.value("role", std::string{});

        if (role == "user") {
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << USER_OPEN << content;
            pending_asst_marker = true;
            pending_tool_marker = false;
        } else if (role == "assistant") {
            const bool is_after_last_user = static_cast<int>(i) > last_user_idx;

            if (pending_asst_marker) {
                out << ASST_OPEN;
                if (is_after_last_user && thinking) {
                    out << "<think>";
                    if (message.contains("reasoning_content") &&
                        message["reasoning_content"].is_string() &&
                        !message["reasoning_content"].get<std::string>().empty()) {
                        out << message["reasoning_content"].get<std::string>();
                    }
                    out << "</think>";
                } else {
                    out << "</think>";
                }
            } else if (pending_tool_marker) {
                if (is_after_last_user && thinking) {
                    out << "\n\n<think>";
                    if (message.contains("reasoning_content") &&
                        message["reasoning_content"].is_string() &&
                        !message["reasoning_content"].get<std::string>().empty()) {
                        out << message["reasoning_content"].get<std::string>();
                    }
                    out << "</think>";
                } else {
                    out << "\n\n</think>";
                }
            }
            pending_asst_marker = false;
            pending_tool_marker = false;

            // Content.
            if (message.contains("content") && message["content"].is_string() &&
                !message["content"].get<std::string>().empty()) {
                out << message["content"].get<std::string>();
            }

            // Tool calls.
            if (message.contains("tool_calls") && message["tool_calls"].is_array() &&
                !message["tool_calls"].empty()) {
                const std::string d = DSML;
                out << "\n\n<" << d << "function_calls>\n";
                for (const auto & tool : message["tool_calls"]) {
                    if (!tool.contains("function") || !tool["function"].is_object()) continue;
                    const auto & func = tool["function"];
                    out << "<" << d << "invoke name=\"" << func.value("name", std::string{}) << "\">\n";

                    // Resolve arguments to a JSON object (parse if string).
                    ordered_json args = func.value("arguments", ordered_json::object());
                    if (args.is_string()) {
                        try {
                            args = ordered_json::parse(args.get<std::string>());
                        } catch (...) {
                            args = ordered_json::object();
                        }
                    }
                    if (args.is_object()) {
                        for (auto it = args.begin(); it != args.end(); ++it) {
                            const std::string & key = it.key();
                            const ordered_json & val = it.value();
                            if (val.is_string()) {
                                out << "<" << d << "parameter name=\"" << key << "\" string=\"true\">"
                                    << val.get<std::string>()
                                    << "</" << d << "parameter>\n";
                            } else {
                                out << "<" << d << "parameter name=\"" << key << "\" string=\"false\">"
                                    << to_jinja_json(val)
                                    << "</" << d << "parameter>\n";
                            }
                        }
                    }
                    out << "</" << d << "invoke>\n";
                }
                out << "</" << d << "function_calls>";
            }

            out << EOS_MARKER;
        } else if (role == "tool") {
            // Find the previous assistant message (with tool_calls) before this index.
            int assistant_idx = -1;
            for (size_t j = 0; j < i; ++j) {
                if (inputs.messages[j].value("role", std::string{}) == "assistant" &&
                    inputs.messages[j].contains("tool_calls") &&
                    inputs.messages[j]["tool_calls"].is_array() &&
                    !inputs.messages[j]["tool_calls"].empty()) {
                    assistant_idx = static_cast<int>(j);
                }
            }
            if (assistant_idx < 0) continue;

            const int call_order = static_cast<int>(i) - assistant_idx;
            const auto & assistant_msg = inputs.messages[assistant_idx];
            const int tool_call_count = static_cast<int>(assistant_msg["tool_calls"].size());

            if (call_order == 1) {
                out << "\n\n<function_results>";
            }
            std::string content;
            if (message.contains("content") && message["content"].is_string()) {
                content = message["content"].get<std::string>();
            }
            out << "\n<result>" << content << "</result>";

            if (call_order == tool_call_count) {
                out << "\n</function_results>";
                pending_asst_marker = false;
                pending_tool_marker = true;
            }
        }
        // system messages are absorbed into system_prompt above; skip in loop.
    }

    // 6. Generation prompt.
    if (inputs.add_generation_prompt) {
        if (pending_asst_marker) {
            out << ASST_OPEN;
            if (thinking) {
                out << "<think>";
            } else {
                out << "<think></think>";
            }
        } else if (pending_tool_marker) {
            if (thinking) {
                out << "\n\n<think>";
            } else {
                out << "\n\n<think></think>";
            }
        }
    }

    return out.str();
}
