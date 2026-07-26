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

// ──────────────────────────────────────────────────────────────────────────────
// Gemma 4 prompt writer
// Mirrors models/templates/google-gemma-4-31B-it.jinja. The Jinja template
// uses elaborate macros (format_argument, format_parameters,
// format_function_declaration, format_tool_response_block, strip_thinking)
// plus a per-message loop with continuation detection and forward-scan of
// tool messages. Each helper here is a 1:1 port.
// ──────────────────────────────────────────────────────────────────────────────

namespace {
using ordered_json = nlohmann::ordered_json;

// String trimming helpers (matching Jinja `| trim`).
std::string trim_ws(const std::string & s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return std::string{};
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// strip_thinking macro: split on `<channel|>` and strip text between
// `<|channel>` and that close marker on each split. Returns trimmed result.
// Mirrors lines 148-158 of the template.
std::string strip_thinking(const std::string & text) {
    std::string result;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t end_pos = text.find("<channel|>", pos);
        std::string part;
        if (end_pos == std::string::npos) {
            part = text.substr(pos);
            pos = text.size() + 1;  // exit loop
        } else {
            part = text.substr(pos, end_pos - pos);
            pos = end_pos + sizeof("<channel|>") - 1;
        }
        // If part contains `<|channel>`, take the prefix before it.
        size_t open_pos = part.find("<|channel>");
        if (open_pos != std::string::npos) {
            result += part.substr(0, open_pos);
        } else {
            result += part;
        }
    }
    return trim_ws(result);
}

// Forward declarations.
std::string format_argument(const ordered_json & argument, bool escape_keys);
std::string format_parameters(const ordered_json & properties, const ordered_json & required, int depth);

// Mirrors `format_argument` macro (lines 118-147).
std::string format_argument(const ordered_json & argument, bool escape_keys) {
    if (argument.is_string()) {
        return "<|\"|>" + argument.get<std::string>() + "<|\"|>";
    }
    if (argument.is_boolean()) {
        return argument.get<bool>() ? "true" : "false";
    }
    if (argument.is_object()) {
        std::string out = "{";
        bool first = true;
        // Jinja `| dictsort` sorts by key. nlohmann::ordered_json preserves
        // insertion order; we sort manually.
        std::vector<std::string> keys;
        for (auto it = argument.begin(); it != argument.end(); ++it) {
            keys.push_back(it.key());
        }
        std::sort(keys.begin(), keys.end());
        for (const auto & key : keys) {
            if (!first) out += ",";
            first = false;
            if (escape_keys) {
                out += "<|\"|>" + key + "<|\"|>";
            } else {
                out += key;
            }
            out += ":";
            out += format_argument(argument[key], escape_keys);
        }
        out += "}";
        return out;
    }
    if (argument.is_array()) {
        std::string out = "[";
        for (size_t i = 0; i < argument.size(); ++i) {
            if (i > 0) out += ",";
            out += format_argument(argument[i], escape_keys);
        }
        out += "]";
        return out;
    }
    if (argument.is_number()) {
        return argument.dump();
    }
    if (argument.is_null()) {
        return "null";
    }
    return argument.dump();
}

// Returns `true` for the JSON "type" string check, mirroring Jinja `| upper`
// comparisons. Compares case-insensitively.
bool type_eq(const ordered_json & schema, const char * upper_target) {
    if (!schema.is_object() || !schema.contains("type") || !schema["type"].is_string()) {
        return false;
    }
    const std::string & t = schema["type"].get_ref<const std::string &>();
    if (std::strlen(upper_target) != t.size()) return false;
    for (size_t i = 0; i < t.size(); ++i) {
        char c = t[i];
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
        if (c != upper_target[i]) return false;
    }
    return true;
}

// Mirrors `format_parameters` macro (lines 1-84). Walks `properties` and emits
// per-property typescript-like definitions inside an enclosing dict. Recurses
// for nested object properties. Skips standard JSON-schema keys.
std::string format_parameters(const ordered_json & properties, const ordered_json & /*required*/, int depth) {
    static const std::vector<std::string> standard_keys = {
        "description", "type", "properties", "required", "nullable",
    };

    std::string out;
    bool found_first = false;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        const std::string & key = it.key();
        const ordered_json & val = it.value();
        if (!val.is_object()) continue;

        bool add_comma = false;
        if (std::find(standard_keys.begin(), standard_keys.end(), key) != standard_keys.end()) {
            continue;
        }
        if (found_first) out += ",";
        found_first = true;
        out += key + ":{";

        if (val.contains("description") && val["description"].is_string()) {
            out += "description:<|\"|>" + val["description"].get<std::string>() + "<|\"|>";
            add_comma = true;
        }

        if (type_eq(val, "STRING")) {
            if (val.contains("enum") && val["enum"].is_array()) {
                if (add_comma) out += ","; else add_comma = true;
                out += "enum:" + format_argument(val["enum"], true);
            }
        } else if (type_eq(val, "ARRAY")) {
            if (val.contains("items") && val["items"].is_object() && !val["items"].empty()) {
                if (add_comma) out += ","; else add_comma = true;
                out += "items:{";
                bool items_first = false;
                const ordered_json & items = val["items"];
                // Jinja `| dictsort` sorts items by key alphabetically.
                std::vector<std::string> keys;
                for (auto k = items.begin(); k != items.end(); ++k) keys.push_back(k.key());
                std::sort(keys.begin(), keys.end());
                for (const auto & item_key : keys) {
                    const ordered_json & item_value = items[item_key];
                    if (item_value.is_null()) continue;
                    if (items_first) out += ",";
                    items_first = true;
                    if (item_key == "properties") {
                        out += "properties:{";
                        if (item_value.is_object()) {
                            out += format_parameters(item_value,
                                                     val["items"].value("required", ordered_json::array()),
                                                     depth + 1);
                        }
                        out += "}";
                    } else if (item_key == "required") {
                        out += "required:[";
                        for (size_t i = 0; i < item_value.size(); ++i) {
                            if (i > 0) out += ",";
                            out += "<|\"|>" + item_value[i].get<std::string>() + "<|\"|>";
                        }
                        out += "]";
                    } else if (item_key == "type") {
                        std::string tval;
                        if (item_value.is_string()) {
                            tval = item_value.get<std::string>();
                            for (auto & c : tval) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                            out += "type:" + format_argument(ordered_json(tval), true);
                        } else if (item_value.is_array()) {
                            ordered_json upper_arr = ordered_json::array();
                            for (const auto & v : item_value) {
                                std::string s = v.get<std::string>();
                                for (auto & c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                                upper_arr.push_back(s);
                            }
                            out += "type:" + format_argument(upper_arr, true);
                        }
                    } else {
                        out += item_key + ":" + format_argument(item_value, true);
                    }
                }
                out += "}";
            }
        }

        if (val.value("nullable", false)) {
            if (add_comma) out += ","; else add_comma = true;
            out += "nullable:true";
        }

        if (type_eq(val, "OBJECT")) {
            if (val.contains("properties") && val["properties"].is_object()) {
                if (add_comma) out += ","; else add_comma = true;
                out += "properties:{";
                out += format_parameters(val["properties"],
                                         val.value("required", ordered_json::array()),
                                         depth + 1);
                out += "}";
            } else if (val.is_object()) {
                if (add_comma) out += ","; else add_comma = true;
                out += "properties:{";
                out += format_parameters(val,
                                         val.value("required", ordered_json::array()),
                                         depth + 1);
                out += "}";
            }
            if (val.contains("required") && val["required"].is_array() && !val["required"].empty()) {
                if (add_comma) out += ","; else add_comma = true;
                out += "required:[";
                for (size_t i = 0; i < val["required"].size(); ++i) {
                    if (i > 0) out += ",";
                    out += "<|\"|>" + val["required"][i].get<std::string>() + "<|\"|>";
                }
                out += "]";
            }
        }

        if (add_comma) out += ","; else add_comma = true;
        std::string tup;
        if (val.contains("type") && val["type"].is_string()) {
            tup = val["type"].get<std::string>();
            for (auto & c : tup) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        out += "type:<|\"|>" + tup + "<|\"|>}";
    }
    return out;
}

// Mirrors `format_function_declaration` macro (lines 86-117).
std::string format_function_declaration(const ordered_json & tool_data) {
    if (!tool_data.is_object()) return std::string{};
    if (!tool_data.contains("function") || !tool_data["function"].is_object()) {
        return std::string{};
    }
    const ordered_json & fn = tool_data["function"];

    std::string out = "declaration:" + fn.value("name", std::string{}) + "{";
    out += "description:<|\"|>" + fn.value("description", std::string{}) + "<|\"|>";

    if (fn.contains("parameters") && fn["parameters"].is_object()) {
        const ordered_json & params = fn["parameters"];
        out += ",parameters:{";
        if (params.contains("properties") && params["properties"].is_object()) {
            out += "properties:{";
            out += format_parameters(params["properties"],
                                     params.value("required", ordered_json::array()),
                                     0);
            out += "},";
        }
        if (params.contains("required") && params["required"].is_array() && !params["required"].empty()) {
            out += "required:[";
            for (size_t i = 0; i < params["required"].size(); ++i) {
                if (i > 0) out += ",";
                out += "<|\"|>" + params["required"][i].get<std::string>() + "<|\"|>";
            }
            out += "],";
        }
        if (params.contains("type") && params["type"].is_string()) {
            std::string tval = params["type"].get<std::string>();
            for (auto & c : tval) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            out += "type:<|\"|>" + tval + "<|\"|>}";
        }
    }

    if (fn.contains("response") && fn["response"].is_object()) {
        const ordered_json & resp = fn["response"];
        out += ",response:{";
        if (resp.contains("description") && resp["description"].is_string()) {
            out += "description:<|\"|>" + resp["description"].get<std::string>() + "<|\"|>,";
        }
        if (type_eq(resp, "OBJECT")) {
            std::string tval = resp["type"].get<std::string>();
            for (auto & c : tval) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            out += "type:<|\"|>" + tval + "<|\"|>}";
        }
    }

    out += "}";
    return out;
}

// Mirrors `format_tool_response_block` macro (lines 160-173).
std::string format_tool_response_block(const std::string & tool_name, const ordered_json & response) {
    std::string out = "<|tool_response>";
    if (response.is_object()) {
        out += "response:" + tool_name + "{";
        std::vector<std::string> keys;
        for (auto it = response.begin(); it != response.end(); ++it) keys.push_back(it.key());
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size(); ++i) {
            if (i > 0) out += ",";
            out += keys[i] + ":" + format_argument(response[keys[i]], false);
        }
        out += "}";
    } else {
        out += "response:" + tool_name + "{value:" + format_argument(response, false) + "}";
    }
    out += "<tool_response|>";
    return out;
}

}  // namespace

std::string common_chat_gemma4_render(const autoparser::generation_params & inputs,
                                      const std::string & bos_token) {
    std::ostringstream out;

    // 1. BOS token.
    out << bos_token;

    // 2. System block: emitted when enable_thinking, tools, or first message system/developer.
    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    const bool first_is_system = inputs.messages.is_array() && !inputs.messages.empty() &&
        inputs.messages[0].is_object() &&
        (inputs.messages[0].value("role", std::string{}) == "system" ||
         inputs.messages[0].value("role", std::string{}) == "developer");

    enum prev_t { PREV_NONE, PREV_THINK, PREV_TOOL, PREV_TOOL_CALL, PREV_TOOL_RESPONSE, PREV_IMAGE, PREV_AUDIO, PREV_VIDEO };
    prev_t prev_message_type = PREV_NONE;

    ordered_json loop_messages = inputs.messages;
    if (inputs.enable_thinking || has_tools || first_is_system) {
        out << "<|turn>system\n";
        if (inputs.enable_thinking) {
            out << "<|think|>\n";
            prev_message_type = PREV_THINK;
        }
        if (first_is_system) {
            const auto & content = loop_messages[0].value("content", ordered_json{});
            std::string s;
            if (content.is_string()) {
                s = content.get<std::string>();
            }
            out << trim_ws(s);
            loop_messages.erase(0);
        }
        if (has_tools) {
            for (const auto & tool : inputs.tools) {
                out << "<|tool>" << trim_ws(format_function_declaration(tool)) << "<tool|>";
            }
            prev_message_type = PREV_TOOL;
        }
        out << "<turn|>\n";
    }

    // 3. Find last user index for reasoning guard.
    int last_user_idx = -1;
    if (loop_messages.is_array()) {
        for (size_t i = 0; i < loop_messages.size(); ++i) {
            if (loop_messages[i].is_object() &&
                loop_messages[i].value("role", std::string{}) == "user") {
                last_user_idx = static_cast<int>(i);
            }
        }
    }

    // 4. Per-message loop.
    if (!loop_messages.is_array()) {
        loop_messages = ordered_json::array();
    }
    for (size_t i = 0; i < loop_messages.size(); ++i) {
        const auto & message = loop_messages[i];
        const std::string msg_role = message.value("role", std::string{});
        if (msg_role == "tool") {
            // Consumed by the assistant message's forward-scan; skip here.
            continue;
        }
        prev_message_type = PREV_NONE;
        const std::string role = msg_role == "assistant" ? "model" : msg_role;

        // Continuation detection: scan backwards for previous non-tool message.
        std::string prev_role;
        bool prev_found = false;
        if (i > 0) {
            for (int j = static_cast<int>(i) - 1; j >= 0 && !prev_found; --j) {
                if (loop_messages[j].value("role", std::string{}) != "tool") {
                    prev_role = loop_messages[j].value("role", std::string{});
                    prev_found = true;
                }
            }
        }
        const bool continue_same_model_turn = (role == "model" && prev_role == "assistant");
        if (!continue_same_model_turn) {
            out << "<|turn>" << role << "\n";
        }

        // Reasoning emission.
        std::string thinking_text;
        if (message.contains("reasoning") && message["reasoning"].is_string()) {
            thinking_text = message["reasoning"].get<std::string>();
        } else if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
            thinking_text = message["reasoning_content"].get<std::string>();
        }
        const bool has_tool_calls = message.contains("tool_calls") &&
            message["tool_calls"].is_array() && !message["tool_calls"].empty();
        if (!thinking_text.empty() && static_cast<int>(i) > last_user_idx && has_tool_calls) {
            out << "<|channel>thought\n" << thinking_text << "\n<channel|>";
        }

        // Tool calls.
        if (has_tool_calls) {
            for (const auto & tool_call : message["tool_calls"]) {
                if (!tool_call.contains("function") || !tool_call["function"].is_object()) continue;
                const auto & function = tool_call["function"];
                out << "<|tool_call>call:" << function.value("name", std::string{}) << "{";
                const auto & args = function.value("arguments", ordered_json{});
                if (args.is_object()) {
                    bool first_arg = true;
                    std::vector<std::string> keys;
                    for (auto it = args.begin(); it != args.end(); ++it) keys.push_back(it.key());
                    std::sort(keys.begin(), keys.end());
                    for (const auto & key : keys) {
                        if (!first_arg) out << ",";
                        first_arg = false;
                        out << key << ":" << format_argument(args[key], /*escape_keys=*/false);
                    }
                } else if (args.is_string()) {
                    out << args.get<std::string>();
                }
                out << "}<tool_call|>";
            }
            prev_message_type = PREV_TOOL_CALL;
        }

        // Tool responses (forward-scan or legacy `tool_responses`).
        bool tr_out_flag = false;
        if (message.contains("tool_responses") && message["tool_responses"].is_array()) {
            for (const auto & tool_response : message["tool_responses"]) {
                std::string tname = tool_response.value("name", std::string{"unknown"});
                out << format_tool_response_block(tname, tool_response.value("response", ordered_json{}));
                tr_out_flag = true;
                prev_message_type = PREV_TOOL_RESPONSE;
            }
        } else if (has_tool_calls) {
            // Forward-scan consecutive tool messages following this assistant.
            for (size_t k = i + 1; k < loop_messages.size(); ++k) {
                if (loop_messages[k].value("role", std::string{}) != "tool") {
                    break;
                }
                const auto & follow = loop_messages[k];
                std::string tname = follow.value("name", std::string{"unknown"});
                // Resolve tool_call_id to function name when available.
                if (follow.contains("tool_call_id") && follow["tool_call_id"].is_string()) {
                    const std::string id = follow["tool_call_id"].get<std::string>();
                    for (const auto & tc : message["tool_calls"]) {
                        if (tc.contains("id") && tc["id"].is_string() && tc["id"].get<std::string>() == id) {
                            tname = tc["function"].value("name", tname);
                        }
                    }
                }
                const auto & tool_body = follow.value("content", ordered_json{});
                if (tool_body.is_string()) {
                    out << format_tool_response_block(tname, tool_body);
                } else if (tool_body.is_array()) {
                    std::string text;
                    for (const auto & part : tool_body) {
                        if (part.is_object() && part.value("type", std::string{}) == "text") {
                            text += part.value("text", std::string{});
                        }
                    }
                    out << format_tool_response_block(tname, ordered_json(text));
                } else {
                    out << format_tool_response_block(tname, tool_body);
                }
                tr_out_flag = true;
                prev_message_type = PREV_TOOL_RESPONSE;
            }
        }

        // Content rendering.
        if (message.contains("content")) {
            const auto & content = message["content"];
            if (content.is_string()) {
                if (role == "model") {
                    out << strip_thinking(content.get<std::string>());
                } else {
                    out << trim_ws(content.get<std::string>());
                }
            } else if (content.is_array()) {
                for (const auto & item : content) {
                    if (!item.is_object()) continue;
                    const std::string type = item.value("type", std::string{});
                    if (type == "text") {
                        const std::string txt = item.value("text", std::string{});
                        if (role == "model") {
                            out << strip_thinking(txt);
                        } else {
                            out << trim_ws(txt);
                        }
                    } else if (type == "image") {
                        out << "<|image|>";
                        prev_message_type = PREV_IMAGE;
                    } else if (type == "audio") {
                        out << "<|audio|>";
                        prev_message_type = PREV_AUDIO;
                    } else if (type == "video") {
                        out << "<|video|>";
                        prev_message_type = PREV_VIDEO;
                    }
                }
            }
        }

        // End-of-turn marker.
        const bool has_content = message.contains("content") && !message["content"].is_null() &&
            !(message["content"].is_string() && message["content"].get<std::string>().empty()) &&
            !(message["content"].is_array() && message["content"].empty());
        if (prev_message_type == PREV_TOOL_CALL && !tr_out_flag) {
            out << "<|tool_response>";
        } else if (!(tr_out_flag && !has_content)) {
            out << "<turn|>\n";
        }
    }

    // 5. Generation prompt.
    if (inputs.add_generation_prompt &&
        prev_message_type != PREV_TOOL_RESPONSE &&
        prev_message_type != PREV_TOOL_CALL) {
        out << "<|turn>model\n";
        if (!inputs.enable_thinking) {
            out << "<|channel>thought\n<channel|>";
        }
    }

    return out.str();
}
