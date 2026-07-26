#include "chat.h"

#include "chat-render-params.h"
#include "chat-formats/format-pipeline.h"
#include "chat-formats/gemma4-format.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "gbnf-to-lark.h"
#include "gbnf-to-peg.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "lark-to-peg.h"
#include "log.h"

#include "peg-parser.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

// Was in chat-auto-parser-helpers, which went with the auto-parser. Kept here
// because content normalisation is not an auto-parser concern.
static std::string trim_whitespace(const std::string & str) {
    const auto b = str.find_first_not_of(" \t\n\r");
    if (b == std::string::npos) {
        return {};
    }
    const auto e = str.find_last_not_of(" \t\n\r");
    return str.substr(b, e - b + 1);
}

// ---------------------------------------------------------------------------
// Chat grammar registry
// ---------------------------------------------------------------------------
// Grammar files (*.lark, *.gbnf) are read from disk once at startup and cached
// in process memory. Format init functions call common_chat_grammar_get() to
// retrieve the grammar for their format key.
//
// This replaces deriving a grammar from the chat template's text. Format
// selection is by name only -- see docs/fork/POLICIES.md#nothing-is-inferred.
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, std::string> s_chat_grammar_registry;

// Defined below, next to the transpiler it needs. Runs once the registry is
// populated: see the note at its definition for why this is a startup check.
static void common_chat_check_format_contracts();

void common_chat_grammar_init(const std::string & grammars_dir) {
    s_chat_grammar_registry.clear();

    if (grammars_dir.empty()) {
        return;
    }

    std::error_code ec;
    if (!std::filesystem::is_directory(grammars_dir, ec)) {
        LOG_WRN("Chat grammar directory not found: %s\n", grammars_dir.c_str());
        return;
    }

    for (const auto & entry : std::filesystem::directory_iterator(grammars_dir, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto & p   = entry.path();
        const auto   ext = p.extension().string();
        if (ext != ".lark" && ext != ".gbnf") {
            continue;
        }

        std::ifstream f(p);
        if (!f.is_open()) {
            LOG_WRN("Failed to open grammar file: %s\n", p.string().c_str());
            continue;
        }
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        // Key = "<stem><ext>", e.g. "gemma4.lark", "gemma4.gbnf"
        std::string key = p.stem().string() + ext;
        s_chat_grammar_registry[key] = std::move(content);
    }

    LOG_INF("Loaded %zu chat grammar file(s) from %s\n",
            s_chat_grammar_registry.size(), grammars_dir.c_str());

    common_chat_check_format_contracts();
}

std::string common_chat_grammar_get(const std::string & model_key) {
    // Lark when llguidance is compiled in, GBNF otherwise.
#ifdef LLAMA_USE_LLGUIDANCE
    const std::string key = model_key + ".lark";
#else
    const std::string key = model_key + ".gbnf";
#endif
    auto it = s_chat_grammar_registry.find(key);
    if (it == s_chat_grammar_registry.end()) {
        return {};
    }
    return it->second;
}

// Registry lookup that refuses to degrade. An empty grammar is not a mild
// fallback: it means the sampler runs with NO constraint at all, which silently
// defeats the entire point of a grammar-file-driven format. Upstream's failure
// mode here was a single LOG_WRN followed by data.grammar = "" at 30+ call
// sites. Fail loudly instead.
static std::string common_chat_grammar_require(const std::string & model_key) {
    auto grammar = common_chat_grammar_get(model_key);
    if (grammar.empty()) {
        throw std::runtime_error(
            "chat grammar '" + model_key + "' not found in the grammar registry. "
            "Set --chat-grammars-dir to the directory holding the .lark/.gbnf files "
            "(default: " + std::string(DEFAULT_CHAT_GRAMMARS_DIR) + ").");
    }
    return grammar;
}

static std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto               time       = std::chrono::system_clock::to_time_t(now);
    auto               local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    auto res = ss.str();
    return res;
}

static json safe_args_parse(const std::string & to_parse) {
    std::string stripped = to_parse;
    if (to_parse.at(0) == '"' && to_parse.at(to_parse.length() - 1) == '"') {
        stripped = to_parse.substr(1, to_parse.length() - 1);
    }
    try {
        return json::parse(stripped);
    } catch (json::exception & e) {
        return stripped;
    }
}

static std::string string_diff(const std::string & last, const std::string & current) {
    if (last.empty()) {
        return current;
    }
    if (!string_starts_with(current, last)) {
        if (string_starts_with(last, current)) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return "";
        }
        throw std::runtime_error("Invalid diff: '" + last + "' not found at start of '" + current + "'");
    }
    return current.substr(last.size());
}

static bool has_content_or_tool_calls(const common_chat_msg & msg) {
    return !msg.content.empty() || !msg.tool_calls.empty();
}

std::string common_chat_msg::render_content(const std::string & delimiter) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    if (!content.empty()) {
        return content;
    }

    std::string text;
    for (const auto & part : content_parts) {
        if (part.type == "text") {
            if (!text.empty()) {
                text += delimiter;
            }
            text += part.text;
        }
    }
    return text;
}

common_chat_role common_chat_role_from_string(const std::string & role) {
    if (role == "system")    { return COMMON_CHAT_ROLE_SYSTEM;    }
    if (role == "assistant") { return COMMON_CHAT_ROLE_ASSISTANT; }
    if (role == "user")      { return COMMON_CHAT_ROLE_USER;      }
    if (role == "tool")      { return COMMON_CHAT_ROLE_TOOL;      }
    return COMMON_CHAT_ROLE_UNKNOWN;
}

const char * common_chat_role_to_string(common_chat_role role) {
    switch (role) {
        case COMMON_CHAT_ROLE_SYSTEM:    return "system";
        case COMMON_CHAT_ROLE_ASSISTANT: return "assistant";
        case COMMON_CHAT_ROLE_USER:      return "user";
        case COMMON_CHAT_ROLE_TOOL:      return "tool";
        case COMMON_CHAT_ROLE_UNKNOWN:   return "";
    }
    return "";
}

json common_chat_msg_delimiters::to_json() const {
    json result = json::array();
    for (const auto & d : delimiters) {
        result.push_back({
            { "role",      common_chat_role_to_string(d.role) },
            { "delimiter", d.delimiter                        },
        });
    }
    return result;
}

common_chat_msg_delimiters common_chat_msg_delimiters_parse(const json & delimiters) {
    common_chat_msg_delimiters result;

    if (!delimiters.is_array()) {
        return result;
    }

    result.delimiters.reserve(delimiters.size());
    for (const auto & d : delimiters) {
        if (!d.is_object()) {
            continue;
        }
        result.delimiters.push_back({
            common_chat_role_from_string(d.value("role", std::string())),
            d.value("delimiter", std::string()),
        });
    }

    return result;
}

void common_chat_msg_delimiters::tokenize(const llama_vocab * vocab) {
    for (auto & d : delimiters) {
        d.tokens = common_tokenize(vocab, d.delimiter, false, true);
    }
}

common_chat_msg_spans common_chat_msg_delimiters::split(const llama_tokens & tokens, const std::map<size_t, size_t> & skips) const {
    std::vector<std::pair<common_chat_role, size_t>> matches;

    auto skip = skips.begin();
    for (size_t i = 0; i < tokens.size();) {
        if (skip != skips.end() && i == skip->first) {
            i += skip->second;
            ++skip;
            continue;
        }
        for (const auto & d : delimiters) {
            if (i + d.tokens.size() > tokens.size()) {
                continue;
            }
            if (std::equal(d.tokens.begin(), d.tokens.end(), tokens.begin() + i)) {
                matches.emplace_back(d.role, i);
                break;
            }
        }
        i++;
    }

    matches.emplace_back(COMMON_CHAT_ROLE_UNKNOWN, tokens.size());

    common_chat_msg_spans spans;
    for (size_t i = 0; i + 1 < matches.size(); i++) {
        const auto & curr = matches[i];
        const auto & next = matches[i + 1];
        spans.add(curr.first, curr.second, next.second - curr.second);
    }

    return spans;
}

json common_chat_msg::to_json_oaicompat(bool concat_typed_text) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    json jmsg {
        {"role", role},
    };
    if (!content.empty()) {
        jmsg["content"] = content;
    } else if (!content_parts.empty()) {
        if (concat_typed_text || contains_media()) {
            std::string text;
            bool last_was_media_marker = false;
            // join parts with newline, do not add newline before or after media markers
            for (const auto & part : content_parts) {
                bool add_new_line = true;
                if (part.type == "text") {
                    add_new_line = !last_was_media_marker && !text.empty();
                    last_was_media_marker = false;
                } else if (part.type == "media_marker") {
                    add_new_line = false;
                    last_was_media_marker = true;
                } else {
                    LOG_WRN("Ignoring content part type: %s\n", part.type.c_str());
                    continue;
                }

                if (add_new_line) {
                    text += '\n';
                }

                text += part.text;
            }
            jmsg["content"] = text;
        } else {
            auto & parts = jmsg["content"] = json::array();
            for (const auto & part : content_parts) {
                parts.push_back({
                    {"type", part.type},
                    {"text", part.text},
                });
            }
        }
    } else {
        jmsg["content"] = "";
    }
    if (!reasoning_content.empty()) {
        jmsg["reasoning_content"] = reasoning_content;
    }
    if (!tool_name.empty()) {
        jmsg["name"] = tool_name;
    }
    if (!tool_call_id.empty()) {
        jmsg["tool_call_id"] = tool_call_id;
    }
    if (!tool_calls.empty()) {
        jmsg["tool_calls"] = json::array();
        auto & jtool_calls = jmsg["tool_calls"];
        for (const auto & tool_call : tool_calls) {
            json tc {
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", json(tool_call.arguments)},
                }},
            };
            if (!tool_call.id.empty()) {
                tc["id"] = tool_call.id;
            }
            // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
            // We only generate a random id for the ones that don't generate one by themselves
            // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
            // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            jtool_calls.push_back(tc);
        }
    }

    return jmsg;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & msg_prv,
                                                                      const common_chat_msg & msg_new) {
    std::vector<common_chat_msg_diff> diffs;
    if (msg_new.tool_calls.size() > msg_prv.tool_calls.size()) {
        diffs.reserve(msg_new.tool_calls.size() - msg_prv.tool_calls.size() + 3);
    } else {
        diffs.reserve(3);
    }

    // TODO: these can become expensive for long messages - how to optimize?
    if (msg_prv.reasoning_content != msg_new.reasoning_content) {
        auto & diff                  = diffs.emplace_back();
        diff.reasoning_content_delta = string_diff(msg_prv.reasoning_content, msg_new.reasoning_content);
    }
    if (msg_prv.content != msg_new.content) {
        auto & diff        = diffs.emplace_back();
        diff.content_delta = string_diff(msg_prv.content, msg_new.content);
    }

    if (msg_new.tool_calls.size() < msg_prv.tool_calls.size()) {
        std::string err = "Invalid diff: now finding less tool calls!\n";
        err += "  Previous (" + std::to_string(msg_prv.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_prv.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current (" + std::to_string(msg_new.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_new.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current msg text content:\n" + msg_new.content + "\n";
        throw std::runtime_error(err);
    }

    if (!msg_prv.tool_calls.empty()) {
        const auto   idx  = msg_prv.tool_calls.size() - 1;
        const auto & pref = msg_prv.tool_calls[idx];
        const auto & newf = msg_new.tool_calls[idx];
        // Allow tool name to change during incremental parsing:
        // - empty -> non-empty (initial discovery)
        // - prefix -> longer string (name grows as more input is parsed)
        if (pref.name != newf.name && !pref.name.empty() && !newf.name.empty()) {
            // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
            bool is_prefix = (newf.name.rfind(pref.name, 0) == 0);
            if (!is_prefix) {
                LOG_ERR("Tool call mismatch: prev='%s' new='%s'\n", pref.name.c_str(), newf.name.c_str());
                throw std::runtime_error("Invalid diff: tool call mismatch!");
            }
        }
        const auto args_diff = string_diff(pref.arguments, newf.arguments);
        if (!args_diff.empty() || pref.id != newf.id || pref.name != newf.name) {
            auto & diff          = diffs.emplace_back();
            diff.tool_call_index = idx;
            if (pref.id != newf.id || pref.name != newf.name) {
                diff.tool_call_delta.id   = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
        }
    }
    for (size_t idx = msg_prv.tool_calls.size(); idx < msg_new.tool_calls.size(); ++idx) {
        auto & diff          = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = msg_new.tool_calls[idx];
    }

    return diffs;
}


struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template;  // Model had builtin template or template overridden was specified.
    // The format plugin this model resolved to at load: --chat-format if given,
    // else the model's declared general.architecture. A request may still name a
    // different one, which is why this is a default rather than the answer.
    std::string chat_format;
    std::unique_ptr<common_chat_template> template_default;  // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

std::string common_chat_templates_format(const struct common_chat_templates * tmpls) {
    return tmpls == nullptr ? std::string{} : tmpls->chat_format;
}

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice) {
    if (tool_choice == "auto") {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    if (tool_choice == "none") {
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    if (tool_choice == "required") {
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    }
    throw std::invalid_argument("Invalid tool_choice: " + tool_choice);
}

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates) {
    common_chat_templates_inputs inputs;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = "test";
    inputs.messages = { msg };
    inputs.enable_thinking = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    auto params = common_chat_templates_apply(chat_templates, inputs);
    return params.supports_thinking;
}

std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {
        if (!messages.is_array()) {
            throw std::invalid_argument("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::invalid_argument("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::invalid_argument("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            auto has_content    = message.contains("content");
            auto has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::invalid_argument("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text" && type != "media_marker") {
                            throw std::invalid_argument("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::invalid_argument("Invalid 'content' type: expected string or array, got " +
                                                content.dump() +
                                                " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::invalid_argument("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::invalid_argument("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::invalid_argument("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::invalid_argument("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name           = fc.at("name");
                    const auto & args = fc.at("arguments");
                    if (args.is_string()) {
                        tc.arguments = args;
                    } else {
                        tc.arguments = args.dump();
                    }
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id");
                    }
                    msg.tool_calls.push_back(tc);
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::invalid_argument(
                    "Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & "
                    "https://github.com/ggml-org/llama.cpp/issues/12279)");
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name");
            }
            if (message.contains("tool_call_id")) {
                msg.tool_call_id = message.at("tool_call_id");
            }

            msgs.push_back(msg);
        }
    } catch (const std::exception & e) {
        // @ngxson : disable otherwise it's bloating the API response
        // printf("%s\n", std::string("; messages = ") + messages.dump(2));
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()));
    }

    return msgs;
}

static json render_message_to_json(const std::vector<common_chat_msg> & msgs, const chat_template_caps & c) {
    if (!c.supports_string_content && !c.supports_typed_content) {
        LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\n", __func__);
    }

    bool only_string_accepted =  c.supports_string_content && !c.supports_typed_content;
    bool only_typed_accepted  = !c.supports_string_content &&  c.supports_typed_content;

    json messages = json::array();
    for (const auto & msg : msgs) {
        if (only_string_accepted) {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ true);
            messages.push_back(jmsg);
        } else if (only_typed_accepted) {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ false);
            if (jmsg.at("content").is_string()) {
                jmsg["content"] = json::array({
                    json{
                        {"type", "text"},
                        {"text", jmsg.at("content").get<std::string>()},
                    }
                });
            }
            messages.push_back(jmsg);
        } else {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ false);
            messages.push_back(jmsg);
        }
    }
    return messages;
}

// DEPRECATED: only used in tests
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    chat_template_caps c;
    c.supports_string_content = true;
    c.supports_typed_content = !concat_typed_text;
    return render_message_to_json(msgs, c);
}

json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            { "type",     "function" },
            { "function", {
                { "name", tool.name },
                { "description", tool.description },
                { "parameters", json::parse(tool.parameters) },
            }},
        });
    }
    return result;
}

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::invalid_argument("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::invalid_argument("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::invalid_argument("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::invalid_argument("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.value("description", ""),
                    /* .parameters = */ function.value("parameters", json::object()).dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

common_chat_continuation common_chat_continuation_parse(const nlohmann::ordered_json & value) {
    if (value.is_boolean() && value.get<bool>()) {
        return COMMON_CHAT_CONTINUATION_AUTO;
    }
    if (value.is_string()) {
        auto value_str = value.get<std::string>();
        if (value_str == "reasoning_content") {
            return COMMON_CHAT_CONTINUATION_REASONING;
        }
        if (value_str == "content") {
            return COMMON_CHAT_CONTINUATION_CONTENT;
        }
    }
    return COMMON_CHAT_CONTINUATION_NONE;
}

// common_chat_verify_template was here. It applied a candidate template to a
// one-message conversation and reported whether that threw -- the validation
// behind --chat-template. Both flags that fed it now reject outright, so there
// is no candidate template to verify.
//
// Note the shape it had: an unconditional `return` inside a bare block, followed
// by unreachable llama_chat_apply_template code. Half of it had already stopped
// running before this fork touched it.

std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass) {
    common_chat_templates_inputs inputs;
    inputs.add_bos   = tmpls->add_bos;
    inputs.add_eos   = tmpls->add_eos;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages              = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg                 = common_chat_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg             = common_chat_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.add_bos              = tmpls->add_bos;
    inputs.add_eos              = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;
    auto add_simple_msg         = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role    = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system", "You are a helpful assistant");
    add_simple_msg("user", "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user", "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

#define CHATML_TEMPLATE_SRC                                                               \
    "{%- for message in messages -%}\n"                                                   \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n"                                                                    \
    "{%- if add_generation_prompt -%}\n"                                                  \
    "  {{- '<|im_start|>assistant\n' -}}\n"                                               \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

bool common_chat_templates_was_explicit(const struct common_chat_templates * tmpls) {
    return tmpls->has_explicit_template;
}

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates) {
    (void) chat_templates;
    // The LFM2 branch that used to be here sniffed the template text for
    // `<|tool_list_start|>` to swap in a different ASR prompt. LFM2 is not a
    // format this build serves, and the probe was the last of its kind outside
    // the two template patches below -- all of them reading a template's TEXT to
    // decide behaviour, which is what --chat-format replaced.
    common_chat_prompt_preset asr_preset;
    asr_preset.system = "";
    asr_preset.user   = "Transcribe audio to text";
    return asr_preset;
}

std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant) {
    if (!variant.empty()) {
        if (variant == "tool_use") {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source();
            }
            return "";
        }
        LOG_DBG("%s: unknown template variant: %s\n", __func__, variant.c_str());
    }
    return tmpls->template_default->source();
}

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override,
                                                     const std::string &        eos_token_override,
                                                     const std::string &        chat_format_override) {
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    // No model AND no template override is now a normal case: a caller can name
    // a chat_format and never touch a template at all, which is what the format
    // tests do. It used to assert, because a template was the only way to say
    // anything about the format.
    if (chat_template_override.empty() && model != nullptr) {
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src  = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }

    // Two upstream patches lived here, rewriting a GPT-OSS template's `<|channel|>`
    // test and a Mistral template's `[TOOL_CALLS]` guard so that Minja would not
    // throw on them. Both edited the template SOURCE, by searching it for the
    // text that broke. Nothing renders a template in this build, so there is
    // nothing to keep working around: `src` survives only as an opaque string
    // reported by /props.

    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    bool        add_bos   = false;
    bool        add_eos   = false;
    if (model) {
        const auto * vocab     = llama_model_get_vocab(model);
        // The warning used to fire only when the template's TEXT mentioned
        // `bos_token` / `eos_token` -- the last of the src.find() probes, and the
        // subtlest, because it read as a sensible relevance check rather than as
        // inference. A renderer that never sees a template cannot be asked what
        // it references, and Gemma 4's renderer takes the BOS token directly, so
        // a missing one matters whatever any template says.
        const auto   get_token = [&](llama_token token, const char * name) {
            if (token == LLAMA_TOKEN_NULL) {
                LOG_WRN("common_chat_templates_init: warning: vocab has no %s token; the renderer "
                        "will not work as intended.\n",
                        name);
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS");
        add_bos   = llama_vocab_get_add_bos(vocab);
        add_eos   = llama_vocab_get_add_eos(vocab);
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    tmpls->add_bos               = add_bos;
    tmpls->add_eos               = add_eos;

    // Resolve the format plugin ONCE, here, and say where the answer came from.
    //
    // Logged rather than left implicit because format selection has been the
    // single most confusing thing about this code path: it used to be inferred
    // from template text, so "which format am I getting" had no answer you could
    // read. Now there is one line at load and it names its source.
    //
    // Throws when the name has no plugin, which stops the server. That is the
    // point: a model this build cannot serve should fail at load naming the
    // architecture it declared, not half-work.
    {
        std::string source;
        tmpls->chat_format = common_chat_format_resolve(model, chat_format_override, &source);
        if (tmpls->chat_format.empty()) {
            LOG_DBG("%s", "chat format unresolved (no --chat-format and no declared architecture)\n");
        } else {
            LOG_INF("chat format '%s' from %s\n", tmpls->chat_format.c_str(), source.c_str());
        }
    }
    try {
        tmpls->template_default = std::make_unique<common_chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        LOG_ERR("%s: failed to initialize chat template\n", __func__);
        throw e;
    }
    if (!template_tool_use_src.empty()) {
        try {
            tmpls->template_tool_use = std::make_unique<common_chat_template>(template_tool_use_src, token_bos, token_eos);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, e.what());
        }
    }
    return tmpls;
}

const char * common_chat_format_name(common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            return "Content-only";
        case COMMON_CHAT_FORMAT_PEG_SIMPLE:
            return "peg-simple";
        case COMMON_CHAT_FORMAT_PEG_NATIVE:
            return "peg-native";
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return "peg-gemma4";
        default:
            throw std::runtime_error("Unknown chat format");
    }
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    switch (format) {
        case COMMON_REASONING_FORMAT_NONE:
            return "none";
        case COMMON_REASONING_FORMAT_AUTO:
            return "auto";
        case COMMON_REASONING_FORMAT_DEEPSEEK:
            return "deepseek";
        case COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY:
            return "deepseek-legacy";
        default:
            throw std::runtime_error("Unknown reasoning format");
    }
}

common_reasoning_format common_reasoning_format_from_name(const std::string & format) {
    if (format == "none") {
        return COMMON_REASONING_FORMAT_NONE;
    }
    if (format == "auto") {
        return COMMON_REASONING_FORMAT_AUTO;
    }
    if (format == "deepseek") {
        return COMMON_REASONING_FORMAT_DEEPSEEK;
    }
    if (format == "deepseek-legacy") {
        return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    }
    throw std::runtime_error("Unknown reasoning format: " + format);
}

static void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

static void foreach_parameter(const json &                                                         function,
                              const std::function<void(const std::string &, const json &, bool)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    const auto & params = function.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto &          props = params.at("properties");
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        params.at("required").get_to(required);
    }
    for (const auto & [name, prop] : props.items()) {
        bool is_required = (required.find(name) != required.end());
        fn(name, prop, is_required);
    }
}





// ──────────────────────────────────────────────────────────────────────────────
// Grammar-file-driven chat parsing
// ──────────────────────────────────────────────────────────────────────────────
// A format's grammar lives in a file (grammars/chat/<key>.lark or .gbnf) rather
// than in hand-written PEG builder lambdas. The same grammar drives BOTH
// sampling (llguidance masks tokens from it) and extraction (it is transpiled to
// a PEG arena), so the two cannot drift apart.
//
// Placeholders in the grammar template are filled per request:
//   {{TOOL_SCHEMA}}     -> the tool argument schemas
//   {{RESPONSE_SCHEMA}} -> the request's response_format schema
// ──────────────────────────────────────────────────────────────────────────────

static bool is_lark_grammar(const std::string & grammar) {
    return grammar.find("%llguidance") != std::string::npos;
}

// --- Gemma 4 tool-argument constraint ----------------------------------------
//
// Gemma 4's argument syntax is NOT JSON: bare keys, `<|"|>`-delimited strings,
// `{city:<|"|>London<|"|>}`. So `%json` emits the wrong language entirely and
// cannot be dropped in, which is what upstream's
//   TODO @aldehir: need to extend json-schema-to-grammar to produce more than
//   JSON rules
// is about, and why ten other model handlers constrain their tool arguments and
// this one did not. Left unconstrained the model may emit any argument at all;
// measured adversarially, 1 call in 25 conformed.
//
// Anything the emitter cannot express falls back to the unconstrained dict for
// THAT tool only. A fallback is not silent: it is per-tool, so one exotic schema
// cannot quietly widen the others.
static std::string gemma4_value_rule(const json & schema, int depth);

static std::string gemma4_string_literal(const std::string & s) {
    // A Lark string literal. Escape only what the syntax requires.
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; }
        out += c;
    }
    return out + "\"";
}

// A quoted Gemma 4 string: <|"|> ... <|"|>. Both delimiters are the same token,
// and the format defines no escape for it -- see ARCHITECTURE.md.
static std::string gemma4_quoted(const std::string & body) {
    return "(str_delim_tag " + body + " str_delim_tag)";
}

static std::string gemma4_value_rule(const json & schema, int depth) {
    // Bound the recursion; a self-referential schema would otherwise not
    // terminate. Beyond it, accept the generic value.
    if (depth > 8 || !schema.is_object()) { return "gemma4_value"; }

    // An enum is the tightest thing we can emit: a literal alternation.
    if (schema.contains("enum") && schema.at("enum").is_array() && !schema.at("enum").empty()) {
        std::vector<std::string> alts;
        for (const auto & e : schema.at("enum")) {
            if (e.is_string()) {
                alts.push_back(gemma4_quoted(gemma4_string_literal(e.get<std::string>())));
            } else if (e.is_number() || e.is_boolean()) {
                alts.push_back(gemma4_string_literal(e.dump()));
            } else {
                return "gemma4_value";
            }
        }
        return "(" + string_join(alts, " | ") + ")";
    }

    const std::string type = schema.value("type", std::string{});

    if (type == "string")  { return gemma4_quoted("gemma4_str_content"); }
    if (type == "integer" || type == "number") { return "GEMMA4_NUMBER"; }
    if (type == "boolean") { return "GEMMA4_BOOL"; }
    if (type == "null")    { return "\"null\""; }

    if (type == "array") {
        const json items = schema.contains("items") ? schema.at("items") : json::object();
        const std::string item = gemma4_value_rule(items, depth + 1);
        return "(\"[\" (" + item + " (\",\" " + item + ")*)? \"]\")";
    }

    if (type == "object") {
        if (!schema.contains("properties") || !schema.at("properties").is_object()) {
            return "gemma4_dict";
        }
        std::vector<std::string> required;
        for (const auto & r : schema.value("required", json::array())) {
            if (r.is_string()) { required.push_back(r.get<std::string>()); }
        }
        // Emit properties in DECLARED order. Required ones are mandatory, the
        // rest optional, each preceded by its separating comma so the comma
        // disappears with the property it belongs to. This is why the order is
        // fixed rather than a permutation: a permutation of N optionals is N!
        // alternatives, which is not a grammar anyone wants to compile.
        std::vector<std::string> parts;
        bool first = true;
        for (const auto & [key, prop] : schema.at("properties").items()) {
            const bool is_required = std::find(required.begin(), required.end(), key) != required.end();
            const std::string kv   = gemma4_string_literal(key) + " \":\" " + gemma4_value_rule(prop, depth + 1);
            if (is_required) {
                parts.push_back(first ? kv : "\",\" " + kv);
                first = false;
            } else {
                parts.push_back(first ? "(" + kv + ")?" : "(\",\" " + kv + ")?");
            }
        }
        if (parts.empty()) { return "(\"{\" \"}\")"; }
        return "(\"{\" " + string_join(parts, " ") + " \"}\")";
    }

    return "gemma4_value";
}

// One alternative per declared tool, pairing the tool's NAME LITERAL with that
// tool's own argument shape.
//
// The correlation is the point. `func_name` was a free identifier and the schema
// a `oneOf` across every tool, so a call could name tool A and carry tool B's
// arguments and still be well-formed. Pairing them makes that unrepresentable.
//
// It also retires the FUNC_NAME charset limit: names come from literals, so a
// dotted or dashed MCP name like `filesystem.read_file` needs nothing special.
//
// With no tool declared there is nothing to correlate against, so the shape is
// the format's own: any identifier, any dict. The dict is optional here for the
// same reason it is optional per-tool below.
static const char * const gemma4_tool_call_any = "tool_call_directive \":\" func_name gemma4_dict?";

static std::string build_gemma4_tool_schema(const json & tools) {
    if (!tools.is_array() || tools.empty()) {
        return gemma4_tool_call_any;
    }
    std::vector<std::string> alts;
    foreach_function(tools, [&](const json & tool) {
        const auto & function = tool.at("function");
        const auto   name     = function.value("name", std::string{});
        if (name.empty()) { return; }
        const json params = function.contains("parameters") && function.at("parameters").is_object()
            ? function.at("parameters")
            : json::object();

        // Arguments are a DICT or nothing; a bare scalar is not a call. So a
        // schema the emitter cannot read falls back to `gemma4_dict`, not to
        // gemma4_value_rule's generic any-value -- which would have let a tool
        // whose `parameters` omits `"type": "object"` be called with a string.
        std::string args = gemma4_value_rule(params, 0);
        if (args == "gemma4_value") { args = "gemma4_dict"; }

        // S1 is `functionCall: CALL COLON ID object?`, so `call:name` with the
        // braces omitted is a well-formed zero-argument call -- and zero-argument
        // tools are the commonest kind an agent declares.
        //
        // Generation gets that freedom exactly when an EMPTY object would also be
        // legal, i.e. the tool declares no required property. Omitting the braces
        // therefore cannot become a way around a required argument, and the
        // writer's own `{}` spelling stays available either way.
        const json required = params.value("required", json::array());
        if (!required.is_array() || required.empty()) {
            args = "(" + args + ")?";
        }

        alts.push_back("(tool_call_directive \":\" " + gemma4_string_literal(name) + " " + args + ")");
    });
    if (alts.empty()) {
        return gemma4_tool_call_any;
    }
    return "(" + string_join(alts, " | ") + ")";
}

static std::string build_lark_tool_schema(const json & tools) {
    if (!tools.is_array() || tools.empty()) {
        return "%json {}";
    }
    json one_of = json::array();
    foreach_function(tools, [&](const json & tool) {
        const auto & function = tool.at("function");
        if (function.contains("parameters") && function.at("parameters").is_object()) {
            one_of.push_back(function.at("parameters"));
        } else {
            one_of.push_back(json::object());
        }
    });
    if (one_of.size() == 1) {
        return "%json " + one_of[0].dump();
    }
    return "%json {\"oneOf\": " + one_of.dump() + "}";
}

// GBNF equivalent: generate a JSON-schema-constrained rule per tool and join
// them as alternatives. Same oneOf-style correlation defect as the Lark path.
static std::string build_gbnf_tool_schema(const json & tools) {
    if (!tools.is_array() || tools.empty()) {
        return "([^]*)";
    }
    std::vector<std::string> schemas;
    foreach_function(tools, [&](const json & tool) {
        const auto & function = tool.at("function");
        auto schema = function.contains("parameters") ? function.at("parameters") : json::object();
        std::string gbnf = build_grammar([&](const common_grammar_builder & builder) {
            builder.resolve_refs(schema);
            builder.add_schema("root", schema);
        });
        // The generated GBNF starts with 'root ::= ...'; take the body only.
        for (auto & line : string_split<std::string>(gbnf, '\n')) {
            if (line.substr(0, 9) == "root ::= ") {
                schemas.push_back("(" + line.substr(9) + ")");
                break;
            }
        }
    });
    if (schemas.empty()) {
        return "([^]*)";
    }
    if (schemas.size() == 1) {
        return schemas[0];
    }
    std::string result = schemas[0];
    for (size_t i = 1; i < schemas.size(); i++) {
        result += " | " + schemas[i];
    }
    return result;
}

// Replace EVERY occurrence, not just the first. Chat grammars repeat the same
// placeholder in several rules (once for the tool-call body, once inside an
// optional response-format alternative); a single-substring replace would leave
// the second occurrence literal in the rendered grammar.
static std::string replace_all(const std::string & str,
                               const std::string & placeholder,
                               const std::string & replacement) {
    std::string result;
    result.reserve(str.size());
    size_t start = 0;
    size_t pos;
    while ((pos = str.find(placeholder, start)) != std::string::npos) {
        result.append(str, start, pos - start);
        result.append(replacement);
        start = pos + placeholder.size();
    }
    result.append(str, start, str.size() - start);
    return result;
}

static std::string inject_tool_schema(const std::string & grammar_template, const json & tools) {
    const std::string placeholder = "{{TOOL_SCHEMA}}";
    if (grammar_template.find(placeholder) == std::string::npos) {
        return grammar_template;
    }
    // Gemma 4's argument syntax is not JSON, so its Lark grammar gets the
    // Gemma-dict emitter rather than `%json`. build_lark_tool_schema stays for
    // grammars whose arguments really are JSON.
    std::string schema = is_lark_grammar(grammar_template)
        ? build_gemma4_tool_schema(tools)
        : build_gbnf_tool_schema(tools);
    return replace_all(grammar_template, placeholder, schema);
}

// The caller's own grammar, as a SUBGRAMMAR reference.
//
// `@response_grammar` names the second entry of the llguidance grammar list this
// request compiles to (see gemma4_compose_grammars below). Referencing rather
// than pasting is what makes a caller rule named `content` or `start` harmless:
// llguidance gives each grammar in the list its own lexeme class, so nothing has
// to be renamed and nothing can collide.
//
// With no caller grammar the production is still substituted -- with the format's
// own free-text `content` -- because the placeholder has to become valid Lark
// either way. Leaving `{{USER_GRAMMAR}}` in the text was the exact shape of an
// earlier bug: llguidance rejected the grammar and, rejecting it, constrained
// nothing at all.
static std::string inject_user_grammar(const std::string & grammar_template, bool have_user_grammar) {
    return replace_all(grammar_template, "{{USER_GRAMMAR}}",
                       have_user_grammar ? "@response_grammar" : "content");
}

// Whether a turn may open a tool call at all.
//
// This is a property of the REQUEST -- did it declare tools, and did it allow
// them to be used -- so it cannot live in the grammar file as a fixed
// alternative. With tools the placeholder becomes `tool_call_request |`; without
// them it becomes nothing, and `<|tool_call>` leaves the mask entirely.
//
// Unconditional was a hole: `{{TOOL_SCHEMA}}` degrades to any-name/any-args when
// no tools are declared, so a plain chat request could emit a call to a function
// nobody offered. The trailing `|` belongs to the placeholder because the empty
// substitution has to remove the alternative, not leave an empty one behind --
// `( | x)` matches the empty string and would make the whole group optional.
static std::string inject_tool_call_alt(const std::string & grammar_template, bool tools_available) {
    return replace_all(grammar_template, "{{TOOL_CALL_ALT}}",
                       tools_available ? "tool_call_request |" : "");
}

// Additional thought channels after the first.
//
// EMPTY for sampling: the documented ordering has one thought channel per model
// turn, and permitting more was an unbounded loop -- `content` matches empty, so
// `<|channel>thought<channel|>` with nothing in it was legal and repeatable
// forever. Measured live, once the tool branch stopped being available as an
// escape hatch, the model fell straight into it and ran to max_tokens.
//
// PRESENT for extraction, because the two directions answer different questions:
// what may the model emit, versus what might it have emitted. A prefilled or
// replayed turn can carry a second channel, and a parser that refuses it drops
// the thought into content instead -- caught by test-chat, which is what this
// split exists for.
static std::string inject_extra_channels(const std::string & grammar_template, bool allow) {
    return replace_all(grammar_template, "{{EXTRA_CHANNELS}}",
                       allow ? "(channel_block content)*" : "");
}

// One llguidance grammar LIST: the format grammar, plus the caller's under the
// name the format grammar references.
//
// llguidance takes a list via its "llguidance" constraint type and lets an entry
// reference another by name (docs/syntax.md, "Multiple grammars"); nested Lark
// grammars are explicitly not supported yet, so this is the sanctioned way to
// compose two Lark grammars. ONE matcher results, and therefore one mask -- the
// point of §1.7: two samplers over one token stream intersect their languages,
// and an empty intersection is every logit -INF with no diagnostic.
static std::string gemma4_compose_grammars(const std::string & format_grammar,
                                           const std::string & user_lark) {
    if (user_lark.empty()) {
        return format_grammar;
    }
    json grammars = json::array();
    grammars.push_back({ { "lark_grammar", format_grammar } });
    grammars.push_back({ { "name", "response_grammar" }, { "lark_grammar", user_lark } });
    return json{ { "grammars", grammars } }.dump();
}

static std::string inject_response_schema(const std::string & grammar_template, const json & json_schema) {
    const std::string placeholder = "{{RESPONSE_SCHEMA}}";
    if (grammar_template.find(placeholder) == std::string::npos) {
        return grammar_template;
    }
    std::string schema;
    if (is_lark_grammar(grammar_template)) {
        // BOUND THE WHITESPACE. llguidance's %json defaults to
        // `whitespace_pattern: [\x20\x0A\x0D\x09]+` -- unbounded -- so between any
        // two JSON tokens the model may emit newlines forever and the grammar is
        // satisfied at every single step. It is the same shape as every runaway
        // in gemma4.lark, one layer down and not ours.
        //
        // Measured on E2B, thinking off, a two-property schema, N=100: 12 of 100
        // generations ran to max_tokens, every one of them like
        //
        //   {\n  "canonical": "writer of Dune"\n  \n  \n  \n  \n  ... (512 tokens)
        //
        // stuck after the first value, never reaching the comma. That is 12% of
        // the calls Protean's Canonicalizer and CacheGate make.
        //
        // COMPACT, not merely bounded, and that was measured rather than chosen.
        // Bounding first looked right -- let the model indent, forbid only the
        // loop -- so `whitespace_pattern: "[ \n\r\t]{0,16}"` was tried first and
        // did nothing: 9/50, against 12/100 for the default. A capped whitespace
        // lexeme can still match the EMPTY string, and matching empty over and
        // over is the same runaway with extra steps. `whitespace_flexible: false`
        // removes the lexeme instead, which is why it works:
        //
        //     default (unbounded)                12/100 failed, avg 88 tokens
        //     whitespace_pattern {0,16}           9/50  failed, avg 117
        //     whitespace_flexible: false          1/50  failed, avg 29
        //
        // The cost is that responses come back compact rather than pretty. That
        // is a formatting difference in a machine-read field, and it also more
        // than halves the tokens spent getting there.
        //
        // Set only when the caller did not ask for something else -- their own
        // x-guidance wins.
        auto s = json_schema;
        if (s.is_object() && !s.contains("x-guidance")) {
            s["x-guidance"] = json{ { "whitespace_flexible", false } };
        }
        schema = "%json " + s.dump();
    } else {
        std::string gbnf = build_grammar([&](const common_grammar_builder & builder) {
            auto s = json_schema;
            builder.resolve_refs(s);
            builder.add_schema("root", s);
        });
        for (auto & line : string_split<std::string>(gbnf, '\n')) {
            if (line.substr(0, 9) == "root ::= ") {
                schema = line.substr(9);
                break;
            }
        }
        if (schema.empty()) {
            schema = "([^]*)";
        }
    }
    return replace_all(grammar_template, placeholder, schema);
}

// Transpile a grammar template to a PEG arena for EXTRACTION.
//
// The placeholders become generic JSON matchers here on purpose: the precise
// schema constraint was already enforced at sampling time by llguidance/GBNF,
// and re-imposing it during extraction would reject output the sampler had
// legitimately produced.
static common_peg_arena chat_grammar_to_peg(const std::string & grammar_template,
                                           const std::string & root_rule = "turn_start") {
    std::string base = grammar_template;
    const bool  lark = is_lark_grammar(base);

    // Extraction gets the UNCONSTRAINED tool-call shape, never the request's
    // schema. The schema was already enforced at sampling time; re-imposing it
    // here would reject output the sampler had legitimately produced, and would
    // make parsing depend on which tools a request happened to declare.
    // The argument object is optional on the PARSE side because S1 is
    // `functionCall: CALL COLON ID object?` -- `call:name` with no braces is a
    // well-formed zero-argument call, and zero-argument tools are the commonest
    // kind an agent declares. Requiring the braces here dropped such a call on
    // the floor entirely rather than reporting it with `{}` for arguments.
    base = replace_all(base, "{{TOOL_SCHEMA}}",
                       lark ? "tool_call_directive \":\" func_name gemma4_dict?" : "([^]*)");
    base = replace_all(base, "{{RESPONSE_SCHEMA}}", lark ? "__JSON_VALUE__"  : "([^]*)");
    // Likewise the caller's grammar: extraction reads back what the sampler
    // produced, and the sampler already required it to match. Re-imposing it
    // here could only reject legitimate output -- and the subgrammar reference
    // it is replaced with means nothing to the PEG transpiler anyway.
    base = replace_all(base, "{{USER_GRAMMAR}}", lark ? "content" : "([^]*)");
    // Extraction ALWAYS keeps the tool-call alternative, whatever the request
    // declared. It reads back what the sampler produced, and a call the model
    // made has to be parseable -- withholding the branch here would drop it on
    // the floor instead. Same reasoning as withholding the tool schema.
    base = replace_all(base, "{{TOOL_CALL_ALT}}", lark ? "tool_call_request |" : "");
    base = replace_all(base, "{{EXTRA_CHANNELS}}", lark ? "(channel_block content)*" : "");

    return lark ? common_lark_to_peg(base, root_rule) : common_gbnf_to_peg(base);
}

// Every registered format's FSM<->grammar contract, checked once at startup.
//
// ARCHITECTURE.md states the contract as binding -- every state reachable in a
// per-format FSM corresponds to a named rule in that format's Lark grammar, and
// the C++ tracker mirrors llguidance's rule path. It has been a convention until
// now, and a convention cannot catch drift: IN_TOOL_ARGS sat in the registry
// naming a rule the tracker never entered, which is precisely what the registry
// existed to prevent.
//
// STARTUP, not per request, because the failure is silent in the worst possible
// way. The generation entry is composed as `start: <rule>`; a `<rule>` llguidance
// cannot resolve makes it fail OPEN, and a server that cannot constrain its
// output must not come up pretending it can.
//
// Checked against the grammar as PARSED, not as text: a name that appears only
// inside another rule's body is not a definition, and a text scan cannot tell
// the two apart.
static void common_chat_check_format_contracts() {
    const auto grammar = common_chat_grammar_get("gemma4");
    if (grammar.empty()) {
        // No grammar loaded is a separate failure, reported by
        // common_chat_grammar_require at the point of use with a message that
        // says what to set. Nothing to check here.
        return;
    }
    // Any root that exists will do -- has_rule() looks at the whole rule table,
    // not the entry. `conversation` is the widest and reaches every production.
    common_chat_gemma4_check_state_rule_contract(chat_grammar_to_peg(grammar, "conversation"));
    LOG_DBG("%s", "gemma4: FSM state<->grammar rule contract OK\n");
}

// Append `start: <root_rule>` so the chosen production becomes the entry rule.
//
// The sampler cannot be told where to begin. llguidance compiles the rule
// literally named "start" and offers no way to choose another --
// `let start_name = "start";` (parser/src/lark/compiler.rs:612), with %override
// explicitly rejected (:705) -- and GBNF has the same constraint on "root". So
// entry selection is structural.
//
// The grammar FILES therefore define no entry rule at all: they are libraries of
// productions, and the entry is composed here, per request. That is why this can
// be an append rather than surgery on the grammar text. An earlier version of
// this function scanned for the `start:` definition and spliced its body,
// continuation lines and all -- string manipulation of a grammar source, which
// is exactly the sort of thing this fork exists to remove.
//
// Keeping sampling and extraction on the same entry is the point. Extraction
// takes its root as a parameter; the sampling grammar was always compiled from
// `start`. Resuming inside a prefilled thought, that is wrong in both
// directions: it permits a fresh `<|channel>thought` opener (a second thought
// inside the first) and permits dropping into content (the first thought never
// closed). The model was free to emit either, and only extraction would have
// noticed.
std::string common_chat_grammar_set_entry(const std::string & grammar, const std::string & root_rule) {
    const bool lark = is_lark_grammar(grammar);

    // Lark rule names use '_', GBNF uses '-'. Callers name rules the Lark way;
    // this is the same normalization lark-to-peg applies.
    std::string root = root_rule;
    if (!lark) {
        for (char & c : root) {
            if (c == '_') { c = '-'; }
        }
    }

    return grammar + (lark ? "\nstart: " : "\nroot ::= ") + root + "\n";
}

static common_chat_params common_chat_params_init_gemma4(const common_chat_template &    tmpl,
                                                         const common_chat_render_params & inputs) {
    common_chat_params data;

    // Rendered in C++, not by Jinja. The renderer, the validator (the
    // `conversation` rule), the tracker FSM and the grammar are one plugin and
    // must agree; a Jinja template is a fifth artefact none of them can check.
    // See docs/fork/ARCHITECTURE.md.
    // The format plugin owns the entire conversation shape -- input, output,
    // decisions and parse tree -- as one FSM. It reports the prompt and its
    // generation tail; nothing here reconstructs, appends to, or subtracts from
    // either. Every previous attempt to do so from out here fought the renderer
    // and lost: three sequential blocks each rewrote generation_prompt, one
    // invalidated the next one's end-of-turn test, and a double-render diff
    // could not isolate the tail under continuation.
    const auto rendered    = common_chat_gemma4_render(inputs, tmpl.bos_token());
    data.chat_prompt       = { rendered.prompt, rendered.entry_state };
    data.prompt            = data.chat_prompt.text;


    // Hand extraction the prompt and a parser rooted at the `conversation` rule.
    // The SAME pipeline walks the prompt and then the generation, so validation
    // of the input and establishment of the FSM state are one traversal -- and
    // whatever the open assistant turn already contains lands in the output
    // message on the way through, rather than being summarised here.
    {
        const auto conv_grammar = common_chat_grammar_require("gemma4");
        data.prompt_validator    = common_chat_prompt_validator(
            chat_grammar_to_peg(conv_grammar, "conversation"));
    }

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_USER,      "<|turn>user"  },
        { COMMON_CHAT_ROLE_ASSISTANT, "<|turn>model" },
    };

    data.format            = COMMON_CHAT_FORMAT_PEG_GEMMA4;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<|channel>thought";
    data.thinking_end_tag   = "<channel|>";

    // Every marker the model can emit. The server converts each generated token
    // with `special = params_base.special || preserved_tokens.count(tok)`, and
    // llama_vocab::token_to_piece renders nothing for a CONTROL token when
    // `special` is false -- so an unlisted CONTROL marker never reaches the text
    // the parser is given, and the final non-partial parse fails on the missing
    // literal.
    //
    // Measured: dropping `<|tool_response>` from this list does NOT visibly break
    // tool calls, because server-task.cpp only replaces the accumulated message
    // `if (!new_msg.empty())` and the streaming partial parses have already built
    // it. So this is not what makes extraction work -- it is what makes the parse
    // SUCCEED rather than fail into that fallback. Do not read the fallback as
    // permission to leave a marker out.
    //
    // Which markers are affected cannot be read off the spelling. Gemma 4 splits
    // them across two attribute classes, and llama.cpp then FORCES CONTROL onto
    // any token in its hardcoded gemma4 EOG list (`<eos>`, `<turn|>`,
    // `<|tool_response>`), so the class is partly llama.cpp's doing and not the
    // publisher's. Listing every marker the grammar can emit removes the
    // dependency on which class each one ended up in.
    data.preserved_tokens = {
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<|\"|>",
        "<|turn>",
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    // A caller-supplied grammar, from `response_format: {"type": "lark_grammar"}`
    // / `{"type": "gbnf_grammar"}` or the legacy top-level `grammar` field.
    //
    // Normalized to Lark HERE rather than at the request boundary so every
    // caller of this function gets the same treatment -- the server, the CLI's
    // --grammar, and the tests. Which of the two syntaxes a string is follows
    // the convention the sampler has always used for this field (`%llguidance`
    // opens a Lark grammar); that is a declared marker in the text, not a guess
    // about what the grammar looks like.
    //
    // A malformed grammar throws, and the throw becomes a request error. That is
    // the whole point of converting eagerly: the alternative is llguidance
    // refusing the grammar at sampling time, which it expresses by constraining
    // NOTHING.
    std::string user_lark;
    if (!inputs.grammar.empty()) {
        user_lark = is_lark_grammar(inputs.grammar) ? inputs.grammar
                                                    : common_gbnf_to_lark(inputs.grammar);
    }
    const bool has_user_grammar = !user_lark.empty();

    // The grammar comes from grammars/chat/gemma4.lark (or .gbnf), not from
    // hand-written PEG builder lambdas. One artefact drives BOTH sampling and
    // extraction, so the two cannot drift apart -- which is the whole point of
    // the FSM<->grammar contract in docs/fork/ARCHITECTURE.md.
    //
    // grammar_lazy is false and there are no triggers: the format grammar covers
    // the entire output from token 1 (content, reasoning, tool calls), rather
    // than only switching on once a "<|tool_call>" trigger word is seen. Lazy
    // triggering leaves everything before the trigger unconstrained.
    {
        // ONE grammar, for sampling and for extraction, whatever the caller wants
        // done with reasoning. There used to be a second file selected here when
        // reasoning_format was NONE; it described the same wire format and had
        // drifted badly. Whether a thought is surfaced as reasoning or folded
        // back into content is a presentation choice and lives in the
        // transformer -- see common_chat_gemma4_transformer::shape.
        const auto parser_base   = common_chat_grammar_require("gemma4");

        // Generation starts where the prompt left the model, so the entry rule
        // follows entry_state. Resuming inside a prefilled thought means the
        // model continues mid-`reasoning`, with the `<|channel>thought` opener
        // already in the prompt; `start` expects that opener.
        //
        // entry_state was computed by the renderer and plumbed through two structs
        // but read by nothing -- the same dead-plumbing shape that let the FSM
        // itself run as unused code.
        //
        // `tool_choice: "required"` selects the variant of the same entry whose
        // turn cannot end without a call. Only meaningful with tools declared:
        // "required" with an empty tool list would otherwise compose a grammar
        // demanding a call to nothing, which is unsatisfiable and would strand
        // the sampler with every token masked.
        //
        // A requested response_format is likewise a demand, not a suggestion: its
        // branch used to be one alternative in the turn, so a schema could be
        // compiled into the grammar and then simply not taken. Measured on the
        // 12B, a two-property schema with additionalProperties:false came back
        // with three properties of the model's own choosing.
        //
        // tool_choice wins when both are set, because a turn cannot be required
        // to both call a tool and emit a schema block, and the tool call is the
        // more specific instruction.
        //
        // A caller-supplied grammar is the same kind of demand and sits in the
        // same precedence order, below tool_choice and above the JSON schema:
        // asking for BOTH a schema and a grammar is contradictory, and the
        // grammar is the more specific of the two.
        //
        // Note this is precedence between ANSWER SHAPES only. Declaring tools no
        // longer competes with either: every entry carries the tool-call
        // alternative when the request offered tools (inject_tool_call_alt), so
        // "call one of these, or else answer like THIS" is one grammar rather
        // than a choice between two demands. Only tool_choice=required still
        // wins outright, because it removes the "or else" by definition.
        auto demand = COMMON_CHAT_GEMMA4_ENTRY_ANY;
        if (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            demand = COMMON_CHAT_GEMMA4_ENTRY_TOOL_CALL;
        } else if (has_user_grammar) {
            demand = COMMON_CHAT_GEMMA4_ENTRY_USER_GRAMMAR;
        } else if (has_response_format) {
            demand = COMMON_CHAT_GEMMA4_ENTRY_RESPONSE_FORMAT;
        }
        const std::string gen_root =
            common_chat_gemma4_entry_root(rendered.entry_state, demand);

        // BOTH sides enter at the same rule. Extraction takes it as a parameter;
        // the sampler cannot be told, so the grammar is rewritten to enter there
        // (see common_chat_grammar_set_entry). Skipping this on the sampling side is
        // what left the model unconstrained at a resume point -- free to open a
        // second thought inside the first, or to drop into content leaving the
        // first unclosed -- with only extraction any the wiser.
        // The response-schema placeholder is substituted ALWAYS, not only when the
        // caller asked for a response_format.
        //
        // It was conditional, which meant that on every ordinary request the
        // literal text `{{RESPONSE_SCHEMA}}` was handed to llguidance, where it is
        // not valid Lark -- the grammar failed to compile and the request sampled
        // with no format constraint at all. Measured, before this line changed:
        //   llg error: 173(19): Expected value (at "{" ('{'))
        // Invisible to the suite because test-chat never compiles Lark grammars;
        // test-grammar-llguidance's gemma4 case is what caught it and is what
        // keeps it caught.
        //
        // With no schema requested, an empty JSON Schema is used: it accepts any
        // JSON value, so a fenced ```json block stays samplable without
        // constraining its contents. Dropping the alternative instead would leave
        // the model unable to emit ``` at all, since `content` excludes it
        // precisely so `response_format` can match.
        std::string sampling_grammar = inject_response_schema(
            common_chat_grammar_set_entry(parser_base, gen_root),
            has_response_format ? inputs.json_schema : json::object());

        // The declared tools' schemas, as productions inside the tool-call scope.
        // Once the model has opened a call it can only name a declared tool, and
        // having named one it can only emit that tool's arguments in that tool's
        // shape. With no tools declared this is the unconstrained form, so a
        // request without tools is unaffected.
        sampling_grammar = inject_tool_schema(sampling_grammar, inputs.tools);
        sampling_grammar = inject_user_grammar(sampling_grammar, has_user_grammar);
        // The tool-call branch exists only for a request that offered tools AND
        // allowed them to be called. `tool_choice: "none"` means exactly "do not
        // call", so it drops the branch the same way declaring no tools does --
        // enforced in the grammar rather than asked for in a prompt.
        sampling_grammar = inject_tool_call_alt(
            sampling_grammar, has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
        sampling_grammar = inject_extra_channels(sampling_grammar, /* allow = */ false);
        // Last, because it stops being a Lark grammar here and becomes an
        // llguidance grammar LIST. Every substitution above operates on Lark
        // text and would have to be JSON-aware otherwise.
        sampling_grammar = gemma4_compose_grammars(sampling_grammar, user_lark);

        // Extraction enters at the PERMISSIVE root even when sampling was pinned
        // to the tool-required one. The required variant's language is a strict
        // subset -- it only removes the "no call" alternative -- so the permissive
        // root accepts everything the sampler could have produced, and re-imposing
        // the narrower one could only reject output that was legitimately
        // generated. Same reasoning as withholding the tool schema from the parser.
        const std::string parse_root =
            common_chat_gemma4_entry_root(rendered.entry_state, COMMON_CHAT_GEMMA4_ENTRY_ANY);

        data.grammar             = sampling_grammar;
        data.parser              = chat_grammar_to_peg(parser_base, parse_root).save();
        data.grammar_file_parser = true;
        data.grammar_lazy        = false;
        data.grammar_triggers    = {};
        data.reasoning_format    = inputs.reasoning_format;

        (void) include_grammar;  // grammar is always present for this format now
    }

    return data;
}

// The DeepSeek V4 reference implementation renders consecutive tool results into a single
// user block, ordered by the tool call order of the preceding assistant message (matched
// by tool call id) rather than by the order they appear in the conversation.
static json deepseek_v4_sort_tool_results(const json & messages) {
    json adjusted = messages;
    std::map<std::string, size_t> call_order;

    for (size_t i = 0; i < adjusted.size();) {
        const auto & msg  = adjusted[i];
        const auto   role = msg.value("role", "");

        if (role == "assistant" && msg.contains("tool_calls") &&
                msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty()) {
            call_order.clear();
            const auto & tool_calls = msg.at("tool_calls");
            for (size_t idx = 0; idx < tool_calls.size(); idx++) {
                auto id = tool_calls[idx].value("id", "");
                if (!id.empty()) {
                    call_order[id] = idx;
                }
            }
            i++;
            continue;
        }

        if (role != "user" && role != "tool") {
            i++;
            continue;
        }

        // collect a maximal run of user/tool messages - they render into one user block
        std::vector<size_t> tool_positions;
        size_t run_end = i;
        for (; run_end < adjusted.size(); run_end++) {
            const auto r = adjusted[run_end].value("role", "");
            if (r == "tool") {
                tool_positions.push_back(run_end);
            } else if (r != "user") {
                break;
            }
        }

        if (tool_positions.size() > 1 && !call_order.empty()) {
            std::vector<json> results;
            results.reserve(tool_positions.size());
            for (auto pos : tool_positions) {
                results.push_back(adjusted[pos]);
            }
            std::stable_sort(results.begin(), results.end(), [&](const json & a, const json & b) {
                const auto order = [&](const json & m) {
                    auto it = call_order.find(m.value("tool_call_id", ""));
                    return it == call_order.end() ? (size_t) 0 : it->second;
                };
                return order(a) < order(b);
            });
            for (size_t k = 0; k < tool_positions.size(); k++) {
                adjusted[tool_positions[k]] = std::move(results[k]);
            }
        }

        i = run_end;
    }

    return adjusted;
}

namespace workaround {

static void map_developer_role_to_system(json & messages) {
    for (auto & message : messages) {
        if (message.contains("role")) {
            if (message["role"] == "developer") {
                message["role"] = "system";
            }
        }
    }
}


// if first message is system and template does not support it, merge it with next message
static void system_message_not_supported(json & messages) {
    if (!messages.empty() && messages.front().at("role") == "system") {
        if (messages.size() > 1) {
            LOG_DBG("Merging system prompt into next message\n");
            auto & first_msg = messages.front();
            auto & second_msg = messages[1];
            second_msg["content"] = first_msg.at("content").get<std::string>()
                + "\n" + second_msg.at("content").get<std::string>();
            messages.erase(messages.begin());
        } else {
            LOG_WRN("Removing system prompt due to template not supporting system role\n");
            messages.erase(messages.begin());
        }
    }
}

static void requires_non_null_content(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls") && !message.contains("content")) {
            message["content"] = "";
        }
    }
}

// Gemma4 uses a custom tool_responses field instead of role:tool messages.
//
// This will transform a sequence of messages:
//   assistant(tool_call+) -> tool+ -> assistant(content)
//
// Into a single assistant message containing a tool_responses field:
//   assistant(content + tool_call + tool_responses)
//
// This is necessary for the Gemma4 chat template to properly format the prompt.
// See https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
struct gemma4_model_turn_builder {
    json & messages;
    size_t pos;
    json tool_calls = json::array();
    json tool_responses = json::array();
    json content;
    json reasoning_content;

    gemma4_model_turn_builder(json & msgs, size_t pos) : messages(msgs), pos(pos) {}

    void collect() {
        // Collect the first assistant message
        auto & msg = messages[pos];
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            // According to the prompt formatting guide, we need to preserve reasoning_content
            // between function calls. The current chat templates do not support this, but we will do it anyway.
            reasoning_content = msg.at("reasoning_content");
        }
        for (auto & tc : msg.at("tool_calls")) {
            tool_calls.push_back(tc);
        }
        pos++;

        // Collect tool call results
        while (pos < messages.size() && messages[pos].value("role", "") == "tool") {
            collect_result(messages[pos]);
            pos++;
        }

        // Check if the next assistant message is the final message
        if (pos < messages.size() && messages[pos].value("role", "") == "assistant") {
            auto & next = messages[pos];
            if (!has_tool_calls(next) && has_content(next)) {
                content = next.at("content");
                pos++;
            }
        }
    }

    void collect_result(const json & curr) {
        json response;
        if (curr.contains("content")) {
            const auto & content = curr.at("content");
            if (content.is_string()) {
                // Try to parse the content as JSON; fall back to raw string
                try {
                    response = json::parse(content.get<std::string>());
                } catch (...) {
                    response = content;
                }
            } else {
                response = content;
            }
        }

        std::string name;

        // Match name with corresponding tool call
        size_t idx = tool_responses.size();
        if (idx < tool_calls.size()) {
            auto & tc = tool_calls[idx];
            if (tc.contains("function")) {
                name = tc.at("function").value("name", "");
            }
        }

        // Fallback to the tool call id
        if (name.empty()) {
            name = curr.value("tool_call_id", "");
        }

        tool_responses.push_back({{"name", name}, {"response", response}});
    }

    json build() {
        collect();

        json msg = {
            {"role", "assistant"},
            {"tool_calls", tool_calls},
        };
        if (!tool_responses.empty()) {
            msg["tool_responses"] = tool_responses;
        }
        if (!content.is_null()) {
            msg["content"] = content;
        }
        if (!reasoning_content.is_null()) {
            msg["reasoning_content"] = reasoning_content;
        }
        return msg;
    }

    static bool has_content(const json & msg) {
        if (!msg.contains("content") || msg.at("content").is_null()) {
            return false;
        }
        const auto & content = msg.at("content");
        if (content.is_string() && !content.get<std::string>().empty()) {
            return true;
        }
        if (content.is_array() && !content.empty()) {
            return true;
        }
        return false;
    }

    static bool has_tool_calls(const json & msg) {
        return msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty();
    }
};

static void convert_tool_responses_gemma4(json & messages) {
    json result = json::array();
    size_t i = 0;

    while (i < messages.size()) {
        auto & msg = messages[i];

        if (msg.value("role", "") != "assistant" || !msg.contains("tool_calls") ||
            !msg.at("tool_calls").is_array() || msg.at("tool_calls").empty()) {
            result.push_back(msg);
            i++;
            continue;
        }

        gemma4_model_turn_builder builder(messages, i);
        result.push_back(builder.build());
        i = builder.pos;
    }

    messages = result;
}

static void func_args_not_string(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls")) {
            for (auto & tool_call : message["tool_calls"]) {
                if (tool_call.contains("function") && tool_call["function"].contains("arguments")) {
                    auto & args = tool_call["function"]["arguments"];
                    if (args.is_string()) {
                        try {
                            args = json::parse(args.get<std::string>());
                        } catch (const std::exception & e) {
                            throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(e.what()));
                        }
                    }
                }
            }
        }
    }
}

// Trim leading/trailing whitespace from message contents before rendering. This
// has to run on the messages (not on the rendered JSON) because templates with
// string-only content caps concatenate typed content parts into a single string
// during rendering, after which the per-part whitespace can no longer be reached.
// Both the plain string content and the text of typed content parts are trimmed.
static void trim_all_content(std::vector<common_chat_msg> & messages) {
    for (auto & message : messages) {
        message.content           = trim_whitespace(message.content);
        message.reasoning_content = trim_whitespace(message.reasoning_content);
        for (auto & part : message.content_parts) {
            if (part.type == "text") {
                part.text = trim_whitespace(part.text);
            }
        }
    }
}

}

static json common_chat_extra_context() {
    json ctx = json::object();
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::string datetime_str = format_time(now, "%b %d %Y");
    std::string date_str = format_time(now, "%d %b %Y");
    ctx["datetime"] = datetime_str;
    ctx["date_string"] = date_str;
    return ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// The format-plugin registry
// ─────────────────────────────────────────────────────────────────────────────
//
// One entry per format, keyed by the name `chat_format` / `--chat-format` /
// `general.architecture` all speak. Adding a format means implementing its four
// pieces -- renderer, validator, tracker, grammar -- and adding a line here;
// that nothing else has to change is the test of whether the interface is right.
//
// This replaced `common_chat_try_specialized_template`, a chain of
// `src.find("...")` probes over the chat template's TEXT. Editing a template
// silently changed how a model was parsed, which is the opposite of what a
// format declaration should do.
using common_chat_format_init_fn =
    common_chat_params (*)(const common_chat_template &, const common_chat_render_params &);

static const std::map<std::string, common_chat_format_init_fn> & chat_format_registry() {
    static const std::map<std::string, common_chat_format_init_fn> registry = {
        { "gemma4", common_chat_params_init_gemma4 },
    };
    return registry;
}

std::vector<std::string> common_chat_format_names() {
    std::vector<std::string> names;
    for (const auto & [name, fn] : chat_format_registry()) {
        (void) fn;
        names.push_back(name);
    }
    return names;
}

bool common_chat_format_is_registered(const std::string & name) {
    return chat_format_registry().count(name) != 0;
}

static std::string chat_format_known_list() {
    return string_join(common_chat_format_names(), ", ");
}

std::string common_chat_format_resolve(const struct llama_model * model,
                                       const std::string &        config_override,
                                       std::string *              source_out) {
    std::string name   = config_override;
    std::string source = "--chat-format";

    if (name.empty()) {
        // Declared GGUF metadata. `general.architecture` IS the plugin name for
        // Gemma 4 (measured: the 12B and E2B GGUFs both declare `gemma4`), which
        // is what lets a stock OpenAI client talk to an unconfigured model with
        // no extra parameters -- §1.0 rather than a shortcut.
        source = "GGUF general.architecture";
        if (model != nullptr) {
            char buf[128];
            const int32_t n = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
            if (n > 0) {
                name.assign(buf, (size_t) n);
            }
        }
    }

    if (name.empty()) {
        // Nothing to resolve FROM -- no override and no model (or a model
        // declaring no architecture). Not an error here: callers that never
        // apply a template, such as common_chat_verify_template, legitimately
        // reach this. The error belongs where a format is actually needed, and
        // common_chat_templates_apply raises it there with the same advice.
        if (source_out != nullptr) {
            *source_out = "unresolved";
        }
        return {};
    }
    if (!common_chat_format_is_registered(name)) {
        throw std::invalid_argument("chat format '" + name + "' (from " + source +
                                    ") has no registered plugin. Known formats: " +
                                    chat_format_known_list());
    }
    if (source_out != nullptr) {
        *source_out = source;
    }
    return name;
}

static common_chat_params common_chat_templates_apply_impl(const struct common_chat_templates *        tmpls,
                                                            const struct common_chat_templates_inputs & inputs) {
    common_chat_render_params params;
    params.tools = common_chat_tools_to_json_oaicompat(inputs.tools);
    const auto & tmpl =
        params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
    // No `src` here any more: nothing in this function reads the template's TEXT.
    // That is the whole of the Jinja removal in one line -- rendering, format
    // selection and every per-family workaround used to start by searching it.
    const auto & caps            = tmpl.original_caps();
    // A StepFun workaround stood here, trimming message contents when the
    // template's text contained "You have access to the following functions in
    // JSONSchema format". StepFun is not a format this build serves, and the
    // probe was template-text inference like the rest.
    params.messages              = render_message_to_json(inputs.messages, tmpl.original_caps());
    params.tool_choice           = inputs.tool_choice;
    params.reasoning_format      = inputs.reasoning_format;
    params.enable_thinking       = inputs.enable_thinking;
    params.preserve_thinking     = inputs.preserve_thinking;
    params.grammar               = inputs.grammar;
    params.now                   = inputs.now;
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.add_bos               = tmpls->add_bos;
    params.add_eos               = tmpls->add_eos;

    params.continue_final_message = inputs.continue_final_message;
    if (params.continue_final_message != COMMON_CHAT_CONTINUATION_NONE) {
        params.add_generation_prompt = false;

        if (!inputs.messages.empty()) {
            // Render messages[:-1] and store continuation message separately
            params.continue_msg = inputs.messages.back();
            params.messages.erase(params.messages.size() - 1);
        }

        if (params.continue_final_message == COMMON_CHAT_CONTINUATION_AUTO && !inputs.messages.empty()) {
            // Resolve based on message content
            params.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
            if (!params.continue_msg.reasoning_content.empty() &&
                params.continue_msg.content.empty() &&
                params.continue_msg.content_parts.empty()) {
                params.continue_final_message = COMMON_CHAT_CONTINUATION_REASONING;
            }
        }
    }

    // Unconditional now. This was gated on the template text NOT containing
    // `<|channel|>`, i.e. "every model except GPT-OSS" expressed as a substring
    // search. Gemma 4 has a real system turn and no `developer` role, so the
    // mapping always applies.
    workaround::map_developer_role_to_system(params.messages);

    if (!tmpl.original_caps().supports_system_role) {
        workaround::system_message_not_supported(params.messages);
    }

    if (tmpl.original_caps().supports_tool_calls) {
        // some templates will require the content field in tool call messages
        // to still be non-null, this puts an empty string everywhere where the
        // content field is null
        workaround::requires_non_null_content(params.messages);
    }

    if (tmpl.original_caps().supports_object_arguments) {
        workaround::func_args_not_string(params.messages);
    }

    params.extra_context = common_chat_extra_context();
    for (auto el : inputs.chat_template_kwargs) {
        params.extra_context[el.first] = json::parse(el.second);
    }

    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }

    params.parallel_tool_calls = inputs.parallel_tool_calls;

    if (params.tools.is_array()) {
        // The second of the two "no grammar with tools" guards, one layer below
        // the server's. Both are gone for the same reason: the caller's grammar
        // is composed into the format grammar as a production, so there are not
        // two samplers to intersect. See gemma4_compose_grammars, and the note
        // in oaicompat_chat_params_parse.
        //
        // Removing only the outer one left this reachable, and it answered a
        // valid request with a 500 -- found live, not by the suite, because no
        // test sent both.
        if (caps.supports_tool_calls && !caps.supports_tools) {
            LOG_WRN(
                "Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
                "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    if (inputs.force_pure_content) {
        LOG_WRN("Forcing pure content template, will not render reasoning or tools separately.");
        // Create the result structure
        common_chat_params data;
        auto params_copy               = params;
        params_copy.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
        // Same rule as the main path: the plugin reports both values, nothing
        // here derives one from the other.
        const auto rendered            = common_chat_gemma4_render(params_copy, tmpl.bos_token());
        data.chat_prompt               = { rendered.prompt, rendered.entry_state };
        data.prompt                    = data.chat_prompt.text;
        data.format                    = COMMON_CHAT_FORMAT_PEG_NATIVE;
        auto parser                    = build_chat_peg_parser([&data](common_chat_peg_builder &p) {
            return p.literal(data.generation_prompt) << p.content(p.rest());
        });
        data.parser                    = parser.save();
        return data;
    }

    // Dispatch by NAME, through the registry.
    //
    // The request may name a format; otherwise the model's resolved one is used
    // -- which came from --chat-format, or from declared GGUF metadata. Three
    // sources, in that order, and no fourth.
    //
    // What was here before the registry was the "differential autoparser": it
    // applied the Jinja template several times with varied inputs, diffed the
    // outputs, and inferred a grammar from where they differed. Inference twice
    // over -- from a template this fork no longer renders, into a grammar nobody
    // wrote. A format gets a hand-written grammar plus a tracker that mirrors
    // it, or it is not supported. See docs/fork/POLICIES.md#nothing-is-inferred.
    const std::string format = inputs.chat_format.empty() ? tmpls->chat_format : inputs.chat_format;
    if (format.empty()) {
        throw std::invalid_argument(
            "no chat format resolved for this model. Set --chat-format, or send "
            "\"chat_format\" on the request. Known formats: " + chat_format_known_list());
    }
    const auto it = chat_format_registry().find(format);
    if (it == chat_format_registry().end()) {
        // std::invalid_argument so the server answers 400: a `chat_format` the
        // CALLER named is a bad request, not a server fault.
        throw std::invalid_argument("chat format '" + format + "' has no registered plugin. Known formats: " +
                                    chat_format_known_list());
    }
    return it->second(tmpl, params);
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(const struct common_chat_templates *        tmpls,
                                                             const struct common_chat_templates_inputs & inputs) {
    size_t                          alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string>        contents;

    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text" && part.type != "media_marker") {
                LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";
                ;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg     = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({ msg.role.c_str(), content.c_str() });
        size_t msg_size = msg.role.size() + content.size();
        alloc_size += msg_size + (msg_size / 4);  // == msg_size * 1.25 but avoiding float ops
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t      res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt,
                                                 buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("unsupported chat template; this build serves Gemma 4 only (see FORK.md)");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(),
                                        buf.size());
    }

    // for safety, we check the result again
    if (res < 0 || (size_t) res > buf.size()) {
        throw std::runtime_error("failed to render the chat prompt (see FORK.md)");
    }

    common_chat_params params;
    params.prompt = std::string(buf.data(), res);
    if (!inputs.json_schema.empty()) {
        params.grammar = json_schema_to_grammar(json::parse(inputs.json_schema));
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}

common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                               const struct common_chat_templates_inputs & inputs) {
    GGML_ASSERT(tmpls != nullptr);
    // One path. There is no template engine to select between any more; the
    // format plugin's renderer builds the prompt. See FORK.md.
    return common_chat_templates_apply_impl(tmpls, inputs);
}

common_chat_msg common_chat_parse(const std::string &               input,
                                  bool                              is_partial,
                                  const common_chat_parser_params & params) {
    return common_chat_peg_parse(params.parser, input, is_partial, params);
}

common_chat_msg common_chat_peg_parse(const common_peg_arena &          src_parser,
                                      const std::string &               input,
                                      bool                              is_partial,
                                      const common_chat_parser_params & params) {
    const common_peg_arena & parser = src_parser.empty() ?
        build_chat_peg_parser([](common_chat_peg_builder & p) { return p.content(p.rest()) + p.end(); }) :
        src_parser;

    if (src_parser.empty()) {
        LOG_DBG("No parser definition detected, assuming pure content parser.");
    }

    const std::string effective_input = params.generation_prompt.empty()
        ? input
        : params.generation_prompt + input;


    //LOG_DBG("Parsing PEG input with format %s: %s\n", common_chat_format_name(params.format), effective_input.c_str());

    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_LENIENT;
    if (params.debug) {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
    }

    common_peg_parse_context ctx(effective_input, flags);
    auto result = parser.parse(ctx);

    // Extraction goes through the format's own pipeline -- tracker, decoder,
    // transformer, presenter -- seeded with the state the prompt left the model
    // in. The legacy mapper is the fallback for formats with no pipeline.
    //
    // This dispatch was lost during the re-derivation onto b10121: the pipeline
    // files were ported but nothing called the factory, so the FSM this fork is
    // built around was dead code and extraction silently ran on the mapper.
    auto extract_message = [&](common_chat_msg & msg) {
        auto pipeline = common_chat_make_format_pipeline(params.format, msg, is_partial, params.reasoning_format);
        if (pipeline.valid()) {
            // ONE tracker across BOTH parses. The pipeline owns the tracker,
            // decoder, transformer and presenter, so running it over the prompt
            // and then over the generation is a single continuous walk: the
            // tracker carries its state forward, and anything the open assistant
            // turn already contains lands in `msg` on the way through.
            //
            // This is what removes the last derived artefact. Nothing summarises
            // the entry state or the prefilled content for the FSM any more --
            // it walks the conversation and knows.
            // Input has its own type and its own parser; how a prompt is
            // validated and walked belongs there, not inline in the middle of
            // parsing model OUTPUT. See chat-formats/prompt.h.
            params.prompt_validator.walk(params.chat_prompt, pipeline, msg, params.debug);
            pipeline.run(ctx.ast, result);
            return;
        }
        // No fallback. A format either has a pipeline or is not served.
        //
        // The legacy mapper used to sit here, and it is exactly why the FSM ran
        // as dead code without anyone noticing: when the pipeline dispatch went
        // missing, extraction quietly kept working on the mapper instead of
        // failing. Removing it makes that class of silent bypass impossible.
        throw std::runtime_error(
            std::string("no extraction pipeline for chat format ") +
            common_chat_format_name(params.format) + " (see FORK.md)");
    };

    if (result.fail()) {
        // During partial parsing, return partial results if any AST nodes were captured
        // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
        if (is_partial && result.end > 0) {
            // Try to extract any partial results from what was successfully parsed
            common_chat_msg msg;
            msg.role = "assistant";
            extract_message(msg);

            if (ctx.is_debug()) {
                fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", ctx.ast.dump().c_str());
                fflush(stderr);
            }
            return msg;
        }
        LOG_WRN("%s: unparsed %s output: %s\n", __func__, common_chat_format_name(params.format), effective_input.substr(result.end).c_str());
        LOG_DBG("%s: full %s output triggering error:\n=== BEGIN ===\n%s\n=== END ===\n", __func__, common_chat_format_name(params.format), effective_input.c_str());
        throw std::runtime_error(std::string("The model produced output that does not match the expected ") + common_chat_format_name(params.format) + " format");
    }

    common_chat_msg msg;
    msg.role = "assistant";

    extract_message(msg);

    if (ctx.is_debug()) {
        fprintf(stderr, "\nAST for %s parse:\n%s\n", is_partial ? "partial" : "full", ctx.ast.dump().c_str());
        fflush(stderr);
    }

    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat({ msg }).at(0).dump().c_str());
    }
    return msg;
}

std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates) {
    GGML_ASSERT(chat_templates != nullptr);
    GGML_ASSERT(chat_templates->template_default != nullptr);
    if (chat_templates->template_tool_use != nullptr) {
        // take the more expressive template when available
        return chat_templates->template_tool_use->caps.to_map();
    }
    return chat_templates->template_default->caps.to_map();
}
