// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/prompt.h"
#include "peg-parser.h"

#include "nlohmann/json_fwd.hpp"

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct common_chat_templates;

namespace autoparser {
struct generation_params;
}  // namespace autoparser

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const common_chat_tool_call & other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;

    // TODO @ngxson : no known chat templates support reasoning_content in content parts yet
    //                this can be useful for models with interleaved thinking (like Kimi-K2)
    //                if you see any templates explicitly support this, please ping me
    // std::string reasoning_content;

    bool operator==(const common_chat_msg_content_part & other) const {
        return type == other.type && text == other.text;
    }
};

// Capabilities of a chat format.
//
// These used to be DERIVED by executing the Jinja template and inspecting which
// branches were reachable (jinja::caps). That made model behaviour a property of
// a template's text, so editing the template silently changed what the server
// reported it could do. They are now stated facts about the format plugin.
//
// The defaults below are Gemma 4's, which is the only format this build serves:
// it has a real system turn, native tool declarations and tool calls, object-
// valued arguments, and accepts both string and typed content parts.
struct chat_template_caps {
    bool supports_system_role      = true;
    bool supports_tools            = true;
    bool supports_tool_calls       = true;
    bool supports_object_arguments = true;
    bool supports_string_content   = true;
    bool supports_typed_content    = true;
    // Gemma 4 keeps thoughts across an active tool-calling turn (Google's Rule 2),
    // so preserve_reasoning is supported. See docs/fork/ARCHITECTURE.md.
    bool supports_preserve_reasoning = true;

    std::map<std::string, bool> to_map() const {
        return {
            { "supports_system_role",      supports_system_role      },
            { "supports_tools",            supports_tools            },
            { "supports_tool_calls",       supports_tool_calls       },
            { "supports_object_arguments", supports_object_arguments },
            { "supports_string_content",   supports_string_content   },
            { "supports_typed_content",    supports_typed_content    },
            { "supports_preserve_reasoning", supports_preserve_reasoning },
        };
    }
};

// Carries the model's special tokens, and the template source purely as an
// opaque string for reporting (/props, --verbose).
//
// It no longer parses anything. This used to hold a jinja::program plus derived
// jinja::caps, which made the chat template an executable artefact the renderer,
// validator, tracker and grammar all had to stay consistent with -- a fifth
// source of truth none of the other four could check. Prompts are now built by a
// format plugin's renderer; see FORK.md.
struct common_chat_template {
    std::string bos_tok;
    std::string eos_tok;
    std::string src;

    common_chat_template(const std::string & src, const std::string & bos_token, const std::string & eos_token)
        : bos_tok(bos_token), eos_tok(eos_token), src(src) {}

    chat_template_caps caps;

    const std::string & source() const { return src; }
    const std::string & bos_token() const { return bos_tok; }
    const std::string & eos_token() const { return eos_tok; }

    chat_template_caps original_caps() const { return caps; }
};

struct common_chat_msg {
    std::string                               role;
    std::string                               content;
    std::vector<common_chat_msg_content_part> content_parts;
    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               tool_name;
    std::string                               tool_call_id;

    nlohmann::ordered_json to_json_oaicompat(bool concat_typed_text = false) const;

    std::string render_content(const std::string & delimiter = "\n\n") const;

    bool empty() const {
        return content.empty() && content_parts.empty() && tool_calls.empty() && reasoning_content.empty() &&
               tool_name.empty() && tool_call_id.empty();
    }

    bool contains_media() const {
        for (const auto & part : content_parts) {
            if (part.type == "media_marker") {
                return true;
            }
        }
        return false;
    }

    void set_tool_call_ids(std::vector<std::string> &           ids_cache,
                           const std::function<std::string()> & gen_tool_call_id) {
        for (auto i = 0u; i < tool_calls.size(); i++) {
            if (ids_cache.size() <= i) {
                auto id = tool_calls[i].id;
                if (id.empty()) {
                    id = gen_tool_call_id();
                }
                ids_cache.push_back(id);
            }
            tool_calls[i].id = ids_cache[i];
        }
    }

    bool operator==(const common_chat_msg & other) const {
        return role == other.role && content == other.content && content_parts == other.content_parts &&
               tool_calls == other.tool_calls && reasoning_content == other.reasoning_content &&
               tool_name == other.tool_name && tool_call_id == other.tool_call_id;
    }

    bool operator!=(const common_chat_msg & other) const { return !(*this == other); }
};

struct common_chat_msg_diff {
    std::string           reasoning_content_delta;
    std::string           content_delta;
    size_t                tool_call_index = std::string::npos;
    common_chat_tool_call tool_call_delta;

    static std::vector<common_chat_msg_diff> compute_diffs(const common_chat_msg & msg_prv,
                                                           const common_chat_msg & msg_new);

    bool operator==(const common_chat_msg_diff & other) const {
        return content_delta == other.content_delta && tool_call_index == other.tool_call_index &&
               tool_call_delta == other.tool_call_delta;
    }
};

enum common_chat_role {
    COMMON_CHAT_ROLE_UNKNOWN,
    COMMON_CHAT_ROLE_SYSTEM,
    COMMON_CHAT_ROLE_ASSISTANT,
    COMMON_CHAT_ROLE_USER,
    COMMON_CHAT_ROLE_TOOL
};

common_chat_role common_chat_role_from_string(const std::string & role);
const char *     common_chat_role_to_string(common_chat_role role);

struct common_chat_msg_span {
    common_chat_role role = COMMON_CHAT_ROLE_UNKNOWN;
    std::size_t pos = 0;
    std::size_t len = 0;

    bool valid() const {
        return role != COMMON_CHAT_ROLE_UNKNOWN;
    }
};

struct common_chat_msg_spans {
    std::vector<common_chat_msg_span> spans;

    void add(common_chat_role role, size_t pos, size_t len) {
        spans.push_back({ role, pos, len });
    }

    bool is_user_start(int32_t pos) const {
        for (auto it = spans.begin(); it != spans.end(); ++it) {
            if (it->role == COMMON_CHAT_ROLE_USER && pos == (int32_t) it->pos) {
                return true;
            }
        }
        return false;
    }

    int32_t last_user_message_pos() const {
        for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
            if (it->role == COMMON_CHAT_ROLE_USER) {
                return (int32_t) it->pos;
            }
        }
        return -1;
    }
};

struct common_chat_msg_delimiter {
    common_chat_role role = COMMON_CHAT_ROLE_UNKNOWN;
    std::string      delimiter;
    llama_tokens     tokens = {};
};

struct common_chat_msg_delimiters {
    std::vector<common_chat_msg_delimiter> delimiters;

    common_chat_msg_delimiters() = default;
    common_chat_msg_delimiters(std::initializer_list<common_chat_msg_delimiter> delims) : delimiters(delims) {}

    void add(common_chat_role role, const std::string & delimiter) {
        delimiters.push_back({ role, delimiter });
    }

    void tokenize(const llama_vocab * vocab);

    // split tokens into message spans. skips maps a start index to a length of a region to jump over without matching
    common_chat_msg_spans split(const llama_tokens & tokens, const std::map<size_t, size_t> & skips = {}) const;

    nlohmann::ordered_json to_json() const;
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,

    // These are intended to be parsed by the PEG parser
    COMMON_CHAT_FORMAT_PEG_SIMPLE,
    COMMON_CHAT_FORMAT_PEG_NATIVE,
    COMMON_CHAT_FORMAT_PEG_GEMMA4,

    COMMON_CHAT_FORMAT_COUNT,  // Not a format, just the # formats
};


// Continuation method provided via `continue_final_message`
enum common_chat_continuation {
    COMMON_CHAT_CONTINUATION_NONE,
    COMMON_CHAT_CONTINUATION_AUTO,
    COMMON_CHAT_CONTINUATION_REASONING,
    COMMON_CHAT_CONTINUATION_CONTENT,
};

struct common_chat_templates_inputs {
    std::vector<common_chat_msg>          messages;
    // Which format plugin renders, validates, tracks and constrains THIS
    // conversation. Empty means "whatever the model resolved to at load".
    //
    // Highest of three sources, and the only per-request one:
    //   1. this field             -- the `chat_format` request parameter
    //   2. --chat-format          -- per-model server configuration
    //   3. general.architecture   -- declared GGUF metadata
    //
    // What separates those three from what they replaced is DECLARED versus
    // GUESSED. A metadata key the publisher wrote is data, the same as a config
    // value somebody wrote down; string-matching the Jinja blob in
    // tokenizer.chat_template to infer a format is a guess, and is gone.
    std::string                           chat_format;
    std::string                           grammar;
    std::string                           json_schema;
    bool                                  add_generation_prompt  = true;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    std::vector<common_chat_tool>         tools;
    common_chat_tool_choice               tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool                                  parallel_tool_calls = false;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool enable_thinking"
    bool                                  enable_thinking     = true;
    // Keep reasoning on assistant turns that carry tool calls, so a multi-hop
    // tool chain does not lose the thoughts connecting its steps. Google:
    // "thoughts must NOT be removed between the function calls."
    // Defaults off, matching the writer spec, where an unset template variable
    // is falsy. Reasoning on the LAST assistant turn is preserved regardless --
    // see the gate in common_chat_gemma4_render.
    bool                                  preserve_thinking   = false;
    std::chrono::system_clock::time_point now                 = std::chrono::system_clock::now();
    std::map<std::string, std::string>    chat_template_kwargs;
    bool                                  add_bos = false;
    bool                                  add_eos = false;
    bool                                  force_pure_content = false;
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string                         prompt;
    std::string                         grammar;
    bool                                grammar_lazy         = false;
    std::string                         generation_prompt;
    bool                                supports_thinking    = false;
    std::string                         thinking_start_tag;  // e.g., "<think>"
    std::string                         thinking_end_tag;    // e.g., "</think>"
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
    std::string                         parser;
    common_chat_msg_delimiters          message_delimiters;
    // When true, generation_prompt is NOT prepended to the effective input in
    // common_chat_peg_parse. Grammar-file-driven parsers consume the model's raw
    // output directly and do not need the prefix.
    bool                                grammar_file_parser  = false;
    // Model input as its own type, plus the parser that checks it. `prompt`
    // above is the same bytes, kept because it is what callers send; this is the
    // object the format produced and the validator consumes. See prompt.h.
    common_chat_prompt                  chat_prompt;
    common_chat_prompt_validator        prompt_validator;
    // Carried through from common_chat_templates_inputs.reasoning_format so a
    // per-format pipeline can branch on whether the caller wants reasoning
    // extracted (AUTO/DEEPSEEK) or folded into content (NONE).
    common_reasoning_format             reasoning_format     = COMMON_REASONING_FORMAT_NONE;
};

// per-message parsing syntax
// should be derived from common_chat_params
struct common_chat_parser_params {
    common_chat_format      format               = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format     = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool parse_reasoning"
    // Whether reasoning_content should be inlined in the content (e.g. for reasoning_format=deepseek in stream mode)
    bool                    reasoning_in_content = false;
    std::string             generation_prompt;
    bool                    parse_tool_calls     = true;
    bool                    is_continuation      = false;
    bool                    echo                 = false;  // Include assistant prefilled msg in output
    bool                    debug                = false;  // Enable debug output for PEG parser
    common_peg_arena        parser               = {};
    bool                    grammar_file_parser  = false;  // mirrors common_chat_params::grammar_file_parser
    std::string             override_grammar;              // Lark or GBNF override; used instead of the serialized parser
    // Model input and its parser. Extraction walks the prompt FIRST with the
    // same pipeline, so the tracker arrives at generation already holding where
    // it is and what the open turn contains -- established by walking, not
    // summarised. See prompt.h.
    common_chat_prompt           chat_prompt;
    common_chat_prompt_validator prompt_validator;
    common_chat_parser_params() = default;
    common_chat_parser_params(const common_chat_params & chat_params) {
        format              = chat_params.format;
        grammar_file_parser = chat_params.grammar_file_parser;
        chat_prompt         = chat_params.chat_prompt;
        prompt_validator    = chat_params.prompt_validator;
        reasoning_format    = chat_params.reasoning_format;
        // Grammar-file parsers consume raw model output and must not be handed
        // the generation_prompt prefix.
        generation_prompt   = chat_params.grammar_file_parser ? std::string{} : chat_params.generation_prompt;
    }
};

// common_chat_verify_template was declared here, validating "--chat-template".
// That flag now rejects, so there is no template to validate.

void common_chat_templates_free(struct common_chat_templates * tmpls);

struct common_chat_templates_deleter {
    void operator()(common_chat_templates * tmpls) { common_chat_templates_free(tmpls); }
};

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

// ─────────────────────────────────────────────────────────────────────────────
// Chat format plugins, selected by NAME
// ─────────────────────────────────────────────────────────────────────────────
//
// A format plugin is four pieces that must agree -- renderer, validator,
// tracker, grammar -- registered under one name. `gemma4` is the only one today;
// adding another means implementing the four and registering it, and nothing
// else should need to change. See docs/fork/ARCHITECTURE.md.

// Every registered plugin name, sorted. For error messages and for --help.
std::vector<std::string> common_chat_format_names();

bool common_chat_format_is_registered(const std::string & name);

// Resolve which plugin a MODEL uses, and say where the answer came from.
//
// `config_override` is --chat-format; empty falls back to the model's declared
// `general.architecture`. Throws when the result names no registered plugin --
// at load, so a server that cannot serve a model does not come up claiming it
// can. `source_out` receives a human phrase for the log line.
std::string common_chat_format_resolve(const struct llama_model * model,
                                       const std::string &        config_override,
                                       std::string *              source_out = nullptr);

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override = "",
                                                     const std::string &        eos_token_override = "",
                                                     const std::string &        chat_format_override = "");

// The plugin name this model resolved to at load, i.e. the default for every
// request that does not name one itself.
std::string common_chat_templates_format(const struct common_chat_templates * tmpls);

bool        common_chat_templates_was_explicit(const struct common_chat_templates * tmpls);
std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant = "");

struct common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                                      const struct common_chat_templates_inputs & inputs);

// Format single message, while taking into account the position of that message in chat history
std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass);

// Returns an example of formatted chat
std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       const std::map<std::string, std::string> & chat_template_kwargs);

const char *    common_chat_format_name(common_chat_format format);
common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_parser_params & params);
common_chat_msg common_chat_peg_parse(const common_peg_arena & src_parser, const std::string & input, bool is_partial, const common_chat_parser_params & params);

// used by arg and server
const char *            common_reasoning_format_name(common_reasoning_format format);
common_reasoning_format common_reasoning_format_from_name(const std::string & format);

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice);

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates);

// Parses a JSON array of messages in OpenAI's chat completion API format.
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const nlohmann::ordered_json & messages);

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const nlohmann::ordered_json & tools);

common_chat_continuation common_chat_continuation_parse(const nlohmann::ordered_json & value);

// DEPRECATED: only used in tests
nlohmann::ordered_json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

nlohmann::ordered_json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);

// Default install location for chat grammar files. Overridable with
// --chat-grammars-dir / LLAMA_ARG_CHAT_GRAMMARS_DIR.
#define DEFAULT_CHAT_GRAMMARS_DIR "/usr/share/llama.cpp/grammars/chat"

// Chat grammar registry: loads chat grammar files from a directory at startup.
// Files are named <format-key>.lark and <format-key>.gbnf (e.g. "gemma4.lark");
// the .lark variant is used when built with LLAMA_USE_LLGUIDANCE, .gbnf otherwise.
//
// Call common_chat_grammar_init() once at startup, before any chat template
// processing. The registry lives in process memory and is read on every request,
// so grammar files are never re-read from disk mid-run.
//
// common_chat_grammar_get() returns an empty string on a registry miss. Callers
// MUST treat that as fatal rather than proceeding: an empty grammar means no
// sampling constraint at all, which silently defeats the entire point of a
// grammar-file-driven format. See docs/fork/POLICIES.md#nothing-is-inferred.
void        common_chat_grammar_init(const std::string & grammars_dir);
std::string common_chat_grammar_get(const std::string & model_key);

// Rewrite a grammar so `root_rule` becomes its entry rule.
//
// The sampler cannot be told where to start: llguidance compiles the rule
// literally named "start" (parser/src/lark/compiler.rs:612) and rejects
// %override (:705); GBNF has the same constraint on "root". Entry selection is
// therefore structural. Rule names are given the Lark way ('_'); the GBNF
// spelling ('-') is derived.
//
// Throws if the grammar has no entry rule, or no rule by the requested name.
std::string common_chat_grammar_set_entry(const std::string & grammar, const std::string & root_rule);

// get template caps, useful for reporting to server /props endpoint
std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates);

// common_chat_template_direct_apply / common_chat_template_generation_prompt were
// declared here and implemented nowhere -- the last two entry points into the
// Jinja renderer, left behind when it was deleted. A prompt is built by a format
// plugin's renderer now (common_chat_gemma4_render); there is no other path.
//
// The renderer's parameter type is common_chat_render_params, in
// chat-render-params.h.


// specialized per-task preset
struct common_chat_prompt_preset {
    std::string system;
    std::string user;
};

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates);

common_chat_msg_delimiters common_chat_msg_delimiters_parse(const nlohmann::ordered_json & delimiters);
