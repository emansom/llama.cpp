#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "peg-parser.h"

#include <string>

#include <unordered_set>

// Gemma 4 chat format — mirrors the official LiteRT-LM ANTLR grammar:
//   https://github.com/google-ai-edge/LiteRT-LM/tree/main/runtime/components/tool_use/antlr
//
// Wire format (model-output side):
//   * Reasoning channel: <|channel>thought ...thoughts... <channel|>
//   * Tool call:         <|tool_call>call:FUNC_NAME{key:value, ...}<tool_call|>
//   * Response (JSON):   ```json\n{schema-conformant value}\n```
//
// Tool-call value types: ESCAPED_STRING (delimited by <|"|>, <escape>, or
// <ctrl46>), NUMBER, BOOLEAN, null, nested object, or array. Object keys are
// bare identifiers (no quoting). The transformer walks the dict subtree and
// produces a canonical JSON `arguments` string — that walk is the only place
// in this pipeline that synthesizes JSON structural characters, and only
// because the source format is non-JSON.

class common_chat_gemma4_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_gemma4_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;

    // The Gemma 4 decoder needs the arena to walk the tool-call subtree on
    // close (to assemble the JSON args from `gemma4_dict` children). The
    // pipeline coordinator hands the arena in via this hook before each
    // decode pass.
    void set_arena(const common_peg_ast_arena * arena) override { arena_ = arena; }

  private:
    const common_peg_ast_arena * arena_ = nullptr;

    // Walk a `gemma4_dict` / `gemma4_value` / `gemma4_array` / `gemma4_string`
    // subtree and emit its canonical JSON serialization. Strings are JSON-escaped;
    // numbers/bools/null are passed through verbatim.
    std::string gemma4_to_json(common_peg_ast_id id);

    // AST IDs whose events were already emitted as part of a parent visit
    // (e.g. the inner `tag content` of an analysis-content rule).
    std::unordered_set<common_peg_ast_id> handled_ids_;
};

class common_chat_gemma4_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
};

extern const common_chat_format_state_rules gemma4_state_rules;

// Render the Gemma 4 prompt. Mirrors `models/templates/google-gemma-4-31B-it.jinja`
// byte-for-byte. The template is the most intricate of the migrated formats:
//   * System block emitted when tools, system message, or `enable_thinking`.
//   * TypeScript-style schema with `<|"|>...<|"|>` Gemma string markers.
//   * Continuation detection that suppresses duplicate `<|turn>model` when
//     consecutive non-tool messages are both assistant.
//   * Forward-scan of consecutive `tool` messages following an assistant
//     `tool_calls` message (legacy `tool_responses` field also supported).
//   * Per-content-type rendering (text / image / audio / video).
std::string common_chat_gemma4_render(const autoparser::generation_params & inputs,
                                      const std::string & bos_token);
