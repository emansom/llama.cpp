#pragma once

#include "chat-formats/format-decoder.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "peg-parser.h"

#include <unordered_set>

// DeepSeek-V3.2 chat format (DSML XML param):
//   <｜DSML｜function_calls>
//   <｜DSML｜invoke name="NAME">
//   <｜DSML｜parameter name="K" string="true|false">V</｜DSML｜parameter>
//   ...
//   </｜DSML｜invoke>
//   </｜DSML｜function_calls>
//
// The arg pairs come as `<param>` blocks rather than a JSON object. The
// decoder accumulates each (param-name, bool-str, param-value) triple and
// emits a TOOL_ARG_KV event. The transformer buffers KV events and assembles
// a single TOOL_ARGS_JSON on TOOL_CLOSE — that's where the deliberate `{`/`}`
// synthesis lives, scoped to this one transformer because DSML has no JSON
// envelope.

class common_chat_deepseek_v3_2_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_deepseek_v3_2_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;
    std::vector<common_chat_decoded_event> on_finalize() override;

    void set_arena(const common_peg_ast_arena * arena) override { arena_ = arena; }

  private:
    const common_peg_ast_arena * arena_ = nullptr;
    // Track whether we're inside a tool call so a fresh 'tool' visit can
    // close the previous one before opening the new one.
    bool        in_tool_ = false;
    // True when the last 'tool' tag visit was non-partial — only then do we
    // emit a TOOL_CLOSE in on_finalize (a partial tool means args text is
    // still in flight; closing it here would prematurely seal the buffer).
    bool        last_tool_complete_ = false;
    // AST IDs whose events were already emitted as part of a parent 'arg'
    // rule visit; the visitor's later traversal of those same nodes must
    // not double-emit.
    std::unordered_set<common_peg_ast_id> handled_ids_;
};

// The transformer assembles the JSON args object from per-arg KV events. This
// is the deliberate, scoped exception to the no-character-synthesis policy:
// DSML's wire shape isn't JSON, so converting it to JSON requires synthesizing
// `{`, `}`, `,`, `:`. All other layers (tracker, decoder, presenter) emit only
// model-provided characters.
class common_chat_deepseek_v3_2_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
    std::vector<common_chat_shaped_event> on_finalize() override;

  private:
    // Buffer for KV events between TOOL_OPEN and TOOL_CLOSE.
    bool        in_tool_   = false;
    bool        first_arg_ = true;
    std::string args_json_buffer_;  // accumulating "{...}" text
};
