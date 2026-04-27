#pragma once

#include "chat-formats/decoded-event.h"
#include "chat-formats/format-tracker.h"

#include <vector>

struct common_peg_ast_node;
class common_peg_ast_arena;

// Per-format decoder: reads AST nodes (in lock-step with the tracker) and
// emits typed common_chat_decoded_event values. The decoder is the only layer
// that knows the model's wire shape (DSML `<param>` tags, Python `name(k=v)`,
// JSON `{"k":v}`, Kimi `functions.NAME:IDX`, Gemma's `<|"|>`-delimited dict, etc.).
//
// Decoders MUST consult state_.current_state() on every call to interpret the
// node correctly (e.g. a `<` token means different things in IN_CONTENT vs
// IN_TOOL_CALL). Decoders MUST NOT mutate tracker state and MUST NOT touch
// common_chat_msg.
class common_chat_format_decoder {
  public:
    explicit common_chat_format_decoder(const common_chat_format_state_view & state) : state_(state) {}
    virtual ~common_chat_format_decoder() = default;

    // Per-node decode step. Returns zero or more decoded events.
    virtual std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) = 0;

    // Optional flush at end-of-AST-walk. Default: no events.
    virtual std::vector<common_chat_decoded_event> on_finalize() { return {}; }

    // The pipeline coordinator hands the arena in before kicking off the AST
    // walk, so decoders that need to look up nodes by tag/rule (e.g. Gemma 4
    // walking the gemma4_dict subtree) have access. Default no-op for decoders
    // that walk only the per-node tag/text (most formats).
    virtual void set_arena(const common_peg_ast_arena * /*arena*/) {}

  protected:
    const common_chat_format_state_view & state_;
};
