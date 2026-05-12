#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "peg-parser.h"

#include <string>
#include <unordered_set>

// MiniMax-M2 chat format. Tool wire shape:
//   <minimax:tool_call>
//   <invoke name="NAME">
//   <parameter name="K">VALUE</parameter>
//   ...
//   </invoke>
//   </minimax:tool_call>
//
// Reasoning via `<think>...</think>` (gen prompt prepends synthetic
// `<think>` so the grammar always sees a complete think block).

class common_chat_minimax_m2_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_minimax_m2_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;
    std::vector<common_chat_decoded_event> on_finalize() override;

    void set_arena(const common_peg_ast_arena * arena) override { arena_ = arena; }

  private:
    const common_peg_ast_arena * arena_ = nullptr;
    bool in_tool_ = false;
    bool last_tool_complete_ = false;
    std::unordered_set<common_peg_ast_id> handled_ids_;
};

class common_chat_minimax_m2_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
    std::vector<common_chat_shaped_event> on_finalize() override;

  private:
    bool        in_tool_   = false;
    bool        first_arg_ = true;
    std::string args_json_buffer_;
};

extern const common_chat_format_state_rules minimax_m2_state_rules;

// Render the MiniMax-M2 prompt. Minimal --- test fixtures don't exercise
// `expect_reconstruction()` so byte-exact Jinja parity isn't required.
std::string common_chat_minimax_m2_render(const autoparser::generation_params & inputs);
