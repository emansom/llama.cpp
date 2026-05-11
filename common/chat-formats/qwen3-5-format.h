#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "peg-parser.h"

#include <string>
#include <unordered_set>

// Qwen3.5 chat format. Wire shape for the model's emitted assistant turn:
//   <reasoning></think>\n\n<content>(
//     <tool_call>\n<function=NAME>\n
//       (<parameter=K>\nVALUE\n</parameter>\n)*
//     </function>\n</tool_call>
//   )*
//
// Per-arg XML format (like DeepSeek-V3.2): each arg is wrapped in its own
// <parameter=K>...</parameter> block rather than a JSON args object. The
// decoder accumulates each (param-name, param-value) pair into a
// TOOL_ARG_KV event and the transformer assembles the JSON args buffer.
// Unlike DeepSeek-V3.2 there is no `string="true|false"` attribute --- all
// values are wrapped as JSON strings on the output side.

class common_chat_qwen3_5_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_qwen3_5_decoder : public common_chat_format_decoder {
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

class common_chat_qwen3_5_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
    std::vector<common_chat_shaped_event> on_finalize() override;

  private:
    bool        in_tool_   = false;
    bool        first_arg_ = true;
    std::string args_json_buffer_;
};

extern const common_chat_format_state_rules qwen3_5_state_rules;

// Render the Qwen3.5 prompt. Mirrors `models/templates/Qwen3.5-4B.jinja`
// byte-for-byte for the non-vision message shapes (vision content lists are
// rejected with a runtime error to match the template's raise_exception path).
//
// Notable behaviours:
// - last_query_index logic: a backwards scan finds the most recent non-
//   tool-response user message; assistant messages past that index keep
//   their `<think>...</think>` block intact, older ones have it stripped.
// - System block: tools-bearing requests get a system message with the
//   canonical "# Tools" instructions; the first system message (if any)
//   is appended after the tools text.
// - Generation prompt: `<|im_start|>assistant\n<think>\n` (thinking on) or
//   `<|im_start|>assistant\n<think>\n\n</think>\n\n` (thinking off).
std::string common_chat_qwen3_5_render(const autoparser::generation_params & inputs);
