#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"

#include <string>

// DeepSeek-V3.1 chat format. Tool wire shape:
//   <｜tool▁calls▁begin｜>
//     (<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{args_json}<｜tool▁call▁end｜>)+
//   <｜tool▁calls▁end｜>
// Reasoning: `REASONING</think>` (forced-open `<think>` via gen prompt
// when thinking-on; no reasoning marker when thinking-off).

class common_chat_deepseek_v3_1_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_deepseek_v3_1_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;
    std::vector<common_chat_decoded_event> on_finalize() override;

  private:
    bool in_tool_ = false;
    bool last_tool_complete_ = false;
};

class common_chat_deepseek_v3_1_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
};

extern const common_chat_format_state_rules deepseek_v3_1_state_rules;

// Render DeepSeek-V3.1 prompt. Minimal --- test fixtures don't exercise
// reconstruction, so byte-exact Jinja parity isn't required.
std::string common_chat_deepseek_v3_1_render(const autoparser::generation_params & inputs,
                                             const std::string & bos_token);
