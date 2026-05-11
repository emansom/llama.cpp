#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "peg-parser.h"

#include <string>
#include <unordered_set>

// Apriel chat format (both unsloth-Apriel-1.5 and Apriel-1.6-Thinker).
// Tool wire shape:
//   <tool_calls>[{"name":"X","arguments":{...}}, ...]</tool_calls>
// 1.6-Thinker additionally has a leading `<reasoning>\n[BEGIN FINAL RESPONSE]\n`
// segment; 1.5 has plain content.

class common_chat_apriel_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_apriel_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;
    std::vector<common_chat_decoded_event> on_finalize() override;

  private:
    bool in_tool_ = false;
    bool last_tool_complete_ = false;
};

class common_chat_apriel_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
};

extern const common_chat_format_state_rules apriel_state_rules;

// Apriel-1.5 prompt writer (no reasoning).
std::string common_chat_apriel_render(const autoparser::generation_params & inputs,
                                      const std::string & bos_token);

// Apriel-1.6-Thinker prompt writer (reasoning model).
std::string common_chat_apriel_thinker_render(const autoparser::generation_params & inputs,
                                              const std::string & bos_token);
