#pragma once

#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"

#include <string>

// Shared codec for content-only-with-reasoning chat formats: model emits
// `<think>{reasoning}</think>{content}` (or equivalent reasoning markers
// the grammar tags as `reasoning` / `content`). No tool calls in scope.
//
// Used by DeepSeek-R1-Distill, MiniMax-M2, Apriel, etc. --- each provides
// its own grammar and writer; the codec is identical.

class common_chat_simple_reasoning_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_simple_reasoning_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;
};

class common_chat_simple_reasoning_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
};

extern const common_chat_format_state_rules simple_reasoning_state_rules;
