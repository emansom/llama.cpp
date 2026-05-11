#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "peg-parser.h"

#include <string>
#include <unordered_set>

// Cohere c4ai Command-R chat format. Wire shape:
//   <|START_THINKING|>{reasoning}<|END_THINKING|>
//     (<|START_ACTION|>[{json_tool_call}, ...]<|END_ACTION|>
//      | <|START_RESPONSE|>{content}<|END_RESPONSE|>)
//
// Tool calls are a JSON array of `{"tool_call_id", "tool_name", "parameters"}`
// objects. The decoder parses this envelope on TOOL_OPEN and emits one event
// sequence per tool call.

class common_chat_cohere_c4ai_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_cohere_c4ai_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;
    std::vector<common_chat_decoded_event> on_finalize() override;

    void set_arena(const common_peg_ast_arena * arena) override { arena_ = arena; }

  private:
    const common_peg_ast_arena * arena_ = nullptr;
    // IDs of AST nodes whose content events were already emitted as part of
    // a parent `analysis-content` visit (used to suppress double-emission
    // when the visitor descends into the inner `content`-tagged node).
    std::unordered_set<common_peg_ast_id> handled_ids_;
    bool in_tool_ = false;
    bool last_tool_complete_ = false;
};

class common_chat_cohere_c4ai_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
};

extern const common_chat_format_state_rules cohere_c4ai_state_rules;

// Render the Cohere c4ai Command-R prompt. Mirrors
// `models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja`
// byte-for-byte for the typical chat/tool-call flow (skipping documents
// injection and citation grounding, which are not exercised by the test
// fixtures).
std::string common_chat_cohere_c4ai_render(const autoparser::generation_params & inputs,
                                           const std::string & bos_token);
