#pragma once

#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"

// GLM-4.7-Flash chat format (per-arg XML):
//   <tool_call>NAME<arg_key>K1</arg_key><arg_value>V1</arg_value>...<arg_key>Kn</arg_key><arg_value>Vn</arg_value></tool_call>
//
// Values arrive as raw model bytes — strings pass through verbatim; non-strings
// (numbers, bools, null, arrays, objects) are emitted by the chat template via
// tojson(), so they already have JSON shape on the wire. The transformer
// detects which case applies per value (try-parse JSON; on failure, wrap as
// string) and assembles `{"k1": v1, "k2": v2, ...}` JSON args.
//
// The grammar tags both the outer `tool_call` rule and the inner `tool_open`
// (`<tool_call>` literal) with `tool-open`, so the AST has a nested tool-open.
// The decoder tracks depth like the LFM2 decoder.

class common_chat_glm_4_7_flash_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_glm_4_7_flash_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;

  private:
    int         tool_open_depth_ = 0;
    std::string pending_key_;
};

class common_chat_glm_4_7_flash_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
    std::vector<common_chat_shaped_event> on_finalize() override;

  private:
    bool        in_tool_   = false;
    bool        first_arg_ = true;
    std::string args_json_buffer_;
};

extern const common_chat_format_state_rules glm_4_7_flash_state_rules;
