#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/lfm2-format.h"

#include <string>

// LFM2.5 chat format: same Python-style `[name(k=v)]` as LFM2, but the
// `<|tool_call_start|>...<|tool_call_end|>` wrapper tokens are optional.
// The grammar handles that difference; the four pipeline layers behave the
// same as LFM2.

class common_chat_lfm2_5_tracker     : public common_chat_lfm2_tracker {};
class common_chat_lfm2_5_decoder     : public common_chat_lfm2_decoder {
  public:
    using common_chat_lfm2_decoder::common_chat_lfm2_decoder;
};
class common_chat_lfm2_5_transformer : public common_chat_lfm2_transformer {
  public:
    using common_chat_lfm2_transformer::common_chat_lfm2_transformer;
};

extern const common_chat_format_state_rules lfm2_5_state_rules;

// Render the LFM2.5 prompt. Mirrors `models/templates/LFM2.5-Instruct.jinja`
// byte-for-byte. Differs from LFM2 in:
//   * No `<|tool_list_start|>` / `<|tool_list_end|>` markers around the
//     tool list (just `List of tools: [...]`).
//   * No `<|tool_response_start|>` / `<|tool_response_end|>` wrapping for
//     tool messages.
//   * Past-but-not-last assistant messages have their `<think>...</think>`
//     reasoning prefix stripped (`keep_past_thinking` defaults to false).
std::string common_chat_lfm2_5_render(const autoparser::generation_params & inputs,
                                      const std::string & bos_token);
