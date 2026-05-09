#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/json-tagged-format.h"

#include <string>

// Kimi-K2 chat format:
//   <|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{json}<|tool_call_end|><|tool_calls_section_end|>
//
// Wire shape is JSON-tagged (tool-args is the model's verbatim JSON object).
// The `tool-id` text is `functions.NAME:IDX`; the `tool-name` rule inside
// captures just `NAME`. Both are emitted by the decoder and the test suite
// expects:
//   call.name = "NAME"          (from tool-name)
//   call.id   = "functions.NAME:IDX"  (from tool-id, verbatim)
//
// No per-format overrides needed — the json-tagged base does the right thing.

class common_chat_kimi_k2_tracker     : public common_chat_json_tagged_tracker {};
class common_chat_kimi_k2_decoder     : public common_chat_json_tagged_decoder {
  public:
    using common_chat_json_tagged_decoder::common_chat_json_tagged_decoder;
};
class common_chat_kimi_k2_transformer : public common_chat_json_tagged_transformer {
  public:
    using common_chat_json_tagged_transformer::common_chat_json_tagged_transformer;
};

// FSM-state-to-grammar-rule registry. See format-state-registry.h.
extern const common_chat_format_state_rules kimi_k2_state_rules;

// Render the Kimi-K2 prompt for `inputs` (messages + tools + flags). Mirrors
// `models/templates/moonshotai-Kimi-K2.jinja` byte-for-byte. Replaces the
// Jinja path for Kimi-K2 in `common_chat_params_init_kimi_k2`.
std::string common_chat_kimi_k2_render(const autoparser::generation_params & inputs);
