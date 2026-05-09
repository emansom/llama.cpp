#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/json-tagged-format.h"

#include <string>

// Functionary-v3.2 chat format: `>>>NAME\n{json_args}`. Pure JSON-tagged shape:
// the `tool_call` rule wraps `func_name "\n" tool_args`. No explicit
// tool-close tag — the presenter's on_finalize() commits the still-pending
// tool call when the AST walk ends.

class common_chat_functionary_v3_2_tracker     : public common_chat_json_tagged_tracker {};
class common_chat_functionary_v3_2_decoder     : public common_chat_json_tagged_decoder {
  public:
    using common_chat_json_tagged_decoder::common_chat_json_tagged_decoder;
};
class common_chat_functionary_v3_2_transformer : public common_chat_json_tagged_transformer {
  public:
    using common_chat_json_tagged_transformer::common_chat_json_tagged_transformer;
};

extern const common_chat_format_state_rules functionary_v3_2_state_rules;

// Render the Functionary v3.2 prompt. Mirrors
// `models/templates/meetkai-functionary-medium-v3.2.jinja` byte-for-byte,
// including the system+tools header that includes a TypeScript-style schema
// for each tool.
std::string common_chat_functionary_v3_2_render(const autoparser::generation_params & inputs,
                                                const std::string & bos_token);
