#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/json-tagged-format.h"

#include <string>

// IBM Granite 4.0 chat format. Wire shape for the model's emitted assistant
// turn is identical to Hermes (`<content>(<tool_call>\n{"name":"X",
// "arguments":{json}}\n</tool_call>)*`), so this codec reuses the json-tagged
// base unchanged. Only the prompt-rendering side differs --- Granite uses
// `<|start_of_role|>...<|end_of_role|>...<|end_of_text|>` role markers and a
// different system-message text plus optional documents (RAG) channel.
class common_chat_granite_4_tracker     : public common_chat_json_tagged_tracker {};
class common_chat_granite_4_decoder     : public common_chat_json_tagged_decoder {
  public:
    using common_chat_json_tagged_decoder::common_chat_json_tagged_decoder;
};
class common_chat_granite_4_transformer : public common_chat_json_tagged_transformer {
  public:
    using common_chat_json_tagged_transformer::common_chat_json_tagged_transformer;
};

extern const common_chat_format_state_rules granite_4_state_rules;

// Render the Granite 4.0 prompt. Mirrors
// `models/templates/ibm-granite-granite-4.0.jinja` byte-for-byte for the
// non-vision path. Notable behaviours:
//   - Default system message when none provided.
//   - Tools system message: canonical "You are a helpful assistant..." text,
//     tools rendered as JSON lines, appended to system_message.
//   - Documents (RAG): same shape as tools, with a separate intro text. Not
//     surfaced through `generation_params` --- the writer ignores any
//     `documents` field for now.
//   - Tool responses: `<|start_of_role|>user<|end_of_role|>` opener on
//     transition into tool sequence; `<|end_of_text|>\n` close on exit.
std::string common_chat_granite_4_render(const autoparser::generation_params & inputs);
