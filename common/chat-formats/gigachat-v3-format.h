#pragma once

#include "chat-formats/format-state-registry.h"
#include "chat-formats/json-tagged-format.h"

// GigaChat-v3 chat format: `<|message_sep|>\n\nfunction call<|role_sep|>\n{json}`.
// Pure JSON-tagged: tool-args is the model-emitted JSON object verbatim.

class common_chat_gigachat_v3_tracker     : public common_chat_json_tagged_tracker {};
class common_chat_gigachat_v3_decoder     : public common_chat_json_tagged_decoder {
  public:
    using common_chat_json_tagged_decoder::common_chat_json_tagged_decoder;
};
class common_chat_gigachat_v3_transformer : public common_chat_json_tagged_transformer {
  public:
    using common_chat_json_tagged_transformer::common_chat_json_tagged_transformer;
};

extern const common_chat_format_state_rules gigachat_v3_state_rules;
