#pragma once

#include "chat-formats/format-state-registry.h"
#include "chat-formats/json-tagged-format.h"

// Ministral-3 chat format: `[THINK]…[/THINK][TOOL_CALLS]name[ARGS]{json}`.
//
// This is a pure JSON-tagged format — the wire shape is `tool-open` / `tool-name` /
// `tool-args(JSON)` / `tool-close`, so all four pipeline layers come from the
// shared json_tagged family. This header exists for explicit per-format
// dispatch and as a place to hang any future Ministral-3-specific shaping.

class common_chat_ministral_3_tracker     : public common_chat_json_tagged_tracker {};
class common_chat_ministral_3_decoder     : public common_chat_json_tagged_decoder {
  public:
    using common_chat_json_tagged_decoder::common_chat_json_tagged_decoder;
};
class common_chat_ministral_3_transformer : public common_chat_json_tagged_transformer {
  public:
    using common_chat_json_tagged_transformer::common_chat_json_tagged_transformer;
};

extern const common_chat_format_state_rules ministral_3_state_rules;
