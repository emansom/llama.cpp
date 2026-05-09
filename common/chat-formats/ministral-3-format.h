#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/json-tagged-format.h"

#include <string>

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

// Render the Ministral-3 prompt. Mirrors
// `models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja`
// byte-for-byte. Performs the same `reasoning_content` → typed-content
// preprocessing the existing init function did before passing to Jinja
// (`thinking` block + `text` block on system/assistant messages).
//
// The Ministral-3 template emits no explicit generation prompt --- the
// model continues right after the last `[/INST]` or `[/TOOL_RESULTS]`.
// `add_generation_prompt` is therefore unused.
std::string common_chat_ministral_3_render(const autoparser::generation_params & inputs,
                                           const std::string & bos_token,
                                           const std::string & eos_token);
