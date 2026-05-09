#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/json-tagged-format.h"

#include <string>

// GPT-OSS chat format (OpenAI Harmony):
//   https://github.com/openai/gpt-oss
//   https://developers.openai.com/cookbook/articles/openai-harmony
//
// Wire format (model output):
//
//   * Content turn:  <|channel|>final|commentary<|message|>{content}
//   * Analysis turn: <|channel|>analysis<|message|>{reasoning}
//   * Tool call:     [<|channel|>{TYPE}] [ to=functions.NAME] [<|constrain|>...] <|message|>{json}
//                    — recipient may appear in role header or channel header
//   * Builtin call:  [<|channel|>{TYPE}] [ to=BUILTIN_NAME] ... <|message|>{body}
//                    — silently dropped (NOT surfaced as tool_calls)
//   * Multi-turn:    <|end|><|start|>assistant separates intermediate turns
//
// All four pipeline layers come from the shared json_tagged family — args are
// already JSON in the wire format. The tracker/decoder/transformer here just
// provide a typed alias so format-pipeline.cpp can dispatch by enum.

class common_chat_gpt_oss_tracker     : public common_chat_json_tagged_tracker {};
class common_chat_gpt_oss_decoder     : public common_chat_json_tagged_decoder {
  public:
    using common_chat_json_tagged_decoder::common_chat_json_tagged_decoder;

    // Harmony channel markers around the no-reasoning analysis-content body.
    // The opener and the trailing `<|end|>` are matched as literals in the
    // enclosing `analysis_turn` rule; this decoder re-injects them around
    // the captured body so the surfaced content matches the wire shape.
    std::string analysis_marker_open()  const override { return "<|channel|>analysis<|message|>"; }
    std::string analysis_marker_close() const override { return "<|end|>"; }
};
class common_chat_gpt_oss_transformer : public common_chat_json_tagged_transformer {
  public:
    using common_chat_json_tagged_transformer::common_chat_json_tagged_transformer;
};

extern const common_chat_format_state_rules gpt_oss_state_rules;

// Render the GPT-OSS prompt. Mirrors `models/templates/openai-gpt-oss-120b.jinja`
// byte-for-byte. The template uses OpenAI's Harmony multi-turn format with
// channel-tagged messages. Notable mechanisms:
//   * Always-emitted system message with model identity, knowledge cutoff,
//     current date (from `inputs.now`), reasoning effort, and the
//     "Valid channels" closer.
//   * Optional developer message (extracted from the first system/developer
//     message + tool namespace).
//   * Per-message rendering with channel tags: `analysis` for thinking,
//     `final` for content, `commentary` for tool calls/responses.
//   * Future-final-message lookahead: tool-call messages drop their analysis
//     channel emission when a later assistant final message exists.
//   * `<|return|>` terminator for the last assistant message when
//     `add_generation_prompt=false` (training shape); `<|end|>` otherwise.
std::string common_chat_gpt_oss_render(const autoparser::generation_params & inputs);
