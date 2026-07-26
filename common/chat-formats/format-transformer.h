#pragma once

#include "chat-formats/decoded-event.h"
#include "chat-formats/format-tracker.h"
#include "common.h"  // common_reasoning_format

#include <vector>

// Per-format transformer: consumes common_chat_decoded_event values from the
// decoder and emits common_chat_shaped_event values. Applies model-specific
// transforms: content prefix stripping, reasoning channel routing, tool-args
// JSON construction. Transformers MUST consult state_.current_state() and
// MUST NOT mutate tracker state or touch common_chat_msg.
//
// `reasoning_format` is plumbed through to every transformer so per-format
// subclasses can branch on whether the API caller wants reasoning extracted
// (AUTO/DEEPSEEK) vs. rolled into content (NONE). Most transformers ignore
// it; those that need it (GPT-OSS) read `reasoning_format()` accessor.
//
// The transformer is the only layer permitted to synthesize JSON-shape
// characters (`{`, `}`, `:`, `,`) — and only when converting non-JSON wire
// shapes (Python `name(k=v)`, DSML `<param>`) into the JSON `arguments`
// string the API requires. Each transformer that does so must comment the
// scoped exception in its file header.
class common_chat_format_transformer {
  public:
    explicit common_chat_format_transformer(const common_chat_format_state_view & state,
                                            common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE)
        : state_(state), reasoning_format_(reasoning_format) {}
    virtual ~common_chat_format_transformer() = default;

    // Per-event shaping. Returns zero or more shaped events. The transformer
    // MAY buffer events (e.g. accumulate TOOL_ARG_KV until TOOL_CLOSE) and
    // emit them later in shape() or on_finalize().
    virtual std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) = 0;

    // Optional final flush at end-of-AST-walk. Default: no events.
    virtual std::vector<common_chat_shaped_event> on_finalize() { return {}; }

  protected:
    const common_chat_format_state_view & state_;
    common_reasoning_format               reasoning_format_ = COMMON_REASONING_FORMAT_NONE;

  public:
    common_reasoning_format reasoning_format() const { return reasoning_format_; }
};
