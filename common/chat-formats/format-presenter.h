#pragma once

#include "chat-formats/decoded-event.h"
#include "chat-formats/format-tracker.h"
#include "chat.h"

// Presenter: consumes common_chat_shaped_event values from the transformer and
// writes to common_chat_msg. One default implementation handles the
// OpenAI-compatible message shape for all formats:
//   REASONING_TEXT  -> append to result.reasoning_content
//   CONTENT_TEXT    -> append to result.content
//   TOOL_OPEN       -> push a new pending common_chat_tool_call
//   TOOL_NAME       -> set pending tool .name
//   TOOL_ID         -> set pending tool .id
//   TOOL_ARGS_JSON  -> set pending tool .arguments (verbatim, no synthesis)
//   TOOL_CLOSE      -> commit pending tool to result.tool_calls
//
// Per-format subclasses exist only if a model needs message-shape-level
// peculiarities (rare). The presenter MUST consult state_.current_state() to
// disambiguate (e.g. TOOL_NAME in IN_TOOL_NAME starts a new pending call;
// TOOL_NAME in DONE means a follow-up call). The presenter MUST NOT mutate
// tracker state.
class common_chat_format_presenter {
  public:
    common_chat_msg & result;
    bool is_partial_parse = false;

    common_chat_format_presenter(common_chat_msg & m, const common_chat_format_state_view & state)
        : result(m), state_(state) {}
    virtual ~common_chat_format_presenter() = default;

    // Per-event projection into common_chat_msg. The default implementation
    // covers all formats; subclasses override only for format-specific
    // message-shape peculiarities.
    virtual void present(const common_chat_shaped_event & event);

    // Final cross-cutting cleanups: discard whitespace-only reasoning_content,
    // ensure default role="assistant", flush any pending tool call that was
    // never closed (in non-partial mode).
    virtual void on_finalize();

  protected:
    const common_chat_format_state_view & state_;

    // The pending tool call accumulating between TOOL_OPEN and TOOL_CLOSE.
    // Pushed to result.tool_calls on TOOL_CLOSE (or on the next TOOL_OPEN if
    // the model skipped emitting a close, which only happens in partial parse
    // tail trimming).
    bool                  has_pending_ = false;
    common_chat_tool_call pending_;
};
