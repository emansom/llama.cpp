#include "chat-formats/format-presenter.h"

#include <cctype>

void common_chat_format_presenter::present(const common_chat_shaped_event & event) {
    using kind = common_chat_shaped_event::kind;
    switch (event.k) {
        case kind::REASONING_TEXT:
            result.reasoning_content += event.text;
            break;

        case kind::CONTENT_TEXT:
            result.content += event.text;
            break;

        case kind::TOOL_OPEN:
            // Begin a new pending tool call. If a previous one was never
            // closed (only happens during partial-parse tail trimming),
            // commit it now so we don't lose it.
            if (has_pending_ && !pending_.name.empty()) {
                result.tool_calls.push_back(pending_);
            }
            pending_     = common_chat_tool_call{};
            has_pending_ = true;
            break;

        case kind::TOOL_NAME:
            if (has_pending_) {
                pending_.name = event.text;
            }
            break;

        case kind::TOOL_ID:
            if (has_pending_) {
                pending_.id = event.text;
            }
            break;

        case kind::TOOL_ARGS_JSON:
            if (has_pending_) {
                pending_.arguments = event.text;
            }
            break;

        case kind::TOOL_CLOSE:
            if (has_pending_ && !pending_.name.empty()) {
                result.tool_calls.push_back(pending_);
            }
            has_pending_ = false;
            pending_     = common_chat_tool_call{};
            break;
    }
}

void common_chat_format_presenter::on_finalize() {
    // Commit any still-pending tool call that's named but never closed.
    // Streaming mode often hits this: the close marker hasn't been emitted
    // yet but enough of the call is fully formed for it to appear.
    if (has_pending_ && !pending_.name.empty()) {
        result.tool_calls.push_back(pending_);
        has_pending_ = false;
        pending_     = common_chat_tool_call{};
    }

    // Discard whitespace-only reasoning content (e.g. produced by an empty
    // <think></think> prefill).
    if (!result.reasoning_content.empty()) {
        bool all_ws = true;
        for (char c : result.reasoning_content) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                all_ws = false;
                break;
            }
        }
        if (all_ws) {
            result.reasoning_content.clear();
        }
    }

    if (result.role.empty()) {
        result.role = "assistant";
    }
}
