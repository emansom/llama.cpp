#include "chat-formats/gigachat-v3-format.h"

// GigaChat-v3 FSM-state-to-grammar-rule registry. No reasoning channel and
// no tool-id; the grammar is content + a single function-call turn.
const common_chat_format_state_rules gigachat_v3_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};
