#include "chat-formats/ministral-3-format.h"

// Ministral-3 FSM-state-to-grammar-rule registry. Includes a reasoning
// channel ([THINK]/[/THINK]) but no tool-id rule.
const common_chat_format_state_rules ministral_3_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};
