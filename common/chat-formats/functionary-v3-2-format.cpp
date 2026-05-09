#include "chat-formats/functionary-v3-2-format.h"

// Functionary-v3.2 FSM-state-to-grammar-rule registry. The grammar has no
// reasoning channel and no tool-id rule, so those states are absent.
// Rule names use hyphens (lark-to-peg normalization).
const common_chat_format_state_rules functionary_v3_2_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};
