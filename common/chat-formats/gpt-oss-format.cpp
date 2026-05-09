#include "chat-formats/gpt-oss-format.h"

// GPT-OSS (Harmony) FSM-state-to-grammar-rule registry. The grammar has
// reasoning (analysis turn), content turn (final/commentary), and function
// tool calls; no tool-id rule.
const common_chat_format_state_rules gpt_oss_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,   "content" },
        { common_chat_format_state::IN_REASONING, "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL, "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME, "func-name" },
        { common_chat_format_state::IN_TOOL_ARGS, "tool-args" },
    }
};
