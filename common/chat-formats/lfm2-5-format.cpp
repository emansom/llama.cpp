#include "chat-formats/lfm2-5-format.h"

// LFM2.5 FSM-state-to-grammar-rule registry. Same Python-style call shape
// as LFM2 (and same FSM states); only the optional outer wrapper tokens
// (<|tool_call_start|>...<|tool_call_end|>) differ in the grammar.
const common_chat_format_state_rules lfm2_5_state_rules = {
    {
        { common_chat_format_state::IN_CONTENT,      "content" },
        { common_chat_format_state::IN_REASONING,    "reasoning" },
        { common_chat_format_state::IN_TOOL_CALL,    "tool-call" },
        { common_chat_format_state::IN_TOOL_NAME,    "func-name" },
        { common_chat_format_state::IN_TOOL_ARG_KEY, "arg-name" },
        { common_chat_format_state::IN_TOOL_ARG_VAL, "arg-value" },
    }
};
