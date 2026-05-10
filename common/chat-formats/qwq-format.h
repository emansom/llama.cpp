#pragma once

#include "chat-auto-parser.h"

#include <string>

// QwQ (Qwen-QwQ-32B) chat format. Wire shape for the model's emitted
// assistant turn is identical to Hermes/Qwen2.5: `<content>(<tool_call>\n
// {"name":"X","arguments":{json}}\n</tool_call>)*`. The output-parsing
// pipeline reuses the Hermes grammar/codec; only the prompt-rendering side
// differs --- QwQ's system message has no "You are Qwen..." default,
// supports past-assistant `<think>...</think>` stripping, and emits a
// `<|im_start|>assistant\n<think>\n` generation prompt.
std::string common_chat_qwq_render(const autoparser::generation_params & inputs);
