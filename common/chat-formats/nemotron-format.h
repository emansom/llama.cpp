#pragma once

#include "chat-auto-parser.h"

#include <string>

// NVIDIA Nemotron-3 chat format. The model's emitted assistant turn uses the
// same per-arg XML tool wire shape as Qwen3.5
// (`<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n...\n
// </function>\n</tool_call>`), so the output codec, grammar, and FSM state
// registry are reused from qwen3-5; only prompt rendering differs.
//
// Notable differences from Qwen3.5 on the writer side:
//   - System+tools message uses XML schema blocks (`<function><name>X
//     </name><description>...</description><parameters><parameter>...
//     </parameter>...</parameters></function>`) instead of JSON.
//   - Default content for assistant messages without explicit reasoning gets
//     wrapped in `<think></think>` (empty think prefix).
//   - Generation prompt: `<|im_start|>assistant\n<think>\n` (thinking on) or
//     `<|im_start|>assistant\n<think></think>` (thinking off).
//   - `truncate_history_thinking` (default true) strips reasoning from past
//     assistant turns (loop index < last_user_idx).
std::string common_chat_nemotron_render(const autoparser::generation_params & inputs);
