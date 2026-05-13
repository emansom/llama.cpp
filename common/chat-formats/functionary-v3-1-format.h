#pragma once

#include "chat-auto-parser.h"

#include <string>

// Meetkai Functionary medium v3.1 chat format. Llama-3-style role markers
// (`<|start_header_id|>`/`<|end_header_id|>`/`<|eot_id|>`/`<|eom_id|>`),
// with `<function=NAME>{args_json}</function>` tool wire shape. No
// reasoning channel.
//
// Reuses the Apriel JSON-tagged tool codec (auto-tagging maps
// `tool_call` → "tool", `func_name` → "tool-name", `tool_args` →
// "tool-args") since the per-tool structure surfaces the same events.
std::string common_chat_functionary_v3_1_render(const autoparser::generation_params & inputs,
                                                const std::string & bos_token);
