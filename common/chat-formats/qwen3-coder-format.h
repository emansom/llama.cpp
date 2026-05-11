#pragma once

#include "chat-auto-parser.h"

#include <string>

// Qwen3-Coder chat format. Wire shape:
//   <content>(<tool_call>\n<function=NAME>\n
//     (<parameter=K>\nVALUE\n</parameter>\n)*
//   </function>\n</tool_call>)*
//
// Per-arg XML same as Qwen3.5/Nemotron but no thinking block (Qwen3-Coder
// is a coder model). Output codec reused from Qwen3.5; only the writer
// and grammar differ.
std::string common_chat_qwen3_coder_render(const autoparser::generation_params & inputs);
