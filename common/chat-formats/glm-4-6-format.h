#pragma once

#include "chat-auto-parser.h"

#include <string>

// GLM-4.6 chat format. Same wire shape as GLM-4.7-Flash but with newlines
// between arg_key/arg_value pairs (`<tool_call>NAME\n<arg_key>K</arg_key>\n
// <arg_value>V</arg_value>\n...</tool_call>`). The output codec is reused
// from GLM-4.7-Flash; only the writer and grammar differ.
//
// The template's generation prompt is just `<|assistant|>` (thinking on,
// model emits its own `\n<think>...</think>\n...` body) or
// `<|assistant|>\n<think></think>` (thinking off, the empty think block is
// prepended).
std::string common_chat_glm_4_6_render(const autoparser::generation_params & inputs);
