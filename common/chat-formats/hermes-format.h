#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-state-registry.h"

#include <string>

// Hermes (NousResearch Hermes-2-Pro / Hermes-3 / Qwen2.5 / MiMo-VL family)
// chat format. Wire shape:
//   <|im_start|>system\n<system_with_tool_descriptions><|im_end|>\n
//   <|im_start|>user\n<content><|im_end|>\n
//   <|im_start|>assistant\n<content><|im_end|>\n
//   <|im_start|>assistant\n<content>\n<tool_call>\n{json}\n</tool_call><|im_end|>\n
//   <|im_start|>tool\n<tool_response>\n<content>\n</tool_response><|im_end|>\n
//
// Tool calls are JSON objects with `name` and `arguments` fields, wrapped in
// `<tool_call>\n...\n</tool_call>` markers. Content can precede tool calls.

extern const common_chat_format_state_rules hermes_state_rules;

// Render the Hermes prompt. Mirrors `models/templates/NousResearch-Hermes-3-
// Llama-3.1-8B-tool_use.jinja` byte-for-byte. The Hermes template's system
// message includes the canonical tool-calling instructions plus per-tool
// "name(args) -> ret - description\n\n    Args:\n        ..." synthesised
// docstrings.
std::string common_chat_hermes_render(const autoparser::generation_params & inputs,
                                      const std::string & bos_token);
