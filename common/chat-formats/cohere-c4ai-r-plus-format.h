#pragma once

#include "chat-auto-parser.h"

#include <string>

// CohereForAI Command-R-Plus chat format. Distinct from r7b: tool calls
// use a markdown code block (`Action:\n```json\n[...]\n```` ) and
// `{"tool_name", "parameters"}` JSON shape rather than r7b's
// `<|START_ACTION|>[{...}]<|END_ACTION|>` + `{"tool_call_id", "tool_name",
// "parameters"}` shape.
//
// Reuses the Apriel output codec (COMMON_CHAT_FORMAT_PEG_APRIEL) since
// both formats use the same auto-tagging structure for tool calls
// (`func_name` → `tool-name`, `tool_args` → `tool-args`).
std::string common_chat_cohere_c4ai_r_plus_render(const autoparser::generation_params & inputs);
