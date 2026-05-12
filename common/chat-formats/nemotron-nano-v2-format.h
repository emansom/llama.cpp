#pragma once

#include "chat-auto-parser.h"

#include <string>

// NVIDIA Nemotron-Nano-v2 chat format. Wire shape:
//   <content>(<TOOLCALL>[json_array_of_tool_calls]</TOOLCALL>)?(<SPECIAL_12>)?
//
// Uses `<SPECIAL_10>System`, `<SPECIAL_11>User`/`Assistant` role markers and
// `<SPECIAL_12>` as end-of-turn. Tool calls are a JSON array of
// `{"name": "...", "arguments": {...}}` objects --- the same envelope shape
// as Apriel, so this format reuses the Apriel output codec
// (COMMON_CHAT_FORMAT_PEG_APRIEL) via a Nemotron-specific grammar.
//
// Writer is minimal: test fixtures don't exercise expect_reconstruction(),
// so byte-exact Jinja parity isn't required.
std::string common_chat_nemotron_nano_v2_render(const autoparser::generation_params & inputs);
