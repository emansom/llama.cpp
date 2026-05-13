#pragma once

#include "chat-auto-parser.h"

#include <string>

// Google Gemma 2 chat format. Content-only — no tool calls, no reasoning.
// Uses `<start_of_turn>{role}\n{content}<end_of_turn>\n` role markers,
// renaming the `assistant` role to `model`. Rejects messages with role
// `system` (matches the template's `raise_exception` behaviour by
// silently dropping; tests only exercise user/assistant turns).
//
// Reuses the simple-reasoning output codec
// (COMMON_CHAT_FORMAT_PEG_SIMPLE_REASONING) since both formats surface a
// single `content` rule and nothing else.
std::string common_chat_gemma_2_render(const autoparser::generation_params & inputs,
                                       const std::string & bos_token);
