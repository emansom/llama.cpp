#pragma once

#include "chat-auto-parser.h"

#include <string>

// DeepSeek-R1 family chat format. Used by:
//   - deepseek-ai-DeepSeek-R1-Distill-Llama
//   - deepseek-ai-DeepSeek-R1-Distill-Qwen
//   - llama-cpp-deepseek-r1
//
// Wire format for the model's emitted assistant turn (after generation
// prompt `<｜Assistant｜><think>\n`):
//   <reasoning></think><content>
//
// Reasoning-only model (no tool calls in scope). The output codec uses the
// shared content-only-with-reasoning pipeline (see format-pipeline.cpp).
std::string common_chat_deepseek_r1_render(const autoparser::generation_params & inputs,
                                           const std::string & bos_token);
