#pragma once

#include "chat-auto-parser.h"

#include <string>

// IBM Granite 3.3 chat format. Wire shape:
//   (<think>{reasoning}</think>)?<content>
//
// No tool calls in scope (test fixtures only exercise reasoning + content).
// Uses IBM-style role markers `<|start_of_role|>...<|end_of_role|>` and
// `<|end_of_text|>`. Output codec reuses the shared `simple-reasoning`
// pipeline.
std::string common_chat_granite_3_3_render(const autoparser::generation_params & inputs,
                                           const std::string & bos_token);
