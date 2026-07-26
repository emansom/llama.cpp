#pragma once

#include "chat.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <string>

// What a format plugin's renderer is given.
//
// Its own header, rather than a member of chat.h, only because chat.h carries
// `nlohmann/json_fwd.hpp` -- forward declarations cannot be held by value, and
// pulling the full json header into chat.h would land on every translation unit
// that includes it.
//
// This was `autoparser::generation_params` in chat-auto-parser.h (deleted), beside the
// differential auto-parser's own types: `template_params`, `diff_split` and
// `compare_variants_result`, which existed to apply a Jinja template several
// times with varied inputs, diff the outputs, and infer a grammar from where
// they differed. That machinery is gone and those three went with it -- this one
// was simply stranded in their header, under their namespace, describing
// something else entirely.
//
// It differs from common_chat_templates_inputs by being the DECODED form: JSON
// messages rather than common_chat_msg, thinking already resolved from whichever
// spelling the caller used, and any continuation already split off the message
// list.
struct common_chat_render_params {
    nlohmann::ordered_json                messages;
    nlohmann::ordered_json                tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    nlohmann::ordered_json                json_schema;
    bool                                  parallel_tool_calls = true;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_AUTO;
    bool                                  stream              = true;
    std::string                           grammar;
    bool                                  add_generation_prompt  = false;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    common_chat_msg                       continue_msg;
    bool                                  enable_thinking        = true;
    // See common_chat_templates_inputs::preserve_thinking.
    bool                                  preserve_thinking      = false;
    std::chrono::system_clock::time_point now                    = std::chrono::system_clock::now();
    nlohmann::ordered_json                extra_context;
    bool                                  add_bos       = false;
    bool                                  add_eos       = false;
    bool                                  is_inference  = true;
    bool                                  add_inference = false;
    bool                                  mark_input    = true;  // mark input strings when rendering
    // See common_params::chat_thought_prefill.
    bool                                  thought_prefill = true;

    bool has_continuation() const {
        return continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !continue_msg.empty();
    }
};
