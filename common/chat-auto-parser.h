#pragma once

#include "chat.h"
#include "common.h"
#include "peg-parser.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

class common_chat_peg_builder;

// ============================================================================
// Parameters for template application (low-level, used by diff analysis)
// ============================================================================
struct template_params {
    json                messages;
    json                tools;
    bool                add_generation_prompt = false;
    bool                enable_thinking       = true;
    std::optional<json> extra_context         = std::nullopt;
};

struct diff_split {
    std::string prefix;
    std::string suffix;
    std::string left;
    std::string right;

    bool operator==(struct diff_split & other) const {
        return prefix == other.prefix && suffix == other.suffix && left == other.left && right == other.right;
    }
};

// Result of compare_variants containing diff and original outputs
struct compare_variants_result {
    diff_split  diff;
    std::string output_A;
    std::string output_B;
};

namespace autoparser {

// ============================================================================
// High-level params for parser generation
// ============================================================================

struct generation_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    json                                  json_schema;
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
    json                                  extra_context;
    bool                                  add_bos       = false;
    bool                                  add_eos       = false;
    bool                                  is_inference  = true;
    bool                                  add_inference = false;
    bool                                  mark_input    = true;  // whether to mark input strings when rendering

    bool has_continuation() const {
        return continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !continue_msg.empty();
    }
};

}  // namespace autoparser
