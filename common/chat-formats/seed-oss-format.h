#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "peg-parser.h"

#include <string>
#include <unordered_set>

// ByteDance Seed-OSS chat format. Wire shape:
//   (<seed:think>{reasoning}</seed:think>\n)?
//   {content}
//   (<seed:tool_call>\n<function=NAME>\n
//     (<parameter=K>VALUE</parameter>\n)*
//   </function>\n</seed:tool_call>)*
//
// Per-arg XML format (like Qwen3.5/Nemotron) but with inline VALUE (no
// surrounding newlines around the value text). The codec reuses the
// DeepSeek-V3.2 per-arg KV pattern; arg-value parsing emits raw text and
// the transformer wraps it as JSON.

class common_chat_seed_oss_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_seed_oss_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;
    std::vector<common_chat_decoded_event> on_finalize() override;

    void set_arena(const common_peg_ast_arena * arena) override { arena_ = arena; }

  private:
    const common_peg_ast_arena * arena_ = nullptr;
    bool in_tool_ = false;
    bool last_tool_complete_ = false;
    std::unordered_set<common_peg_ast_id> handled_ids_;
};

class common_chat_seed_oss_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
    std::vector<common_chat_shaped_event> on_finalize() override;

  private:
    bool        in_tool_   = false;
    bool        first_arg_ = true;
    std::string args_json_buffer_;
};

extern const common_chat_format_state_rules seed_oss_state_rules;

// Render the Seed-OSS prompt. Mirrors models/templates/ByteDance-Seed-OSS.jinja
// for the typical chat/tool-call flow. Skips thinking_budget mode + dictsort
// budget reflection lookup (use defaults), JSON tool definition mode (uses
// Python-docstring format by default).
std::string common_chat_seed_oss_render(const autoparser::generation_params & inputs);
