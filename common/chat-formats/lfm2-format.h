#pragma once

#include "chat-auto-parser.h"
#include "chat-formats/format-decoder.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"

#include <string>

// LFM2 chat format (Python-style):
//   <|tool_call_start|>[name(k="v",...)]<|tool_call_end|>
//
// The grammar tags `tool_call`/`tool_open`/`tool_close`/`func_name`/`arg_name`/
// `arg_value`. Notably, both the outer `tool_call` rule AND the inner
// `tool_open` rule (which wraps the `(`) get the `tool-open` tag, so the AST
// has nested tool-open nodes. The decoder tracks "depth" via the tracker
// state to ignore the inner one.
//
// The transformer assembles per-arg KV events into a JSON args object — this
// is the deliberate, scoped synthesis exception (Python `name(k=v)` has no
// JSON envelope, so we synthesize `{`, `}`, `,`, `:`).

class common_chat_lfm2_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;
    std::vector<std::string> expected_productions() const override;
};

class common_chat_lfm2_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;
    std::vector<common_chat_decoded_event> on_finalize() override;

  private:
    // Track whether we're currently inside a tool call so the next visit of a
    // 'tool' node can emit a TOOL_CLOSE for the previous call before opening
    // the new one.
    bool        in_tool_ = false;
    // Buffered key for the next TOOL_ARG_KV (between arg-name and arg-value).
    std::string pending_key_;
};

class common_chat_lfm2_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
    std::vector<common_chat_shaped_event> on_finalize() override;

  private:
    bool        in_tool_   = false;
    bool        first_arg_ = true;
    std::string args_json_buffer_;
};

extern const common_chat_format_state_rules lfm2_state_rules;

// Render the LFM2 prompt. Mirrors `models/templates/LFM2-8B-A1B.jinja`
// byte-for-byte. The writer also covers LFM2.5: that variant differs only
// in the system-prompt tools wrapping (no `<|tool_list_start|>` markers and
// different separator) — it has its own writer in lfm2-5-format.cpp.
std::string common_chat_lfm2_render(const autoparser::generation_params & inputs,
                                    const std::string & bos_token);
