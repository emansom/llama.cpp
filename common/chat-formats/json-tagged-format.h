#pragma once

#include "chat-formats/format-decoder.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "peg-parser.h"

#include <unordered_set>

// Family base for JSON-tagged chat formats — i.e. formats whose grammar uses
// the conventional `tool-open` / `tool-name` / `tool-id` / `tool-args` /
// `tool-close` tags AND whose `tool-args` AST node carries a complete JSON
// object verbatim. Examples: Kimi-K2, Ministral-3, GPT-OSS, Functionary-v3.2,
// GigaChat-v3, Hermes, Granite, QwQ, etc.
//
// Each per-format family member typically reuses these three classes and only
// overrides the bits that differ (e.g. Kimi parses `functions.NAME:IDX` from
// the tool-id text; GPT-OSS routes channel events).

class common_chat_json_tagged_tracker : public common_chat_format_tracker {
  public:
    void advance(const common_peg_ast_node & node) override;

    std::vector<std::string> expected_productions() const override;
};

class common_chat_json_tagged_decoder : public common_chat_format_decoder {
  public:
    using common_chat_format_decoder::common_chat_format_decoder;

    std::vector<common_chat_decoded_event> decode(const common_peg_ast_node & node) override;

    void set_arena(const common_peg_ast_arena * arena) override { arena_ = arena; }

    // Wire-shape markers around the no-reasoning analysis-content body. The
    // base class returns Ministral-3's '[THINK]' / '[/THINK]' so simply-named
    // subclasses Just Work. Per-format subclasses with different markers
    // (e.g. '<think>'/'</think>') override.
    virtual std::string analysis_marker_open()  const { return "[THINK]"; }
    virtual std::string analysis_marker_close() const { return "[/THINK]"; }

  protected:
    const common_peg_ast_arena * arena_ = nullptr;
    // AST IDs whose events were already emitted as part of a parent "tool"
    // or "analysis-content" node visit; the later child-walk visit must not
    // double-emit.
    std::unordered_set<common_peg_ast_id> handled_ids_;
};

class common_chat_json_tagged_transformer : public common_chat_format_transformer {
  public:
    using common_chat_format_transformer::common_chat_format_transformer;

    std::vector<common_chat_shaped_event> shape(const common_chat_decoded_event & event) override;
};
