#include "chat-formats/format-pipeline.h"

void common_chat_format_pipeline::run(const common_peg_ast_arena & arena,
                                      const common_peg_parse_result & result) {
    if (!valid()) {
        return;
    }

    // Decoders that need to look up sibling/descendant nodes (e.g. assembling
    // tool args from a structured subtree) get the arena here; most decoders
    // ignore this and work off the per-node tag/text.
    decoder->set_arena(&arena);

    // Walk the AST in tree order. Each node: tracker advances FSM, decoder
    // emits decoded events (reading tracker state), transformer shapes them,
    // presenter writes to common_chat_msg.
    arena.visit(result, [this](const common_peg_ast_node & node) {
        tracker->advance(node);
        for (const auto & decoded : decoder->decode(node)) {
            for (const auto & shaped : transformer->shape(decoded)) {
                presenter->present(shaped);
            }
        }
    });

    // End-of-walk: drain decoder buffered events, then transformer, then
    // finalize the presenter (default impl trims whitespace-only reasoning,
    // commits any still-pending tool call, sets default role).
    for (const auto & decoded : decoder->on_finalize()) {
        for (const auto & shaped : transformer->shape(decoded)) {
            presenter->present(shaped);
        }
    }
    for (const auto & shaped : transformer->on_finalize()) {
        presenter->present(shaped);
    }
    presenter->on_finalize();
}

// Factory: dispatch on format. Returns an empty pipeline (.valid() == false)
// for formats not yet migrated; the chat.cpp dispatcher then falls back to the
// legacy common_chat_peg_mapper. Per-format pipelines are wired in by
// subsequent patches as each model family is migrated.
common_chat_format_pipeline common_chat_make_format_pipeline(
    common_chat_format      format,
    common_chat_msg &       msg,
    bool                    is_partial_parse,
    common_reasoning_format reasoning_format) {
    (void)format;
    (void)msg;
    (void)is_partial_parse;
    (void)reasoning_format;
    return common_chat_format_pipeline{};
}
