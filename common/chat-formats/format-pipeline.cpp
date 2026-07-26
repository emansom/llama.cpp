#include "chat-formats/format-pipeline.h"

#include "chat-formats/gemma4-format.h"

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

// Factory: dispatch on format.
//
// This fork serves Gemma 4 and nothing else (see FORK.md). The other per-family
// pipelines were removed rather than carried: their conformance could not be
// held to the standard docs/fork/ARCHITECTURE.md sets for Gemma 4, and keeping
// them would have made the format plugin interface aspirational instead of real.
//
// An empty pipeline (.valid() == false) is a hard error at the call site: there
// is no fallback extractor. A format either has a pipeline or is not served.
common_chat_format_pipeline common_chat_make_format_pipeline(
    common_chat_format      format,
    common_chat_msg &       msg,
    bool                    is_partial_parse,
    common_reasoning_format reasoning_format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return make_pipeline_default<
                common_chat_gemma4_tracker,
                common_chat_gemma4_decoder,
                common_chat_gemma4_transformer>(msg, is_partial_parse, reasoning_format);

        default:
            return common_chat_format_pipeline{};
    }
}
