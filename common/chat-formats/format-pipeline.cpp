#include "chat-formats/format-pipeline.h"

#include "chat-formats/functionary-v3-2-format.h"
#include "chat-formats/gemma4-format.h"
#include "chat-formats/gigachat-v3-format.h"
#include "chat-formats/glm-4-7-flash-format.h"
#include "chat-formats/kimi-k2-format.h"
#include "chat-formats/lfm2-5-format.h"
#include "chat-formats/lfm2-format.h"
#include "chat-formats/ministral-3-format.h"

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
    switch (format) {
        case COMMON_CHAT_FORMAT_PEG_FUNCTIONARY_V3_2:
            return make_pipeline_default<
                common_chat_functionary_v3_2_tracker,
                common_chat_functionary_v3_2_decoder,
                common_chat_functionary_v3_2_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return make_pipeline_default<
                common_chat_gemma4_tracker,
                common_chat_gemma4_decoder,
                common_chat_gemma4_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_GIGACHAT_V3:
            return make_pipeline_default<
                common_chat_gigachat_v3_tracker,
                common_chat_gigachat_v3_decoder,
                common_chat_gigachat_v3_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_GLM_4_7_FLASH:
            return make_pipeline_default<
                common_chat_glm_4_7_flash_tracker,
                common_chat_glm_4_7_flash_decoder,
                common_chat_glm_4_7_flash_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_KIMI_K2:
            return make_pipeline_default<
                common_chat_kimi_k2_tracker,
                common_chat_kimi_k2_decoder,
                common_chat_kimi_k2_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_LFM2:
            return make_pipeline_default<
                common_chat_lfm2_tracker,
                common_chat_lfm2_decoder,
                common_chat_lfm2_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_LFM2_5:
            return make_pipeline_default<
                common_chat_lfm2_5_tracker,
                common_chat_lfm2_5_decoder,
                common_chat_lfm2_5_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_MINISTRAL_3:
            return make_pipeline_default<
                common_chat_ministral_3_tracker,
                common_chat_ministral_3_decoder,
                common_chat_ministral_3_transformer>(msg, is_partial_parse, reasoning_format);

        default:
            return common_chat_format_pipeline{};
    }
}
