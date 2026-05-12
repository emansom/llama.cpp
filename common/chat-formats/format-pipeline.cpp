#include "chat-formats/format-pipeline.h"

#include "chat-formats/deepseek-v3-2-format.h"
#include "chat-formats/functionary-v3-2-format.h"
#include "chat-formats/gemma4-format.h"
#include "chat-formats/gigachat-v3-format.h"
#include "chat-formats/glm-4-7-flash-format.h"
#include "chat-formats/gpt-oss-format.h"
#include "chat-formats/hermes-format.h"
#include "chat-formats/kimi-k2-format.h"
#include "chat-formats/lfm2-5-format.h"
#include "chat-formats/lfm2-format.h"
#include "chat-formats/apriel-format.h"
#include "chat-formats/cohere-c4ai-format.h"
#include "chat-formats/deepseek-v3-1-format.h"
#include "chat-formats/granite-4-format.h"
#include "chat-formats/minimax-m2-format.h"
#include "chat-formats/ministral-3-format.h"
#include "chat-formats/qwen3-5-format.h"
#include "chat-formats/seed-oss-format.h"
#include "chat-formats/simple-reasoning-format.h"

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
        case COMMON_CHAT_FORMAT_PEG_DEEPSEEK_V3_2:
            return make_pipeline_default<
                common_chat_deepseek_v3_2_tracker,
                common_chat_deepseek_v3_2_decoder,
                common_chat_deepseek_v3_2_transformer>(msg, is_partial_parse, reasoning_format);

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

        case COMMON_CHAT_FORMAT_PEG_HERMES:
            return make_pipeline_default<
                common_chat_hermes_tracker,
                common_chat_hermes_decoder,
                common_chat_hermes_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_GPT_OSS:
            return make_pipeline_default<
                common_chat_gpt_oss_tracker,
                common_chat_gpt_oss_decoder,
                common_chat_gpt_oss_transformer>(msg, is_partial_parse, reasoning_format);

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

        case COMMON_CHAT_FORMAT_PEG_QWEN3_5:
            return make_pipeline_default<
                common_chat_qwen3_5_tracker,
                common_chat_qwen3_5_decoder,
                common_chat_qwen3_5_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_GRANITE_4:
            return make_pipeline_default<
                common_chat_granite_4_tracker,
                common_chat_granite_4_decoder,
                common_chat_granite_4_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_COHERE_C4AI:
            return make_pipeline_default<
                common_chat_cohere_c4ai_tracker,
                common_chat_cohere_c4ai_decoder,
                common_chat_cohere_c4ai_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_SEED_OSS:
            return make_pipeline_default<
                common_chat_seed_oss_tracker,
                common_chat_seed_oss_decoder,
                common_chat_seed_oss_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_SIMPLE_REASONING:
            return make_pipeline_default<
                common_chat_simple_reasoning_tracker,
                common_chat_simple_reasoning_decoder,
                common_chat_simple_reasoning_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_APRIEL:
            return make_pipeline_default<
                common_chat_apriel_tracker,
                common_chat_apriel_decoder,
                common_chat_apriel_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_DEEPSEEK_V3_1:
            return make_pipeline_default<
                common_chat_deepseek_v3_1_tracker,
                common_chat_deepseek_v3_1_decoder,
                common_chat_deepseek_v3_1_transformer>(msg, is_partial_parse, reasoning_format);

        case COMMON_CHAT_FORMAT_PEG_MINIMAX_M2:
            return make_pipeline_default<
                common_chat_minimax_m2_tracker,
                common_chat_minimax_m2_decoder,
                common_chat_minimax_m2_transformer>(msg, is_partial_parse, reasoning_format);

        default:
            return common_chat_format_pipeline{};
    }
}
