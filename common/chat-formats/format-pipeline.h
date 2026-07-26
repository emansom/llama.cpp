#pragma once

#include "chat-formats/format-decoder.h"
#include "chat-formats/format-presenter.h"
#include "chat-formats/format-tracker.h"
#include "chat-formats/format-transformer.h"
#include "chat.h"
#include "peg-parser.h"

#include <memory>

// A fully-wired four-layer pipeline for one chat format. Built by
// common_chat_make_format_pipeline() based on params.format.
//
// Layer ordering during AST walk (per node):
//   1. tracker.advance(node)                    -> updates FSM state
//   2. for each event in decoder.decode(node):  -> wire-shape extraction
//        for each shaped in transformer.shape(event):  -> model-specific shaping
//          presenter.present(shaped)            -> common_chat_msg projection
// At end-of-walk:
//   for each shaped in transformer.on_finalize():
//     presenter.present(shaped)
//   presenter.on_finalize()
//
// The tracker is the single source of truth for "where in the conversation
// are we"; the other three layers are stateless w.r.t. conversation position
// and consult tracker.current_state() at every step.
struct common_chat_format_pipeline {
    std::unique_ptr<common_chat_format_tracker>     tracker;
    std::unique_ptr<common_chat_format_decoder>     decoder;
    std::unique_ptr<common_chat_format_transformer> transformer;
    std::unique_ptr<common_chat_format_presenter>   presenter;

    // Walk the AST, drive the four layers in order. Returns when the walk is
    // complete; presenter.result is populated as a side effect.
    void run(const common_peg_ast_arena & arena, const common_peg_parse_result & result);

    bool valid() const { return tracker && decoder && transformer && presenter; }
};

// Factory: build the pipeline for the given chat format. Returns a pipeline
// with all four layers null if the format does not yet have a pipeline impl
// (caller should fall back to legacy mapper).
//
// `reasoning_format` is plumbed through for formats whose transformer needs
// to switch behavior based on whether the API caller wants reasoning
// extracted — e.g. GPT-OSS treats `analysis` channel content differently
// when reasoning_format == NONE (concatenated as content with the channel
// markers preserved). Most pipelines ignore this argument.
common_chat_format_pipeline common_chat_make_format_pipeline(
    common_chat_format      format,
    common_chat_msg &       msg,
    bool                    is_partial_parse,
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE);

// Helper: assemble a pipeline from a tracker / decoder / transformer triple
// + the default presenter. Per-format factory branches use this so they
// only need to specify the four type parameters; tracker/decoder are
// constructed without arguments, the transformer receives the tracker
// state-view AND the reasoning_format so it can branch on caller intent.
template <typename Tracker, typename Decoder, typename Transformer>
common_chat_format_pipeline make_pipeline_default(common_chat_msg & msg, bool is_partial,
                                                  common_reasoning_format reasoning_format) {
    auto tracker     = std::make_unique<Tracker>();
    auto * tracker_p = tracker.get();
    auto decoder     = std::make_unique<Decoder>(*tracker_p);
    auto transformer = std::make_unique<Transformer>(*tracker_p, reasoning_format);
    auto presenter   = std::make_unique<common_chat_format_presenter>(msg, *tracker_p);
    presenter->is_partial_parse = is_partial;

    common_chat_format_pipeline pipeline;
    pipeline.tracker     = std::move(tracker);
    pipeline.decoder     = std::move(decoder);
    pipeline.transformer = std::move(transformer);
    pipeline.presenter   = std::move(presenter);
    return pipeline;
}
