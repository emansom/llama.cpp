#pragma once

#include "chat-formats/format-tracker.h"
#include "peg-parser.h"

#include <string>

// Forward-declared, not included: chat.h includes this header, and
// format-pipeline.h includes chat.h. Only the .cpp needs their definitions.
struct common_chat_format_pipeline;
struct common_chat_msg;

// Model INPUT, as its own type and with its own parser.
//
// Input and output are different things and had been sharing plumbing: the
// rendered bytes travelled as a bare std::string duplicated across two structs,
// the state they left the model in travelled beside it as a loose enum, and the
// parser that checks them was an arena field parsed inline in the middle of the
// generation-parsing function. Nothing named "the prompt" or "the thing that
// validates a prompt", so nothing could state their invariants either.
//
// A prompt is a PREFIX of a conversation: it stops exactly where generation is
// about to begin. That single fact is what separates it from model output, and
// it is why it needs its own parser rather than the generation grammar's entry.
// See the validator below.

// The bytes sent to the model, and the state they leave it in.
//
// Produced only by a format's renderer, which knows both because it wrote them.
// The entry state is REPORTED, not inferred from the text afterwards -- there is
// no "generation prompt" to recover and re-prepend.
struct common_chat_prompt {
    std::string              text;
    common_chat_format_state entry_state = common_chat_format_state::INITIAL;

    bool empty() const { return text.empty(); }
};

// The parser for model input, rooted at a format's `conversation` rule.
//
// Deliberately NOT the generation parser. That one is rooted at an entry
// production and consumes what the model emits; this one consumes what the
// caller is about to send, and its job is to refuse anything the model was not
// designed to receive.
//
// It also carries prior emission forward: walking the prompt with the SAME
// pipeline that will parse the generation means the tracker arrives at
// generation already holding where it is, and anything the open assistant turn
// already contains lands in the output message on the way through.
class common_chat_prompt_validator {
  public:
    common_chat_prompt_validator() = default;
    explicit common_chat_prompt_validator(common_peg_arena conversation)
        : conversation_(std::move(conversation)) {}

    bool empty() const { return conversation_.empty(); }

    // Validate `prompt` and walk it through `pipeline`.
    //
    // Throws std::runtime_error naming the offset if the prompt does not fit the
    // format. Malformed input is rejected HERE rather than sent to the model.
    void walk(const common_chat_prompt &          prompt,
              common_chat_format_pipeline &       pipeline,
              common_chat_msg &                   msg,
              bool                                debug = false) const;

  private:
    common_peg_arena conversation_ = {};
};
