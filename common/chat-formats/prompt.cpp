#include "chat-formats/prompt.h"

#include "chat-formats/format-pipeline.h"
#include "chat.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

void common_chat_prompt_validator::walk(const common_chat_prompt &    prompt,
                                        common_chat_format_pipeline & pipeline,
                                        common_chat_msg &             msg,
                                        bool                          debug) const {
    if (conversation_.empty() || prompt.empty()) {
        return;
    }

    // A rendered prompt is a PREFIX of a conversation -- it stops where
    // generation begins -- so the last turn is legitimately incomplete. Both
    // flags are needed to say that:
    //
    //   STREAMING makes a repetition report NEED_MORE at EOF instead of quietly
    //   ending, so the incomplete turn's partial nodes still reach the decoder.
    //
    //   LENIENT makes running OUT of input a partial match rather than a
    //   failure (peg-parser, literal at EOF). Despite the name it does not
    //   loosen what is accepted: a byte that does not match still fails. It is
    //   the difference between "no more input" and "wrong input", and only the
    //   second is malformed.
    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_STREAMING | COMMON_PEG_PARSE_FLAG_LENIENT;
    if (debug) {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
    }

    common_peg_parse_context ctx(prompt.text, flags);
    auto                     result = conversation_.parse(ctx);

    // Every byte must be accounted for. Checked here rather than by anchoring
    // the grammar at end-of-input, because "wants more input" is a success for a
    // prefix and an anchor cannot express that. Without the check a rule like
    // `closed_turn*` reports success having read nothing at all -- validation
    // that validates nothing.
    if (result.fail() || result.end != prompt.text.size()) {
        throw std::runtime_error(
            "rendered prompt failed `conversation` validation at offset " +
            std::to_string(result.end) + ": " + prompt.text.substr(result.end, 80));
    }

    // Set GEMMA4_DUMP_WALK to inspect this walk: it prints the prompt, the
    // conversation AST, and what the pipeline emitted from it. Kept behind an
    // env var because it is the only way to see whether the walk produced
    // decoder events at all -- a walk that emits nothing is otherwise silent.
    const bool dump = getenv("GEMMA4_DUMP_WALK") != nullptr;
    if (dump) {
        fprintf(stderr, "\n=== WALK prompt ===\n%s\n=== WALK AST ===\n%s\n",
                prompt.text.c_str(), ctx.ast.dump().c_str());
    }

    pipeline.run(ctx.ast, result);

    if (dump) {
        fprintf(stderr, "=== WALK emitted: content=[%s] reasoning=[%s] ===\n",
                msg.content.c_str(), msg.reasoning_content.c_str());
    }
}
