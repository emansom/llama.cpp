#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Forward declaration to keep this header free of peg-parser includes (the
// tracker only sees AST nodes through its concrete subclass).
struct common_peg_ast_node;

// Universal "where are we in the conversation" states. Each per-format FSM
// uses a SUBSET of these. New states are added centrally and per-format
// trackers declare which they actually use.
enum class common_chat_format_state : uint8_t {
    INITIAL,         // nothing emitted yet
    IN_CONTENT,      // free-form assistant content
    IN_REASONING,    // inside <think>/<channel|>thought/etc.
    IN_TOOL_CALL,    // inside a tool call envelope (after open, before close)
    IN_TOOL_NAME,    // currently emitting the function name
    IN_TOOL_ID,      // currently emitting the tool-call id (Kimi-style)
    IN_TOOL_ARGS,    // inside the args region (json or per-format)
    IN_TOOL_ARG_KEY, // tagged-args formats: between arg-name open and close
    IN_TOOL_ARG_VAL, // tagged-args formats: between arg-value open and close
    IN_TOOL_PARAM,   // DSML <param>...</param>
    DONE,            // tool-call closed; back at top-level
};

// Read-only window into a tracker's FSM. Exposed to the PEG parser context AND
// to the decoder/transformer/presenter so none of them need the full tracker
// type. The PEG runtime sees only this interface, keeping the parser layer
// free of chat-format coupling.
class common_chat_format_state_view {
  public:
    virtual ~common_chat_format_state_view() = default;

    virtual common_chat_format_state current_state() const = 0;

    // Names of grammar rules valid as the next production from the current
    // state. The PEG parser intersects this with its `next_sequence_delimiters`
    // when set; an empty list means "any production" (grammar-driven only).
    virtual std::vector<std::string> expected_productions() const = 0;
};

// Abstract per-format FSM. Pure state-keeping: trackers MUST NOT touch
// common_chat_msg, emit decoded events, or mutate any other layer. They
// inspect node.tag/node.rule/node.is_partial only.
class common_chat_format_tracker : public common_chat_format_state_view {
  public:
    common_chat_format_tracker() = default;
    ~common_chat_format_tracker() override = default;

    // Single transition entry-point. Concrete trackers MUST update `state_`
    // per their declared FSM.
    virtual void advance(const common_peg_ast_node & node) = 0;

    // common_chat_format_state_view
    common_chat_format_state current_state() const override { return state_; }
    std::vector<std::string> expected_productions() const override = 0;

  protected:
    common_chat_format_state state_ = common_chat_format_state::INITIAL;
};
