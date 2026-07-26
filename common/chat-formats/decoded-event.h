#pragma once

#include <cstdint>
#include <string>

// A typed enumeration of the wire-format-independent things a model can emit
// during a single AST-walk step. Decoders read AST nodes and translate them
// into a sequence of these events; transformers consume them and produce
// shaped events; presenters consume shaped events and write to common_chat_msg.
//
// All four pipeline layers (tracker, decoder, transformer, presenter) operate
// in lock-step with the tracker's FSM — every decoder/transformer/presenter
// step consults the tracker's current_state() to interpret the event.
struct common_chat_decoded_event {
    enum class kind : uint8_t {
        REASONING_TEXT,   // text fragment to append to .reasoning_content
        CONTENT_TEXT,     // text fragment to append to .content
        TOOL_OPEN,        // begin a new tool call envelope
        TOOL_NAME,        // function name (full string; partial fragments suppressed)
        TOOL_ID,          // tool call id (e.g. Kimi's "functions.NAME:IDX")
        TOOL_ARGS_RAW,    // pre-formed JSON args text (model emitted JSON directly)
        TOOL_ARG_KV,      // a single (key, value, is_string) pair (formats with per-arg KV)
        TOOL_CLOSE,       // end of tool call envelope
    };

    kind k;

    // For REASONING_TEXT, CONTENT_TEXT, TOOL_NAME, TOOL_ID, TOOL_ARGS_RAW.
    std::string text;

    // For TOOL_ARG_KV.
    std::string key;
    std::string value;       // raw model bytes; transformer is responsible for shaping into JSON
    bool        is_string = false;  // true => string-typed arg (quote+escape); false => emit verbatim

    // Marks an event whose source AST node was partial. Transformers may choose to
    // suppress partial events to keep streaming output monotonic.
    bool        is_partial = false;
};

// Output of the transformer layer. Same conceptual variants as
// common_chat_decoded_event but post-shaping: per-format transformers may
// e.g. assemble TOOL_ARG_KV events from the decoder into a single
// TOOL_ARGS_JSON event that the presenter can copy verbatim.
struct common_chat_shaped_event {
    enum class kind : uint8_t {
        REASONING_TEXT,
        CONTENT_TEXT,
        TOOL_OPEN,
        TOOL_NAME,
        TOOL_ID,
        TOOL_ARGS_JSON,   // always JSON-string form, regardless of source format
        TOOL_CLOSE,
    };

    kind k;
    std::string text;
    bool is_partial = false;
};
