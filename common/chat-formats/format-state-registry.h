#pragma once

#include "chat-formats/format-tracker.h"
#include "peg-parser.h"

#include <map>
#include <string>
#include <vector>

// Per-format declaration mapping FSM states to grammar rule names.
//
// Each format's codec declares which `common_chat_format_state` values it
// reaches and the grammar rule that backs each one. This makes the
// FSM-to-grammar contract explicit rather than implicit (i.e. relying on
// the C++ tracker and llguidance's internal grammar state to agree by
// virtue of consuming the same byte stream).
//
// The registry is consulted to:
//   1. Validate at startup that every state's declared rule exists in the
//      format's grammar arena (catches grammar/tracker drift; fail-fast).
//   2. Provide per-state rule-name lookups for diagnostics and the
//      bidirectional pipeline (Package C+).
//
// Rule names are FORMAT-SPECIFIC (e.g. Kimi-K2's `func_name` vs another
// format's `function_name`). Tag names (e.g. `tool-name`) live elsewhere
// — they are auto-attached by lark-to-peg based on rule-name conventions
// and are queried at parse time rather than declared statically.
struct common_chat_format_state_rules {
    std::map<common_chat_format_state, std::string> by_state;

    // Returns the rule name for `state`, or an empty string if the state
    // is not declared by this format.
    std::string rule_for(common_chat_format_state state) const;

    // Returns the de-duplicated list of all rule names declared for any
    // state. Used by validation and tests.
    std::vector<std::string> declared_rules() const;

    // Returns the names of declared rules that DO NOT exist in `arena`.
    // An empty vector means the registry is consistent with the grammar.
    std::vector<std::string> missing_in(const common_peg_arena & arena) const;
};
