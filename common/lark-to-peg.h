#pragma once

#include "peg-parser.h"

#include <string>

/**
 * Translate a Lark grammar string into a common_peg_arena.
 *
 * The translation preserves the full hierarchical AST structure of the Lark grammar:
 * each named rule becomes a named PEG rule node, ensuring the PEG parser's AST mirrors
 * what llguidance enforces during constrained sampling.
 *
 * Supported Lark syntax:
 *   rule_name: body           -> p.rule("rule_name", body)
 *   TERMINAL: /regex/         -> p.rule("TERMINAL", p.chars("regex"))
 *   "literal"                 -> p.literal("literal")
 *   /regex/                   -> p.chars("regex")
 *   a b c  (sequence)         -> p.sequence({a, b, c})
 *   a | b  (alternation)      -> p.choice({a, b})
 *   rule?                     -> p.optional(rule)
 *   rule*                     -> p.zero_or_more(rule)
 *   rule+                     -> p.one_or_more(rule)
 *   rule_name  (ref)          -> p.ref("rule_name")
 *   %ignore /pattern/         -> ignored (whitespace handling via PEG context)
 *   %llguidance {}            -> stripped directive
 *   (group)                   -> inline group (no named node)
 *
 * Throws std::runtime_error on parse errors.
 */
// Transpile a Lark grammar to a PEG arena.
//
// root_rule selects the entry production. "start" is the assistant-turn
// continuation used for generation; "conversation" is the full wire shape, used
// to VALIDATE a rendered prompt and walk the FSM up to the point generation
// begins. Both live in the same grammar, so the input and output views of the
// format cannot drift apart.
//
// Naming a production the grammar does not define throws, rather than silently
// falling back to the default entry and parsing the wrong thing.
//
// require_eof anchors the root at end of input. A validator MUST set it: without
// it the root may match a PREFIX and report success, which is not validation --
// `conversation: closed_turn* open_model_turn?` happily matches the empty string
// and leaves the entire prompt unread. It also makes ordered choice work as
// written, since an alternative that matches vacuously no longer wins over the
// one that consumes the input.
//
// Generation must leave it OFF: that parse is incremental and a partial response
// is expected to stop mid-rule.
common_peg_arena common_lark_to_peg(const std::string & lark_grammar,
                                    const std::string & root_rule = "start",
                                    bool                require_eof = false);
