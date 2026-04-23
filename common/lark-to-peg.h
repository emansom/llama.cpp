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
common_peg_arena common_lark_to_peg(const std::string & lark_grammar);
