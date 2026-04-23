#pragma once

#include "peg-parser.h"

#include <string>

/**
 * Translate a GBNF grammar string into a common_peg_arena.
 *
 * Each named rule (rule-name ::= body) becomes a named PEG rule node,
 * preserving the full hierarchical structure of the GBNF grammar.
 *
 * Supported GBNF syntax:
 *   rule-name ::= body          -> p.rule("rule-name", body)
 *   "literal"                   -> p.literal("literal")
 *   [char-class]                -> p.chars("char-class")
 *   rule1 rule2  (sequence)     -> p.sequence({rule1, rule2})
 *   rule1 | rule2 (alternation) -> p.choice({rule1, rule2})
 *   rule?                       -> p.optional(rule)
 *   rule*                       -> p.zero_or_more(rule)
 *   rule+                       -> p.one_or_more(rule)
 *   rule-name  (ref)            -> p.ref("rule-name")
 *   (group)                     -> inline group
 *
 * Throws std::runtime_error on parse errors.
 */
common_peg_arena common_gbnf_to_peg(const std::string & gbnf_grammar);
