#pragma once

#include <string>

// Convert a GBNF grammar to llguidance's Lark syntax.
//
// Why this exists: llguidance is the only sampler this fork constrains with, and
// it does not read GBNF. Its grammar list (`{"grammars":[...]}`) accepts exactly
// two entry kinds, `lark_grammar` and `json_schema` -- see GrammarWithLexer in
// llguidance's parser/src/api.rs -- so a caller's GBNF has to become Lark before
// it can be composed into a chat format's grammar as a subgrammar.
//
// This is a port of llguidance's OWN reference converter,
// python/llguidance/gbnf_to_lark.py, not a fresh interpretation of GBNF. The
// conversion decisions that are not obvious -- promoting all-terminal rules to
// uppercase lexemes, renaming `root` to `start`, camelCase to snake_case,
// stripping leading zeros from \u escapes -- are that script's, and are kept
// deliberately so a grammar converts identically here and there.
//
// Throws std::invalid_argument with a line number on malformed input. That type
// specifically: the server maps it to a 400, and a grammar the CALLER wrote is a
// bad request, not a server fault. Failing loudly matters more here than usual —
// a grammar that does not compile is one llguidance will not constrain with.
std::string common_gbnf_to_lark(const std::string & gbnf);
