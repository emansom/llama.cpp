#!/usr/bin/env python3
"""response_format enforcement, live, against the OpenAI-compatible server.

Four types: json_object / json_schema, unchanged from stock, plus the two vendor
extensions lark_grammar and gbnf_grammar. All of them compose into the Gemma 4
format grammar as a production, so ONE llguidance matcher and one token mask
result -- never a second sampler beside the format's own.

What each constrained case asks is not "did the model answer" but "could it have
answered any other way", so every one is paired with a prompt that pulls AWAY
from the constraint: asked for paragraphs of reasoning when the grammar admits
two words. A case that passes because the model felt like conforming proves
nothing, and that is exactly how an entirely unenforced response_format was once
reported clean.

Usage:
    llama-server -m gemma-4-12b-it.gguf --alias gemma-4-12b --port 8099 \\
        --chat-grammars-dir grammars/chat
    scripts/gemma4-response-format.py [base-url]
"""
import json
import re
import sys

import jsonschema
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
MODEL = "gemma-4-12b"

BASEBODY = {
    "model": MODEL,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "seed": 7,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": False},
}

GET_TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

# A prompt that actively wants prose, used wherever the grammar says otherwise.
PROSE_BAIT = ("Is the sky blue? Please explain your reasoning in a few paragraphs, "
              "with examples, before giving any answer.")

results = []


def post(body):
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=600)
    ctype = r.headers.get("content-type", "")
    return r.status_code, (r.json() if ctype.startswith("application/json") else r.text)


def case(title, body, *, expect_status=200, content_re=None, schema=None,
         expect_error_re=None, expect_calls=None):
    print(f"\n=== {title} ===")
    status, d = post(body)
    problems = []

    if status != expect_status:
        problems.append(f"http {status} (expected {expect_status}): {json.dumps(d)[:300]}")
    elif status != 200:
        msg = json.dumps(d)
        print(f"  error: {msg[:240]}")
        if expect_error_re and not re.search(expect_error_re, msg):
            problems.append(f"error body does not match /{expect_error_re}/")
    else:
        ch = d["choices"][0]
        content = ch["message"].get("content") or ""
        calls = ch["message"].get("tool_calls") or []
        print(f"  finish_reason: {ch['finish_reason']}  content: {content[:120]!r}"
              + (f"  calls: {[c['function']['name'] for c in calls]}" if calls else ""))
        # A generation that ran out of budget did not end on its own terms.
        if ch["finish_reason"] == "length":
            problems.append("finish_reason=length (ran to the token limit)")
        if content_re is not None and not re.fullmatch(content_re, content):
            problems.append(f"content {content!r} does not match /{content_re}/")
        if schema is not None:
            try:
                jsonschema.validate(json.loads(content), schema)
            except json.JSONDecodeError as e:
                problems.append(f"content is not JSON ({e})")
            except jsonschema.ValidationError as e:
                problems.append(f"schema: {e.message}")
        if expect_calls is not None:
            names = [c["function"]["name"] for c in calls]
            if names != expect_calls:
                problems.append(f"expected calls {expect_calls}, got {names}")

    for p in problems:
        print(f"  VIOLATION: {p}")
    ok = not problems
    print(f"  {'PASS' if ok else 'FAIL'}")
    results.append(ok)
    return ok


def main():
    yes_no_lark = '%llguidance {}\nstart: "yes" | "no"\n'
    yes_no_gbnf = 'root ::= "yes" | "no"\n'

    # ── the two new types actually constrain ─────────────────────────────────
    case("lark_grammar enforced against a prose-seeking prompt",
         {**BASEBODY,
          "response_format": {"type": "lark_grammar", "lark_grammar": yes_no_lark},
          "messages": [{"role": "user", "content": PROSE_BAIT}]},
         content_re=r"yes|no")

    # The same language in GBNF, which llguidance cannot read: it is converted to
    # Lark first, by a port of llguidance's own gbnf_to_lark.py.
    case("gbnf_grammar enforced (converted to Lark)",
         {**BASEBODY,
          "response_format": {"type": "gbnf_grammar", "gbnf_grammar": yes_no_gbnf},
          "messages": [{"role": "user", "content": PROSE_BAIT}]},
         content_re=r"yes|no")

    # The parts of GBNF that are not a transliteration: a character class,
    # bounded repetition, and a referenced sub-rule.
    case("gbnf_grammar with char class, repetition and a sub-rule",
         {**BASEBODY,
          "response_format": {"type": "gbnf_grammar",
                              "gbnf_grammar": 'root ::= "[" item ("," item)* "]"\n'
                                              'item ::= [A-Z] [a-z]{2,8}\n'},
          "messages": [{"role": "user", "content":
                        "Name three colours. Answer however you like, prose is fine."}]},
         content_re=r"\[[A-Z][a-z]{2,8}(,[A-Z][a-z]{2,8})*\]")

    # ── the type this all sits on top of, still binding ──────────────────────
    verdict = {"type": "object",
               "properties": {"ok": {"type": "boolean"}, "why": {"type": "string"}},
               "required": ["ok", "why"], "additionalProperties": False}
    case("json_schema still binding",
         {**BASEBODY,
          "response_format": {"type": "json_schema",
                              "json_schema": {"name": "verdict", "schema": verdict}},
          "messages": [{"role": "user", "content": "Is 17 prime? Answer as the schema."}]},
         schema=verdict)

    # ── composition with tools ───────────────────────────────────────────────
    # Stock llama-server refuses this outright ("Cannot use custom grammar
    # constraints with tools"), and has to: there the two are separate samplers
    # over one token stream, and two grammars whose languages do not nest can
    # intersect to nothing -- every logit -INF, with no diagnostic.
    case("lark_grammar + tools composes (stock rejects this)",
         {**BASEBODY, "tools": [GET_TIME], "tool_choice": "auto",
          "response_format": {"type": "lark_grammar", "lark_grammar": yes_no_lark},
          "messages": [{"role": "user", "content": "Is the sky blue?"}]},
         content_re=r"yes|no", expect_calls=[])

    # Precedence is stated, not a silent drop of one of the two: a turn cannot be
    # required to both call a tool and answer in a grammar, and the call is the
    # more specific instruction.
    case("tool_choice=required wins over a caller grammar",
         {**BASEBODY, "tools": [GET_TIME], "tool_choice": "required",
          "response_format": {"type": "lark_grammar", "lark_grammar": yes_no_lark},
          "messages": [{"role": "user", "content": "What time is it in Oslo?"}]},
         expect_calls=["get_time"])

    # ── a grammar that cannot be honoured must be an ERROR ───────────────────
    # Not prose. llguidance answers a grammar it cannot build by constraining
    # nothing, and the sampler used to pass that off as success.
    case("malformed lark_grammar is a request error, not silent prose",
         {**BASEBODY,
          "response_format": {"type": "lark_grammar",
                              "lark_grammar": '%llguidance {}\nstart: "unterminated\n'},
          "messages": [{"role": "user", "content": "Is the sky blue?"}]},
         expect_status=400, expect_error_re=r"(?i)grammar")

    case("malformed gbnf_grammar is a request error, naming the rule",
         {**BASEBODY,
          "response_format": {"type": "gbnf_grammar", "gbnf_grammar": 'root ::= undefined-rule\n'},
          "messages": [{"role": "user", "content": "Is the sky blue?"}]},
         expect_status=400, expect_error_re=r"undefined-rule")

    case("unknown response_format type is rejected, listing what is accepted",
         {**BASEBODY,
          "response_format": {"type": "yaml"},
          "messages": [{"role": "user", "content": "hi"}]},
         expect_status=400, expect_error_re=r"lark_grammar")

    # The declared marker that tells the two syntaxes apart downstream. Missing
    # it, the grammar would be parsed as GBNF and fail somewhere less obvious.
    case("lark_grammar without %llguidance is rejected",
         {**BASEBODY,
          "response_format": {"type": "lark_grammar", "lark_grammar": 'start: "yes"\n'},
          "messages": [{"role": "user", "content": "hi"}]},
         expect_status=400, expect_error_re=r"llguidance")

    # ── and the ordinary path is untouched ───────────────────────────────────
    case("no response_format: ordinary prose, unchanged",
         {**BASEBODY,
          "messages": [{"role": "user", "content": "Say hello in one short sentence."}]},
         content_re=r"(?s).{3,}")

    print(f"\n\n===== {sum(results)}/{len(results)} cases clean =====")
    print("Cross-check the server log: one 'constraint ACTIVE' per generating "
          "request, and no 'DROPPED' or 'UNCONSTRAINED' lines at all.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
