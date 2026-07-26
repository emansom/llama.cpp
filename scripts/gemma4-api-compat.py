#!/usr/bin/env python3
"""OpenAI-compatibility corpus: the fork's wire surface against stock llama-server.

FORK.md §1.0 makes one binding promise -- a client that works against stock
llama-server works against the fork -- and this is what checks it. Everything the
fork does internally (no Jinja, no format detection, one grammar engine) is an
implementation change behind an API that must not move.

WHAT IS COMPARED IS SHAPE, NOT TEXT. Two servers running a model at temperature
1.0 never produce the same words, and the same server does not produce them twice.
So each case is reduced to a STRUCTURAL DIGEST: HTTP status, which keys exist and
at what types, `finish_reason`, whether tool-call arguments parse as JSON, the SSE
chunk shape. Content, token counts, ids and timestamps are excluded by
construction -- comparing them would be noise that always fails.

Two groups, and the split is the point:

  COMPAT     the stock-supported surface. Any difference here is a REGRESSION:
             the fork changed something a client can see.
  EXTENSION  surface the fork deliberately adds or widens (response_format
             grammars, `tools` alongside a grammar). Differences are EXPECTED and
             are reported rather than failed -- they are the feature.

Usage:
    gemma4-api-compat.py capture <base-url> <out.json>
    gemma4-api-compat.py compare <stock.json> <fork.json>

Run the two captures SEQUENTIALLY, not side by side: two 12B instances will not
share a 16 GB card.
"""
import json
import sys

import requests

MODEL = "gemma-4-12b"

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

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["ok", "why"],
    "additionalProperties": False,
}

BASE = {
    "model": MODEL,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "seed": 11,
    "max_tokens": 96,
}

# Tool and structured cases get room to FINISH, and thinking off -- which is what
# Protean sends for them anyway.
#
# They were on BASE's 96 tokens with thinking on, and the reasoning channel ate
# the whole budget: stock ended `finish_reason: length` having produced no call,
# the fork ended `tool_calls` having produced one, and the digest dutifully
# reported ten structural differences. None were real. Comparing a response that
# never finished against one that did is comparing noise, so the budget is raised
# here and `compare` refuses such a case outright (see INCONCLUSIVE).
ROOMY = {**BASE, "max_tokens": 384, "chat_template_kwargs": {"enable_thinking": False}}


def U(msg):
    return [{"role": "user", "content": msg}]


# (name, group, method, path, body)  -- group is "compat" or "extension".
CORPUS = [
    # ── the plain surface ────────────────────────────────────────────────────
    ("chat.plain", "compat", "POST", "/v1/chat/completions",
     {**BASE, "messages": U("Say hello in one short sentence.")}),
    ("chat.system+multiturn", "compat", "POST", "/v1/chat/completions",
     {**BASE, "messages": [
         {"role": "system", "content": "You are terse."},
         {"role": "user", "content": "Name a colour."},
         {"role": "assistant", "content": "Blue."},
         {"role": "user", "content": "Another."}]}),
    ("chat.stop_sequence", "compat", "POST", "/v1/chat/completions",
     {**BASE, "stop": ["\n"], "messages": U("List three fruits, one per line.")}),
    ("chat.max_tokens_truncates", "compat", "POST", "/v1/chat/completions",
     {**BASE, "max_tokens": 8, "messages": U("Write a long paragraph about the sea.")}),

    # ── thinking, both ways, via the extension Protean actually sends ────────
    # Its own budget: a thought plus an answer does not fit in BASE's 96 tokens,
    # and a case that truncates is a case that does not get compared.
    ("chat.thinking_on", "compat", "POST", "/v1/chat/completions",
     {**BASE, "max_tokens": 768, "chat_template_kwargs": {"enable_thinking": True},
      "messages": U("What is 2+2?")}),
    ("chat.thinking_off", "compat", "POST", "/v1/chat/completions",
     {**BASE, "chat_template_kwargs": {"enable_thinking": False},
      "messages": U("What is 2+2?")}),

    # ── tools ────────────────────────────────────────────────────────────────
    ("tools.auto_expect_call", "compat", "POST", "/v1/chat/completions",
     {**ROOMY, "tools": [GET_TIME], "tool_choice": "auto",
      "messages": U("What time is it in Oslo?")}),
    ("tools.required", "compat", "POST", "/v1/chat/completions",
     {**ROOMY, "tools": [GET_TIME], "tool_choice": "required",
      "messages": U("What time is it in Oslo?")}),
    # Answerable WITHOUT the tool, deliberately. Asking the time with
    # `tool_choice: none` strands the model -- it may not call, and it does not
    # know -- and a stranded model rambles: measured 4/5 stop, 1/5 length on the
    # fork, and stock is no steadier. That tests the model's composure, not the
    # API. What this case is for is the SHAPE: tool_choice none yields an ordinary
    # message with no tool_calls.
    ("tools.none", "compat", "POST", "/v1/chat/completions",
     {**ROOMY, "tools": [GET_TIME], "tool_choice": "none",
      "messages": U("What is the capital of Norway? One word.")}),
    ("tools.roundtrip", "compat", "POST", "/v1/chat/completions",
     {**ROOMY, "tools": [GET_TIME],
      "messages": [
          {"role": "user", "content": "What time is it in Oslo?"},
          {"role": "assistant", "content": "",
           "tool_calls": [{"id": "call_1", "type": "function",
                           "function": {"name": "get_time",
                                        "arguments": '{"city":"Oslo"}'}}]},
          {"role": "tool", "tool_call_id": "call_1", "content": "13:45"}]}),

    # ── structured output ────────────────────────────────────────────────────
    ("format.json_schema", "compat", "POST", "/v1/chat/completions",
     {**ROOMY, "response_format": {"type": "json_schema",
                                   "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}},
      "messages": U("Is 17 prime? Answer as the schema.")}),
    ("format.json_object", "compat", "POST", "/v1/chat/completions",
     {**ROOMY, "response_format": {"type": "json_object"},
      "messages": U("Give a JSON object with one key 'x'.")}),
    ("format.text", "compat", "POST", "/v1/chat/completions",
     {**BASE, "response_format": {"type": "text"}, "messages": U("Say hi.")}),

    # ── streaming ────────────────────────────────────────────────────────────
    ("stream.plain", "compat", "STREAM", "/v1/chat/completions",
     {**BASE, "stream": True, "messages": U("Count to three.")}),
    ("stream.tools", "compat", "STREAM", "/v1/chat/completions",
     {**ROOMY, "stream": True, "tools": [GET_TIME], "tool_choice": "required",
      "messages": U("What time is it in Oslo?")}),
    ("stream.json_schema", "compat", "STREAM", "/v1/chat/completions",
     {**ROOMY, "stream": True,
      "response_format": {"type": "json_schema",
                          "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}},
      "messages": U("Is 17 prime? Answer as the schema.")}),

    # ── non-chat endpoints ───────────────────────────────────────────────────
    ("models.list", "compat", "GET", "/v1/models", None),
    ("health", "compat", "GET", "/health", None),
    ("props", "compat", "GET", "/props", None),

    # ── errors keep the OpenAI shape ─────────────────────────────────────────
    ("error.no_messages", "compat", "POST", "/v1/chat/completions", {**BASE}),
    ("error.bad_tool_choice", "compat", "POST", "/v1/chat/completions",
     {**BASE, "tools": [GET_TIME], "tool_choice": "sometimes",
      "messages": U("hi")}),
    ("error.bad_response_format", "compat", "POST", "/v1/chat/completions",
     {**BASE, "response_format": {"type": "yaml"}, "messages": U("hi")}),

    # ── EXTENSIONS: expected to differ, reported not failed ──────────────────
    ("ext.lark_grammar", "extension", "POST", "/v1/chat/completions",
     {**ROOMY, "response_format": {"type": "lark_grammar",
                                   "lark_grammar": '%llguidance {}\nstart: "yes" | "no"\n'},
      "messages": U("Is the sky blue?")}),
    ("ext.gbnf_grammar", "extension", "POST", "/v1/chat/completions",
     {**ROOMY, "response_format": {"type": "gbnf_grammar",
                                   "gbnf_grammar": 'root ::= "yes" | "no"\n'},
      "messages": U("Is the sky blue?")}),
    ("ext.grammar_with_tools", "extension", "POST", "/v1/chat/completions",
     {**ROOMY, "tools": [GET_TIME], "tool_choice": "auto",
      "response_format": {"type": "lark_grammar",
                          "lark_grammar": '%llguidance {}\nstart: "yes" | "no"\n'},
      "messages": U("Is the sky blue?")}),
]


def typename(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    if v is None:
        return "null"
    return type(v).__name__


def digest_message(m):
    """Structure of choices[0].message, with volatile parts abstracted away."""
    d = {"keys": sorted(k for k in m if k not in ("content", "reasoning_content"))}
    d["content_type"] = typename(m.get("content"))
    # Presence is model-dependent (did it think?), so this is reported, never
    # compared -- see SOFT below.
    d["_soft_has_reasoning"] = "reasoning_content" in m and bool(m.get("reasoning_content"))
    calls = m.get("tool_calls") or []
    d["n_tool_calls_class"] = "0" if not calls else ("1" if len(calls) == 1 else "many")
    if calls:
        c = calls[0]
        d["tool_call_keys"] = sorted(c.keys())
        d["tool_call_type"] = c.get("type")
        fn = c.get("function", {})
        d["tool_fn_keys"] = sorted(fn.keys())
        d["tool_name_is_declared"] = fn.get("name") == "get_time"
        try:
            json.loads(fn.get("arguments", ""))
            d["tool_args_parse"] = True
        except Exception:
            d["tool_args_parse"] = False
    return d


def digest_body(body):
    if not isinstance(body, dict):
        return {"toplevel_type": typename(body)}
    d = {"keys": sorted(k for k in body if k not in ("id", "created", "timings"))}
    if "error" in body:
        e = body["error"]
        d["error_keys"] = sorted(e.keys()) if isinstance(e, dict) else typename(e)
        d["error_type"] = e.get("type") if isinstance(e, dict) else None
        return d
    ch = (body.get("choices") or [{}])[0]
    d["choice_keys"] = sorted(ch.keys())
    d["finish_reason"] = ch.get("finish_reason")
    if "message" in ch:
        d["message"] = digest_message(ch["message"])
    if "usage" in body and isinstance(body["usage"], dict):
        d["usage_keys"] = sorted(body["usage"].keys())
        d["usage_all_int"] = all(isinstance(v, int) for v in body["usage"].values())
    return d


def capture_stream(url, body):
    r = requests.post(url, json=body, stream=True, timeout=600)
    d = {"status": r.status_code}
    if r.status_code != 200:
        try:
            d.update(digest_body(r.json()))
        except Exception:
            d["nonjson_error"] = True
        return d
    n, done, objs, first_delta, saw_finish, saw_role, saw_tool = 0, False, set(), None, False, False, False
    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        payload = raw[6:]
        if payload.strip() == "[DONE]":
            done = True
            continue
        n += 1
        try:
            c = json.loads(payload)
        except Exception:
            continue
        objs.add(c.get("object"))
        ch = (c.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        if first_delta is None and delta:
            first_delta = sorted(delta.keys())
        if "role" in delta:
            saw_role = True
        if delta.get("tool_calls"):
            saw_tool = True
        if ch.get("finish_reason"):
            saw_finish = True
    d.update({"had_chunks": n > 0, "saw_DONE": done, "chunk_objects": sorted(o for o in objs if o),
              "first_delta_keys": first_delta, "saw_role_delta": saw_role,
              "saw_finish_reason": saw_finish, "saw_tool_call_delta": saw_tool})
    return d


def capture(base):
    out = {}
    for name, group, method, path, body in CORPUS:
        url = base + path
        print(f"  {name} ...", end="", flush=True)
        try:
            if method == "STREAM":
                d = capture_stream(url, body)
            elif method == "GET":
                r = requests.get(url, timeout=60)
                d = {"status": r.status_code}
                try:
                    d.update(digest_body(r.json()))
                except Exception:
                    d["nonjson"] = True
            else:
                r = requests.post(url, json=body, timeout=600)
                d = {"status": r.status_code}
                try:
                    d.update(digest_body(r.json()))
                except Exception:
                    d["nonjson"] = True
        except Exception as e:
            d = {"transport_error": type(e).__name__}
        d["_group"] = group
        out[name] = d
        print(f" {d.get('status', d.get('transport_error'))}")
    return out


def flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


def compare(a, b):
    regressions, expected, soft, inconclusive = [], [], [], []
    for name in sorted(set(a) | set(b)):
        da, db = a.get(name), b.get(name)
        group = (db or da or {}).get("_group", "compat")
        if da is None or db is None:
            regressions.append((name, "present in only one capture", da, db))
            continue

        # A response that ran out of tokens never finished, so it has no shape to
        # compare -- half a message is not a different message. Skip the case and
        # say so, rather than reporting the difference between a truncated
        # generation and a complete one.
        #
        # Not a loophole: it fires only where the corpus did NOT ask for
        # truncation, and it is loud. Ten "regressions" in the first run were all
        # this, from a max_tokens too small for a tool call with thinking on.
        if name != "chat.max_tokens_truncates" and \
                "length" in (da.get("finish_reason"), db.get("finish_reason")):
            inconclusive.append((name, "finish_reason=length: generation did not complete",
                                 da.get("finish_reason"), db.get("finish_reason")))
            continue

        fa, fb = flatten(da), flatten(db)
        for k in sorted(set(fa) | set(fb)):
            if k == "_group":
                continue
            va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
            if va == vb:
                continue
            entry = (name, k, va, vb)
            if k.split(".")[-1].startswith("_soft"):
                soft.append(entry)
            elif group == "extension":
                expected.append(entry)
            else:
                regressions.append(entry)

    def show(title, rows):
        print(f"\n{title} ({len(rows)})")
        for name, k, va, vb in rows:
            print(f"  {name:28s} {k:34s} stock={va!r:24s} fork={vb!r}")

    show("REGRESSIONS -- compat surface moved", regressions)
    show("EXPECTED -- deliberate fork extensions", expected)
    show("SOFT -- model-dependent, informational", soft)
    show("INCONCLUSIVE -- truncated, nothing to compare", inconclusive)
    n_compat = sum(1 for v in b.values() if v.get("_group") == "compat")
    print(f"\n{n_compat} compat cases; {len(regressions)} regression(s), "
          f"{len(inconclusive)} inconclusive.")
    return 1 if regressions else 0


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "capture":
        out = capture(sys.argv[2].rstrip("/"))
        with open(sys.argv[3], "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print(f"wrote {sys.argv[3]}")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        with open(sys.argv[2]) as f:
            a = json.load(f)
        with open(sys.argv[3]) as f:
            b = json.load(f)
        return compare(a, b)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
