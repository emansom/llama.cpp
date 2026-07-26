#!/usr/bin/env python3
"""Protean-shaped tool calls against the OpenAI-compatible server.

Not the toy one-tool cases: the shapes Protean's Phase 3/4 knowledge stack and
scoped-tool stages actually produce -- several tools declared at once, dotted MCP
names, arrays of enums, nested objects, parallel fan-out, and the full tool
round-trip where results are fed back and the model continues INSIDE the same
turn via the <|tool_response> handshake.
"""
import json
import sys

import jsonschema
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
MODEL = "gemma-4-12b"

WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web.search",  # dotted, MCP-style
        "description": "Search the web via SearxNG.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "engines": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["duckduckgo", "wikipedia", "stackexchange"]},
                    "description": "Engines to query.",
                },
                "time_range": {"type": "string", "enum": ["day", "week", "month", "year"]},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

KB_RETRIEVE = {
    "type": "function",
    "function": {
        "name": "knowledge.retrieve",
        "description": "Retrieve passages from the local ZIM/RAG corpora.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "corpus": {
                    "type": "string",
                    "enum": ["devdocs", "archwiki", "wikipedia", "stackexchange"],
                },
                "top_k": {"type": "integer"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "lang": {"type": "string", "enum": ["en", "nl", "de"]},
                        "min_score": {"type": "number"},
                        "exclude_stale": {"type": "boolean"},
                    },
                    "required": ["lang"],
                    "additionalProperties": False,
                },
            },
            "required": ["query", "corpus"],
            "additionalProperties": False,
        },
    },
}

READ_FILE = {
    "type": "function",
    "function": {
        "name": "filesystem.read_file",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

RUN_SQL = {
    "type": "function",
    "function": {
        "name": "postgres.query",
        "description": "Run a read-only SQL query against the memory store.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "params": {"type": "array", "items": {"type": "string"}},
                "timeout_ms": {"type": "integer"},
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
}

EMBED = {
    "type": "function",
    "function": {
        "name": "embed.text",
        "description": "Embed text with the Granite embedder.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "normalize": {"type": "boolean"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}

ALL_TOOLS = [WEB_SEARCH, KB_RETRIEVE, READ_FILE, RUN_SQL, EMBED]
SCHEMA_BY_NAME = {t["function"]["name"]: t["function"]["parameters"] for t in ALL_TOOLS}


def post(body: dict) -> dict:
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=600)
    if r.status_code != 200:
        return {"__error__": f"http {r.status_code}: {r.text[:400]}"}
    return r.json()


def check_calls(calls: list) -> list[str]:
    """Validate every returned call against ITS OWN declared schema."""
    problems = []
    for c in calls or []:
        fn = c.get("function", {})
        name = fn.get("name")
        if name not in SCHEMA_BY_NAME:
            problems.append(f"undeclared tool {name!r}")
            continue
        try:
            args = json.loads(fn.get("arguments", ""))
        except json.JSONDecodeError as e:
            problems.append(f"{name}: arguments not JSON ({e})")
            continue
        try:
            jsonschema.validate(args, SCHEMA_BY_NAME[name])
        except jsonschema.ValidationError as e:
            problems.append(f"{name}: {e.message}")
    return problems


def scenario(title: str, body: dict, expect_calls: int | None = None) -> bool:
    print(f"\n=== {title} ===")
    d = post(body)
    if "__error__" in d:
        print(f"  FAIL {d['__error__']}")
        return False
    ch = d["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    print(f"  finish_reason: {ch['finish_reason']}   calls: {len(calls)}")
    for c in calls:
        print(f"    -> {c['function']['name']}({c['function']['arguments']})")
    if ch["message"].get("reasoning_content"):
        rc = ch["message"]["reasoning_content"].replace("\n", " ")[:110]
        print(f"    reasoning: {rc}...")
    if ch["message"].get("content"):
        print(f"    content: {ch['message']['content'][:160]!r}")

    problems = check_calls(calls)

    # Validate response_format too. Without this the no-tool-call scenarios pass
    # by default, which is exactly how a completely unenforced schema was first
    # reported as clean.
    rf = body.get("response_format")
    if rf and rf.get("type") == "json_schema":
        schema = rf["json_schema"]["schema"]
        raw = ch["message"].get("content") or ""
        try:
            jsonschema.validate(json.loads(raw), schema)
        except json.JSONDecodeError as e:
            problems.append(f"response_format: content is not JSON ({e})")
        except jsonschema.ValidationError as e:
            problems.append(f"response_format: {e.message}")

    # A generation that ran out of budget did not end on its own terms; treat it
    # as a failure rather than reading whatever it managed to emit.
    if ch["finish_reason"] == "length":
        problems.append("finish_reason=length (ran to the token limit)")

    for p in problems:
        print(f"  VIOLATION: {p}")
    ok = not problems
    if expect_calls is not None and len(calls) != expect_calls:
        print(f"  NOTE: expected {expect_calls} call(s), got {len(calls)}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    base = {
        "model": MODEL,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": 3,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    results = []

    # 1. Five tools declared; the model must pick the right one and fill a nested
    #    object with its own required field.
    results.append(scenario(
        "5 tools declared, nested object argument",
        {**base, "tools": ALL_TOOLS, "tool_choice": "required",
         "messages": [{"role": "user", "content":
                       "Look up how systemd socket activation works in the Arch wiki, "
                       "English only, give me the top 3 passages."}]},
        expect_calls=1))

    # 2. Array-of-enum argument, dotted tool name.
    results.append(scenario(
        "array-of-enum argument, dotted MCP name",
        {**base, "tools": ALL_TOOLS, "tool_choice": "required",
         "messages": [{"role": "user", "content":
                       "Search the web for pgvector HNSW tuning, using duckduckgo and "
                       "stackexchange, limited to the past month."}]},
        expect_calls=1))

    # 3. Parallel fan-out -- Protean's `parallel` strategy shape.
    results.append(scenario(
        "parallel tool calls",
        {**base, "tools": ALL_TOOLS, "tool_choice": "required",
         "parallel_tool_calls": True,
         "messages": [{"role": "user", "content":
                       "Do these two independent things: read /etc/protean/env, and "
                       "embed the sentence 'hello world'."}]}))

    # 4. THE ROUND TRIP. An assistant turn that called a tool, the tool's result,
    #    and the model continuing inside the same turn. This is what every agent
    #    loop does and what the <|tool_response> handshake exists for.
    results.append(scenario(
        "tool round-trip: result fed back, model continues",
        {**base, "tools": ALL_TOOLS,
         "messages": [
             {"role": "user", "content": "What's in /etc/protean/env?"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"id": "call_1", "type": "function",
                              "function": {"name": "filesystem.read_file",
                                           "arguments": '{"path":"/etc/protean/env"}'}}]},
             {"role": "tool", "tool_call_id": "call_1",
              "content": "LLAMACPP_MODEL=gemma-4-12b\nREDIS_URL=unix:///run/protean/redis/redis.sock"},
         ]}))

    # 5. Round trip WITH thinking on -- the post-tool-response thought re-opener,
    #    which is a different grammar entry (resume_reasoning).
    results.append(scenario(
        "tool round-trip with thinking enabled (resume_reasoning entry)",
        {**base, "tools": ALL_TOOLS,
         "chat_template_kwargs": {"enable_thinking": True},
         "messages": [
             {"role": "user", "content": "How many rows are in the memory table?"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"id": "call_9", "type": "function",
                              "function": {"name": "postgres.query",
                                           "arguments": '{"sql":"select count(*) from memory_entries"}'}}]},
             {"role": "tool", "tool_call_id": "call_9", "content": "count = 40213"},
         ]}))

    # 6. tools + response_format together -- stock 400s on grammar+tools; the
    #    fork is supposed to compose them.
    results.append(scenario(
        "tools + response_format json_schema together",
        {**base, "tools": ALL_TOOLS, "tool_choice": "none",
         "response_format": {"type": "json_schema", "json_schema": {
             "name": "verdict",
             "schema": {"type": "object",
                        "properties": {"ok": {"type": "boolean"}, "why": {"type": "string"}},
                        "required": ["ok", "why"], "additionalProperties": False}}},
         "messages": [{"role": "user", "content": "Is 17 a prime number? Answer as the schema."}]},
        expect_calls=0))

    print(f"\n\n===== {sum(results)}/{len(results)} scenarios clean =====")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
