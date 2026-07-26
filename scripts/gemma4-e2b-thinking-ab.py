#!/usr/bin/env python3
"""E2B thinking A/B: does the empty-thought prefill still suppress ghost channels?

Gemma 4 emits a "ghost" thought channel on a thinking-OFF, schema-constrained
call. Google's fix is a prefill -- an opened-and-immediately-closed thought at the
end of the prompt -- shipped in the gemma-4-12B-it template and never backported
to the E2B repo. Measured in the Jinja era on this machine: E2B on its own
template failed 10/100 (avg 435 completion tokens, finish_reason=length, empty
content); E2B on the 12B-it template failed 0/100 (avg 24).

That fix now lives in the RENDERER, which emits the prefill for every Gemma 4
variant with no template involved. This re-runs the measurement against the
renderer, to confirm the property survived the port from Jinja to C++ -- a
regression check, not a decision about whether to keep it.

A failure is a call that produced no usable schema-conformant object: an empty or
truncated completion, or output the schema rejects. That is what the ghost
channel does -- it burns the budget and returns nothing.

Usage:
    gemma4-e2b-thinking-ab.py <base-url> <label> [N]

To produce the OFF arm, build with the prefill disabled in
common_chat_gemma4_render and run again with a different label.
"""
import json
import sys

import jsonschema
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "arm"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
MODEL = "gemma-4-e2b"

# The shape Protean's Canonicalizer and CacheGate actually send: a small object,
# every property required, nothing extra allowed, thinking off.
SCHEMA = {
    "type": "object",
    "properties": {
        "canonical": {"type": "string"},
        "confident": {"type": "boolean"},
    },
    "required": ["canonical", "confident"],
    "additionalProperties": False,
}

PROMPT = ("Rewrite this request as a short, phrasing-invariant search query, and say "
          "whether you are confident: 'hey, could you tell me who wrote Dune?'")


def main() -> int:
    fails, empties, lengths, completion_tokens = 0, 0, 0, []
    for seed in range(1, N + 1):
        body = {
            "model": MODEL,
            "temperature": 1.0, "top_p": 0.95, "top_k": 64,
            "seed": seed, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "canonical_query", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": PROMPT}],
        }
        try:
            r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=600)
            d = r.json()
            ch = d["choices"][0]
            content = ch["message"].get("content") or ""
            completion_tokens.append(d.get("usage", {}).get("completion_tokens", 0))
            bad = False
            if ch["finish_reason"] == "length":
                lengths += 1
                bad = True
            if not content.strip():
                empties += 1
                bad = True
            else:
                try:
                    jsonschema.validate(json.loads(content), SCHEMA)
                except Exception:
                    bad = True
            fails += bad
        except Exception as e:
            fails += 1
            print(f"  seed {seed}: transport {type(e).__name__}")
        if seed % 20 == 0:
            print(f"  {seed}/{N}  failures so far: {fails}", flush=True)

    avg = sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0
    print(f"\n=== {LABEL} (E2B, thinking off, schema-constrained) ===")
    print(f"    failures            : {fails}/{N}")
    print(f"      finish_reason=length: {lengths}")
    print(f"      empty content       : {empties}")
    print(f"    avg completion tokens: {avg:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
