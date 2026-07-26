#!/usr/bin/env python3
"""Did llguidance actually steer the model, on a LIVE OpenAI-compatible server?

`tests/test-grammar-llguidance.cpp` walks the mask offline and proves the grammar
admits only the FSM-correct tokens. That is the stronger check, but it exercises
the sampler in isolation. This one asks the same question of the running server,
because that -- not the library -- is what a client talks to.

The method: ask for `logprobs`, which report the model's own PRE-mask
distribution, and compare its top-1 at each position against the token actually
emitted. A mismatch at a position the grammar had pinned is the mask overriding
the model, visible from outside.

COMPARE TOKEN IDS, NEVER THE RENDERED TEXT. The first version of this script
compared `token` strings and reported a false override on every `<|tool_response>`:
that field is rendered with special=false, and llama_vocab::token_to_piece returns
EMPTY for a CONTROL token, so id 50 shows up as '' in the candidate list and as
'<|tool_response>' in the emitted field. Two renderings of one token read as a
disagreement. It produced a confident and completely wrong claim -- that the model
wanted to end the turn and was overruled into the hand-over -- when measurement on
an unconstrained server shows it emits that token itself at p=0.9998.

Two more things NOT to conclude from a mismatch:

  * Inside a free string body (a tool argument's text) the grammar admits ~262k
    tokens, so a mismatch there is ordinary temperature sampling, not the mask.
    Cross-reference the offline walk for which positions are actually pinned.
  * `post_sampling_probs: true` does NOT show the mask. It reports the
    distribution after the whole sampler chain, which has already collapsed to
    the selected token -- measured: exactly one candidate at every position,
    including positions where 262044 tokens were legal.

The prompt is deliberately one the model wants to answer conversationally, with
tool_choice=required. Its own first choice is therefore prose, and the grammar
has to overrule it.

Usage:  gemma4-live-mask-probe.py <base-url> [model]
"""
import json
import sys

import requests

PROMPT = "hello, how are you today?"
TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def main() -> int:
    base_url = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma-4-12b"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [TOOL],
        "tool_choice": "required",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": 1,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "logprobs": True,
        "top_logprobs": 5,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=300)
    if r.status_code != 200:
        print(f"http {r.status_code}: {r.text[:300]}")
        return 1

    choice = r.json()["choices"][0]
    lp = choice.get("logprobs")
    if not lp:
        print("server returned no logprobs; cannot probe the mask")
        return 1

    def show(entry: dict) -> str:
        """Render a candidate unambiguously: CONTROL tokens come back empty."""
        text = (entry.get("token") or "").replace("\n", ".")
        return f"{text or '<control>'}#{entry['id']}"

    print(f"prompt: {PROMPT!r}   tool_choice=required")
    print(f"{'step':<5}{'EMITTED':<24}{'model top-1 (pre-mask)':<26}")
    overrides = 0
    for i, t in enumerate(lp["content"]):
        tops = t.get("top_logprobs") or []
        top = tops[0] if tops else None
        # IDs, not strings -- see the note at the top of this file.
        differs = top is not None and top["id"] != t["id"]
        overrides += differs
        raw = show(top) if top else "?"
        print(f"{i:<5}{show(t):<24}{raw:<26}{'<-- overruled' if differs else ''}")

    print(f"\npositions where the emitted token was not the model's top choice: "
          f"{overrides}/{len(lp['content'])}")
    print("  (that count mixes MASK events with ordinary sampling -- a mismatch")
    print("   inside a free string body is temperature, not the grammar. Cross-")
    print("   reference test_gemma4_mask_walk for which positions are pinned.)")
    print(f"finish_reason: {choice['finish_reason']}")
    print(f"tool_calls:    {json.dumps(choice['message'].get('tool_calls'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
