#!/usr/bin/env python3
"""Adversarial tool-argument conformance, N samples against one llama-server.

The measurement Phase 1 exists to move: ONE declared tool with a strict schema,
tool_choice=required, and a user message written so that the natural answer does
NOT fit the schema. Selection cannot fail (there is one tool); the only thing
being measured is whether the ARGUMENTS conform.

Conformance is the schema, checked with jsonschema -- required properties
present, declared types, enum membership, and no invented properties. Property
ORDER is deliberately not checked: it is not part of JSON Schema, and the Gemma 4
emitter fixes an order only because permuting N optionals is N! alternatives.

Usage:  adversarial_tool_args.py <base-url> <label> [N]
"""
import json
import sys

import jsonschema
import requests

"""CASES

`meeting` is the realistic one: an ordinary tool, a request phrased the way a
person phrases things. `incident` deliberately maximises pressure on every axis
a schema can constrain at once -- enum, integer, boolean, array-of-enum, nested
object, several required fields the prose never mentions -- because the point of
an adversarial measurement is the ceiling of the failure, not the average.

Both were designed once, from that description, and run as-is. Neither was
adjusted after seeing a score; tuning an adversarial case until the baseline
fails is how you measure your own patience rather than the software.
"""

PARAMETERS = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short title for the meeting."},
        "day": {
            "type": "string",
            "enum": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "description": "Weekday to schedule the meeting on.",
        },
        "hour": {"type": "integer", "description": "Start hour on a 24-hour clock."},
        "recurring": {"type": "boolean", "description": "Whether it repeats weekly."},
    },
    "required": ["title", "day", "hour"],
    "additionalProperties": False,
}

TOOL = {
    "type": "function",
    "function": {
        "name": "schedule_meeting",
        "description": "Schedule a meeting on the shared work calendar.",
        "parameters": PARAMETERS,
    },
}

# Adversarial on every axis the schema constrains, and it names none of the
# fields:
#   "the weekend"        -> outside the `day` enum (mon-fri)
#   "late afternoon-ish" -> not an integer `hour`
#   attendees / link / reminder -> properties that do not exist
#   "every week"         -> the one detail that does fit, so the request is not
#                           uniformly hopeless and the model has a reason to try
USER_MESSAGE = (
    "Set something up for the weekend, late afternoon-ish, and call it "
    "'design sync'. Add Priya and Tom to it, drop the video link in, and remind "
    "me fifteen minutes before. It should happen every week."
)

# --- case 2: maximum pressure on every axis a schema can constrain -------------
INCIDENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One-line summary."},
        "severity": {
            "type": "string",
            "enum": ["sev1", "sev2", "sev3"],
            "description": "Severity band.",
        },
        "affected_minutes": {"type": "integer", "description": "Whole minutes of impact."},
        "customer_facing": {"type": "boolean", "description": "Did customers see it."},
        "components": {
            "type": "array",
            "items": {"type": "string", "enum": ["api", "database", "cache", "frontend"]},
            "description": "Components implicated.",
        },
        "oncall": {
            "type": "object",
            "properties": {
                "primary": {"type": "string", "description": "Primary responder."},
                "escalated": {"type": "boolean", "description": "Was it escalated."},
            },
            "required": ["primary", "escalated"],
            "additionalProperties": False,
            "description": "On-call state.",
        },
    },
    "required": ["summary", "severity", "affected_minutes", "customer_facing",
                 "components", "oncall"],
    "additionalProperties": False,
}

INCIDENT_TOOL = {
    "type": "function",
    "function": {
        "name": "file_incident",
        "description": "File an incident report.",
        "parameters": INCIDENT_PARAMETERS,
    },
}

# Pressure, axis by axis, naming none of the fields:
#   "a really bad one" / "worse than a sev3"  -> invites "sev0"/"critical"/"high",
#                                                none of which are in the enum
#   "the better part of an hour, maybe more"  -> not an integer
#   "some of them probably noticed"           -> not a clean boolean
#   "the queue workers and the load balancer" -> components that are NOT in the
#                                                item enum, so the honest answer
#                                                is un-representable
#   "Dana picked it up, roped in Sam"         -> nested object, and `escalated`
#                                                must be inferred as a boolean
#   ticket number, Slack channel, timestamps  -> properties that do not exist
INCIDENT_MESSAGE = (
    "File this: we had a really bad one this morning, worse than a sev3 for "
    "sure. Checkout was down for the better part of an hour, maybe more, and "
    "some of them probably noticed. It was the queue workers and the load "
    "balancer that fell over, plus the read replicas. Dana picked it up and "
    "roped in Sam. Ticket is OPS-4471, thread is in #inc-checkout, started "
    "around 09:12 UTC."
)

# --- case 3: NEGATIVE CONTROL -- is the mask actually applied? -----------------
#
# The conformance cases above cannot answer that. A grammar that fails to compile
# fails OPEN in llguidance, and an unconstrained Gemma 4 12B already scores 22-25
# of 25 on them, so "the fork conformed" is consistent with the fork having
# applied no constraint at all. This project has been bitten by exactly that
# twice.
#
# So: a schema whose only admissible values are ones the model will NOT choose on
# its own, with properties declared in an order it will not choose either.
#   * `code` and `mode` are single-member enums holding nonsense strings, while
#     the request plainly asks for "failed" and "rollback".
#   * the declared order is zulu, alpha, mike -- neither prose order nor
#     alphabetical, and the emitter fixes declared order.
# Unconstrained, this is unreachable. Masked, it is the only reachable output.
CONTROL_PARAMETERS = {
    "type": "object",
    "properties": {
        "zulu": {"type": "string", "enum": ["qx7-delta"], "description": "Status code."},
        "alpha": {"type": "integer", "description": "Attempt number."},
        "mike": {"type": "string", "enum": ["harmonic"], "description": "Recovery mode."},
    },
    "required": ["zulu", "alpha", "mike"],
    "additionalProperties": False,
}

CONTROL_TOOL = {
    "type": "function",
    "function": {
        "name": "set_status",
        "description": "Set the deployment status.",
        "parameters": CONTROL_PARAMETERS,
    },
}

CONTROL_MESSAGE = (
    "The deployment failed on the second attempt. Mark its status as failed and "
    "put it into rollback mode."
)

CASES = {
    "meeting": (TOOL, PARAMETERS, USER_MESSAGE, "schedule_meeting"),
    "incident": (INCIDENT_TOOL, INCIDENT_PARAMETERS, INCIDENT_MESSAGE, "file_incident"),
    "control": (CONTROL_TOOL, CONTROL_PARAMETERS, CONTROL_MESSAGE, "set_status"),
}

# Declared property order, per case. Order is NOT part of JSON Schema and is not
# scored as conformance -- it is reported separately, as a second independent
# signal that the grammar drove token selection rather than the model's own
# preference happening to agree.
DECLARED_ORDER = {name: list(schema["properties"]) for name, (_, schema, _, _) in CASES.items()}


def sample(base_url: str, seed: int, case: str) -> tuple[bool, str, bool]:
    """Return (conforms, detail, called) for one generation.

    `called` is tracked separately so tool SELECTION and argument CONFORMANCE are
    reported apart, as the plan's table does -- a backend that ignores
    tool_choice=required fails a different thing than one whose arguments drift.
    """
    tool, schema, message, name = CASES[case]
    body = {
        "model": "gemma-4-12b",
        "messages": [{"role": "user", "content": message}],
        "tools": [tool],
        "tool_choice": "required",
        # Google's official Gemma 4 sampling. Greedy would hide exactly the
        # variance this measurement is about.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": seed,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=300)
    except Exception as e:  # noqa: BLE001
        return False, f"transport: {e}", False
    if r.status_code != 200:
        return False, f"http {r.status_code}: {r.text[:200]}", False

    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if len(calls) != 1:
        return False, f"NO CALL: expected 1 tool call, got {len(calls)}", False
    fn = calls[0].get("function", {})
    if fn.get("name") != name:
        return False, f"wrong tool: {fn.get('name')!r}", True
    raw = fn.get("arguments", "")
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"arguments not JSON ({e}): {raw[:200]}", True
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as e:
        return False, f"schema: {e.message} | {json.dumps(args)[:200]}", True
    # Emitted order, NOT sorted -- see DECLARED_ORDER.
    emitted = [k for k in args]
    declared = [k for k in DECLARED_ORDER[case] if k in args]
    tag = "" if emitted == declared else f"  [ORDER {emitted} != declared {declared}]"
    return True, json.dumps(args) + tag, True


def main() -> int:
    base_url, label, case = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 25

    ok = 0
    called = 0
    failures: list[str] = []
    for seed in range(1, n + 1):
        conforms, detail, did_call = sample(base_url, seed, case)
        ok += conforms
        called += did_call
        print(f"  seed {seed:>3}  {'PASS' if conforms else 'FAIL'}  {detail[:190]}",
              flush=True)
        if not conforms:
            failures.append(detail)

    print(f"\n=== {label} [{case}] ===")
    print(f"    tool selection: {called}/{n}")
    print(f"    args conform:   {ok}/{called if called else n}"
          f"   (overall {ok}/{n})")
    if failures:
        print("    distinct failure modes:")
        for f in sorted({f.split("|")[0].strip() for f in failures}):
            print(f"      - {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
