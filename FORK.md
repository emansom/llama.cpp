# This fork: a Gemma 4 conversation FSM for llama.cpp

Orientation file for anyone — human or agent — working on this fork. Read it
first. Detail lives in `docs/fork/`.

## The goal, stated once

**Reliable input to the model, and reliable structured, guided output from it,
behind an unchanged OpenAI-compatible API, with nothing left to inference.**

Everything here serves that sentence. Work that doesn't, doesn't belong.

## Why this fork exists

Stock llama.cpp does not constrain Gemma 4 tool-call arguments.
`common_chat_params_init_gemma4` passes a generic `gemma4-dict` rule instead of
the per-tool JSON Schema — the schema line is commented out under
`// TODO @aldehir : need to extend json-schema-to-grammar to produce more than
JSON rules`. Ten other model handlers in the same file *do* compile the schema.
Gemma 4 can't, because its wire format uses `<|"|>`-delimited strings and bare
keys, and `json-schema-to-grammar` only emits JSON rules.

Measured against stock b10068, Gemma 4 12B QAT, N=25 per configuration:

| configuration | tool selection | args conform |
|---|---|---|
| 5 tools, long prompt, prior history, natural input | 25/25 | 25/25 |
| 1 tool, `tool_choice:required`, natural input | 25/25 | 25/25 |
| 1 tool, **adversarial** input | 25/25 | **1/25** |
| `response_format.json_schema`, same schema, adversarial | n/a | **25/25** |

Scoping a call to a single tool buys reliability in practice but provides no
guarantee. The guarantee lives in the grammar, and for Gemma 4 tool arguments
the grammar was never asked for. This fork asks for it.

## Scope: Gemma 4 only, but pluggable

This fork serves **Gemma 4 and nothing else**. Jinja templating is gone, along
with template sniffing and the multi-format grammar registry. A model whose
format resolves to no registered plugin **fails loudly at load** — there is no
silent fallback, because the entire value of this fork is that what reaches the
model is always well-formed.

Adding a format (IBM Granite 4.1 is the intended second) means implementing four
pieces and registering them under a name. Nothing else should need to change;
that is the test of whether the interface is right.

| piece | responsibility |
|---|---|
| **renderer** | message array → prompt bytes. Replaces Jinja. |
| **validator** | reject a conversation the model was never designed to receive |
| **tracker** | FSM over generation state; mirrors llguidance's rule path |
| **grammar** | constrains generation; hosts composed per-request schemas |

See `docs/fork/ARCHITECTURE.md` for the layer contracts and the FSM↔grammar
contract that binds tracker and grammar together.

## The API does not move

The OpenAI-compatible surface is **preserved and only extended**. A client that
works against stock `llama-server` works against this fork.

Unchanged: `/v1/chat/completions` (`messages`, `tools`, `tool_choice`,
`response_format`, `stream`, sampling params, `chat_template_kwargs`), the
response shape (`content`, `reasoning_content`, `tool_calls`, `finish_reason`,
`usage`, SSE deltas), `/v1/models`, `/props`, `/health`, `/v1/embeddings`, and
the OpenAI error-body shape — including for every failure mode this fork adds.

Extended, additively — `response_format` gains two `type` values, per-request:

```jsonc
{"response_format": {"type": "lark_grammar", "lark_grammar": "..."}}
{"response_format": {"type": "gbnf_grammar", "gbnf_grammar": "..."}}
```

These may be combined with `tools`: the two constraints become branches of one
grammar (see "composition" below), not competing masks.

## Nothing is inferred

The server is **told**, never guesses. Three sources of truth, in precedence
order, and no fourth:

1. **Request parameters** — `chat_format`, `thinking`.
2. **Per-model server config** — `--chat-format`.
3. **Declared GGUF metadata** — `general.architecture`.

The line is between *declared* and *guessed*. A metadata key the publisher wrote
is authoritative data, the same as a config value someone wrote down. Reading it
is not detection. String-matching the Jinja blob in `tokenizer.chat_template` to
infer a format **is**, and it is gone. The resolved source is logged once at
model load, so resolution is never a mystery.

`chat_template_kwargs.enable_thinking` keeps working by requirement — with no
template, the renderer reads it directly.

## Composition, not a second sampler

A per-request grammar becomes a **production inside the format grammar**, with
the tracker deciding where it applies: the request's grammar is active in
`IN_CONTENT`, the tool grammar in the tool-call states, as alternatives in one
grammar.

The rejected design — and what the code used to do — was a second sampler
running in parallel (`user_grmr` in `common/sampling.cpp`). Two independent
grammars masking one token stream intersect their languages; if neither is a
superset of the other the intersection can be **empty**, which surfaces as all
logits `-INF` mid-generation with no diagnostic at all.

llguidance has no API for entry-rule selection (`start_name` is hardcoded at
`compiler.rs:612`, `%override` is rejected at `:705`), so composition must emit
one grammar with one start rule and select the branch structurally. The
state→rule registry is the single source of truth for that contract.

## Conformance bar

The tracker and validator must support **every feature of the official Gemma 4
grammar** as implemented in LiteRT-LM and documented by Google — not whatever
the grammar happens to cover today. `docs/fork/ARCHITECTURE.md` carries the
ranked normative sources and the arbitration rules for the places those sources
contradict each other (there are fifteen, several *inside* LiteRT-LM).

## Critical rules

Non-negotiable. Each was set after a concrete incident; violating one repeats
that incident. `docs/fork/POLICIES.md` records why each exists.

1. **Never make closing tokens optional.** They are required literals in chat
   grammars. If a fixture lacks one, fix the fixture, not the grammar.
2. **Never synthesise characters in the decoder or presenter.** Every byte in
   `content` / `reasoning_content` / `tool_calls[].arguments` must have been
   emitted by the model. The only exception is per-format transformers
   converting a non-JSON wire shape into JSON, marked "scoped exception".
3. **Never rewrite or skip tests to match an incomplete implementation.**
4. **No `Co-Authored-By` or other attribution trailers** in commit messages.
5. **Rebase on the latest upstream release tag** before substantial work.
6. **Commit WIP to a backup branch first** before any reset/checkout/stash.
7. **No `/tmp` for WIP backups** — use `.wip-backup/`.
8. **Read files before editing.** The tree may have advanced.
9. **Per-format pipelines, not mapper hacks.** No format-specific branches in
   `common_chat_peg_mapper`.
10. **Nothing is inferred** (see above). No new detection heuristics, ever.

## Build & test

```bash
cmake -B build -DLLAMA_LLGUIDANCE=ON -DGGML_VULKAN=ON
cmake --build build --target test-chat -j$(nproc)

build/bin/test-chat                                # all templates
build/bin/test-chat --template gemma4              # one
build/bin/test-chat --template gemma4 --detailed   # + AST dump and PEG traces
```

`-DLLAMA_LLGUIDANCE=ON` is **not** optional here — upstream defaults it OFF
(`CMakeLists.txt`), and the grammar composition path needs it.

Grammars are loaded from disk at startup; `--chat-grammars-dir` overrides the
default `/usr/share/llama.cpp/grammars/chat`. A registry miss must throw, never
degrade to an unconstrained parse.

## Where to start

| task | read first |
|---|---|
| Add a format plugin (e.g. Granite 4.1) | `docs/fork/ARCHITECTURE.md` |
| Change a grammar | `docs/fork/POLICIES.md` (closing tokens), then the normative-source table |
| Investigate a failing template test | `docs/fork/ARCHITECTURE.md`, then run with `--detailed` |
| Rebase on a new upstream tag | `docs/fork/WORKFLOWS.md` |

Upstream's own `AGENTS.md` and `CONTRIBUTING.md` still apply to anything
destined for upstream. Note upstream rejects fully-AI-generated PRs and requires
disclosure when AI assisted meaningfully.

## On upstreaming

Deferred, not abandoned. This fork carries a permanent local delta against the
Arch `llama-cpp` package, so every upstream bump is a rebase. The cleanest
eventual contribution is a **pluggable value-syntax for
`json-schema-to-grammar`** — precisely what `TODO @aldehir` asks for — which
would let Gemma 4 get per-tool schemas with no special-casing and make most of
this fork's grammar work unnecessary for everyone.

### Report upstream once this settles

Three findings that are not Gemma-4-specific and belong to their projects, not
here. Deliberately parked until the fork stops moving — filing a bug against a
moving reproducer wastes the maintainer's time as much as our own.

**1. llguidance panics on `[lazy]`** (guidance-ai/llguidance). The documented
lazy-lexeme idiom aborts the process. Three lines, no model specifics:

```lark
%llguidance {}
reasoning[lazy]: /(.|\n)*/
start: "<|c>" reasoning "<c|>"
```

matching `<|c>think<c|>` gives `assertion failed: !state.has_lowest_match()` in
`parser/src/panic_utils.rs`. It reproduces with any vocabulary and whatever
follows the lazy rule, including nothing. Worth reporting even though this fork
no longer needs the option — the markers became special tokens instead, and the
whole question went away.

**2. Added-but-not-special tokens are invisible to llguidance**
(ggml-org/llama.cpp, `common/llguidance.cpp`). The bridge marked a token special
by prefixing `\xff` when detokenizing with `special=false` rendered nothing,
which catches `CONTROL` and misses `USER_DEFINED`. Both come from the
tokenizer's added-tokens list — the flag only records whether the entry was
marked special — so a grammar could not name half of Gemma 4's markers:
`unknown special token: "<|channel>"`. Any model whose markers are added rather
than control tokens has this. The fix here is two lines and reads a declared
attribute; it should apply as-is.

**3. llguidance's own test grammars do not survive a large vocabulary**
(ggml-org/llama.cpp, `tests/test-grammar-llguidance.cpp`). Run against Gemma 4's
262k-token vocabulary the `+` quantifier case dies with `Too many items (limit
50000; mask); try avoiding single-byte/short lexemes`. That is a property of
those single-byte test grammars rather than of this fork, and it is why the
Gemma 4 case runs as a separate invocation with its own weights.
