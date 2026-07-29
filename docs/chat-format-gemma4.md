# The native/translated boundary

This fork constrains Gemma 4 to **its own wire format and nothing else**. The
OpenAI shapes a client sees — `tool_calls`, `reasoning_content`,
`finish_reason` — are built by translating that native output at the API
boundary. The model is never asked to emit OpenAI-shaped JSON.

This is worth stating explicitly because it is invisible from outside: a client
sends OpenAI `tools` and receives OpenAI `tool_calls`, so it is a reasonable
guess that the model was constrained to produce them. It was not, and the
distinction decides where a bug belongs.

## The boundary

```
   client                fork                          model
   ------                ----                          -----
   tools: [...]     ──►  render to native prompt  ──►  <|turn>model\n…
                         common_chat_gemma4_render
                         (common/chat-formats/gemma4-format.cpp:923)

                         constrain to native grammar
                         grammars/chat/gemma4.lark
                         + {{TOOL_SCHEMA}} per tool

   tool_calls: [...] ◄─  parse native, build OpenAI  ◄─  <|tool_call>call:NAME{…}<tool_call|>
                         common/chat-peg-parser.cpp
                         tools/server/server-common.cpp:905
```

Everything left of the model is OpenAI's vocabulary. Everything right of it is
Gemma 4's. Nothing crosses.

## What the model is actually constrained to

The generation grammar is the Gemma 4 conversation format, not JSON:

```lark
tool_call_open_tag:   <|tool_call>
tool_call_close_tag:  <tool_call|>
#   <|tool_call>call:NAME{args}<tool_call|>
#   <|turn>model\n<assistant_body>[<|tool_call>…]<turn|>\n
```

Tool **arguments** are constrained too, and also in the native syntax:
`{{TOOL_SCHEMA}}` expands to a per-tool alternation pairing each function-name
literal with a `gemma4_dict` built from that tool's JSON Schema — bare keys,
`<|"|>`-delimited strings. It is not `%json`: llguidance's `%json` emits JSON
syntax, which cannot be dropped into a bare-key dict. The emitter walks the
schema and produces Lark for the dict grammar instead.

The practical consequence, measured: adversarial tool-argument conformance went
**1/25 on stock llama.cpp to 25/25 here**, because stock passes a generic
`gemma4-dict` rule and never compiles the tool's schema at all
(`common/chat.cpp`, the `TODO @aldehir` upstream).

## What is deliberately NOT translated: `response_format.json_schema`

Structured output is the exception, and it is an exception on purpose.

When a caller sends `response_format: {type: "json_schema", …}`, **JSON is the
artifact the caller asked for**, not a protocol wrapper the model is being
forced into. There is nothing to translate: the caller wants a JSON object and
the grammar constrains the model to produce one. Routing it through the native
dict and converting back would be a translation with no client on the other end.

It would also be a **downgrade**, for a reason the format cannot fix:

> the `<|"|>` delimiter is symmetric, and being symmetric is why the format
> cannot escape it

That is a spec hole, not an unimplemented feature — no escape is defined in
LiteRT-LM's lexer, its writer, or any Google documentation, and a string
containing the delimiter corrupts silently (measured 10/10 on stock). The fork's
answer is to **reject** such a payload with an OpenAI-shaped 400.

Rejection is tolerable for tool arguments: the caller supplied them and can be
told they are unrepresentable. It is not tolerable for structured output, where
the unbounded free-text field is the *model's own* — a critique, a rationale, an
assessment — and routinely contains quotes, backticks and code fragments. JSON
has `\"`. Sending that through an unescapable delimiter trades a mechanism that
works for one that corrupts, and there is no one to return a 400 to.

| | constrained as | on the wire to the client | if it contains `<\|"\|>` |
| --- | --- | --- | --- |
| tool call + arguments | native `gemma4_dict` | translated to OpenAI `tool_calls` | 400, naming the argument |
| thinking channel | native `<\|channel>thought` | translated to `reasoning_content` | n/a |
| `response_format.json_schema` | **JSON, as asked for** | **passed through** | representable — `\"` |

## Where a bug belongs

- The model emitted something malformed → the **grammar** (`gemma4.lark`) or the
  **tracker** (`common_chat_gemma4_tracker`), which must stay in lockstep; every
  reachable FSM state has a named rule.
- The client saw the wrong shape → the **translation layer**
  (`chat-peg-parser.cpp`, `server-common.cpp`), not the grammar.
- The model stalled or ran away while still inside the grammar → neither. That
  is llguidance's **skip regex**, which applies between every pair of terminals
  and is what a grammar cannot bound. See `docs/llguidance.md`.

The last one is the easiest to misfile. A grammar constrains *what* may come
next, never *how long* the model may dither before producing it.
