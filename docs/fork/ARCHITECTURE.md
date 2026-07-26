# Architecture — the Gemma 4 conversation FSM

How a chat format is rendered, validated, constrained and extracted in this
fork. Read this before adding a format, modifying one, or debugging extraction.

## A format is four pieces

A format plugin owns the whole round trip, not just output parsing. All four
pieces must agree; the FSM↔grammar contract below is what binds them.

| piece | responsibility | Gemma 4 |
|---|---|---|
| **renderer** | message array → prompt bytes. Replaces Jinja. | `common_chat_gemma4_render()` |
| **validator** | reject a conversation the model was never designed to receive | the `conversation` grammar rule |
| **tracker** | FSM over generation state; mirrors llguidance's rule path | `common_chat_gemma4_tracker` |
| **grammar** | constrains generation; hosts composed per-request schemas | `grammars/chat/gemma4.lark` |

Formats are resolved **by name only** — the name that `chat_format`,
`--chat-format`, and GGUF `general.architecture` all resolve to. There is no
detection; see `docs/fork/POLICIES.md#nothing-is-inferred`.

## Normative sources

**There is no canonical template.** The format plugin -- renderer, validator,
tracker, grammar -- IS the definition of the wire format for this build. A
`.jinja` file is not a spec here; it is at best a historical artefact of how
someone else implemented the same format.

That distinction is load-bearing. While a test asserted render *parity against a
template*, the template was the source of truth and the FSM was derivative --
which is how this tree ended up asserting against a stale vendored copy (16448
bytes vs 18683, missing the post-tool-response thought re-opener) and could not
detect three renderer divergences. Inverting it removes that whole class of
failure: there is nothing left to be stale against.

Tests therefore assert **explicit expected bytes**, not template output.

External references, used to *derive* the plugin and to settle questions during
review -- never consulted at runtime, never asserted against:

| id | source | authority |
|---|---|---|
| **S1** | LiteRT-LM ANTLR grammar -- `runtime/components/tool_use/antlr/AntlrFcLexer.g4`, `AntlrFcParser.g4` | **the parser reference** |
| **S2** | `runtime/components/tool_use/fc_tool_format_utils.cc` (`FormatValueAsFc` / `FormatToolAsFc`) | **the writer reference** |
| **S3** | `runtime/components/logits_processor/constrained_decoding/llg_fc_tool_calls.cc` | **the generation-grammar reference** |
| **S4** | `runtime/conversation/model_data_processor/gemma4_data_processor_config.h` | token constants |
| **S5** | `runtime/components/tool_use/rust/fc_parser.rs` | parse semantics |
| **S8** | ai.google.dev `capabilities/thinking` | official prose |

LiteRT-LM is `github.com/google-ai-edge/LiteRT-LM`, inventoried at `7231f65`.

> Do not cite `ai.google.dev/gemma/docs/core/prompt-structure`. It is Gemma-3
> era: it documents `<start_of_turn>` and states *"the `system` role or a system
> turn is not supported"* -- wrong for Gemma 4 and contradicted by every source
> above.

### Arbitration where sources disagree

Fifteen conflicts exist, several *inside* LiteRT-LM itself. The governing
principle: **accept liberally per the parser (S1), emit conservatively per the
writer (S2), and generate at the intersection (S3).** Postel, with the parser
as arbiter.

| conflict | resolution |
|---|---|
| Trailing `<\|tool_response>` after the last tool call — LiteRT-LM's own Jinja never emits it; its generation grammar makes it **mandatory** (`fc_resp`) | **Mandatory.** S3 requires it and LiteRT-LM's own shipped template is the outlier |
| Identifier charset — S1 parser allows `.` and `-`; S3 generation grammar does not | **Parse** `[a-zA-Z_][a-zA-Z0-9_.-]*`, **emit** `[a-zA-Z_][a-zA-Z0-9_]*` for argument keys. Function *names* stay dotted: S3 emits them as a literal alternation, so MCP-style `server.tool-name` is samplable |
| String terminator — S1 parser non-greedy, S3 generation greedy | **Non-greedy.** The parser is the arbiter |
| NUMBER — S1 permits bare `.5` / `-.5` and forbids leading zeros; S3 differs | **Union on parse, intersection on generate** |
| `null` argument value — LiteRT-LM's Jinja renders Python `None`; its own C++ writer emits `null` | **`null`.** The shipped template is buggy |
| OBJECT `properties` comma — LiteRT-LM's Jinja emits malformed `{,properties:{…}}` for a description-less object | **Guard the comma** — the unguarded form is malformed output |
| `<\|think\|>` trailing newline; the `\n\n` separator before tool declarations; image rendering | **Decided by the renderer**, and pinned by explicit expected bytes in its tests |

### The `<|"|>` delimiter has no escape mechanism

This is a **hole in the format, not a gap in the implementation.** S1 is
`ESCAPED_STRING : ESCAPE .*? ESCAPE` (non-greedy — the first inner delimiter
terminates the string); S2 emits `escape_tag << raw_string << escape_tag` with no
escaping; no documentation defines an escape. Measured on stock: a payload
containing a literal `<|"|>` is silently corrupted 10/10.

It cannot be fixed, only detected. **The renderer must reject such a payload with
an OpenAI-shaped 400 naming the offending argument**, never silently truncate.

## Why per-format pipelines

The legacy `common_chat_peg_mapper::map()` was a 280-line method conflating four
responsibilities (state-keeping, wire-shape extraction, model-specific shaping,
message-shape projection) and accumulating format-specific hacks — phantom
tool-open guards, nested-tool-open extent checks, name-before-open recovery, JSON
brace synthesis. Adding a format meant adding branches to `map()`.

The per-format pipeline splits these into four cooperating layers, each
per-format, operating in lock-step with a tracker FSM that mirrors llguidance's
internal grammar state.

## The four output layers

```
AST node N
   ├─> tracker.advance(N)                     // update FSM state -> S
   │
   ├─> events = decoder.decode(N)             // reads S to interpret N
   │     // a `<` in S=IN_CONTENT is a content char;
   │     // a `<` in S=IN_TOOL_OPEN is the start of a tool tag
   │
   ├─> for each event in events:
   │     shaped = transformer.shape(event)    // reads S to shape the event
   │
   └─> for each shaped event:
         presenter.present(shaped)            // reads S to land in common_chat_msg
```

The tracker is the single source of truth for "where in the conversation are we".
The other three layers are stateless w.r.t. conversation position; any state they
keep (e.g. the transformer buffering KV events for args assembly) is local to a
single state-bracketed span and reset on the relevant FSM transition.

### 1. Tracker — `common_chat_format_tracker`

- Owns the FSM (`common_chat_format_state` enum).
- Per-node transition entry point: `void advance(const common_peg_ast_node &)`.
- Read-only state view: `current_state()`, `expected_productions()`.
- Pure state-keeping. **No side effects** — does not write to `common_chat_msg`,
  does not emit events, does not mutate another layer.
- `expected_productions()` returns the names of grammar rules that could legally
  follow. The PEG parser uses this to bound speculative matching; the llguidance
  setup verifies those rules exist in the per-format grammar.

### 2. Decoder — `common_chat_format_decoder`

- `vector<common_chat_decoded_event> decode(const common_peg_ast_node &)`.
- Reads `state_view.current_state()` to interpret the node correctly.
- Emits typed events — `TOOL_OPEN` / `TOOL_NAME` / `TOOL_ID` / `TOOL_ARGS_RAW` /
  `TOOL_ARG_KV` / `TOOL_CLOSE` / `REASONING_TEXT` / `CONTENT_TEXT`.
- Knows the wire format. JSON-args formats emit `TOOL_ARGS_RAW` (the object
  verbatim); per-arg formats emit one `TOOL_ARG_KV{key, value, is_string}` per arg.
- **Never** writes to `common_chat_msg`. **Never** synthesises characters.
- `on_finalize()` may emit a final batch at end-of-walk.

### 3. Transformer — `common_chat_format_transformer`

- `vector<common_chat_shaped_event> shape(const common_chat_decoded_event &)`.
- Applies model-specific shaping: reasoning-channel routing, content-prefix
  stripping, and **tool-args JSON assembly** (per-arg KV → JSON object) — the
  deliberate, scoped synthesis exception, confined to per-format transformer
  files and marked with an explicit "scoped exception" comment.
- May buffer state across calls (accumulate `TOOL_ARG_KV` until `TOOL_CLOSE`,
  emit one `TOOL_ARGS_JSON`). State is local to one tool-call span.
- `on_finalize()` flushes buffered events.

### 4. Presenter — `common_chat_format_presenter`

- `void present(const common_chat_shaped_event &)` → writes to `common_chat_msg`.
- **One default implementation** covers all formats; per-format overrides exist
  only for message-shape idiosyncrasies (none currently).
- Default mapping: `REASONING_TEXT` → `reasoning_content`; `CONTENT_TEXT` →
  `content`; `TOOL_OPEN` → push pending tool call; `TOOL_NAME`/`TOOL_ID` → set
  pending fields; `TOOL_ARGS_JSON` → set `.arguments`; `TOOL_CLOSE` → commit.
- `on_finalize()` handles cross-cutting cleanup: discard whitespace-only
  reasoning, default `role="assistant"`, commit a still-pending streaming call.

## State enum

```cpp
enum class common_chat_format_state : uint8_t {
    INITIAL,            // nothing emitted yet
    IN_CONTENT,         // free-form assistant content
    IN_REASONING,       // inside <|channel>thought ... <channel|>
    IN_TOOL_CALL,       // inside a tool call envelope (after open, before close)
    IN_TOOL_NAME,       // currently emitting the function name
    IN_TOOL_ID,         // currently emitting the tool-call id
    IN_TOOL_ARGS,       // inside the args region
    IN_TOOL_ARG_KEY,    // tagged-args formats: between arg-name open and close
    IN_TOOL_ARG_VAL,    // tagged-args formats: between arg-value open and close
    IN_TOOL_PARAM,      // DSML <param>...</param>
    DONE,               // tool-call closed; back at top-level
};
```

Each per-format FSM uses a **subset**. A state that a format declares but never
reaches is a bug, not a spare — see the startup check below.

**Known gap:** these states model only the *assistant-turn interior*. A tracker
that also drives input validation needs conversation-scope states —
`IN_SYSTEM_TURN`, `IN_TOOL_DECLARATIONS`, `IN_TOOL_RESPONSE`,
`AWAITING_TOOL_RESPONSE`, `IN_GENERATION_PROMPT` — plus per-call identity so
parallel tool calls don't all collapse into one `IN_TOOL_CALL`.

### There is no generation prompt — only an input state and an output state

The format owns the whole conversation: input, output, decisions, parse tree.
Two states, both locally scoped to the format, and nothing else:

```cpp
struct <format>_input {
    std::string              prompt;       // the bytes the model receives
    common_chat_format_state entry_state;  // the state those bytes leave it in
};
```

Generation is then parsed by the tracker **starting from `entry_state`**. That
is the entire contract.

`generation_prompt` is not part of it, and should not be. It exists in upstream
only because callers had to recover *where generation begins* in order to make
the grammar's `start` rule match — `common_chat_peg_parse` re-prepends it to the
model's output as a text prefix. Seed the FSM instead and the need evaporates:
the parser stops re-scanning bytes the model never emitted, and the turn opener
becomes a required literal rather than an optional one.

**Everything that tried to reconstruct it from outside the format has failed, in
a different way each time.** Three sequential blocks in the init function each
rewrote it, and one invalidated the next one's end-of-turn test. A double-render
diff could not isolate it, because under continuation both renders emit the
channel marker so the difference is not the tail. Having the renderer *report* a
tail alongside the prompt is the same mistake one layer in — it keeps a derived
artefact alive instead of deleting the concept.

The states are:

| state | meaning |
|---|---|
| `IN_GENERATION_PROMPT` | prompt ended at a fresh turn opener |
| `IN_REASONING` | prompt ended inside a thought — thinking-off prefill, or a continuation carrying `reasoning_content` |
| `IN_CONTENT` | prompt ended with the model mid-content (continuation on content) |

A test asserts the prompt bytes and the entry state. It does not assert a
"generation prompt", any more than it asserts against a template.

## FSM↔grammar contract

This is what lets the C++ tracker and llguidance's internal state agree:

- Every `common_chat_format_state` reachable in a per-format FSM **must**
  correspond to a named rule (or set of rules) in that format's Lark grammar.
- `expected_productions()` **must** return the names of those rules.
- llguidance needs no direct hooks: it tracks the rule path internally from the
  same Lark grammar, computing token masks at every position. The C++ tracker is
  the **mirror** of llguidance's internal state, observable to the parser and the
  downstream layers.

If a grammar drops a rule, `expected_productions()` names a non-existent rule and
the parser stops constraining properly. **Always update tracker and grammar
together**, and let the startup check enforce it: a declared state with no
matching rule, or a rule named by `expected_productions()` that doesn't exist,
must fail loudly at load rather than degrade silently.

## Per-request grammar composition

A per-request grammar (`response_format` with `json_schema`, `lark_grammar` or
`gbnf_grammar`) becomes a **production inside the format grammar**, with the
tracker deciding where it applies: active in `IN_CONTENT`, while the tool grammar
is active in the tool-call states. Alternatives in one grammar, never
simultaneous masks. This is what allows `response_format` and `tools` in the same
request.

The rejected design was a second sampler in parallel (`user_grmr`). Two
independent grammars masking one token stream intersect their languages; if
neither is a superset of the other the intersection can be **empty** — all logits
`-INF` mid-generation, with no diagnostic.

**Constraint:** llguidance has no API for entry-rule selection (`start_name` is
hardcoded at `compiler.rs:612`; `%override` is rejected at `:705`). Composition
must therefore emit one grammar with one start rule and select the branch
structurally. The state→rule registry is the single source of truth.

## Streaming monotonicity invariant

Every streaming-prefix parse must produce a `common_chat_msg` whose `content` /
`reasoning_content` / `tool_calls[].arguments` are prefixes of the final
non-partial parse's outputs. `compute_diffs(prev, current)` only emits deltas;
non-monotonic state surfaces as a delta contradicting a prior one, breaking
client-side rendering.

Pitfalls that break it:

- **Closing a tool call early** on a partial AST node when later prefixes extend
  the args. Synthesised `}` seals the buffer; later prefixes re-open and
  overwrite.
- **Emitting a per-arg KV event repeatedly** as the value text grows. Emit
  exactly once, when the parent rule is fully matched.
- **Re-routing `CONTENT_TEXT` to `REASONING_TEXT`** on a transition that happens
  *after* content was surfaced. The tracker must transition BEFORE the
  decoder/transformer runs for that node.

Verified by `test-chat.cpp` — streaming prefixes are accumulated via
`compute_diffs` and compared against each prefix's full parse. Failures surface
as `Error comparing accumulated message to current`.

## How a new format is added

1. Write `grammars/chat/<name>.lark` and its GBNF companion. Cover the
   conversation rule (input validation), tool-call, response-format and
   content-only paths in one grammar. Use the `tool_call` rule name (auto-tagged
   as the `tool` wrapper) and `func_name` / `arg_name` / `arg_value` for the
   json-tagged decoder family.
2. Add `COMMON_CHAT_FORMAT_PEG_<NAME>` to `common/chat.h` and a name string in
   `common_chat_format_name()`.
3. Add `common_chat_params_init_<name>` — set `data.format`,
   `data.grammar_file_parser = true`, point `data.parser` at
   `chat_grammar_to_peg(base_grammar)`.
4. Write the **renderer** (message array → prompt bytes) and the **validator**
   (run the rendered prompt against the `conversation` rule). Neither is
   optional; without them the format has no input path.
5. Create `common/chat-formats/<name>-format.{h,cpp}` — tracker FSM, decoder,
   transformer. Inherit the json-tagged base for JSON-args formats.
6. Register the format **by name** in the plugin registry and add the case to
   `common_chat_make_format_pipeline()`.
7. List the new files in `common/CMakeLists.txt`.
8. Add tests in `tests/test-chat.cpp`, plus render parity against the format's
   canonical normative source.

## Debugging a test failure

```bash
build/bin/test-chat --template <name> --detailed 2>&1 | less
```

`--detailed` enables AST dumps after every parse plus PEG execution traces:

- `CHOICE option N: need_more_input` with no later option tried — alternatives
  don't backtrack on `NEED_MORE_INPUT` in lenient mode.
- `Error comparing accumulated message to current` — streaming non-monotonicity.
- AST nodes containing the entire input as content when tool calls were expected
  — the grammar parse failed entirely; check every literal in the wire shape.

**Rebuild the target first.** A stale test binary reports on a tree that no
longer exists; that has already produced one false failure here.
