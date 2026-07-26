# Policies

These are the non-negotiable rules for working on this fork. Each was set after
a concrete incident; violating them will repeat that incident. The "Why"
sections record the reasoning so you can reapply the spirit when the letter
doesn't quite fit.

## Closing tokens are required

**Rule:** Closing markers in chat grammars (`</think>`, `</tool_call>`,
`<|tool_call_end|>`, `</｜DSML｜invoke>`, `<|tool_calls_section_end|>`, etc.)
must remain required literals. **Never** make them optional.

**Why:** Chat grammars are dual-purpose — they drive sampling (llguidance
masks tokens at every position based on the grammar) AND they drive
post-generation extraction. Making a closer optional lets the model skip it
during sampling, breaking the format contract. Patterns like
`think_block: "<think>" reasoning ("</think>" | )` and `unclosed_think_block`
alternatives were explicitly rejected.

**How to apply:** When a test fixture appears to require an unclosed wire
shape, the right fix is almost always to **edit the test input** — adding the
missing closer. The grammar defines what valid model output looks like; tests
should conform to that. Specifically:
- Do not add an alternative rule for "unclosed" variants
- Do not loosen a closer to `?`
- Do not add C++ fallback recovery that reconstructs what the grammar rejected

If a closer genuinely has two valid forms (e.g. `</think>` OR a tool-call
section marker as an implicit terminator), express that as `("</think>" |
section_marker_lookahead)` only if every branch is itself a required literal —
not by making one branch empty.

**Note the one legitimate exception, which is not a loosened closer:** Gemma 4
defines turn shapes that genuinely terminate with a bare `<|tool_response>`
instead of `<turn|>`, and a turn carrying tool responses with no successor emits
no terminator at all. Those are *distinct required alternatives* in the wire
format, sourced from the canonical template — not an optional closer. Model them
as explicit alternatives, each with its own required literal.

## No character synthesis in the decoder or presenter

**Rule:** Every byte that ends up in `result.content` /
`result.reasoning_content` / `result.tool_calls[].arguments` must have been
emitted by the model. The decoder and presenter must never write a `}`, `{`,
`,`, `:`, `"`, or any character the model didn't produce.

**Why:** "`}` should not be added as it should exist in the model output itself
already". Synthesis hides format mismatches and breaks the
streaming-monotonicity invariant — if the decoder closes a tool call early with
a synthetic `}`, later prefixes that extend the args produce a non-monotonic
args sequence (the `}` would have to disappear, which diffs can't represent).

**How to apply:**
- The decoder reads AST nodes verbatim; it emits `TOOL_ARGS_RAW{node.text}` or
  `TOOL_ARG_KV{key, value}` with the model's bytes unchanged.
- The presenter writes shaped events to `common_chat_msg` verbatim.
- The **transformer** is the ONLY layer where structural JSON characters may be
  synthesised, and only when converting a non-JSON wire shape (DSML `<param>`,
  Python `name(k=v)`, XML-tagged per-arg) into the JSON `arguments` object the
  API requires. Synthesis is confined to per-format transformer files. Each
  instance is marked with an explicit "scoped exception" comment.

The rule applies even when it would be convenient. Fix the grammar or the wire
format expectation; do not paper over with synthesis.

## Don't rewrite tests to match incomplete implementations

**Rule:** When the parser produces output that doesn't match a test's
expectation, fix the implementation, not the test. Tests encode the
user-facing contract.

**Why:** Said twice across sessions: "no, don't skip tests" and "do not rewrite
tests." Rewriting hides regressions, breaks alignment with upstream, and weakens
the suite's signal.

**How to apply:**
- Failing test → change the implementation until the test passes.
- If a test fixture is genuinely malformed (e.g. the wire shape doesn't match
  what the model actually emits), correcting the fixture is acceptable —
  but only if the corrected fixture still expresses a valid scenario. Don't
  simplify a fixture to make it pass; replace it with a different valid
  fixture if that's what's needed.
- Document the fixture correction in the commit message with the rationale.

**Corollary, learned the expensive way — a test can be green and worthless.**
`tests/test-chat-render.cpp` asserted byte-parity against a *stale copy* of the
Gemma 4 template, so it could not detect three real renderer divergences. Before
trusting a passing suite, check that its fixture is the current normative
source, and that the binary you ran was built from the current tree.

## Nothing is inferred

**Rule:** The server is told what to do; it never guesses. No format
auto-detection, no grammar auto-selection, no sniffing of chat-template text or
model filenames. Resolution order is: **request parameter → per-model config →
declared GGUF metadata → error.**

**Why:** Detection heuristics make behaviour depend on things nobody thinks of
as configuration. `common_chat_try_specialized_template` string-matched the
Jinja blob, so editing a template silently changed how a model was parsed. A
registry miss returned `""` and 30+ call sites set `data.grammar = ""` — *no
constraint at all* — after a single `LOG_WRN`.

**How to apply:**
- Declared data is fine; guessed data is not. A GGUF metadata key the publisher
  wrote is authoritative, the same as a config value someone wrote down.
  String-matching `tokenizer.chat_template`'s *text* is not.
- Log the resolved source once at model load, so resolution is never a mystery.
- An unresolvable or unregistered format is a **hard error**, never a fallback.
  A quiet degradation path destroys the only guarantee this fork offers.

## Per-format pipelines, not mapper hacks

**Rule:** Each chat format gets its own pipeline (tracker + decoder +
transformer + presenter) plus a renderer and validator, under
`common/chat-formats/`. Do **not** add format-specific branches to
`common_chat_peg_mapper::map()`.

**Why:** The legacy mapper accumulated phantom-tool-open guards,
nested-tool-open extent checks, DSML param state, name-before-open recovery —
each workaround made the mapper's behaviour harder to reason about for the
OTHER formats. The original ask was for "a state machine per model format to be
used by the parsers to know which state the model output is in".

**How to apply:** Follow `docs/fork/ARCHITECTURE.md#how-a-new-format-is-added`.

## No Co-Authored-By trailers

**Rule:** Do not add `Co-Authored-By:` (or any other attribution trailer) to
any commit message.

**Why:** Preference — keep commit messages clean.

**How to apply:** Use `git commit -m "$(cat <<'EOF' ... EOF)"` with a HEREDOC.
Never paste the boilerplate trailer.

## Always rebase on the latest upstream release tag

**Rule:** Before substantial work, rebase onto the latest upstream release tag.

**Why:** Work against a stale base conflicts with current upstream and rots.
"This is non-negotiable. Don't construct patches against a stale base."

**How to apply:** See `docs/fork/WORKFLOWS.md#rebasing-on-latest-upstream`.
Upstream release tags are `bNNNN`; pick the most recent merged into
`upstream/master` (`git tag --merged upstream/master --list 'b*' | sort -V |
tail -1`). **Fetch first** — the local `upstream/master` ref goes stale silently,
and a stale ref makes the repo believe it is current when it is months behind.

## Backup-branch workflow for risky operations

**Rule:** Before any reset, checkout, stash, or filesystem-level snapshot of
in-flight work, commit ALL uncommitted work to a backup branch — **and push it**.

**Why:** Git refuses to lose committed work; uncommitted work in the working
tree (or in `/tmp`, or even in a `.wip-backup/` dir) can be lost in a single
bad command. The backup branch is the safety net.

**And a local branch is only half a safety net.** This repo once carried 59
commits — the entire conversation-pipeline package, including the Gemma 4
tracker and renderer — on a single local branch that had never been pushed. One
disk failure would have taken all of it. Verify before restructuring:
`git branch -r --contains HEAD | grep origin`.

**How to apply:** See `docs/fork/WORKFLOWS.md#backup-branch-workflow`.

## No `/tmp` for WIP backups

**Rule:** Do not use `/tmp` (or any system-temp path) when snapshotting
working-tree files. Use a dotfile-prefixed dir under the repo root that
`.gitignore` covers (e.g. `.wip-backup/`).

**Why:** The OS can clear `/tmp` at any time — reboots, tmpfs evictions, manual
cleanup. A working-tree snapshot lost there means lost in-flight work that isn't
yet in a git commit. (Near-miss during a patch-stack split.)

**How to apply:** Use `.wip-backup/`. Better: don't snapshot at all — use the
backup-branch workflow so everything is in git.

## Read files before editing

**Rule:** Always read the current state of a file before editing — the tree may
have advanced, the linter may have reformatted, someone may have made a manual
change.

**Why:** Editing stale content produces conflicts and silently overwrites later
changes. The `Edit` tool fails if you didn't read the file in this session, but
it will NOT catch reading at session start and the file changing mid-session.

**How to apply:** Re-read after any of: external `git pull`, a build step that
might run linters/formatters, an acknowledged manual edit, or a long gap.

## Execute git commands directly

**Rule:** Stage, commit, and push autonomously as part of each workflow. Don't
ask someone else to run git commands.

**How to apply:** When the work is reviewed and ready, do the commit; don't say
"you can now commit with...". Same for `git push`.

**Exception:** Destructive operations — force-push, hard reset, branch deletion
— should still be confirmed if not pre-authorised in the current context.

## Patch splits optimise reviewer burden — *dormant*

**Status: dormant.** The patch layer was retired; the fork branch is the
artifact and upstreaming is deferred. This rule applies again the moment
upstream submission resumes, so it is kept rather than deleted.

**Rule:** When splitting work into a patch stack for upstream PR submission, the
metric that matters is reviewer burden, not author convenience. A 3500-line
"architectural" patch is worse than 9 × 400-line patches even if splitting is
harder.

**Why:** Reviewers must internalize all design choices at once and can't push
back on individual pieces in a giant patch.

**How to apply:**
- If the surgical split is hard because the work is interdependent, that's a
  signal — usually the work is genuinely entangled and needs re-decomposition
  during the split.
- Each patch should leave the tree buildable and test-passing. CI should pass on
  every commit, not just the tip.
- Each PR should be self-contained and reviewable on its own.
