# Workflows

Concrete recipes for the operations that come up regularly. Each is designed to
leave the repo in a clean, recoverable state.

> **The fork branch is the artifact.** There is no patch series, no
> `format-patch` regeneration, and no superproject. Work is committed directly to
> the fork branch. The patch-stack workflows that used to live here were removed
> when the patch layer was retired; if upstreaming resumes, reconstruct them from
> git history rather than from memory.

## Rebasing on latest upstream

The base must always be the latest upstream release tag. See
`docs/fork/POLICIES.md#always-rebase-on-the-latest-upstream-release-tag`.

```bash
# 1. Fetch (the local upstream ref goes stale fast — always fetch first)
git fetch upstream --tags

# 2. Identify latest release tag merged into master
LATEST=$(git tag --merged upstream/master --list 'b*' | sort -V | tail -1)
echo "Latest upstream tag: $LATEST"

# 3. Identify current base
CURRENT_BASE=$(git merge-base HEAD upstream/master)
CURRENT_TAG=$(git tag --points-at "$CURRENT_BASE" | head -1)
echo "Current base: $CURRENT_TAG ($CURRENT_BASE)"

# 4. If different, back up and rebase
if [ "$LATEST" != "$CURRENT_TAG" ]; then
    git branch backup/pre-${LATEST}-rebase HEAD 2>/dev/null || true
    git push origin backup/pre-${LATEST}-rebase          # off-machine, not just local

    git rebase --onto "$LATEST" "$CURRENT_TAG"

    cmake --build build --target test-chat -j$(nproc)
    build/bin/test-chat
fi
```

If the rebase conflicts, fix each conflict, `git rebase --continue`, and verify
tests still pass **at each step** — not only at the tip.

**When the drift is large** (hundreds of upstream commits), a mechanical rebase
of a branch that rewrites `common/chat.cpp` is not realistic. Re-derive instead:
take the upstream file as the base and re-apply the fork's intent hunk by hunk,
using the backup branch as the source of truth for what that intent was. Deleting
unneeded formats *before* rebasing shrinks the conflict surface dramatically.

## Backup-branch workflow

Before any reset, checkout, stash, or filesystem snapshot, back the working state
up to a branch — and push it. A local-only backup does not survive a bad day.

```bash
git add -A
git commit -m "wip: full snapshot before <operation>"
git branch backup/<descriptor> HEAD
git reset HEAD^          # back the WIP commit out of the working branch;
                         # the snapshot survives on backup/<descriptor>
git push origin backup/<descriptor>

# To recover:
git checkout backup/<descriptor> -- <files>
```

**Check that work is actually off-machine before restructuring anything:**

```bash
git branch -r --contains HEAD | grep origin || echo "NOT PUSHED — fix before proceeding"
```

## Adding a commit

```bash
# Read files first (policy 8 — the tree may have advanced), then make changes.

cmake --build build --target test-chat -j$(nproc)
build/bin/test-chat

git add <files>          # explicit paths, not -A
git commit -m "$(cat <<'EOF'
<module> : <subject>

<body>
EOF
)"
```

Commit message convention is upstream's: `<module> : <title>`. **No
`Co-Authored-By` or other attribution trailers** (policy 4).

## Investigating a `test-chat` failure

```bash
mkdir -p .wip-backup
build/bin/test-chat --template <template-name> --detailed 2>&1 | tee .wip-backup/trace.txt
```

(`.wip-backup/` rather than `/tmp` — policy 7.)

In the trace, look for:

- `Parsed message:` — the final non-partial parse output.
- `Error comparing accumulated message to current` — streaming non-monotonicity.
  Compare consecutive prefixes' `Parsed message:` lines.
- `Tool call mismatch` — args/name/id divergence.
- `CHOICE option N: need_more_input` with no later option tried —
  alternative-backtracking issue in lenient mode.
- `AST for partial parse:` — what the decoder saw at each prefix.

Then walk the failing fixture:

- Does the **final non-partial** parse produce the expected message? If not, the
  problem is in the grammar or the decoder.
- Do the **streaming-prefix** parses produce monotonic outputs? If not, the
  problem is in the transformer's buffered state or an early `TOOL_CLOSE`.

## Adding a format plugin

The structural checklist is in
`docs/fork/ARCHITECTURE.md#how-a-new-format-is-added`. A format is only complete
when all four pieces exist and agree: **renderer, validator, tracker, grammar**.

Reminders that cost the most time when missed:

1. **Closing tokens stay required** in the grammar —
   `docs/fork/POLICIES.md#closing-tokens-are-required`.
2. **No synthesis** in the decoder or presenter; only the transformer may
   synthesise structural JSON characters, and only for a non-JSON wire shape.
3. **Streaming monotonicity** — emit per-arg KV events exactly once, when the
   parent rule is fully matched, never on partial value-text growth.
4. **The FSM↔grammar contract** — every tracker state must correspond to a named
   rule in the grammar, and `expected_productions()` must name real rules. Update
   tracker and grammar together; the startup check will reject drift.
5. **Register the format by name** — the same name `chat_format` /
   `--chat-format` / GGUF `general.architecture` resolve to. Never add a
   detection heuristic (policy 10).
6. **Test in isolation first** (`--template <name>`), then the full suite.
7. **Update `common/CMakeLists.txt`** with the new `<name>-format.{h,cpp}`.

## Verifying a conformance claim

Conformance is against the normative sources in
`docs/fork/ARCHITECTURE.md#normative-sources`, not against what the grammar
currently accepts. A claim needs **both** a primary source and a measurement:

```bash
# Grammar/parser level
build/bin/test-chat --template gemma4
build/bin/test-chat-grammar
build/bin/test-chat-render      # byte-parity against the CANONICAL template
build/bin/test-chat-grammar-files

# Server level — the only thing that validates the user-facing claim
cmake --build build --target llama-server -j$(nproc)
pytest tools/server/tests/unit/test_chat_completion.py
```

Rebuild a test target before trusting its result: a stale binary reports on a
tree that no longer exists, which has already produced one false failure here.
