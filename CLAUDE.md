# FedInsight — project instructions

## Cost-tiered delegation policy

The main session model is the expensive orchestrator/judge. Keep its context
lean and its output short; push volume work down to the cheap project agents
in `.claude/agents/`:

| Agent | Model | Route these tasks to it |
|---|---|---|
| `scout` | Haiku | "Where is X?", "does Y exist?", summarize a module, read logs/diffs and report the gist |
| `runner` | Haiku | Run pytest/ruff/builds/long commands; report distilled pass/fail + error lines only |
| `mechanic` | Haiku | Precisely specified mechanical edits: lint fixes, renames, version bumps, table entries, constant syncs |
| `writer` | Sonnet | READMEs, docs, HF Space cards, comparison pages, changelogs |

Routing rules:
- **Delegate** when the task is self-contained and describable in one prompt
  paragraph, or when the raw output (file contents, test logs) would bloat the
  orchestrator's context.
- **Don't delegate** work that needs the conversation's accumulated context,
  tight multi-step coupling, debugging of unclear failures, architecture or
  API design, or anything security-sensitive — the orchestrator does those
  itself.
- When delegating, write a self-contained prompt: exact file paths, exact
  expected change/answer, and the verification command. Subagents start cold.
- Verification of delegated edits: trust the agent's reported ruff/pytest
  output; spot-check only when the change touches shared contracts
  (the vector-store interface in `src/vectorstore/base.py`, the LLM provider
  protocol in `src/rag/llm.py`, or public function signatures).

## Verification commands

- Lint: `python -m ruff check src scripts app.py`
- Tests: `python -m pytest -q` (no suite yet — see plan.md, step "Test suite")

## Project facts agents get wrong

- Mermaid renders on GitHub, not on Hugging Face's file viewer.
- Shell is PowerShell 7 on Windows; `&&` works, but `$env:VAR="x"` must be a
  separate statement (not `cd x && $env:V=1`).
