# Repo Conventions

*Read this file at the start of every new Claude session working in this repo.*
*Last updated: 2026-02-21*

---

## Meta-Convention

- **Update this file** whenever the user provides new or changed conventions. This is the single source of truth for all repo conventions.
- **Every session notes file** should begin with a reference to read this file first.

---

## Git Remotes and Branching

- `origin` = `https://github.com/donboyd5/tax-microdata-benchmarking.git` (Don's fork)
- `upstream` = `https://github.com/PSLmodels/tax-microdata-benchmarking.git` (main PSLmodels repo)
- **PRs must be pushed to the `upstream` remote** (PSLmodels), NOT to `origin` (your fork).
  Push command: `git push upstream <branch-name>`
- **Never commit directly to local master.** Master should only be updated by pulling from upstream.

---

## Code Quality

- **Run `make format` and `make lint` before pushing to upstream remote.** This ensures code passes the lint/style checks.
- **Always verify claims before posting PR descriptions.** Do not assert "all tests passing" or "fingerprint unchanged" without running them in the current session.

---

## Data Conventions

- PUF `S006` is stored in hundredths; divide by 100 for actual weights. TMD `s006` is already divided.
- PUF variable names are UPPERCASE (e.g., `E03400`); TMD variable names are lowercase (e.g., `e03400`).

---

## Communication Style

- **When presenting choices to the user, always use identification letters (A, B, C, ...) instead of numbers.** This makes it easier to reference choices in conversation.

---

## Session Notes Workflow

The `session_notes/` directory is tracked on a dedicated orphan branch (`session-notes`) on the `origin` fork, using a git worktree. This keeps session notes:
- Separate from all working branches (no risk of accidentally including them in PRs to upstream)
- Editable from any branch without switching
- Pushed to the origin fork for backup

### Daily workflow for committing session notes changes

**Ask Claude to commit and push session notes** at any point during a session (e.g., "push session notes" or "commit the notes"). Claude should run:

```bash
cd session_notes
git add -A && git commit -m "Update notes" && git push
cd ..
```

Claude should also **proactively offer to commit and push session notes** at the end of a session or after making significant updates to any notes file.

### If the worktree is lost or needs to be recreated

```bash
git worktree add session_notes session-notes
```

This restores the `session_notes/` directory from the `session-notes` branch on origin.
