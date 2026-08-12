# Project Development Rules

## Test Execution
- After writing code, **do not automatically run tests**. Stop and report what was changed and why.
- Wait for explicit confirmation ("run tests" / "ok to test") before executing any test suite.
- If tests fail, report the failure reason. Do not attempt more than one self-fix cycle — pause and wait for confirmation again after that.

## Task Splitting
- For any task touching 2+ files, first present a plan: which files, what changes in each. Wait for confirmation before starting.
- Do not implement everything across all files and report at the end. Break work into steps (by file or by logical unit) and report after each step.
- Split large features into small, independently verifiable steps. Pause after each one.

## Comment Style
- Comments should be minimal and explain "why," not "what" (the code itself should make the "what" clear).
- **No large docstrings/JSDoc blocks**, unless it's a public API and I explicitly ask for one.
- Simple functions need no comments.
- Non-obvious logic, tricky decisions, or workarounds (e.g. why this algorithm, what edge case this avoids) can get a one-line comment.

## General
- If something is ambiguous, ask me directly rather than guessing and proceeding.
- Explain the approach before writing code, not after.