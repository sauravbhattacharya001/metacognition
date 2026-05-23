# Contributing to mBFT

Thanks for your interest in **mBFT — Metacognitive Byzantine Fault Tolerance**. This document
explains how to set up a development environment, the expected quality bar for changes, and
the workflow for issues, pull requests, and protocol-level proposals.

mBFT is a *reference implementation of a research protocol*. That means correctness, falsifiability
of empirical claims, and clarity of the math-to-code mapping all matter more than raw throughput
or feature volume. Please keep that framing in mind.

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [Getting set up](#getting-set-up)
3. [Project layout](#project-layout)
4. [Running tests](#running-tests)
5. [Code style](#code-style)
6. [Branching and commits](#branching-and-commits)
7. [Pull request checklist](#pull-request-checklist)
8. [What we look for in a PR](#what-we-look-for-in-a-pr)
9. [Proposing protocol changes](#proposing-protocol-changes)
10. [Reporting bugs and security issues](#reporting-bugs-and-security-issues)
11. [Release process](#release-process)

---

## Code of conduct

Be kind, be precise, assume good faith. Disagreement on protocol design is welcome; personal
attacks, harassment, or dismissive behaviour are not. Maintainers reserve the right to lock
threads, close issues, or remove comments that don't meet that bar.

---

## Getting set up

Requirements:

- Python **3.10+** (3.12 recommended — CI runs the full matrix on 3.10–3.12)
- `pip` and `venv` (or `uv`, if you prefer)
- A POSIX-like shell (Linux, macOS, or PowerShell on Windows — all are supported by CI)
- Optional: `mkdocs` if you intend to preview the docs site locally

```bash
git clone https://github.com/sauravbhattacharya001/metacognition.git
cd metacognition

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

The editable install pulls in runtime dependencies plus the `dev` extras (pytest, coverage,
type-checking and lint tooling). If `[dev]` isn't defined for your install, fall back to:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy
```

If you plan to touch the docs site:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

---

## Project layout

```
src/             Protocol engines and supporting modules (one concern per file)
tests/           pytest suite — mirrors src/ where practical
docs/            Markdown sources for the mkdocs-material site
paper/           LaTeX / PDF artefacts for the accompanying write-up
.github/         CI, issue templates, dependabot, copilot setup
pyproject.toml   Build + tool config (ruff, mypy, pytest)
```

Most modules in `src/` are intentionally self-contained "engines": calibrator, consensus,
diversity, governance, dreaming, etc. Cross-engine coordination lives in the higher-level
orchestration modules (autopilot, cascade, governance). When adding a new engine, prefer
following that pattern over reaching into other engines' internals.

---

## Running tests

```bash
# Full suite
pytest

# Quick feedback loop while iterating on a single module
pytest tests/test_calibrator.py -x -q

# With coverage (matches what CI reports)
pytest --cov=src --cov-report=term-missing
```

**House rules:**

- Every new public function or class should have at least one test that demonstrates its
  intended behaviour.
- Every bug fix should add a regression test that would have failed before the fix.
- Stochastic tests must seed their RNGs (`random.seed`, `numpy.random.seed`, or `torch.manual_seed`).
  Flaky tests are bugs.
- Tests that exercise consensus or swarm behaviour should keep node counts small
  (`n_agents <= 8`) unless you're specifically measuring scaling.

---

## Code style

We rely on **ruff** for lint + formatting and **mypy** for type checking. Both are run in CI.

```bash
ruff format .
ruff check .
mypy src
```

Project conventions on top of those tools:

- Type hints on all public functions. `Any` is allowed for genuinely dynamic protocol payloads
  but should be the exception.
- Prefer dataclasses or `pydantic` models over loose dicts for messages that cross engine
  boundaries.
- Keep modules under ~800 lines where possible. If an engine is getting larger than that,
  split it (e.g. `consensus.py` → `consensus/` package).
- Public API symbols use `snake_case` for functions and `PascalCase` for classes. Internal
  helpers start with `_`.
- Logging via `logging.getLogger(__name__)` only. Don't `print()` in library code.
- Docstrings: one-line summary, blank line, then details. Include references to the paper
  section number when implementing something the paper describes.

---

## Branching and commits

mBFT uses **trunk-based development on `master`**.

- Open small, focused PRs against `master`. Long-lived feature branches tend to drift.
- Commit messages: short imperative subject (≤ 72 chars), optional body explaining *why*.
- If your change affects observable behaviour, add a line to `CHANGELOG.md` (if present) under
  an `Unreleased` heading.

Suggested commit subject patterns:

```
fix(calibrator): clamp Brier score to [0,1] under degenerate inputs
feat(consensus): add weighted-quorum mode (paper §4.3)
docs(engines): document angiogenesis hyperparameters
test(diversity): cover empty-population branch
```

---

## Pull request checklist

Before opening a PR, please confirm:

- [ ] `pytest` passes locally
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] `mypy src` is clean (or you've explained why a regression is acceptable)
- [ ] New or changed behaviour has tests
- [ ] Public API changes are reflected in `docs/` and (where appropriate) `README.md`
- [ ] You've described **what** and **why** in the PR body, not just *what*

Draft PRs are welcome for early feedback — just mark them as draft.

---

## What we look for in a PR

Maintainers tend to ask the same questions during review. Pre-empting them speeds things up:

1. **Correctness against the protocol.** Does this match the paper? If you're changing the
   protocol, say so explicitly and link the relevant section.
2. **Fault-tolerance assumptions.** Does the change preserve the `n >= 3f + 1` quorum
   condition (or whatever variant the engine in question uses)?
3. **Determinism where it matters.** Engines that need to be replayable must remain seedable.
4. **Cost.** New synchronous network round-trips, O(n²) scans over the agent set, and unbounded
   queues all need a justification.
5. **No silent failure.** Prefer raising a typed exception over returning `None` / `False` from
   an engine when an invariant is violated.

---

## Proposing protocol changes

Changes to the **protocol itself** (not just the implementation) should be discussed before
code is written. The flow is:

1. Open a **Discussion** in the repo's Discussions tab (or an issue tagged `protocol-change`)
   describing the motivation, proposed behaviour, and impact on safety/liveness.
2. Reach rough consensus with maintainers.
3. Open the PR, referencing the discussion. Include updates to:
   - `docs/protocol.md` (or `docs/architecture.md` if cross-engine)
   - the relevant section of `paper/` if the change invalidates a paper claim
   - tests that exercise the new behaviour

Protocol changes that break backward compatibility require a minor or major version bump
(see [Release process](#release-process)).

---

## Reporting bugs and security issues

- **Functional bugs:** use the [Bug report](.github/ISSUE_TEMPLATE/bug_report.yml) template.
- **Documentation problems:** use the [Documentation](.github/ISSUE_TEMPLATE/documentation.yml) template.
- **Feature requests:** use the [Feature request](.github/ISSUE_TEMPLATE/feature_request.yml) template.
- **Security vulnerabilities:** please **do not** open a public issue. Use
  [GitHub Security Advisories](https://github.com/sauravbhattacharya001/metacognition/security/advisories/new)
  so maintainers can coordinate a private fix.

When filing a bug, a minimal reproducer with a fixed seed is worth ten paragraphs of
description.

---

## Release process

Releases are cut from `master` by maintainers using the existing `release.yml` workflow.

- Versioning follows **SemVer** (`MAJOR.MINOR.PATCH`):
  - `PATCH` — bug fixes, doc-only changes, internal refactors with no API impact
  - `MINOR` — new engines, new public APIs, additive protocol options
  - `MAJOR` — breaking API changes or protocol changes that aren't backward compatible
- Tag format: `vMAJOR.MINOR.PATCH` (e.g. `v1.7.3`)
- The CI tag push triggers PyPI publish via `publish.yml`. Make sure `pyproject.toml` is in
  sync with the tag before pushing.

Thanks again for contributing. PRs that move the protocol or the docs forward — even small
ones — are very welcome.
