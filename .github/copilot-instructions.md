# Copilot Instructions — metacognition (MBFT)

This document orients GitHub Copilot coding agents (Claude, Codex, etc.) working
in this repository. Read it before making non-trivial changes.

## What this project is

`metacognition` is the reference implementation of **Metacognitive Byzantine
Fault Tolerance (MBFT)** — a consensus framework where agents reason about
their own reasoning. The package is published to PyPI as `mbft-consensus`.

It is a research codebase: many modules implement biologically- and
neurologically-inspired "engines" (allostasis, autophagy, microbiome,
hibernation, nociception, …) that compose into a multi-agent swarm with
self-regulating behaviors. The code targets **Python 3.10+** and is fully
typed (`py.typed`).

## Repository layout

```
src/
  __init__.py            # exports __version__ and high-level API
  agents/                # individual agent implementations
  core/                  # consensus primitives, message types
  network/               # message transport
  <engine>.py            # ~40 standalone engines (each ~20–60 KB)
  stats_utils.py         # shared math helpers (pearson, gini, cosine_similarity)
tests/
  test_<engine>.py       # one test module per engine + cross-cutting tests
docs/                    # MkDocs site (Material theme)
paper/                   # research paper sources
.github/workflows/       # CI, docker, pages, publish, codeql, dependabot
Dockerfile               # multi-stage container image
mkdocs.yml               # docs site config
pyproject.toml           # build + tooling config (pytest, coverage)
```

## How to work in this repo

### Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install ruff
```

The full reproducible bootstrap lives in
`.github/copilot-setup-steps.yml` — mirror it if the steps above ever drift.

### Test

```bash
pytest -q --no-cov                # fast iteration
pytest                            # full run, includes coverage (configured in pyproject)
pytest tests/test_<engine>.py -q  # single module
```

`pytest-asyncio` is in `auto` mode — `async def test_*` functions are picked
up without decorators. Tests are deterministic; if you add randomness, seed it.

### Lint / format

```bash
ruff check src/ tests/
ruff format src/ tests/
```

CI only fails on the `E9,F63,F7,F82` ruff subset (syntax + undefined names) —
keep new code clean but don't reformat unrelated files in a fix PR.

### Docs

```bash
pip install mkdocs mkdocs-material
mkdocs serve     # http://127.0.0.1:8000
mkdocs build     # static site
```

## Conventions

- **One engine per file** under `src/`, named after the biological/cognitive
  analogue it models. Keep the public surface small; expose helpers through
  module-level functions or a single `*Engine` class.
- **No heavy dependencies.** Only `pydantic>=2.5` is a hard runtime
  requirement. Avoid pulling in numpy/scipy/torch — re-use `stats_utils`
  (`pearson`, `gini`, `cosine_similarity`) instead of reimplementing math.
- **Type everything.** The package ships a `py.typed` marker. New public
  functions and dataclasses must have annotations.
- **Tests live in `tests/test_<module>.py`.** Mirror the source filename.
  Prefer `pytest` style (plain functions or `Test*` classes), use
  `pytest.approx` for floats, and seed any `random.Random` instance.
- **No emojis or decorative output** in library code; CLI/log output may use
  them sparingly if it already does in surrounding context.
- **Backwards compatibility.** This is a published package — don't rename
  public symbols without a deprecation shim and a CHANGELOG entry.

## Pull request checklist

Before opening or merging a PR, make sure:

1. `pytest -q --no-cov` passes locally for affected modules.
2. `ruff check src/ tests/` is clean for files you touched.
3. New public APIs have docstrings and type hints.
4. If you added a new engine, also add `tests/test_<engine>.py` with at least
   construction, happy-path, and one edge-case test.
5. Don't bump `src/__init__.py` `__version__` — releases are tagged manually.

## Things to avoid

- Adding numpy / scipy / pandas / torch as runtime dependencies.
- Touching `paper/` unless explicitly asked — those are LaTeX sources.
- Mass reformatting unrelated files.
- Disabling tests to make CI green — fix them or revert your change.
- Editing `.github/workflows/publish.yml` casually; it pushes to PyPI on tag.

## Useful entry points

- Consensus core: `src/core/`
- Agent base classes: `src/agents/`
- Engine catalog & architecture overview: `docs/`
- Shared math: `src/stats_utils.py`
- Version: `src/__init__.py` (`__version__`)
