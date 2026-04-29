# Publishing to PyPI

This project uses [Trusted Publishers](https://docs.pypi.org/trusted-publishers/) (OIDC) for secure, tokenless publishing from GitHub Actions.

## Automated Publishing

Releases are published automatically when a GitHub Release is created:

1. **Create a release** on GitHub (tag format: `vX.Y.Z`)
2. The `publish.yml` workflow builds, verifies, attests, and publishes to PyPI

### Safety checks (automatic)

- **Version-tag consistency** — the release tag must match `src/__version__`; mismatches abort the pipeline
- **PEP 561 marker** — verifies `src/py.typed` is present so downstream type checkers can use the package
- **Build attestation** — [SLSA provenance](https://slsa.dev/) generated via `actions/attest-build-provenance@v2` for supply-chain verification

## Manual Publishing

Use the **workflow_dispatch** trigger to publish on demand:

1. Go to **Actions → Publish to PyPI**
2. Click **Run workflow**
3. Select target: `testpypi` (staging) or `pypi` (production)

## Setup (One-Time)

### PyPI Trusted Publisher

1. Go to <https://pypi.org/manage/account/publishing/>
2. Add a new pending publisher:
   - **Owner:** `sauravbhattacharya001`
   - **Repository:** `metacognition`
   - **Workflow:** `publish.yml`
   - **Environment:** `pypi`

3. Repeat for TestPyPI at <https://test.pypi.org/manage/account/publishing/> with environment `testpypi`

### GitHub Environments

Create two environments in **Settings → Environments**:

- **`pypi`** — for production releases (optionally add required reviewers)
- **`testpypi`** — for staging/test releases

## Local Build & Check

```bash
pip install build twine
python -m build
twine check dist/*

# Test install
pip install dist/mbft_consensus-*.whl
python -c "from src import __version__; print(__version__)"
python -c "from src.core import MBFTEngine; print('OK')"
```

## Version Bumps

Update `__version__` in `src/__init__.py` before creating a release tag.
The version is read dynamically from `src.__version__` via `pyproject.toml` (`[tool.setuptools.dynamic]`).
