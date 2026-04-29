# Publishing to PyPI

This project uses [Trusted Publishers](https://docs.pypi.org/trusted-publishers/) (OIDC) for secure, tokenless publishing from GitHub Actions.

## Automated Publishing

Releases are published automatically when a GitHub Release is created:

1. **Create a release** on GitHub (tag format: `vX.Y.Z`)
2. The `publish.yml` workflow builds, verifies, and publishes to PyPI

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
python -c "from src.core import MBFTEngine; print('OK')"
```

## Version Bumps

Update `version` in `pyproject.toml` before creating a release tag.
