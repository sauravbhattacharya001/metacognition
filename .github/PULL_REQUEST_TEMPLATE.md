<!--
Thanks for sending a PR. A few things to confirm before requesting review:
-->

## Summary

<!-- One paragraph: what does this change and why? Link the issue if applicable. -->

Fixes #

## Type of change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing behaviour to change)
- [ ] 📚 Documentation only
- [ ] 🧹 Refactor / cleanup (no behaviour change)
- [ ] 🧪 Tests only
- [ ] 🛠️ Build / CI / tooling

## Consensus safety

> If this touches `src/core/*`, `src/agents/*`, or anything in the vote path, please answer:

- [ ] Preserves safety under `f` Byzantine agents (no two honest agents commit conflicting values).
- [ ] Preserves liveness under partial synchrony (no new code paths can hang indefinitely).
- [ ] Did not weaken any existing invariant assertions in the codebase.
- [ ] N/A — change is outside the consensus path.

## Testing

- [ ] `pytest` passes locally.
- [ ] Added tests covering the new behaviour / regression.
- [ ] Coverage did not drop meaningfully (`pytest --cov=src`).

## Checklist

- [ ] Followed the style of surrounding code (type hints, docstrings, no unused imports).
- [ ] Updated docs / README / changelog if behaviour or API changed.
- [ ] No secrets, API keys, or local paths committed.
