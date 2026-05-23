"""Regression tests for issue #19 — simulator must not crash under cp1252.

The documented onboarding command::

    python -m src.network.simulator

used to raise ``UnicodeEncodeError`` on a default Windows shell because
``src/network/simulator.py`` printed Greek glyphs (``Σ``, ``θ``) without
ensuring stdout could encode them. These tests guard against that
regression by:

1. Asserting the source file is ASCII-only.
2. Running ``python -m src.network.simulator`` end-to-end with
   ``PYTHONIOENCODING=cp1252`` and asserting clean exit + expected text.
3. Asserting ``build_demo_swarm`` is stable (5 agents, one Byzantine,
   reasonable confidence shape) since downstream callers
   (``health_demo``, docs) import it.
4. Asserting ``_make_stdout_robust`` is defensive — it silently no-ops on
   streams without ``reconfigure`` rather than blowing up.
"""
from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.network import simulator as sim_module
from src.network.simulator import (
    _make_stdout_robust,
    build_demo_swarm,
    main,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Source file is ASCII-only (so it cannot regress to Σ/θ)
# ---------------------------------------------------------------------------
def test_simulator_source_is_pure_ascii() -> None:
    text = Path(sim_module.__file__).read_text(encoding="utf-8")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:  # pragma: no cover - failure path
        pytest.fail(
            f"src/network/simulator.py contains a non-ASCII character "
            f"({exc.reason!r} at offset {exc.start}); this regresses "
            f"issue #19 since the demo runs under default Windows cp1252."
        )


def test_simulator_source_does_not_print_greek_glyphs() -> None:
    text = Path(sim_module.__file__).read_text(encoding="utf-8")
    # Specifically guard the glyphs from the original crash.
    for glyph in ("\u03a3", "\u03b8"):  # capital sigma, lowercase theta
        assert glyph not in text, (
            f"simulator.py reintroduced {glyph!r}; issue #19 forbids "
            "non-ASCII print output on the onboarding entry point."
        )


# ---------------------------------------------------------------------------
# 2. End-to-end: run the actual documented command under cp1252
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "utf-8"])
def test_simulator_runs_clean_under_strict_encodings(encoding: str) -> None:
    """`python -m src.network.simulator` must exit 0 under cp1252 (issue #19).

    Also exercised under pure ASCII (strictest) and UTF-8 (default Linux).
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = encoding
    # Make sure src/ is importable regardless of CWD.
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-m", "src.network.simulator"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, (
        f"simulator exited {proc.returncode} under PYTHONIOENCODING={encoding}.\n"
        f"--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}"
    )
    # Sanity: the demo should report a commit (the swarm has 3 honest 42 votes).
    assert "COMMITTED" in proc.stdout or "NO CONSENSUS" in proc.stdout
    # Hard guard against the bug coming back: stderr must not mention
    # UnicodeEncodeError under any of these encodings.
    assert "UnicodeEncodeError" not in proc.stderr, proc.stderr


# ---------------------------------------------------------------------------
# 3. build_demo_swarm shape is stable (it's a public-ish import surface)
# ---------------------------------------------------------------------------
def test_build_demo_swarm_shape() -> None:
    swarm = build_demo_swarm()
    assert len(swarm) == 5
    ids = [a.id for a in swarm]
    assert ids == ["a1", "a2", "a3", "a4", "a5"], ids
    assert sum(1 for a in swarm if a.byzantine) == 1
    for a in swarm:
        assert 0.0 <= a.confidence <= 1.0
        assert isinstance(a.answer, str) and a.answer


def test_build_demo_swarm_is_pure() -> None:
    """Two calls must return independent agent instances, not shared state."""
    a, b = build_demo_swarm(), build_demo_swarm()
    assert a is not b
    assert all(x is not y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 4. _make_stdout_robust is defensive
# ---------------------------------------------------------------------------
def test_make_stdout_robust_noop_on_stream_without_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured streams (StringIO, pytest capture) have no ``reconfigure``."""
    fake = io.StringIO()
    assert not hasattr(fake, "reconfigure")
    monkeypatch.setattr(sys, "stdout", fake)
    # Must not raise.
    _make_stdout_robust()


def test_make_stdout_robust_swallows_reconfigure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``reconfigure`` exists but raises, we must keep running."""

    class BoomStream:
        def reconfigure(self, **_kw: object) -> None:
            raise OSError("nope")

    monkeypatch.setattr(sys, "stdout", BoomStream())
    # Must not propagate the OSError.
    _make_stdout_robust()


def test_make_stdout_robust_calls_reconfigure_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class TrackingStream:
        def reconfigure(self, **kw: object) -> None:
            calls.append(kw)

    monkeypatch.setattr(sys, "stdout", TrackingStream())
    _make_stdout_robust()

    assert len(calls) == 1
    assert calls[0]["encoding"] == "utf-8"
    # We pass errors="replace" so production never crashes on stray glyphs.
    assert calls[0]["errors"] == "replace"


# ---------------------------------------------------------------------------
# 5. main() prints ASCII-only output that humans (and cp1252) can read
# ---------------------------------------------------------------------------
def test_main_output_is_ascii_only(capsys: pytest.CaptureFixture[str]) -> None:
    asyncio.run(main())
    captured = capsys.readouterr()
    # The whole printed payload must round-trip through ASCII.
    captured.out.encode("ascii")
    # And it should clearly describe the commit (or absence thereof).
    assert ("COMMITTED" in captured.out) or ("NO CONSENSUS" in captured.out)
    assert "sum V_i" in captured.out or "NO CONSENSUS" in captured.out


def test_main_uses_theta_word_not_glyph(
    capsys: pytest.CaptureFixture[str],
) -> None:
    asyncio.run(main())
    out = capsys.readouterr().out
    assert "\u03b8" not in out  # no Greek theta
    assert "\u03a3" not in out  # no Greek sigma
    if "COMMITTED" in out:
        assert "theta=" in out
