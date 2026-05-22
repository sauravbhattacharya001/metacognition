"""Tests for src.spectral - frequency-domain consensus analysis.

Covers:
- The minimal radix-2 FFT against a hand-computed reference and against a
  known-frequency sinusoid.
- ``power_spectrum`` / ``phase_angles`` boundary behavior and dominant-frequency
  detection for a clean tone.
- ``analyze_spectral`` end-to-end on a deterministic, synthetic ``data`` dict
  (no asyncio / MBFTEngine required), exercising the oscillator, resonance, and
  aggregate-spectrum paths plus the recommendations text.
- A regression test for the perf refactor that previously called
  ``power_spectrum`` 3x per agent: we monkeypatch the module-level helpers and
  assert ``analyze_spectral`` no longer invokes them (it now uses a single
  inlined FFT pass per agent).
- ``generate_html_report`` produces well-formed HTML that contains the
  serialized analysis blobs.
"""
from __future__ import annotations

import json
import math

import pytest

from src import spectral
from src.spectral import (
    _fft,
    analyze_spectral,
    generate_html_report,
    phase_angles,
    power_spectrum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _make_data(voting_series, byzantine_ids=None, aggregate_series=None,
               num_tasks=4, num_rounds=8, threshold=0.6):
    agent_ids = list(voting_series.keys())
    return {
        "agent_ids": agent_ids,
        "byzantine_ids": list(byzantine_ids or []),
        "voting_series": voting_series,
        "aggregate_series": aggregate_series if aggregate_series is not None
            else [sum(v) / max(len(voting_series), 1) for v in zip(*voting_series.values())],
        "commitment_series": [1] * (len(next(iter(voting_series.values()))) if voting_series else 0),
        "num_tasks": num_tasks,
        "num_rounds": num_rounds,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# FFT primitives
# ---------------------------------------------------------------------------

class TestFFT:
    def test_fft_size_one_returns_input(self):
        assert _fft([3 + 0j]) == [3 + 0j]

    def test_fft_size_two_matches_hand_computation(self):
        # FFT of [a, b] = [a+b, a-b]
        out = _fft([1 + 0j, 2 + 0j])
        assert _approx(out[0].real, 3.0) and _approx(out[0].imag, 0.0)
        assert _approx(out[1].real, -1.0) and _approx(out[1].imag, 0.0)

    def test_fft_pads_to_power_of_two(self):
        # length 3 must be padded to 4; result length is 4
        out = _fft([1 + 0j, 0 + 0j, 0 + 0j])
        assert len(out) == 4
        # DC component equals sum of inputs
        assert _approx(out[0].real, 1.0)

    def test_fft_dirac_delta_is_flat_spectrum(self):
        # delta -> all bins have magnitude 1
        out = _fft([1 + 0j] + [0 + 0j] * 7)
        mags = [math.hypot(c.real, c.imag) for c in out]
        for m in mags:
            assert _approx(m, 1.0, tol=1e-9)

    def test_fft_pure_tone_concentrates_power(self):
        n = 64
        k = 4  # 4 cycles in window
        signal = [complex(math.cos(2 * math.pi * k * i / n)) for i in range(n)]
        out = _fft(signal)
        mags = [math.hypot(c.real, c.imag) for c in out]
        # bins k and n-k should dominate (Hermitian symmetry for real input)
        assert max(range(len(mags)), key=lambda i: mags[i]) in (k, n - k)


# ---------------------------------------------------------------------------
# power_spectrum / phase_angles
# ---------------------------------------------------------------------------

class TestPowerSpectrum:
    def test_too_short_returns_empty(self):
        assert power_spectrum([]) == ([], [])
        assert power_spectrum([1.0]) == ([], [])

    def test_returns_half_spectrum(self):
        sig = [float(i) for i in range(8)]
        freqs, pwr = power_spectrum(sig)
        # half-spectrum of N=8 -> 4 bins
        assert len(freqs) == 4
        assert len(pwr) == 4

    def test_dc_removed_so_zero_freq_has_no_power(self):
        # constant signal -> mean-centering yields zero -> all power = 0
        freqs, pwr = power_spectrum([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        for p in pwr:
            assert _approx(p, 0.0, tol=1e-9)

    def test_dominant_frequency_matches_input_tone(self):
        n = 64
        k = 6
        sig = [math.cos(2 * math.pi * k * i / n) for i in range(n)]
        freqs, pwr = power_spectrum(sig)
        dom = max(range(len(pwr)), key=lambda i: pwr[i])
        # k/N should be the dominant normalized frequency
        assert _approx(freqs[dom], k / n, tol=1e-6)

    def test_frequencies_are_monotonic_and_nonneg(self):
        freqs, _ = power_spectrum([float(i % 3) for i in range(16)])
        assert all(0.0 <= f for f in freqs)
        assert freqs == sorted(freqs)


class TestPhaseAngles:
    def test_empty_for_short_input(self):
        assert phase_angles([1.0]) == []

    def test_length_matches_half_spectrum(self):
        ph = phase_angles([float(i) for i in range(16)])
        assert len(ph) == 8

    def test_values_in_minus_pi_pi(self):
        ph = phase_angles([math.sin(i / 3) for i in range(32)])
        for p in ph:
            assert -math.pi <= p <= math.pi


# ---------------------------------------------------------------------------
# analyze_spectral
# ---------------------------------------------------------------------------

class TestAnalyzeSpectral:
    def test_empty_data_yields_default_recommendation(self):
        data = _make_data({})
        result = analyze_spectral(data)
        assert result["agents"] == {}
        assert result["oscillators"] == []
        assert result["resonance_groups"] == []
        assert any("No significant spectral anomalies" in r for r in result["recommendations"])

    def test_too_short_series_skipped(self):
        data = _make_data({"a": [0.5]}, aggregate_series=[0.5])
        result = analyze_spectral(data)
        # length-1 series cannot yield a spectrum
        assert "a" not in result["agents"]

    def test_per_agent_fields_present(self):
        n = 32
        sig = [math.cos(2 * math.pi * 4 * i / n) for i in range(n)]
        data = _make_data({"a": sig}, aggregate_series=sig)
        result = analyze_spectral(data)
        ag = result["agents"]["a"]
        for key in ("dominant_freq", "dominant_power", "period",
                    "spectral_concentration", "total_power",
                    "is_byzantine", "top_freqs"):
            assert key in ag
        assert ag["is_byzantine"] is False
        assert len(ag["top_freqs"]) <= 5
        assert ag["dominant_freq"] > 0

    def test_byzantine_flag_propagates(self):
        sig = [math.sin(i) for i in range(16)]
        data = _make_data({"a": sig, "b": sig},
                          byzantine_ids=["a"], aggregate_series=sig)
        result = analyze_spectral(data)
        assert result["agents"]["a"]["is_byzantine"] is True
        assert result["agents"]["b"]["is_byzantine"] is False

    def test_oscillator_flip_flop_detected(self):
        # Period-4 oscillation (period < 10) with strong spectral concentration.
        # (Pure period-2 / Nyquist falls outside the half-spectrum because the
        # Nyquist bin is not included in range(N//2).)
        flip = [1.0, 1.0, 0.0, 0.0] * 8
        data = _make_data({"flipper": flip}, byzantine_ids=["flipper"],
                          aggregate_series=flip)
        result = analyze_spectral(data)
        osc_agents = [o["agent"] for o in result["oscillators"]]
        assert "flipper" in osc_agents, (
            f"expected flipper in oscillators, got {result['oscillators']}"
        )
        osc = next(o for o in result["oscillators"] if o["agent"] == "flipper")
        assert osc["period"] < 10
        assert osc["is_byzantine"] is True
        # recommendation about Byzantine flip-flop should fire
        assert any("Byzantine" in r and "oscillation" in r for r in result["recommendations"])

    def test_resonance_group_detected_for_identical_signals(self):
        # Two identical, non-trivial signals -> high phase coherence -> grouped
        sig = [math.cos(2 * math.pi * 5 * i / 32) for i in range(32)]
        data = _make_data({"a": sig, "b": sig}, aggregate_series=sig)
        result = analyze_spectral(data)
        # phase coherence between identical signals should be ~1.0
        pcs = [pc for pc in result["phase_coherence"]
               if {pc["agent_a"], pc["agent_b"]} == {"a", "b"}]
        assert pcs and pcs[0]["coherence"] >= 0.99
        # And produce a resonance group of size 2
        assert any(set(g["members"]) == {"a", "b"} for g in result["resonance_groups"])

    def test_phase_coherence_count_is_pairwise(self):
        sig1 = [math.cos(2 * math.pi * 3 * i / 32) for i in range(32)]
        sig2 = [math.cos(2 * math.pi * 5 * i / 32) for i in range(32)]
        sig3 = [math.sin(2 * math.pi * 7 * i / 32) for i in range(32)]
        data = _make_data({"a": sig1, "b": sig2, "c": sig3},
                          aggregate_series=sig1)
        result = analyze_spectral(data)
        # C(3, 2) = 3 phase-coherence entries
        assert len(result["phase_coherence"]) == 3

    def test_aggregate_spectrum_populated_for_long_signal(self):
        agg = [math.cos(2 * math.pi * 4 * i / 64) for i in range(64)]
        data = _make_data({"a": agg}, aggregate_series=agg)
        result = analyze_spectral(data)
        spec = result["aggregate_spectrum"]
        assert spec.get("dominant_freq", 0) > 0
        assert spec.get("period") is not None
        assert len(spec.get("top_freqs", [])) <= 5

    def test_recommendation_when_aggregate_oscillates(self):
        # short-period aggregate (period < 8) should produce the threshold/slash hint.
        # Use period-4 pattern so the dominant bin lies in the visible half-spectrum.
        agg = [1.0, 1.0, -1.0, -1.0] * 8
        data = _make_data({"a": agg}, aggregate_series=agg)
        result = analyze_spectral(data)
        assert any("Aggregate consensus oscillates" in r for r in result["recommendations"])


# ---------------------------------------------------------------------------
# Regression: perf refactor should remove redundant FFT calls
# ---------------------------------------------------------------------------

class TestPerfRegression:
    def test_analyze_spectral_does_not_call_module_helpers(self, monkeypatch):
        """The optimized analyze_spectral runs its own single FFT pass per
        agent. It must NOT delegate to the module-level ``power_spectrum`` or
        ``phase_angles`` helpers (which would re-run the recursive FFT).

        This guards against accidental reintroduction of the 3x-4x redundant
        FFT pattern that motivated the perf fix.
        """
        calls = {"power_spectrum": 0, "phase_angles": 0}

        def _spy_power(*a, **k):
            calls["power_spectrum"] += 1
            return [], []

        def _spy_phase(*a, **k):
            calls["phase_angles"] += 1
            return []

        monkeypatch.setattr(spectral, "power_spectrum", _spy_power)
        monkeypatch.setattr(spectral, "phase_angles", _spy_phase)

        sig = [math.cos(2 * math.pi * 3 * i / 32) for i in range(32)]
        data = _make_data({"a": sig, "b": sig, "c": sig}, aggregate_series=sig)
        analyze_spectral(data)

        # analyze_spectral now inlines the FFT/power/phase computation for
        # per-agent series. The single remaining call to ``power_spectrum`` is
        # for the *aggregate* series (one FFT total, not per-agent). With 3
        # agents the old code path produced 3 (per-agent) + 3 (phase loop) +
        # 3 (spectrogram) = 9 power_spectrum calls; now it should be exactly 1.
        assert calls["power_spectrum"] <= 1, (
            f"analyze_spectral should call power_spectrum at most once "
            f"(for the aggregate series); got {calls['power_spectrum']}. "
            "This guards against reintroducing the 3x-per-agent redundant FFT."
        )
        assert calls["phase_angles"] == 0, (
            f"analyze_spectral should not call phase_angles; got "
            f"{calls['phase_angles']}. Phases are now derived inline from the "
            "single per-agent FFT."
        )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

class TestHtmlReport:
    def test_report_is_self_contained_html(self):
        sig = [math.cos(2 * math.pi * 4 * i / 32) for i in range(32)]
        data = _make_data({"a": sig, "b": sig}, byzantine_ids=["a"],
                          aggregate_series=sig)
        analysis = analyze_spectral(data)
        html = generate_html_report(data, analysis)

        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "Consensus Spectral Analyzer" in html
        # JSON blobs are embedded for the JS renderer
        assert json.dumps(analysis["aggregate_spectrum"]) in html
        # Tab buttons exist
        for tab in ("Overview", "Agent Spectra", "Spectrogram",
                    "Phase Coherence", "Detection"):
            assert tab in html
        # Byzantine agent labeled in agent rows
        assert "Byzantine" in html and "Honest" in html

    def test_report_escapes_recommendations(self):
        # Inject a recommendation with HTML-special chars via crafted oscillator
        # data; easier: build an analysis dict by hand.
        data = _make_data({"a": [0.0] * 4}, aggregate_series=[0.0] * 4)
        analysis = analyze_spectral(data)
        analysis["recommendations"] = ["<script>alert(1)</script>"]
        html = generate_html_report(data, analysis)
        # html.escape converts < and > but leaves the text recognizable
        assert "&lt;script&gt;" in html
        assert "<script>alert(1)</script>" not in html.replace(
            json.dumps(analysis["aggregate_spectrum"]), ""
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
