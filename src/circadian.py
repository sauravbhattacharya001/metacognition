"""Swarm Circadian Engine — autonomous temporal rhythm detection for mBFT.

Detects cyclical performance patterns in swarm agents, identifies optimal
activity windows, classifies agent chronotypes, and detects rhythm
disruptions ("jet-lag") that degrade collective performance.

Capabilities:

- **7 Rhythm Metrics** — task throughput, response latency, error rate,
  collaboration frequency, creativity index, focus duration, recovery time
  — tracked across temporal bins.
- **6 Chronotype Classifications** — Early Bird, Night Owl, Bimodal,
  Steady State, Burst Worker, Irregular — per agent.
- **Phase Detection** — FFT-based dominant period identification with
  amplitude and phase offset extraction.
- **Jet-Lag Detector** — identifies rhythm disruptions via phase shift
  magnitude, recovery time estimation, and performance degradation scoring.
- **Optimal Window Finder** — identifies peak performance windows per
  agent and for the collective, with scheduling recommendations.
- **Entrainment Analysis** — measures phase coupling between agents to
  detect synchronized vs desynchronized swarms.
- **Health Score** — composite 0-100 reflecting rhythm stability and
  alignment.
- **Interactive HTML Dashboard** — polar performance plots, rhythm
  timelines, chronotype distribution, entrainment matrix.

Usage (Python API)::

    from src.circadian import CircadianEngine

    engine = CircadianEngine(bin_hours=1)

    # Record performance samples with timestamps
    engine.record_sample("agent_1", hour=9, metrics={
        "throughput": 12.0,
        "latency": 0.8,
        "error_rate": 0.02,
        "collaboration": 5,
        "creativity": 0.7,
        "focus_duration": 45.0,
        "recovery_time": 2.0,
    })

    # Classify chronotypes
    profile = engine.classify_chronotype("agent_1")
    print(profile.chronotype, profile.peak_hours, profile.confidence)

    # Detect rhythm disruption
    jetlag = engine.detect_jetlag("agent_1")
    print(jetlag.disrupted, jetlag.phase_shift_hours, jetlag.recovery_eta)

    # Find optimal windows
    windows = engine.optimal_windows("agent_1")
    print(windows.peak_start, windows.peak_end, windows.score)

    # Collective analysis
    report = engine.collective_report()
    print(report.health_score, report.entrainment_index)

    # Dashboard
    engine.export_html("circadian_dashboard.html")

CLI::

    python -m src.circadian                      # demo simulation
    python -m src.circadian --agents 10          # simulate 10 agents
    python -m src.circadian --cycles 168         # one week of hourly data
    python -m src.circadian --export html -o dash.html
    python -m src.circadian --json state.json
    python -m src.circadian --health
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "throughput",
    "latency",
    "error_rate",
    "collaboration",
    "creativity",
    "focus_duration",
    "recovery_time",
]

# Higher is better for these metrics
HIGHER_IS_BETTER = {"throughput", "collaboration", "creativity", "focus_duration"}
# Lower is better for these
LOWER_IS_BETTER = {"latency", "error_rate", "recovery_time"}

CHRONOTYPES = [
    "EarlyBird",
    "NightOwl",
    "Bimodal",
    "SteadyState",
    "BurstWorker",
    "Irregular",
]

NUM_BINS = 24  # hourly bins

# Metric weights for composite performance score
METRIC_WEIGHTS: Dict[str, float] = {
    "throughput": 0.20,
    "latency": 0.15,
    "error_rate": 0.15,
    "collaboration": 0.15,
    "creativity": 0.10,
    "focus_duration": 0.15,
    "recovery_time": 0.10,
}

# Chronotype peak hour ranges (inclusive)
CHRONOTYPE_PEAKS: Dict[str, Tuple[int, int]] = {
    "EarlyBird": (5, 11),
    "NightOwl": (20, 3),  # wraps around midnight
    "Bimodal": (9, 11),  # first peak; second detected separately
    "SteadyState": (0, 23),
    "BurstWorker": (0, 23),  # short intense bursts
    "Irregular": (0, 23),
}

# Phase shift threshold for jet-lag detection (hours)
JETLAG_THRESHOLD = 2.0

# Minimum samples per bin for reliable analysis
MIN_SAMPLES_PER_BIN = 3

# History window for jet-lag detection (number of periods to compare)
JETLAG_HISTORY_PERIODS = 3


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PerformanceSample:
    """Single performance observation."""
    agent_id: str
    hour: int  # 0-23
    metrics: Dict[str, float]
    timestamp: float = 0.0  # epoch seconds, optional ordering


@dataclass
class ChronotypeProfile:
    """Chronotype classification result."""
    agent_id: str
    chronotype: str
    confidence: float  # 0-1
    peak_hours: List[int]
    trough_hours: List[int]
    amplitude: float  # performance variation magnitude
    dominant_period: float  # hours
    phase_offset: float  # hour of peak in dominant cycle


@dataclass
class JetlagReport:
    """Rhythm disruption assessment."""
    agent_id: str
    disrupted: bool
    phase_shift_hours: float
    severity: str  # mild, moderate, severe
    recovery_eta_hours: float
    performance_degradation: float  # 0-1 fraction lost
    recent_phase: float
    baseline_phase: float


@dataclass
class OptimalWindow:
    """Peak performance window."""
    agent_id: str
    peak_start: int  # hour
    peak_end: int  # hour
    score: float  # average performance in window
    recommended_tasks: List[str]


@dataclass
class EntrainmentPair:
    """Phase coupling between two agents."""
    agent_a: str
    agent_b: str
    phase_difference: float  # hours
    coupling_strength: float  # 0-1
    synchronized: bool


@dataclass
class CollectiveReport:
    """Swarm-wide circadian health."""
    health_score: float  # 0-100
    entrainment_index: float  # 0-1, how synchronized the swarm is
    chronotype_distribution: Dict[str, int]
    collective_peak_hours: List[int]
    collective_trough_hours: List[int]
    disrupted_agents: List[str]
    recommendations: List[str]
    agent_count: int
    total_samples: int


@dataclass
class AgentRhythmState:
    """Internal state for one agent's rhythm tracking."""
    bins: Dict[int, List[Dict[str, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    phase_history: List[float] = field(default_factory=list)
    sample_count: int = 0


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------


class CircadianEngine:
    """Autonomous temporal rhythm detection and optimization engine."""

    def __init__(self, bin_hours: int = 1, max_samples_per_bin: int = 100):
        self.bin_hours = max(1, min(bin_hours, 12))
        self.num_bins = 24 // self.bin_hours
        self.max_samples_per_bin = max_samples_per_bin
        self._agents: Dict[str, AgentRhythmState] = {}
        self._sample_counter = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_sample(
        self,
        agent_id: str,
        hour: int,
        metrics: Dict[str, float],
        timestamp: float = 0.0,
    ) -> None:
        """Record a performance sample for an agent at a given hour."""
        hour = int(hour) % 24
        bin_idx = hour // self.bin_hours

        if agent_id not in self._agents:
            self._agents[agent_id] = AgentRhythmState(
                bins=defaultdict(list)
            )

        state = self._agents[agent_id]
        state.bins[bin_idx].append(metrics)
        state.sample_count += 1
        self._sample_counter += 1

        # Cap per-bin storage
        if len(state.bins[bin_idx]) > self.max_samples_per_bin:
            state.bins[bin_idx] = state.bins[bin_idx][-self.max_samples_per_bin:]

    # ------------------------------------------------------------------
    # Performance Scoring
    # ------------------------------------------------------------------

    def _compute_performance_score(self, metrics: Dict[str, float]) -> float:
        """Compute composite 0-1 performance score from raw metrics."""
        score = 0.0
        total_weight = 0.0

        for metric, weight in METRIC_WEIGHTS.items():
            if metric not in metrics:
                continue
            val = metrics[metric]
            # Normalize to 0-1 scale
            if metric in HIGHER_IS_BETTER:
                # Use sigmoid-like normalization
                normalized = min(1.0, max(0.0, val / (val + 1.0) if val >= 0 else 0.0))
                if metric == "throughput":
                    normalized = min(1.0, val / 20.0)
                elif metric == "collaboration":
                    normalized = min(1.0, val / 10.0)
                elif metric == "creativity":
                    normalized = min(1.0, max(0.0, val))
                elif metric == "focus_duration":
                    normalized = min(1.0, val / 60.0)
            else:
                # Lower is better — invert
                if metric == "latency":
                    normalized = max(0.0, 1.0 - val / 5.0)
                elif metric == "error_rate":
                    normalized = max(0.0, 1.0 - val)
                elif metric == "recovery_time":
                    normalized = max(0.0, 1.0 - val / 10.0)
                else:
                    normalized = max(0.0, 1.0 - val)
            score += weight * normalized
            total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def _bin_performance_profile(self, agent_id: str) -> List[float]:
        """Get per-bin average performance scores for an agent."""
        if agent_id not in self._agents:
            return [0.0] * self.num_bins

        state = self._agents[agent_id]
        profile = []
        for b in range(self.num_bins):
            samples = state.bins.get(b, [])
            if samples:
                scores = [self._compute_performance_score(s) for s in samples]
                profile.append(statistics.mean(scores))
            else:
                profile.append(0.0)
        return profile

    # ------------------------------------------------------------------
    # FFT-based Phase Detection
    # ------------------------------------------------------------------

    def _simple_dft(self, signal: List[float]) -> List[Tuple[float, float, float]]:
        """Compute DFT and return (period_bins, amplitude, phase) for each frequency."""
        n = len(signal)
        if n == 0:
            return []
        mean_val = statistics.mean(signal)
        centered = [x - mean_val for x in signal]

        results = []
        for k in range(1, n // 2 + 1):
            real = sum(centered[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
            imag = -sum(centered[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
            amplitude = math.sqrt(real**2 + imag**2) * 2.0 / n
            phase = math.atan2(imag, real)
            period_bins = n / k
            results.append((period_bins, amplitude, phase))
        return results

    def _dominant_rhythm(self, profile: List[float]) -> Tuple[float, float, float]:
        """Find dominant period (in hours), amplitude, phase offset (hour)."""
        dft_results = self._simple_dft(profile)
        if not dft_results:
            return (24.0, 0.0, 0.0)

        # Find max amplitude component
        best = max(dft_results, key=lambda x: x[1])
        period_hours = best[0] * self.bin_hours
        amplitude = best[1]
        # Convert phase to hour offset
        phase_hour = (-best[2] / (2 * math.pi)) * period_hours
        phase_hour = phase_hour % period_hours
        return (period_hours, amplitude, phase_hour)

    # ------------------------------------------------------------------
    # Chronotype Classification
    # ------------------------------------------------------------------

    def classify_chronotype(self, agent_id: str) -> ChronotypeProfile:
        """Classify an agent's temporal work pattern."""
        profile = self._bin_performance_profile(agent_id)
        period, amplitude, phase_hour = self._dominant_rhythm(profile)

        # Find peak and trough bins
        if all(v == 0 for v in profile):
            return ChronotypeProfile(
                agent_id=agent_id,
                chronotype="Irregular",
                confidence=0.0,
                peak_hours=[],
                trough_hours=[],
                amplitude=0.0,
                dominant_period=24.0,
                phase_offset=0.0,
            )

        mean_perf = statistics.mean(profile)
        std_perf = statistics.stdev(profile) if len(profile) > 1 else 0.0

        # Peak/trough detection
        threshold_high = mean_perf + 0.3 * std_perf if std_perf > 0 else mean_perf
        threshold_low = mean_perf - 0.3 * std_perf if std_perf > 0 else mean_perf

        peak_bins = [b for b, v in enumerate(profile) if v >= threshold_high]
        trough_bins = [b for b, v in enumerate(profile) if v <= threshold_low]

        peak_hours = [b * self.bin_hours for b in peak_bins]
        trough_hours = [b * self.bin_hours for b in trough_bins]

        # Classify based on peak distribution
        chronotype = self._determine_chronotype(profile, peak_hours, amplitude, std_perf)

        # Confidence based on sample coverage and amplitude
        state = self._agents.get(agent_id)
        coverage = sum(1 for b in range(self.num_bins) if state and len(state.bins.get(b, [])) >= MIN_SAMPLES_PER_BIN) / self.num_bins if state else 0
        confidence = min(1.0, coverage * 0.6 + (amplitude / (amplitude + 0.1)) * 0.4)

        # Update phase history for jet-lag detection
        if state:
            state.phase_history.append(phase_hour)
            if len(state.phase_history) > 20:
                state.phase_history = state.phase_history[-20:]

        return ChronotypeProfile(
            agent_id=agent_id,
            chronotype=chronotype,
            confidence=confidence,
            peak_hours=peak_hours,
            trough_hours=trough_hours,
            amplitude=amplitude,
            dominant_period=period,
            phase_offset=phase_hour,
        )

    def _determine_chronotype(
        self,
        profile: List[float],
        peak_hours: List[int],
        amplitude: float,
        std_perf: float,
    ) -> str:
        """Determine chronotype from peak distribution."""
        if not peak_hours:
            return "Irregular"

        # Low variation = SteadyState
        cv = std_perf / statistics.mean(profile) if statistics.mean(profile) > 0 else 0
        if cv < 0.1:
            return "SteadyState"

        # Check for bimodal distribution (but not wrap-around night owl)
        if len(peak_hours) >= 2:
            sorted_peaks = sorted(peak_hours)
            gaps = []
            for i in range(len(sorted_peaks) - 1):
                gaps.append(sorted_peaks[i + 1] - sorted_peaks[i])
            # Also check circular gap (from last to first wrapping around 24)
            circular_gap = (sorted_peaks[0] + 24) - sorted_peaks[-1]
            # If largest internal gap is big but circular gap is small,
            # it's wrap-around (Night Owl pattern), not bimodal
            if gaps and max(gaps) >= 6:
                if circular_gap >= 6:
                    # Both gaps are large = true bimodal
                    return "Bimodal"
                # else: wrap-around pattern, continue to early/late check

        # Check concentration (use circular mean for wrap-around)
        # Convert hours to angles, compute circular mean
        angles = [2 * math.pi * h / 24 for h in peak_hours]
        sin_mean = statistics.mean(math.sin(a) for a in angles)
        cos_mean = statistics.mean(math.cos(a) for a in angles)
        peak_center = (math.atan2(sin_mean, cos_mean) * 24 / (2 * math.pi)) % 24

        # Circular span (smallest arc containing all peaks)
        sorted_peaks = sorted(peak_hours)
        max_gap = 0
        for i in range(len(sorted_peaks) - 1):
            max_gap = max(max_gap, sorted_peaks[i + 1] - sorted_peaks[i])
        max_gap = max(max_gap, (sorted_peaks[0] + 24) - sorted_peaks[-1])
        peak_span = 24 - max_gap  # actual span is complement of largest gap

        # Burst = very concentrated peaks with high amplitude
        # peak_span already computed above
        if peak_span <= 3 and amplitude > 0.3:
            return "BurstWorker"

        # Early vs Late
        if peak_center < 12:
            return "EarlyBird"
        elif peak_center >= 17 or peak_center <= 4:
            return "NightOwl"

        # Scattered peaks
        if len(peak_hours) >= 4 and peak_span > 12:
            return "Irregular"

        return "SteadyState"

    # ------------------------------------------------------------------
    # Jet-Lag Detection
    # ------------------------------------------------------------------

    def detect_jetlag(self, agent_id: str) -> JetlagReport:
        """Detect rhythm disruption for an agent."""
        state = self._agents.get(agent_id)
        if not state or len(state.phase_history) < 2:
            return JetlagReport(
                agent_id=agent_id,
                disrupted=False,
                phase_shift_hours=0.0,
                severity="none",
                recovery_eta_hours=0.0,
                performance_degradation=0.0,
                recent_phase=0.0,
                baseline_phase=0.0,
            )

        # Compare recent phase to baseline
        history = state.phase_history
        if len(history) >= JETLAG_HISTORY_PERIODS + 1:
            baseline = statistics.mean(history[:-JETLAG_HISTORY_PERIODS])
            recent = statistics.mean(history[-JETLAG_HISTORY_PERIODS:])
        else:
            baseline = history[0]
            recent = history[-1]

        # Circular difference
        shift = recent - baseline
        if shift > 12:
            shift -= 24
        elif shift < -12:
            shift += 24

        abs_shift = abs(shift)
        disrupted = abs_shift >= JETLAG_THRESHOLD

        # Severity classification
        if abs_shift < JETLAG_THRESHOLD:
            severity = "none"
        elif abs_shift < 4:
            severity = "mild"
        elif abs_shift < 8:
            severity = "moderate"
        else:
            severity = "severe"

        # Recovery estimation (roughly 1 hour recovery per 1 hour shift)
        recovery_eta = abs_shift * 1.5 if disrupted else 0.0

        # Performance degradation estimate
        degradation = min(1.0, abs_shift / 12.0) if disrupted else 0.0

        return JetlagReport(
            agent_id=agent_id,
            disrupted=disrupted,
            phase_shift_hours=round(shift, 2),
            severity=severity,
            recovery_eta_hours=round(recovery_eta, 1),
            performance_degradation=round(degradation, 3),
            recent_phase=round(recent, 2),
            baseline_phase=round(baseline, 2),
        )

    # ------------------------------------------------------------------
    # Optimal Windows
    # ------------------------------------------------------------------

    def optimal_windows(self, agent_id: str) -> OptimalWindow:
        """Find the best performance window for an agent."""
        profile = self._bin_performance_profile(agent_id)

        if all(v == 0 for v in profile):
            return OptimalWindow(
                agent_id=agent_id,
                peak_start=9,
                peak_end=17,
                score=0.0,
                recommended_tasks=[],
            )

        # Sliding window of 3-4 bins to find best contiguous block
        window_size = min(4, self.num_bins)
        best_start = 0
        best_score = -1.0

        for start in range(self.num_bins):
            window_scores = []
            for offset in range(window_size):
                idx = (start + offset) % self.num_bins
                window_scores.append(profile[idx])
            avg = statistics.mean(window_scores)
            if avg > best_score:
                best_score = avg
                best_start = start

        peak_start_hour = best_start * self.bin_hours
        peak_end_hour = ((best_start + window_size) * self.bin_hours) % 24

        # Task recommendations based on peak characteristics
        recommendations = self._recommend_tasks(profile, best_start, window_size)

        return OptimalWindow(
            agent_id=agent_id,
            peak_start=peak_start_hour,
            peak_end=peak_end_hour,
            score=round(best_score, 3),
            recommended_tasks=recommendations,
        )

    def _recommend_tasks(
        self, profile: List[float], peak_start: int, window_size: int
    ) -> List[str]:
        """Generate task recommendations for optimal window."""
        tasks = []
        peak_score = statistics.mean(
            profile[(peak_start + i) % self.num_bins] for i in range(window_size)
        )

        if peak_score > 0.7:
            tasks.append("complex_reasoning")
            tasks.append("creative_synthesis")
        if peak_score > 0.5:
            tasks.append("collaborative_tasks")
            tasks.append("decision_making")
        if peak_score > 0.3:
            tasks.append("routine_processing")
        tasks.append("monitoring")
        return tasks

    # ------------------------------------------------------------------
    # Entrainment Analysis
    # ------------------------------------------------------------------

    def compute_entrainment(self) -> List[EntrainmentPair]:
        """Measure phase coupling between all agent pairs."""
        agents = list(self._agents.keys())
        pairs = []

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                profile_a = self._bin_performance_profile(a)
                profile_b = self._bin_performance_profile(b)

                # Cross-correlation to find phase difference
                phase_diff, strength = self._cross_correlate(profile_a, profile_b)

                pairs.append(EntrainmentPair(
                    agent_a=a,
                    agent_b=b,
                    phase_difference=round(phase_diff * self.bin_hours, 2),
                    coupling_strength=round(strength, 3),
                    synchronized=strength > 0.6 and abs(phase_diff * self.bin_hours) < 2,
                ))
        return pairs

    def _cross_correlate(
        self, a: List[float], b: List[float]
    ) -> Tuple[float, float]:
        """Find phase lag and correlation strength between two profiles."""
        n = len(a)
        if n == 0 or all(v == 0 for v in a) or all(v == 0 for v in b):
            return (0.0, 0.0)

        mean_a = statistics.mean(a)
        mean_b = statistics.mean(b)
        std_a = statistics.stdev(a) if len(a) > 1 else 1.0
        std_b = statistics.stdev(b) if len(b) > 1 else 1.0

        if std_a == 0 or std_b == 0:
            return (0.0, 0.0)

        best_lag = 0
        best_corr = -2.0

        for lag in range(n):
            corr = sum(
                (a[t] - mean_a) * (b[(t + lag) % n] - mean_b)
                for t in range(n)
            ) / (n * std_a * std_b)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        return (float(best_lag), max(0.0, best_corr))

    # ------------------------------------------------------------------
    # Collective Report
    # ------------------------------------------------------------------

    def collective_report(self) -> CollectiveReport:
        """Generate swarm-wide circadian health report."""
        if not self._agents:
            return CollectiveReport(
                health_score=0.0,
                entrainment_index=0.0,
                chronotype_distribution={},
                collective_peak_hours=[],
                collective_trough_hours=[],
                disrupted_agents=[],
                recommendations=[],
                agent_count=0,
                total_samples=0,
            )

        # Classify all agents
        chronotypes: Dict[str, int] = defaultdict(int)
        disrupted = []
        all_profiles = []

        for agent_id in self._agents:
            ct = self.classify_chronotype(agent_id)
            chronotypes[ct.chronotype] += 1

            jl = self.detect_jetlag(agent_id)
            if jl.disrupted:
                disrupted.append(agent_id)

            all_profiles.append(self._bin_performance_profile(agent_id))

        # Collective performance profile (average across agents)
        collective_profile = [0.0] * self.num_bins
        for b in range(self.num_bins):
            values = [p[b] for p in all_profiles if p[b] > 0]
            if values:
                collective_profile[b] = statistics.mean(values)

        # Find collective peaks/troughs
        if any(v > 0 for v in collective_profile):
            mean_coll = statistics.mean(v for v in collective_profile if v > 0)
            peak_hours = [b * self.bin_hours for b, v in enumerate(collective_profile) if v >= mean_coll * 1.1]
            trough_hours = [b * self.bin_hours for b, v in enumerate(collective_profile) if 0 < v <= mean_coll * 0.9]
        else:
            peak_hours = []
            trough_hours = []

        # Entrainment index
        pairs = self.compute_entrainment()
        if pairs:
            entrainment_index = statistics.mean(p.coupling_strength for p in pairs)
        else:
            entrainment_index = 1.0 if len(self._agents) == 1 else 0.0

        # Health score components
        disruption_penalty = len(disrupted) / len(self._agents) * 30
        rhythm_strength = statistics.mean(
            self._dominant_rhythm(self._bin_performance_profile(a))[1]
            for a in self._agents
        )
        rhythm_score = min(30, rhythm_strength * 100)
        entrainment_score = entrainment_index * 40

        health_score = max(0, min(100, rhythm_score + entrainment_score + 30 - disruption_penalty))

        # Recommendations
        recommendations = self._generate_recommendations(
            chronotypes, disrupted, entrainment_index, collective_profile
        )

        return CollectiveReport(
            health_score=round(health_score, 1),
            entrainment_index=round(entrainment_index, 3),
            chronotype_distribution=dict(chronotypes),
            collective_peak_hours=peak_hours,
            collective_trough_hours=trough_hours,
            disrupted_agents=disrupted,
            recommendations=recommendations,
            agent_count=len(self._agents),
            total_samples=self._sample_counter,
        )

    def _generate_recommendations(
        self,
        chronotypes: Dict[str, int],
        disrupted: List[str],
        entrainment_index: float,
        collective_profile: List[float],
    ) -> List[str]:
        """Generate actionable scheduling recommendations."""
        recs = []

        if disrupted:
            recs.append(
                f"Rhythm disruption detected in {len(disrupted)} agent(s). "
                "Consider gradual schedule adjustment to restore phase alignment."
            )

        if entrainment_index < 0.4:
            recs.append(
                "Low swarm entrainment — agents are desynchronized. "
                "Schedule shared activities during overlapping peak windows."
            )

        if chronotypes.get("Irregular", 0) > len(disrupted):
            recs.append(
                "Multiple irregular-rhythm agents detected. "
                "Introduce consistent workload anchors to stabilize patterns."
            )

        night_owls = chronotypes.get("NightOwl", 0)
        early_birds = chronotypes.get("EarlyBird", 0)
        if night_owls > 0 and early_birds > 0:
            recs.append(
                "Mixed chronotypes present. Schedule collaborative tasks "
                "during midday overlap window (11:00-15:00)."
            )

        if not recs:
            recs.append("Circadian rhythms are healthy. Maintain current scheduling patterns.")

        return recs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save engine state to JSON."""
        data = {
            "bin_hours": self.bin_hours,
            "max_samples_per_bin": self.max_samples_per_bin,
            "sample_counter": self._sample_counter,
            "agents": {},
        }
        for agent_id, state in self._agents.items():
            data["agents"][agent_id] = {
                "bins": {str(k): v for k, v in state.bins.items()},
                "phase_history": state.phase_history,
                "sample_count": state.sample_count,
            }
        Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: str) -> "CircadianEngine":
        """Load engine state from JSON."""
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        engine = cls(
            bin_hours=raw.get("bin_hours", 1),
            max_samples_per_bin=raw.get("max_samples_per_bin", 100),
        )
        engine._sample_counter = raw.get("sample_counter", 0)
        for agent_id, state_data in raw.get("agents", {}).items():
            state = AgentRhythmState(
                bins=defaultdict(list),
                phase_history=state_data.get("phase_history", []),
                sample_count=state_data.get("sample_count", 0),
            )
            for k, v in state_data.get("bins", {}).items():
                state.bins[int(k)] = v
            engine._agents[agent_id] = state
        return engine

    # ------------------------------------------------------------------
    # HTML Dashboard
    # ------------------------------------------------------------------

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        report = self.collective_report()
        agents_data = []
        for agent_id in self._agents:
            ct = self.classify_chronotype(agent_id)
            jl = self.detect_jetlag(agent_id)
            win = self.optimal_windows(agent_id)
            profile = self._bin_performance_profile(agent_id)
            agents_data.append({
                "id": agent_id,
                "chronotype": ct.chronotype,
                "confidence": ct.confidence,
                "peak_hours": ct.peak_hours,
                "amplitude": ct.amplitude,
                "jetlag": jl.disrupted,
                "phase_shift": jl.phase_shift_hours,
                "optimal_start": win.peak_start,
                "optimal_end": win.peak_end,
                "optimal_score": win.score,
                "profile": profile,
            })

        html = self._render_dashboard(report, agents_data)
        Path(path).write_text(html, encoding='utf-8')

    def _render_dashboard(
        self, report: CollectiveReport, agents_data: List[Dict]
    ) -> str:
        """Render HTML dashboard string."""
        agents_json = json.dumps(agents_data, indent=2)
        report_json = json.dumps(asdict(report), indent=2)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Circadian Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 20px; }}
.header {{ text-align: center; margin-bottom: 30px; }}
.header h1 {{ color: #58a6ff; font-size: 2em; }}
.header .subtitle {{ color: #8b949e; margin-top: 5px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }}
.card h2 {{ color: #58a6ff; margin-bottom: 15px; font-size: 1.1em; }}
.score {{ font-size: 3em; font-weight: bold; text-align: center; }}
.score.high {{ color: #3fb950; }}
.score.mid {{ color: #d29922; }}
.score.low {{ color: #f85149; }}
.metric {{ display: flex; justify-content: space-between; padding: 8px 0;
           border-bottom: 1px solid #21262d; }}
.metric:last-child {{ border: none; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
          font-size: 0.85em; margin: 3px; }}
.badge-eb {{ background: #1f3a1f; color: #3fb950; }}
.badge-no {{ background: #3a1f3a; color: #bc8cff; }}
.badge-bi {{ background: #1f2a3a; color: #58a6ff; }}
.badge-ss {{ background: #2a2a1f; color: #d29922; }}
.badge-bw {{ background: #3a2a1f; color: #f0883e; }}
.badge-ir {{ background: #3a1f1f; color: #f85149; }}
.bar {{ height: 20px; border-radius: 4px; margin: 4px 0; }}
.profile-bar {{ display: flex; align-items: center; margin: 4px 0; }}
.profile-bar .label {{ width: 40px; font-size: 0.8em; color: #8b949e; }}
.profile-bar .fill {{ height: 16px; border-radius: 3px; background: #58a6ff; transition: width 0.3s; }}
.rec {{ padding: 8px 12px; background: #1c2128; border-left: 3px solid #58a6ff;
        margin: 8px 0; border-radius: 4px; font-size: 0.9em; }}
.disrupted {{ color: #f85149; }}
.ok {{ color: #3fb950; }}
</style>
</head>
<body>
<div class="header">
  <h1>&#x1F319; Swarm Circadian Engine</h1>
  <div class="subtitle">Temporal Rhythm Detection &amp; Optimization | {report.agent_count} agents | {report.total_samples} samples</div>
</div>
<div class="grid">
  <div class="card">
    <h2>Health Score</h2>
    <div class="score {'high' if report.health_score >= 70 else 'mid' if report.health_score >= 40 else 'low'}">{report.health_score}</div>
    <div class="metric"><span>Entrainment Index</span><span>{report.entrainment_index:.3f}</span></div>
    <div class="metric"><span>Disrupted Agents</span><span class="{'disrupted' if report.disrupted_agents else 'ok'}">{len(report.disrupted_agents)}</span></div>
  </div>
  <div class="card">
    <h2>Chronotype Distribution</h2>
    {''.join(f'<div class="metric"><span class="badge badge-{ct[:2].lower()}">{ct}</span><span>{count}</span></div>' for ct, count in report.chronotype_distribution.items())}
  </div>
  <div class="card">
    <h2>Collective Peak Hours</h2>
    <div style="color:#3fb950;margin-bottom:10px;">Peak: {', '.join(f'{h}:00' for h in report.collective_peak_hours[:6]) or 'N/A'}</div>
    <div style="color:#f85149;">Trough: {', '.join(f'{h}:00' for h in report.collective_trough_hours[:6]) or 'N/A'}</div>
  </div>
  <div class="card">
    <h2>Recommendations</h2>
    {''.join(f'<div class="rec">{r}</div>' for r in report.recommendations)}
  </div>
</div>
<h2 style="color:#58a6ff;margin:30px 0 15px;">Agent Rhythms</h2>
<div class="grid">
{''.join(self._render_agent_card(a) for a in agents_data)}
</div>
</body>
</html>"""

    def _render_agent_card(self, agent: Dict) -> str:
        """Render individual agent card."""
        profile = agent["profile"]
        max_val = max(profile) if profile and max(profile) > 0 else 1
        bars = ""
        for b, val in enumerate(profile):
            pct = (val / max_val * 100) if max_val > 0 else 0
            hour = b * self.bin_hours
            bars += f'<div class="profile-bar"><span class="label">{hour:02d}h</span><div class="fill" style="width:{pct}%"></div></div>'

        jl_status = f'<span class="disrupted">⚠ Jet-lag ({agent["phase_shift"]:+.1f}h)</span>' if agent["jetlag"] else '<span class="ok">✓ Stable</span>'

        return f"""<div class="card">
    <h2>{agent['id']}</h2>
    <div class="metric"><span>Chronotype</span><span class="badge badge-{agent['chronotype'][:2].lower()}">{agent['chronotype']}</span></div>
    <div class="metric"><span>Rhythm Status</span>{jl_status}</div>
    <div class="metric"><span>Optimal Window</span><span>{agent['optimal_start']:02d}:00-{agent['optimal_end']:02d}:00</span></div>
    <div class="metric"><span>Peak Score</span><span>{agent['optimal_score']:.3f}</span></div>
    <div style="margin-top:10px;">{bars}</div>
</div>"""

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_health(self) -> CollectiveReport:
        """Alias for collective_report."""
        return self.collective_report()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_demo(args: argparse.Namespace) -> None:
    """Run demonstration simulation."""
    num_agents = args.agents
    cycles = args.cycles

    engine = CircadianEngine(bin_hours=1)

    # Create agents with different chronotypes
    agent_configs = []
    for i in range(num_agents):
        # Assign a rhythm pattern
        pattern = random.choice(["early", "late", "bimodal", "steady", "burst"])
        agent_configs.append((f"agent_{i+1:02d}", pattern))

    print(f"Simulating {num_agents} agents over {cycles} hourly cycles...")
    print()

    for cycle in range(cycles):
        hour = cycle % 24
        for agent_id, pattern in agent_configs:
            metrics = _generate_metrics(hour, pattern)
            engine.record_sample(agent_id, hour, metrics, timestamp=cycle * 3600)

    # If perturbation requested, shift some agents
    if args.perturbation:
        print("Injecting rhythm perturbation (phase shift)...")
        # Shift first agent's recent data
        shifted_agent = agent_configs[0][0]
        for cycle in range(24):
            hour = (cycle + 8) % 24  # 8-hour shift
            metrics = _generate_metrics(hour, agent_configs[0][1])
            engine.record_sample(shifted_agent, hour, metrics)
        # Update phase history to reflect shift
        state = engine._agents[shifted_agent]
        state.phase_history.append(state.phase_history[-1] + 8 if state.phase_history else 8.0)

    # Print results
    report = engine.collective_report()
    print(f"{'='*60}")
    print(f"  Swarm Circadian Health Report")
    print(f"{'='*60}")
    print(f"  Health Score:      {report.health_score}/100")
    print(f"  Entrainment Index: {report.entrainment_index:.3f}")
    print(f"  Agent Count:       {report.agent_count}")
    print(f"  Total Samples:     {report.total_samples}")
    print(f"  Disrupted Agents:  {len(report.disrupted_agents)}")
    print()
    print("  Chronotype Distribution:")
    for ct, count in report.chronotype_distribution.items():
        print(f"    {ct:15s} {count}")
    print()
    print("  Collective Peaks: ", [f"{h}:00" for h in report.collective_peak_hours[:5]])
    print()
    print("  Recommendations:")
    for rec in report.recommendations:
        print(f"    • {rec}")
    print()

    # Per-agent details
    print(f"  {'Agent':<12} {'Chronotype':<13} {'Peak Window':<14} {'Jet-Lag':<10}")
    print(f"  {'-'*12} {'-'*13} {'-'*14} {'-'*10}")
    for agent_id, _ in agent_configs:
        ct = engine.classify_chronotype(agent_id)
        jl = engine.detect_jetlag(agent_id)
        win = engine.optimal_windows(agent_id)
        jl_str = f"{jl.phase_shift_hours:+.1f}h" if jl.disrupted else "OK"
        print(f"  {agent_id:<12} {ct.chronotype:<13} {win.peak_start:02d}:00-{win.peak_end:02d}:00   {jl_str:<10}")

    # Export
    if args.export == "html":
        out = args.output or "circadian_dashboard.html"
        engine.export_html(out)
        print(f"\n  Dashboard exported: {out}")

    if args.json:
        engine.save(args.json)
        print(f"  State saved: {args.json}")

    if args.health:
        print(f"\n  Quick Health: {report.health_score}/100 "
              f"({'healthy' if report.health_score >= 70 else 'degraded' if report.health_score >= 40 else 'critical'})")


def _generate_metrics(hour: int, pattern: str) -> Dict[str, float]:
    """Generate synthetic metrics for a given hour and rhythm pattern."""
    noise = lambda: random.gauss(0, 0.05)

    if pattern == "early":
        # Peak 6-11, trough 20-4
        phase = math.sin(math.pi * (hour - 8) / 12)
    elif pattern == "late":
        # Peak 20-2, trough 8-14
        phase = math.sin(math.pi * (hour - 22) / 12)
    elif pattern == "bimodal":
        # Two peaks: 9-11 and 15-17
        phase = 0.5 * (math.sin(math.pi * (hour - 10) / 6) +
                       math.sin(math.pi * (hour - 16) / 6))
    elif pattern == "steady":
        phase = 0.1 * math.sin(math.pi * hour / 12)
    elif pattern == "burst":
        # Sharp peak around hour 14
        phase = math.exp(-((hour - 14) ** 2) / 4) - 0.3
    else:
        phase = random.uniform(-0.5, 0.5)

    # Scale to 0-1 range
    base = 0.5 + 0.3 * phase

    return {
        "throughput": max(0, (base * 15) + random.gauss(0, 1)),
        "latency": max(0.1, (1.5 - base) + noise()),
        "error_rate": max(0, min(1, (0.15 - base * 0.1) + noise())),
        "collaboration": max(0, (base * 8) + random.gauss(0, 0.5)),
        "creativity": max(0, min(1, base * 0.8 + noise())),
        "focus_duration": max(5, (base * 50) + random.gauss(0, 3)),
        "recovery_time": max(0.5, (5 - base * 3) + noise()),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Circadian Engine — temporal rhythm detection"
    )
    parser.add_argument("--agents", type=int, default=6, help="Number of agents")
    parser.add_argument("--cycles", type=int, default=72, help="Hourly simulation cycles")
    parser.add_argument("--perturbation", action="store_true", help="Inject rhythm disruption")
    parser.add_argument("--export", choices=["html"], help="Export format")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--json", help="Save/load state JSON path")
    parser.add_argument("--health", action="store_true", help="Show health summary")
    args = parser.parse_args()
    _run_demo(args)


if __name__ == "__main__":
    main()
