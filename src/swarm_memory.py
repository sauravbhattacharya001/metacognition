"""Swarm Memory — persistent episodic learning for mBFT consensus.

Records every consensus outcome as an *episode*, clusters them into
patterns, and autonomously recommends optimal swarm configurations for
future tasks.  This gives the protocol long-term memory: the swarm
doesn't just run consensus — it **learns from its own history**.

Capabilities:

- **Episode Recording** — captures full context per round: task
  fingerprint, agent roster, threshold, votes, outcome, timing.
- **Pattern Extraction** — identifies recurring success/failure modes
  via configurable similarity bucketing (task tokens, agent mix,
  threshold band).
- **Success Predictor** — estimates commit probability for a given
  configuration using historical episode statistics.
- **Configuration Recommender** — given a new task, searches memory for
  similar past tasks and recommends the agent count, threshold, and
  byzantine-tolerance posture that historically worked best.
- **Forgetting Curve** — exponential decay weights so recent episodes
  matter more than ancient ones, with configurable half-life.
- **Memory Health** — self-monitors for staleness, bias, and coverage
  gaps; flags when the memory should be refreshed.
- **Persistence** — JSON-serializable store; save/load to disk for
  cross-session continuity.
- **Interactive HTML Dashboard** — visualizes memory contents, pattern
  clusters, success rates, and recommendation confidence.

Usage (Python API)::

    from src.swarm_memory import SwarmMemory, Episode

    mem = SwarmMemory()

    # Record outcomes after each consensus round
    mem.record(Episode(
        task="What is 2+2?",
        agent_count=5,
        byzantine_count=1,
        threshold=1.5,
        committed=True,
        solution="4",
        rounds_used=1,
        aggregate_weight=2.3,
        agent_reputations={"a1": 1.0, "a2": 0.8, "a3": 1.0, "a4": 0.5, "a5": 0.3},
        elapsed_s=0.12,
    ))

    # Get recommendation for a new task
    rec = mem.recommend("Solve x^2 - 4 = 0")
    print(rec.suggested_agents, rec.suggested_threshold, rec.confidence)

    # Predict success probability
    prob = mem.predict_success(agent_count=5, threshold=1.5)

    # Save/load for persistence
    mem.save("swarm_memory.json")
    mem = SwarmMemory.load("swarm_memory.json")

    # Generate dashboard
    mem.export_html("memory_dashboard.html")

CLI::

    python -m src.swarm_memory                        # demo with simulated history
    python -m src.swarm_memory --load mem.json        # analyze existing memory
    python -m src.swarm_memory --recommend "new task"  # get recommendation
    python -m src.swarm_memory --export html -o dash.html
    python -m src.swarm_memory --health               # memory health check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single consensus episode recorded in memory."""
    task: str
    agent_count: int
    byzantine_count: int
    threshold: float
    committed: bool
    solution: Optional[str]
    rounds_used: int
    aggregate_weight: float
    agent_reputations: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def task_tokens(self) -> List[str]:
        """Tokenize task into lowercase words for similarity matching."""
        return [w.lower().strip("?.,!;:'\"()[]{}") for w in self.task.split() if len(w) > 1]

    def byzantine_fraction(self) -> float:
        return self.byzantine_count / max(self.agent_count, 1)

    def efficiency(self) -> float:
        """Higher is better: committed in fewer rounds with more weight."""
        if not self.committed:
            return 0.0
        round_factor = 1.0 / self.rounds_used
        weight_factor = min(self.aggregate_weight / max(self.threshold, 0.01), 2.0)
        return round_factor * weight_factor


@dataclass
class Pattern:
    """A cluster of similar episodes forming a recognizable pattern."""
    pattern_id: str
    description: str
    episode_count: int
    success_rate: float
    avg_rounds: float
    avg_threshold: float
    avg_agents: float
    avg_byzantine_fraction: float
    avg_efficiency: float
    common_tags: List[str]
    representative_tasks: List[str]


@dataclass
class Recommendation:
    """A configuration recommendation based on memory analysis."""
    suggested_agents: int
    suggested_threshold: float
    suggested_max_byzantine: int
    confidence: float
    reasoning: str
    similar_episodes: int
    historical_success_rate: float
    predicted_rounds: float


@dataclass
class MemoryHealth:
    """Self-diagnostic report on memory quality."""
    total_episodes: int
    age_days: float
    freshness_score: float      # 0-100, how recent the episodes are
    coverage_score: float       # 0-100, diversity of configurations seen
    bias_score: float           # 0-100 (lower = more biased)
    reliability_score: float    # 0-100, overall memory quality
    warnings: List[str]
    suggestions: List[str]


# ---------------------------------------------------------------------------
# Similarity & Bucketing
# ---------------------------------------------------------------------------

def _jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two token lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _threshold_band(threshold: float) -> str:
    """Bucket thresholds into bands for pattern grouping."""
    if threshold < 1.0:
        return "low"
    elif threshold < 2.0:
        return "medium"
    elif threshold < 3.0:
        return "high"
    else:
        return "very_high"


def _agent_band(count: int) -> str:
    """Bucket agent counts."""
    if count <= 3:
        return "small"
    elif count <= 7:
        return "medium"
    elif count <= 15:
        return "large"
    else:
        return "xlarge"


# ---------------------------------------------------------------------------
# Swarm Memory Core
# ---------------------------------------------------------------------------

class SwarmMemory:
    """Persistent episodic memory for mBFT consensus swarms.

    Records episodes, extracts patterns, predicts outcomes, and recommends
    configurations — all based on historical consensus data.
    """

    def __init__(
        self,
        half_life_days: float = 30.0,
        min_similarity: float = 0.15,
    ) -> None:
        self.episodes: List[Episode] = []
        self.half_life_days = half_life_days
        self.min_similarity = min_similarity
        self._created_at = time.time()

    # -- Recording ----------------------------------------------------------

    def record(self, episode: Episode) -> None:
        """Record a new consensus episode."""
        self.episodes.append(episode)

    def record_from_result(
        self,
        task: str,
        result: Any,  # RoundResult
        agent_count: int,
        byzantine_count: int,
        threshold: float,
        reputations: Optional[Dict[str, float]] = None,
        elapsed_s: float = 0.0,
    ) -> Episode:
        """Convenience: record directly from an MBFTEngine RoundResult."""
        ep = Episode(
            task=task,
            agent_count=agent_count,
            byzantine_count=byzantine_count,
            threshold=threshold,
            committed=result.committed if result else False,
            solution=result.committed_solution if result else None,
            rounds_used=result.round_index + 1 if result else 0,
            aggregate_weight=result.aggregate_weight if result else 0.0,
            agent_reputations=reputations or {},
            elapsed_s=elapsed_s,
        )
        self.record(ep)
        return ep

    # -- Decay Weighting ----------------------------------------------------

    def _weight(self, episode: Episode, now: Optional[float] = None) -> float:
        """Exponential decay weight based on episode age.

        Parameters
        ----------
        now : float, optional
            Pre-captured current timestamp to avoid repeated time.time()
            calls in tight loops.
        """
        age_s = (now if now is not None else time.time()) - episode.timestamp
        age_days = age_s / 86400.0
        return math.exp(-math.log(2) * age_days / self.half_life_days)

    # -- Similarity ---------------------------------------------------------

    def _task_similarity(self, task_a: str, task_b: str) -> float:
        """Compute task-level similarity using token Jaccard."""
        tokens_a = [w.lower().strip("?.,!;:'\"()[]{}") for w in task_a.split() if len(w) > 1]
        tokens_b = [w.lower().strip("?.,!;:'\"()[]{}") for w in task_b.split() if len(w) > 1]
        return _jaccard(tokens_a, tokens_b)

    def _task_similarity_precomputed(
        self, query_tokens_set: set, ep_task: str,
    ) -> float:
        """Task similarity with pre-computed query token set.

        Avoids re-tokenizing the query string on every episode comparison
        in find_similar (O(E) tokenizations -> O(1) for the query side).
        """
        ep_tokens = set(
            w.lower().strip("?.,!;:'\"()[]{}")
            for w in ep_task.split()
            if len(w) > 1
        )
        union = query_tokens_set | ep_tokens
        if not union:
            return 1.0
        return len(query_tokens_set & ep_tokens) / len(union)

    def _config_similarity(self, ep: Episode, agent_count: int, threshold: float) -> float:
        """Configuration similarity (agent count + threshold band)."""
        agent_sim = 1.0 - abs(ep.agent_count - agent_count) / max(ep.agent_count, agent_count, 1)
        thresh_sim = 1.0 if _threshold_band(ep.threshold) == _threshold_band(threshold) else 0.5
        return 0.6 * agent_sim + 0.4 * thresh_sim

    def find_similar(self, task: str, top_k: int = 10) -> List[Tuple[Episode, float]]:
        """Find the top-k most similar episodes to a task.

        Optimised: pre-tokenizes the query once and captures time.time()
        once, avoiding O(E) redundant tokenizations and syscalls.
        """
        query_tokens = set(
            w.lower().strip("?.,!;:'\"()[]{}")
            for w in task.split()
            if len(w) > 1
        )
        now = time.time()
        scored: List[Tuple[Episode, float]] = []
        for ep in self.episodes:
            sim = self._task_similarity_precomputed(query_tokens, ep.task)
            if sim >= self.min_similarity:
                w = self._weight(ep, now=now)
                scored.append((ep, sim * w))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # -- Pattern Extraction -------------------------------------------------

    def extract_patterns(self) -> List[Pattern]:
        """Cluster episodes into recognizable patterns."""
        # Group by (agent_band, threshold_band, outcome)
        buckets: Dict[str, List[Episode]] = defaultdict(list)
        for ep in self.episodes:
            key = f"{_agent_band(ep.agent_count)}_{_threshold_band(ep.threshold)}_{'ok' if ep.committed else 'fail'}"
            buckets[key].append(ep)

        patterns: List[Pattern] = []
        for key, eps in buckets.items():
            if len(eps) < 2:
                continue
            success_eps = [e for e in eps if e.committed]
            all_tags: List[str] = []
            for e in eps:
                all_tags.extend(e.tags)
            tag_counts = Counter(all_tags)
            common = [t for t, _ in tag_counts.most_common(5)] if tag_counts else []

            rounds_list = [e.rounds_used for e in eps]
            thresh_list = [e.threshold for e in eps]
            agent_list = [e.agent_count for e in eps]
            byz_list = [e.byzantine_fraction() for e in eps]
            eff_list = [e.efficiency() for e in eps]

            parts = key.split("_")
            outcome_label = "success" if parts[-1] == "ok" else "failure"
            desc = f"{parts[0]} swarm, {parts[1]} threshold, {outcome_label} pattern"

            patterns.append(Pattern(
                pattern_id=key,
                description=desc,
                episode_count=len(eps),
                success_rate=len(success_eps) / len(eps),
                avg_rounds=statistics.mean(rounds_list),
                avg_threshold=statistics.mean(thresh_list),
                avg_agents=statistics.mean(agent_list),
                avg_byzantine_fraction=statistics.mean(byz_list),
                avg_efficiency=statistics.mean(eff_list),
                common_tags=common,
                representative_tasks=[e.task for e in eps[:3]],
            ))

        patterns.sort(key=lambda p: p.episode_count, reverse=True)
        return patterns

    # -- Success Prediction -------------------------------------------------

    def predict_success(
        self,
        agent_count: int,
        threshold: float,
        byzantine_fraction: float = 0.0,
    ) -> float:
        """Predict commit probability for a given configuration.

        Returns a weighted success rate based on similar configurations
        in memory. Returns 0.5 (uncertain) if no relevant history.

        Optimised: captures time.time() once and pre-computes the
        threshold band for the query to avoid repeated string comparisons.
        """
        if not self.episodes:
            return 0.5

        now = time.time()
        query_band = _threshold_band(threshold)
        weighted_success = 0.0
        total_weight = 0.0
        for ep in self.episodes:
            # Inline config similarity with pre-computed query band
            agent_sim = 1.0 - abs(ep.agent_count - agent_count) / max(ep.agent_count, agent_count, 1)
            thresh_sim = 1.0 if _threshold_band(ep.threshold) == query_band else 0.5
            sim = 0.6 * agent_sim + 0.4 * thresh_sim

            byz_sim = 1.0 - abs(ep.byzantine_fraction() - byzantine_fraction)
            combined = sim * byz_sim * self._weight(ep, now=now)
            if combined > 0.05:
                weighted_success += combined * (1.0 if ep.committed else 0.0)
                total_weight += combined

        if total_weight < 0.1:
            return 0.5
        return weighted_success / total_weight

    # -- Recommendation Engine ----------------------------------------------

    def recommend(self, task: str) -> Recommendation:
        """Recommend an optimal swarm configuration for a task.

        Searches memory for similar tasks and configurations that
        historically produced the best outcomes.
        """
        if not self.episodes:
            return Recommendation(
                suggested_agents=5,
                suggested_threshold=1.5,
                suggested_max_byzantine=1,
                confidence=0.1,
                reasoning="No memory available — using safe defaults.",
                similar_episodes=0,
                historical_success_rate=0.0,
                predicted_rounds=2.0,
            )

        similar = self.find_similar(task, top_k=20)

        if not similar:
            # No similar tasks; use global best configuration
            return self._recommend_from_global()

        # Weighted average of successful similar episode configurations
        w_agents = 0.0
        w_threshold = 0.0
        w_rounds = 0.0
        w_byz = 0.0
        total_w = 0.0
        successes = 0

        for ep, score in similar:
            # score from find_similar already incorporates decay weight,
            # so we only apply the outcome multiplier here.
            w = score * (1.5 if ep.committed else 0.5)
            w_agents += w * ep.agent_count
            w_threshold += w * ep.threshold
            w_rounds += w * ep.rounds_used
            w_byz += w * ep.byzantine_count
            total_w += w
            if ep.committed:
                successes += 1

        if total_w < 0.01:
            return self._recommend_from_global()

        suggested_agents = max(3, round(w_agents / total_w))
        suggested_threshold = round(w_threshold / total_w, 2)
        suggested_max_byz = max(0, round(w_byz / total_w))
        predicted_rounds = round(w_rounds / total_w, 1)
        success_rate = successes / len(similar)
        confidence = min(1.0, len(similar) / 10.0 * success_rate)

        # Build reasoning
        reasons: List[str] = []
        reasons.append(f"Based on {len(similar)} similar episodes (success rate: {success_rate:.0%}).")
        if success_rate >= 0.8:
            reasons.append("High historical success — confident recommendation.")
        elif success_rate >= 0.5:
            reasons.append("Mixed results — consider increasing agent count for safety.")
        else:
            reasons.append("Low historical success — recommending conservative configuration.")
            suggested_agents = max(suggested_agents + 2, 5)
            suggested_threshold = max(suggested_threshold, 1.5)
            confidence *= 0.7

        return Recommendation(
            suggested_agents=suggested_agents,
            suggested_threshold=suggested_threshold,
            suggested_max_byzantine=suggested_max_byz,
            confidence=round(confidence, 3),
            reasoning=" ".join(reasons),
            similar_episodes=len(similar),
            historical_success_rate=round(success_rate, 3),
            predicted_rounds=predicted_rounds,
        )

    def _recommend_from_global(self) -> Recommendation:
        """Fallback: recommend based on global statistics."""
        if not self.episodes:
            return Recommendation(
                suggested_agents=5, suggested_threshold=1.5,
                suggested_max_byzantine=1, confidence=0.1,
                reasoning="Empty memory.", similar_episodes=0,
                historical_success_rate=0.0, predicted_rounds=2.0,
            )
        successes = [e for e in self.episodes if e.committed]
        if not successes:
            avg_agents = statistics.mean([e.agent_count for e in self.episodes])
            return Recommendation(
                suggested_agents=max(5, round(avg_agents) + 2),
                suggested_threshold=1.5,
                suggested_max_byzantine=1,
                confidence=0.2,
                reasoning="No successful episodes in memory — using conservative defaults.",
                similar_episodes=0,
                historical_success_rate=0.0,
                predicted_rounds=3.0,
            )
        avg_agents = statistics.mean([e.agent_count for e in successes])
        avg_thresh = statistics.mean([e.threshold for e in successes])
        avg_rounds = statistics.mean([e.rounds_used for e in successes])
        avg_byz = statistics.mean([e.byzantine_count for e in successes])
        rate = len(successes) / len(self.episodes)
        return Recommendation(
            suggested_agents=max(3, round(avg_agents)),
            suggested_threshold=round(avg_thresh, 2),
            suggested_max_byzantine=max(0, round(avg_byz)),
            confidence=round(min(1.0, len(successes) / 20.0), 3),
            reasoning=f"Global stats from {len(self.episodes)} episodes ({rate:.0%} success).",
            similar_episodes=len(self.episodes),
            historical_success_rate=round(rate, 3),
            predicted_rounds=round(avg_rounds, 1),
        )

    # -- Memory Health ------------------------------------------------------

    def health_check(self) -> MemoryHealth:
        """Self-diagnose memory quality and coverage."""
        n = len(self.episodes)
        if n == 0:
            return MemoryHealth(
                total_episodes=0, age_days=0.0,
                freshness_score=0.0, coverage_score=0.0,
                bias_score=50.0, reliability_score=0.0,
                warnings=["Memory is empty."],
                suggestions=["Run consensus rounds to build memory."],
            )

        now = time.time()
        half_life_s = self.half_life_days * 86400.0
        max_age_s = 0.0
        fresh_count = 0
        success_count = 0
        buckets: set = set()

        # Single-pass: collect freshness, coverage, success counts
        for ep in self.episodes:
            age_s = now - ep.timestamp
            if age_s > max_age_s:
                max_age_s = age_s
            if age_s <= half_life_s:
                fresh_count += 1
            if ep.committed:
                success_count += 1
            buckets.add((_agent_band(ep.agent_count), _threshold_band(ep.threshold)))

        age_days = max_age_s / 86400.0

        # Freshness: what fraction of episodes are within half-life?
        freshness = (fresh_count / n) * 100.0

        # Coverage: how many distinct config buckets are represented?
        max_buckets = 4 * 4  # 4 agent bands × 4 threshold bands
        coverage = min(100.0, (len(buckets) / max_buckets) * 100.0)

        # Bias: is the success/failure ratio extremely skewed?
        ratio = success_count / n
        # Bias score is highest (100) when ratio ~0.5, lowest at extremes
        bias = 100.0 * (1.0 - abs(ratio - 0.5) * 2.0)

        reliability = (0.4 * freshness + 0.3 * coverage + 0.3 * bias)

        warnings: List[str] = []
        suggestions: List[str] = []

        if freshness < 30:
            warnings.append(f"Memory is stale — only {fresh_count}/{n} episodes within half-life.")
            suggestions.append("Run more recent consensus rounds or decrease half_life_days.")
        if coverage < 25:
            warnings.append(f"Low configuration coverage — only {len(buckets)}/{max_buckets} buckets seen.")
            suggestions.append("Try varied agent counts and thresholds to improve coverage.")
        if bias < 20:
            if ratio > 0.8:
                warnings.append(f"Success bias: {ratio:.0%} episodes committed — may overfit to easy tasks.")
            else:
                warnings.append(f"Failure bias: only {ratio:.0%} committed — may underestimate capabilities.")
            suggestions.append("Include a mix of easy and hard consensus tasks.")
        if n < 20:
            warnings.append(f"Small memory ({n} episodes) — predictions will be uncertain.")
            suggestions.append("Accumulate at least 20 episodes for reliable recommendations.")

        return MemoryHealth(
            total_episodes=n,
            age_days=round(age_days, 1),
            freshness_score=round(freshness, 1),
            coverage_score=round(coverage, 1),
            bias_score=round(bias, 1),
            reliability_score=round(reliability, 1),
            warnings=warnings,
            suggestions=suggestions,
        )

    # -- Persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        """Save memory to a JSON file."""
        data = {
            "half_life_days": self.half_life_days,
            "min_similarity": self.min_similarity,
            "created_at": self._created_at,
            "episodes": [asdict(e) for e in self.episodes],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "SwarmMemory":
        """Load memory from a JSON file."""
        raw = json.loads(Path(path).read_text())
        mem = cls(
            half_life_days=raw.get("half_life_days", 30.0),
            min_similarity=raw.get("min_similarity", 0.15),
        )
        mem._created_at = raw.get("created_at", time.time())
        for ep_data in raw.get("episodes", []):
            mem.episodes.append(Episode(**ep_data))
        return mem

    # -- HTML Dashboard -----------------------------------------------------

    def export_html(self, path: str) -> None:
        """Generate an interactive HTML dashboard of memory contents."""
        patterns = self.extract_patterns()
        health = self.health_check()
        n = len(self.episodes)
        successes = sum(1 for e in self.episodes if e.committed)
        success_rate = (successes / n * 100) if n else 0

        # Configuration heatmap data
        heatmap: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        heatmap_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ep in self.episodes:
            ab = _agent_band(ep.agent_count)
            tb = _threshold_band(ep.threshold)
            heatmap_counts[ab][tb] += 1
            if ep.committed:
                heatmap[ab][tb] += 1

        heatmap_rates: Dict[str, Dict[str, float]] = {}
        for ab in heatmap_counts:
            heatmap_rates[ab] = {}
            for tb in heatmap_counts[ab]:
                cnt = heatmap_counts[ab][tb]
                heatmap_rates[ab][tb] = round(heatmap[ab].get(tb, 0) / cnt * 100, 1) if cnt else 0

        # Pre-build dynamic HTML fragments
        gauge_cls_s = 'green' if success_rate >= 70 else ('yellow' if success_rate >= 40 else 'red')
        gauge_cls_r = 'green' if health.reliability_score >= 60 else ('yellow' if health.reliability_score >= 30 else 'red')

        diag_html = ''
        if health.warnings:
            for w in health.warnings:
                diag_html += f'<p class="warning">\u26a0 {w}</p>'
        else:
            diag_html = '<p style="color:#4ade80">\u2713 No warnings</p>'
        for s in health.suggestions:
            diag_html += f'<p class="suggestion">\U0001f4a1 {s}</p>'

        pattern_html = ''
        if patterns:
            for p in patterns[:8]:
                pattern_html += (f'<li><span class="pattern-name">{p.description}</span><br>'
                    f'<span class="pattern-detail">{p.episode_count} episodes \u00b7 '
                    f'{p.success_rate:.0%} success \u00b7 avg {p.avg_rounds:.1f} rounds \u00b7 '
                    f'efficiency {p.avg_efficiency:.2f}</span></li>')
        else:
            pattern_html = '<li style="color:#7a8ba8">Not enough data for pattern detection.</li>'

        heatmap_html = ''
        for ab in ['small', 'medium', 'large', 'xlarge']:
            cells = ''
            ab_data = heatmap_rates.get(ab, {})
            for tb in ['low', 'medium', 'high', 'very_high']:
                val = ab_data.get(tb)
                if val is not None:
                    hue = int(val * 1.2)
                    cells += f'<td style="background:hsl({hue}, 70%, 25%)">{val}%</td>'
                else:
                    cells += '<td>\u2014</td>'
            heatmap_html += f'<tr><th>{ab}</th>{cells}</tr>'

        episode_html = ''
        for ep in reversed(self.episodes[-20:]):
            task_d = ep.task[:60] + ('\u2026' if len(ep.task) > 60 else '')
            cls = 'committed' if ep.committed else 'failed'
            label = '\u2713 Committed' if ep.committed else '\u2717 Failed'
            episode_html += (f'<tr><td>{task_d}</td><td>{ep.agent_count}</td>'
                f'<td>{ep.byzantine_count}</td><td>{ep.threshold}</td>'
                f'<td>{ep.rounds_used}</td><td class="{cls}">{label}</td>'
                f'<td>{ep.efficiency():.2f}</td></tr>')

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mBFT Swarm Memory Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0e17; color: #e0e6f0; padding: 24px; }}
h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
.subtitle {{ color: #7a8ba8; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ background: #131927; border: 1px solid #1e2a3e; border-radius: 12px; padding: 20px; }}
.card h2 {{ font-size: 1.1rem; color: #7ecfff; margin-bottom: 12px; }}
.metric {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a2236; }}
.metric:last-child {{ border: none; }}
.metric-label {{ color: #7a8ba8; }}
.metric-value {{ font-weight: 600; }}
.gauge {{ width: 100%; height: 12px; background: #1a2236; border-radius: 6px; overflow: hidden; margin-top: 4px; }}
.gauge-fill {{ height: 100%; border-radius: 6px; transition: width 0.6s; }}
.green {{ background: linear-gradient(90deg, #22c55e, #4ade80); }}
.yellow {{ background: linear-gradient(90deg, #eab308, #facc15); }}
.red {{ background: linear-gradient(90deg, #ef4444, #f87171); }}
.blue {{ background: linear-gradient(90deg, #3b82f6, #60a5fa); }}
.pattern-list {{ list-style: none; }}
.pattern-list li {{ padding: 8px 0; border-bottom: 1px solid #1a2236; }}
.pattern-list li:last-child {{ border: none; }}
.pattern-name {{ font-weight: 600; color: #a78bfa; }}
.pattern-detail {{ color: #7a8ba8; font-size: 0.9rem; }}
.warning {{ color: #f59e0b; font-size: 0.9rem; padding: 4px 0; }}
.suggestion {{ color: #22d3ee; font-size: 0.9rem; padding: 4px 0; }}
.heatmap {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
.heatmap th {{ color: #7a8ba8; padding: 8px; text-align: center; font-size: 0.85rem; }}
.heatmap td {{ padding: 12px; text-align: center; font-weight: 600; border-radius: 6px; }}
table.episodes {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.85rem; }}
table.episodes th {{ color: #7a8ba8; padding: 8px; text-align: left; border-bottom: 1px solid #1e2a3e; }}
table.episodes td {{ padding: 8px; border-bottom: 1px solid #141c2c; }}
.committed {{ color: #4ade80; }}
.failed {{ color: #f87171; }}
footer {{ text-align: center; color: #4a5568; margin-top: 32px; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>🧠 Swarm Memory Dashboard</h1>
<p class="subtitle">mBFT Consensus Learning &mdash; {n} episodes recorded</p>

<div class="grid">
  <div class="card">
    <h2>📊 Overview</h2>
    <div class="metric"><span class="metric-label">Total Episodes</span><span class="metric-value">{n}</span></div>
    <div class="metric"><span class="metric-label">Committed</span><span class="metric-value committed">{successes}</span></div>
    <div class="metric"><span class="metric-label">Failed</span><span class="metric-value failed">{n - successes}</span></div>
    <div class="metric"><span class="metric-label">Success Rate</span><span class="metric-value">{success_rate:.1f}%</span></div>
    <div class="gauge"><div class="gauge-fill {gauge_cls_s}" style="width:{success_rate:.0f}%"></div></div>
  </div>

  <div class="card">
    <h2>🏥 Memory Health</h2>
    <div class="metric"><span class="metric-label">Freshness</span><span class="metric-value">{health.freshness_score:.0f}%</span></div>
    <div class="gauge"><div class="gauge-fill blue" style="width:{health.freshness_score:.0f}%"></div></div>
    <div class="metric"><span class="metric-label">Coverage</span><span class="metric-value">{health.coverage_score:.0f}%</span></div>
    <div class="gauge"><div class="gauge-fill blue" style="width:{health.coverage_score:.0f}%"></div></div>
    <div class="metric"><span class="metric-label">Bias Balance</span><span class="metric-value">{health.bias_score:.0f}%</span></div>
    <div class="gauge"><div class="gauge-fill blue" style="width:{health.bias_score:.0f}%"></div></div>
    <div class="metric"><span class="metric-label">Reliability</span><span class="metric-value">{health.reliability_score:.0f}%</span></div>
    <div class="gauge"><div class="gauge-fill {gauge_cls_r}" style="width:{health.reliability_score:.0f}%"></div></div>
  </div>

  <div class="card">
  <div class="card">
    <h2>Diagnostics</h2>
    {diag_html}
  </div>
</div>

<div class="grid">
  <div class="card" style="grid-column: span 2;">
    <h2>Detected Patterns</h2>
    <ul class="pattern-list">
      {pattern_html}
    </ul>
  </div>

  <div class="card">
    <h2>Config Success Rate</h2>
    <table class="heatmap">
      <tr><th></th><th>Low θ</th><th>Med θ</th><th>High θ</th><th>V.High θ</th></tr>
      {heatmap_html}
    </table>
  </div>
</div>

<div class="card" style="margin-top: 16px;">
  <h2>Recent Episodes</h2>
  <table class="episodes">
    <tr><th>Task</th><th>Agents</th><th>Byzantine</th><th>Threshold</th><th>Rounds</th><th>Outcome</th><th>Efficiency</th></tr>
    {episode_html}
  </table>
</div>

<footer>mBFT Swarm Memory · Generated by metacognition</footer>
</body>
</html>"""
        Path(path).write_text(html, encoding='utf-8')

    # -- Summary ------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of memory state."""
        health = self.health_check()
        n = len(self.episodes)
        if n == 0:
            return "Swarm Memory: empty (0 episodes)"
        successes = sum(1 for e in self.episodes if e.committed)
        patterns = self.extract_patterns()
        lines = [
            f"Swarm Memory: {n} episodes ({successes} committed, {n - successes} failed)",
            f"  Success rate: {successes / n:.0%}",
            f"  Health: reliability={health.reliability_score:.0f}%, freshness={health.freshness_score:.0f}%, coverage={health.coverage_score:.0f}%",
            f"  Patterns detected: {len(patterns)}",
        ]
        if patterns:
            top = patterns[0]
            lines.append(f"  Top pattern: {top.description} ({top.episode_count} episodes, {top.success_rate:.0%} success)")
        for w in health.warnings[:2]:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo Simulation
# ---------------------------------------------------------------------------

async def _run_demo(args: argparse.Namespace) -> SwarmMemory:
    """Run a simulated swarm and build memory from the outcomes."""
    from src.agents.metacognitive import MockAgent
    from src.core.protocol import MBFTEngine

    mem = SwarmMemory(half_life_days=args.half_life)

    if args.load:
        mem = SwarmMemory.load(args.load)
        print(f"Loaded memory with {len(mem.episodes)} episodes.")

    if not args.recommend and not args.health:
        # Simulate diverse consensus scenarios
        tasks = [
            "What is the capital of France?",
            "Prove that sqrt(2) is irrational.",
            "What caused the 2008 financial crisis?",
            "Translate 'hello world' to Japanese.",
            "Is P = NP?",
            "What is the best sorting algorithm?",
            "Explain quantum entanglement simply.",
            "What year did Apollo 11 land?",
            "Compute the integral of e^(-x^2).",
            "What is the trolley problem?",
            "Factorize x^4 - 1.",
            "Who wrote 'One Hundred Years of Solitude'?",
            "What is the halting problem?",
            "Summarize the theory of relativity.",
            "What is the Nash equilibrium?",
        ]

        configs = [
            {"agents": 5, "byz": 1, "threshold": 1.5},
            {"agents": 7, "byz": 2, "threshold": 2.0},
            {"agents": 3, "byz": 0, "threshold": 0.8},
            {"agents": 9, "byz": 3, "threshold": 2.5},
            {"agents": 5, "byz": 2, "threshold": 1.5},
        ]

        print(f"Simulating {len(tasks)} consensus rounds across {len(configs)} configurations...\n")

        for i, task in enumerate(tasks):
            cfg = configs[i % len(configs)]
            n_agents = cfg["agents"]
            n_byz = cfg["byz"]
            threshold = cfg["threshold"]

            agents = []
            correct_answer = f"answer_{i}"
            for j in range(n_agents):
                if j < n_byz:
                    agents.append(MockAgent(
                        f"byz_{j}", answer=f"wrong_{j}",
                        confidence=random.uniform(0.3, 0.9), byzantine=True,
                    ))
                else:
                    agents.append(MockAgent(
                        f"agent_{j}", answer=correct_answer,
                        confidence=random.uniform(0.6, 0.95),
                    ))

            engine = MBFTEngine(agents=agents, threshold=threshold)
            t0 = time.time()
            result = await engine.run(task)
            elapsed = time.time() - t0

            ep = mem.record_from_result(
                task=task,
                result=result,
                agent_count=n_agents,
                byzantine_count=n_byz,
                threshold=threshold,
                reputations=dict(engine._reputation),
                elapsed_s=elapsed,
            )

            status = "✓ COMMITTED" if ep.committed else "✗ FAILED"
            print(f"  [{i + 1:2d}/{len(tasks)}] {status}  agents={n_agents} byz={n_byz} θ={threshold}  rounds={ep.rounds_used}  |  {task[:50]}")

        print(f"\n{'=' * 60}")

    if args.recommend:
        rec = mem.recommend(args.recommend)
        print(f"\n🎯 Recommendation for: {args.recommend}")
        print(f"  Agents: {rec.suggested_agents}")
        print(f"  Threshold: {rec.suggested_threshold}")
        print(f"  Max Byzantine: {rec.suggested_max_byzantine}")
        print(f"  Confidence: {rec.confidence:.1%}")
        print(f"  Predicted rounds: {rec.predicted_rounds}")
        print(f"  Historical success: {rec.historical_success_rate:.0%}")
        print(f"  Reasoning: {rec.reasoning}")

    if args.health:
        health = mem.health_check()
        print(f"\n🏥 Memory Health Report")
        print(f"  Episodes: {health.total_episodes}")
        print(f"  Freshness: {health.freshness_score:.0f}%")
        print(f"  Coverage: {health.coverage_score:.0f}%")
        print(f"  Bias balance: {health.bias_score:.0f}%")
        print(f"  Reliability: {health.reliability_score:.0f}%")
        for w in health.warnings:
            print(f"  ⚠ {w}")
        for s in health.suggestions:
            print(f"  💡 {s}")

    print(f"\n{mem.summary()}")

    # Patterns
    patterns = mem.extract_patterns()
    if patterns:
        print(f"\n🔍 Detected {len(patterns)} patterns:")
        for p in patterns[:5]:
            print(f"  • {p.description}: {p.episode_count} episodes, {p.success_rate:.0%} success")

    return mem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Memory — persistent learning for mBFT consensus",
    )
    parser.add_argument("--load", type=str, help="Load memory from JSON file")
    parser.add_argument("--save", "-s", type=str, help="Save memory to JSON file")
    parser.add_argument("--recommend", type=str, help="Get recommendation for a task")
    parser.add_argument("--health", action="store_true", help="Run memory health check")
    parser.add_argument("--half-life", type=float, default=30.0, help="Decay half-life in days")
    parser.add_argument(
        "--export", choices=["html", "json"], help="Export format",
    )
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    args = parser.parse_args()

    mem = asyncio.run(_run_demo(args))

    if args.save:
        mem.save(args.save)
        print(f"\n💾 Memory saved to {args.save}")

    if args.export == "html":
        out = args.output or "swarm_memory_dashboard.html"
        mem.export_html(out)
        print(f"\n📊 Dashboard exported to {out}")
    elif args.export == "json":
        out = args.output or "swarm_memory.json"
        mem.save(out)
        print(f"\n📄 Memory exported to {out}")


if __name__ == "__main__":
    main()
