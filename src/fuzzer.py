"""Consensus Protocol Fuzzer — autonomous edge-case discovery via mutation.

Unlike the Adversarial Trainer's fixed attack catalogue, the fuzzer
*generates* novel misbehaviours by randomly mutating protocol messages,
agent timing, confidence values, vote weights, and quorum sizes.  It
tracks coverage of unique failure modes and uses a feedback loop to
concentrate mutations in areas that produced interesting results.

Usage::

    python -m src.fuzzer                            # default 200 iterations
    python -m src.fuzzer --iterations 500
    python -m src.fuzzer --agents 7
    python -m src.fuzzer --seed 42                  # reproducible
    python -m src.fuzzer --export report.html
    python -m src.fuzzer --export results.json
    python -m src.fuzzer --autopilot                # adaptive iteration count
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_mod
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import BaseAgent
from src.core.protocol import MBFTEngine
from src.core.state import Proposal, Vote

# ── Mutation operators ───────────────────────────────────────────────────

MUTATION_OPS = [
    "confidence_noise",
    "vote_weight_noise",
    "confidence_clamp_high",
    "confidence_clamp_low",
    "vote_flip_random",
    "threshold_perturb",
    "quorum_shrink",
    "quorum_bloat",
    "proof_scramble",
    "duplicate_proposal",
    "confidence_nan",
    "vote_all_abstain",
    "leader_impersonate",
    "late_vote_drop",
]

TASKS = [
    "What is 2 + 2?",
    "Is P = NP?",
    "Define consciousness.",
    "What year did the moon landing occur?",
    "Solve x^2 - 4 = 0.",
]


# ── Fuzzer agents ────────────────────────────────────────────────────────


class FuzzableAgent(BaseAgent):
    """Agent whose behaviour can be mutated at runtime."""

    def __init__(self, agent_id: str, *, honest: bool = True) -> None:
        super().__init__(agent_id)
        self.honest = honest
        self._mutations: Dict[str, Any] = {}

    def apply_mutations(self, mutations: Dict[str, Any]) -> None:
        self._mutations = mutations

    def clear_mutations(self) -> None:
        self._mutations = {}

    async def generate_proposal(self, task: str) -> Proposal:
        base_conf = random.uniform(0.6, 0.95) if self.honest else random.uniform(0.1, 0.5)
        solution = "answer-" + self.id
        proof = "proof-" + self.id

        if "confidence_noise" in self._mutations:
            base_conf += random.gauss(0, self._mutations["confidence_noise"])
        if "confidence_clamp_high" in self._mutations:
            base_conf = 1.0
        if "confidence_clamp_low" in self._mutations:
            base_conf = random.uniform(0.0, 0.05)
        if "proof_scramble" in self._mutations:
            proof = "garbled-" + str(random.randint(0, 99999))
        if "duplicate_proposal" in self._mutations:
            solution = self._mutations["duplicate_proposal"]

        base_conf = max(0.0, min(1.0, base_conf))
        return Proposal(agent_id=self.id, solution=solution, proof=proof, confidence=base_conf)

    async def verify_proposal(self, leader_proposal: Proposal) -> Vote:
        weight = random.uniform(0.3, 1.0) if self.honest else random.uniform(-1.0, 0.3)

        if "vote_weight_noise" in self._mutations:
            weight += random.gauss(0, self._mutations["vote_weight_noise"])
        if "vote_flip_random" in self._mutations and random.random() < 0.5:
            weight = -weight
        if "vote_all_abstain" in self._mutations:
            weight = random.uniform(-0.05, 0.05)

        weight = max(-1.0, min(1.0, weight))
        counter = ("counter-" + self.id) if weight < 0 else None
        return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id,
                     weight=weight, counter_proof=counter)


# ── Mutation generator ───────────────────────────────────────────────────


@dataclass
class MutationPlan:
    ops: List[str]
    params: Dict[str, Any]
    affected_agents: List[str]

    @property
    def fingerprint(self) -> str:
        key = "|".join(sorted(self.ops))
        # Non-cryptographic fingerprint (deduplication/identity only). MD5 is
        # used purely for its short, deterministic digest; mark explicitly with
        # ``usedforsecurity=False`` so FIPS/Bandit (B324) won't flag it and so
        # the intent is unambiguous to future readers.
        return hashlib.md5(  # noqa: S324  # nosec B324
            key.encode(), usedforsecurity=False
        ).hexdigest()[:8]


def generate_mutation_plan(agent_ids: List[str], rng: random.Random,
                           energy: Dict[str, float]) -> MutationPlan:
    weights = [energy.get(op, 1.0) for op in MUTATION_OPS]
    total = sum(weights)
    probs = [w / total for w in weights]

    n_ops = rng.randint(1, 3)
    chosen_ops = list(set(rng.choices(MUTATION_OPS, weights=probs, k=n_ops)))

    n_affected = rng.randint(1, max(1, len(agent_ids) // 2))
    affected = rng.sample(agent_ids, min(n_affected, len(agent_ids)))

    params: Dict[str, Any] = {}
    for op in chosen_ops:
        if op == "confidence_noise":
            params["confidence_noise"] = rng.uniform(0.1, 0.5)
        elif op == "vote_weight_noise":
            params["vote_weight_noise"] = rng.uniform(0.1, 0.6)
        elif op == "threshold_perturb":
            params["threshold_perturb"] = rng.uniform(-0.3, 0.3)
        elif op == "quorum_shrink":
            params["quorum_shrink"] = rng.randint(1, max(1, len(agent_ids) // 3))
        elif op == "quorum_bloat":
            params["quorum_bloat"] = rng.randint(1, 3)
        elif op == "late_vote_drop":
            params["late_vote_drop"] = rng.uniform(0.2, 0.6)
        elif op == "duplicate_proposal":
            params["duplicate_proposal"] = "duplicated-answer"
        else:
            params[op] = True

    return MutationPlan(ops=chosen_ops, params=params, affected_agents=affected)


# ── Outcome classification ───────────────────────────────────────────────


@dataclass
class FuzzOutcome:
    iteration: int
    mutation_plan: MutationPlan
    committed: bool
    rounds_used: int
    aggregate_weight: float
    slashed_count: int
    exception: Optional[str] = None
    classification: str = "normal"

    def classify(self) -> str:
        if self.exception:
            return "crash"
        if self.committed and self.slashed_count > 0:
            return "commit_with_slash"
        if not self.committed:
            return "no_commit"
        if self.aggregate_weight > 0.95:
            return "trivial_commit"
        if self.rounds_used > 2:
            return "slow_commit"
        return "normal"


# ── Fuzzer engine ────────────────────────────────────────────────────────


@dataclass
class FuzzerStats:
    total_iterations: int = 0
    classifications: Dict[str, int] = field(default_factory=lambda: Counter())
    unique_fingerprints: int = 0
    crash_count: int = 0
    interesting_count: int = 0
    op_hit_count: Dict[str, int] = field(default_factory=lambda: Counter())
    op_interesting_count: Dict[str, int] = field(default_factory=lambda: Counter())


async def run_one_iteration(
    iteration: int,
    agents: List[FuzzableAgent],
    plan: MutationPlan,
    base_threshold: float,
) -> FuzzOutcome:
    for a in agents:
        a.clear_mutations()

    for aid in plan.affected_agents:
        for a in agents:
            if a.id == aid:
                a.apply_mutations(plan.params)

    active_agents: List[BaseAgent] = list(agents)
    if "quorum_shrink" in plan.params:
        n_remove = min(plan.params["quorum_shrink"], len(active_agents) - 1)
        for _ in range(n_remove):
            if len(active_agents) > 1:
                active_agents.pop(random.randint(0, len(active_agents) - 1))

    if "quorum_bloat" in plan.params:
        for i in range(plan.params["quorum_bloat"]):
            extra = FuzzableAgent("bloat-" + str(i), honest=False)
            extra.apply_mutations(plan.params)
            active_agents.append(extra)

    threshold = base_threshold
    if "threshold_perturb" in plan.params:
        threshold = max(0.01, threshold + plan.params["threshold_perturb"])

    task = random.choice(TASKS)

    try:
        engine = MBFTEngine(active_agents, threshold=threshold, max_rounds=4)
        result = await engine.run(task)
        if result is None:
            return FuzzOutcome(iteration=iteration, mutation_plan=plan,
                               committed=False, rounds_used=0,
                               aggregate_weight=0.0, slashed_count=0)
        slashed = sum(len(r.slashed) for r in engine.history)
        outcome = FuzzOutcome(
            iteration=iteration, mutation_plan=plan,
            committed=result.committed, rounds_used=len(engine.history),
            aggregate_weight=result.aggregate_weight, slashed_count=slashed,
        )
    except Exception as exc:
        outcome = FuzzOutcome(
            iteration=iteration, mutation_plan=plan,
            committed=False, rounds_used=0, aggregate_weight=0.0,
            slashed_count=0, exception=str(exc),
        )

    outcome.classification = outcome.classify()
    return outcome


def is_interesting(outcome: FuzzOutcome) -> bool:
    return outcome.classification in ("crash", "commit_with_slash", "no_commit", "slow_commit")


async def fuzz(
    n_agents: int = 5,
    iterations: int = 200,
    seed: Optional[int] = None,
    autopilot: bool = False,
) -> Tuple[List[FuzzOutcome], FuzzerStats]:
    rng = random.Random(seed)
    if seed is not None:
        random.seed(seed)

    agents = [FuzzableAgent("agent-" + str(i), honest=(i < n_agents * 2 // 3))
              for i in range(n_agents)]
    agent_ids = [a.id for a in agents]
    base_threshold = 0.6

    energy: Dict[str, float] = {op: 1.0 for op in MUTATION_OPS}
    outcomes: List[FuzzOutcome] = []
    stats = FuzzerStats()
    seen_fingerprints: set = set()

    consecutive_boring = 0
    max_consecutive_boring = 50

    for i in range(iterations):
        plan = generate_mutation_plan(agent_ids, rng, energy)
        outcome = await run_one_iteration(i, agents, plan, base_threshold)
        outcomes.append(outcome)
        stats.total_iterations += 1
        stats.classifications[outcome.classification] += 1

        for op in plan.ops:
            stats.op_hit_count[op] += 1

        if plan.fingerprint not in seen_fingerprints:
            seen_fingerprints.add(plan.fingerprint)
            stats.unique_fingerprints += 1

        if outcome.exception:
            stats.crash_count += 1

        if is_interesting(outcome):
            stats.interesting_count += 1
            consecutive_boring = 0
            for op in plan.ops:
                energy[op] = min(5.0, energy[op] + 0.5)
                stats.op_interesting_count[op] += 1
        else:
            consecutive_boring += 1
            for op in plan.ops:
                energy[op] = max(0.2, energy[op] - 0.05)

        if autopilot and consecutive_boring >= max_consecutive_boring and i >= 100:
            break

    return outcomes, stats


# ── HTML report ──────────────────────────────────────────────────────────


def _tag(text: str, cls: str = "") -> str:
    c = " " + cls if cls else ""
    return '<span class="tag' + c + '">' + html_mod.escape(text) + '</span>'


def _generate_html(outcomes: List[FuzzOutcome], stats: FuzzerStats) -> str:
    cls_labels = sorted(stats.classifications.keys())
    cls_values = [stats.classifications[c] for c in cls_labels]

    op_labels = sorted(stats.op_hit_count.keys())
    op_total = [stats.op_hit_count[o] for o in op_labels]
    op_interest = [stats.op_interesting_count.get(o, 0) for o in op_labels]
    op_rate = [round(op_interest[i] / max(1, op_total[i]) * 100, 1) for i in range(len(op_labels))]

    timeline = [{"i": o.iteration, "cls": o.classification,
                 "ops": o.mutation_plan.ops, "agg": round(o.aggregate_weight, 3),
                 "rounds": o.rounds_used}
                for o in outcomes if is_interesting(o)]

    crashes = [{"i": o.iteration, "ops": o.mutation_plan.ops, "error": o.exception}
               for o in outcomes if o.exception]

    # Build table rows as strings
    op_rows = ""
    for i in sorted(range(len(op_labels)), key=lambda x: -op_rate[x]):
        bar_w = str(op_rate[i] * 2)
        op_rows += ("<tr><td>" + html_mod.escape(op_labels[i]) + "</td><td>" + str(op_total[i])
                     + "</td><td>" + str(op_interest[i])
                     + '</td><td><span class="bar" style="width:' + bar_w + 'px"></span> '
                     + str(op_rate[i]) + "%</td></tr>\n")

    timeline_rows = ""
    for t in timeline[:50]:
        tag_cls = "tag-crash" if t["cls"] == "crash" else "tag-interesting"
        cls_cell = _tag(t["cls"], tag_cls)
        ops_cell = "".join(_tag(o) for o in t["ops"])
        timeline_rows += ("<tr><td>" + str(t["i"]) + "</td><td>" + cls_cell
                           + "</td><td>" + ops_cell + "</td><td>" + str(t["agg"])
                           + "</td><td>" + str(t["rounds"]) + "</td></tr>\n")

    crash_section = ""
    if crashes:
        crash_rows = ""
        for c in crashes[:20]:
            ops_cell = "".join(_tag(o, "tag-crash") for o in c["ops"])
            err = html_mod.escape((c["error"] or "")[:120])
            crash_rows += ("<tr><td>" + str(c["i"]) + "</td><td>" + ops_cell
                           + '</td><td style="color:#f85149">' + err + "</td></tr>\n")
        crash_section = ('<div class="card" style="margin-bottom:16px"><h2>💥 Crashes</h2>'
                         '<table><thead><tr><th>#</th><th>Mutations</th><th>Error</th></tr></thead><tbody>'
                         + crash_rows + '</tbody></table></div>')

    more_note = ""
    if len(timeline) > 50:
        more_note = '<p style="color:#8b949e;margin-top:8px">Showing 50 of ' + str(len(timeline)) + ' findings</p>'

    recs = _generate_recommendations(stats, outcomes)
    op_chart_labels = json.dumps([o.replace("_", " ") for o in op_labels])

    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>mBFT Consensus Fuzzer Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f1117;color:#c9d1d9;padding:20px}
h1{text-align:center;color:#58a6ff;margin-bottom:8px;font-size:1.8rem}
.subtitle{text-align:center;color:#8b949e;margin-bottom:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.card h2{color:#58a6ff;font-size:1.1rem;margin-bottom:12px}
.stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #21262d}
.stat-label{color:#8b949e}.stat-value{color:#f0f6fc;font-weight:600}
.crash{color:#f85149}.interesting{color:#d29922}
canvas{max-width:100%;margin:8px 0}
table{width:100%;border-collapse:collapse;margin-top:8px}
th,td{padding:6px 8px;text-align:left;border-bottom:1px solid #21262d;font-size:0.85rem}
th{color:#58a6ff;font-weight:600}
.tag{display:inline-block;background:#1f6feb33;color:#58a6ff;padding:2px 6px;border-radius:4px;font-size:0.75rem;margin:1px}
.tag-crash{background:#f8514933;color:#f85149}
.tag-interesting{background:#d2992233;color:#d29922}
.bar{height:10px;background:#238636;border-radius:4px;display:inline-block}
</style></head><body>
<h1>🔀 mBFT Consensus Protocol Fuzzer</h1>
<p class="subtitle">Autonomous edge-case discovery via protocol mutation</p>

<div class="grid">
<div class="card">
<h2>📊 Summary</h2>
<div class="stat-row"><span class="stat-label">Iterations</span><span class="stat-value">ITER_TOTAL</span></div>
<div class="stat-row"><span class="stat-label">Unique Mutation Combos</span><span class="stat-value">UNIQUE_FP</span></div>
<div class="stat-row"><span class="stat-label">Interesting Outcomes</span><span class="stat-value interesting">INTERESTING</span></div>
<div class="stat-row"><span class="stat-label">Crashes</span><span class="stat-value crash">CRASHES</span></div>
<div class="stat-row"><span class="stat-label">Interesting Rate</span><span class="stat-value">RATE</span></div>
</div>

<div class="card">
<h2>📈 Outcome Classification</h2>
<canvas id="clsChart" height="200"></canvas>
</div>
</div>

<div class="grid">
<div class="card">
<h2>🎯 Mutation Effectiveness</h2>
<canvas id="opChart" height="260"></canvas>
</div>

<div class="card">
<h2>⚡ Most Effective Mutations</h2>
<table><thead><tr><th>Mutation</th><th>Uses</th><th>Interesting</th><th>Rate</th></tr></thead><tbody>
OP_ROWS
</tbody></table>
</div>
</div>

<div class="card" style="margin-bottom:16px">
<h2>🔍 Interesting Findings Timeline</h2>
<table><thead><tr><th>#</th><th>Class</th><th>Mutations</th><th>Agg Weight</th><th>Rounds</th></tr></thead><tbody>
TIMELINE_ROWS
</tbody></table>
MORE_NOTE
</div>

CRASH_SECTION

<div class="card">
<h2>🤖 Proactive Recommendations</h2>
<ul style="padding-left:20px;line-height:1.8">
RECS
</ul>
</div>

<script>
new Chart(document.getElementById('clsChart'),{type:'doughnut',data:{labels:CLS_LABELS,datasets:[{data:CLS_VALUES,backgroundColor:['#238636','#f85149','#d29922','#58a6ff','#8b949e','#bc8cff']}]},options:{plugins:{legend:{labels:{color:'#c9d1d9'}}}}});
new Chart(document.getElementById('opChart'),{type:'bar',data:{labels:OP_CHART_LABELS,datasets:[{label:'Total Uses',data:OP_TOTAL,backgroundColor:'#30363d'},{label:'Interesting',data:OP_INTEREST,backgroundColor:'#d29922'}]},options:{indexAxis:'y',scales:{x:{ticks:{color:'#8b949e'}},y:{ticks:{color:'#c9d1d9',font:{size:10}}}},plugins:{legend:{labels:{color:'#c9d1d9'}}}}});
</script>
</body></html>"""

    rate_str = str(round(stats.interesting_count / max(1, stats.total_iterations) * 100, 1)) + "%"
    html = html.replace("ITER_TOTAL", str(stats.total_iterations))
    html = html.replace("UNIQUE_FP", str(stats.unique_fingerprints))
    html = html.replace("INTERESTING", str(stats.interesting_count))
    html = html.replace("CRASHES", str(stats.crash_count))
    html = html.replace("RATE", rate_str)
    html = html.replace("OP_ROWS", op_rows)
    html = html.replace("TIMELINE_ROWS", timeline_rows)
    html = html.replace("MORE_NOTE", more_note)
    html = html.replace("CRASH_SECTION", crash_section)
    html = html.replace("RECS", recs)
    html = html.replace("CLS_LABELS", json.dumps(cls_labels))
    html = html.replace("CLS_VALUES", json.dumps(cls_values))
    html = html.replace("OP_CHART_LABELS", op_chart_labels)
    html = html.replace("OP_TOTAL", json.dumps(op_total))
    html = html.replace("OP_INTEREST", json.dumps(op_interest))

    return html


def _generate_recommendations(stats: FuzzerStats, outcomes: List[FuzzOutcome]) -> str:
    recs = []
    if stats.crash_count > 0:
        crash_ops: Counter = Counter()
        for o in outcomes:
            if o.exception:
                for op in o.mutation_plan.ops:
                    crash_ops[op] += 1
        top = crash_ops.most_common(3)
        parts = ", ".join("<code>" + o + "</code> (" + str(c) + "x)" for o, c in top)
        recs.append("<li>🚨 <b>" + str(stats.crash_count) + " crashes detected.</b> Top crash-inducing mutations: "
                     + parts + ". Add defensive guards for these edge cases.</li>")

    if stats.op_interesting_count:
        top_op = max(stats.op_interesting_count,
                     key=lambda k: stats.op_interesting_count[k] / max(1, stats.op_hit_count.get(k, 1)))
        rate = stats.op_interesting_count[top_op] / max(1, stats.op_hit_count[top_op]) * 100
        recs.append("<li>🎯 Mutation <code>" + top_op + "</code> has the highest anomaly rate ("
                     + str(int(rate)) + "%). Consider adding protocol hardening specifically for this vector.</li>")

    no_commit = stats.classifications.get("no_commit", 0)
    if no_commit > stats.total_iterations * 0.3:
        recs.append("<li>⚠️ " + str(no_commit) + "/" + str(stats.total_iterations)
                     + " iterations failed to commit. The protocol may be too sensitive to perturbation.</li>")

    slow = stats.classifications.get("slow_commit", 0)
    if slow > 10:
        recs.append("<li>🐌 " + str(slow) + " slow commits (>2 rounds). Investigate leader election stability.</li>")

    if stats.interesting_count < stats.total_iterations * 0.05:
        recs.append("<li>✅ Protocol appears highly robust — less than 5% anomalous iterations.</li>")
    elif stats.interesting_count > stats.total_iterations * 0.5:
        recs.append("<li>🔴 Over half (" + str(stats.interesting_count) + "/" + str(stats.total_iterations)
                     + ") of iterations were anomalous. Significant protocol fragility detected.</li>")

    if not recs:
        recs.append("<li>✅ No major concerns detected. Protocol handled mutations well.</li>")

    return "\n".join(recs)


# ── JSON report ──────────────────────────────────────────────────────────


def _build_json_report(outcomes: List[FuzzOutcome], stats: FuzzerStats) -> Dict[str, Any]:
    return {
        "summary": {
            "total_iterations": stats.total_iterations,
            "unique_fingerprints": stats.unique_fingerprints,
            "interesting_count": stats.interesting_count,
            "crash_count": stats.crash_count,
            "classifications": dict(stats.classifications),
        },
        "mutation_effectiveness": {
            op: {
                "total": stats.op_hit_count[op],
                "interesting": stats.op_interesting_count.get(op, 0),
                "rate": round(stats.op_interesting_count.get(op, 0) / max(1, stats.op_hit_count[op]) * 100, 1),
            }
            for op in sorted(stats.op_hit_count.keys())
        },
        "findings": [
            {
                "iteration": o.iteration,
                "classification": o.classification,
                "mutations": o.mutation_plan.ops,
                "aggregate_weight": round(o.aggregate_weight, 4),
                "rounds_used": o.rounds_used,
                "exception": o.exception,
            }
            for o in outcomes if is_interesting(o)
        ],
    }


# ── CLI ──────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(description="mBFT Consensus Protocol Fuzzer")
    parser.add_argument("--iterations", type=int, default=200, help="Number of fuzz iterations")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents in the network")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--export", type=str, default=None, help="Export report (.html or .json)")
    parser.add_argument("--autopilot", action="store_true", help="Stop early if no interesting results")
    args = parser.parse_args()

    print("🔀 mBFT Consensus Protocol Fuzzer")
    seed_str = str(args.seed) if args.seed is not None else "random"
    print("   Iterations: " + str(args.iterations) + " | Agents: " + str(args.agents)
          + " | Seed: " + seed_str + " | Autopilot: " + str(args.autopilot))
    print()

    t0 = time.time()
    outcomes, stats = await fuzz(
        n_agents=args.agents,
        iterations=args.iterations,
        seed=args.seed,
        autopilot=args.autopilot,
    )
    elapsed = time.time() - t0

    print("✅ Completed " + str(stats.total_iterations) + " iterations in " + str(round(elapsed, 1)) + "s")
    print("   Unique mutation combos: " + str(stats.unique_fingerprints))
    pct = str(int(stats.interesting_count / max(1, stats.total_iterations) * 100))
    print("   Interesting outcomes:   " + str(stats.interesting_count) + " (" + pct + "%)")
    print("   Crashes:                " + str(stats.crash_count))
    print()
    print("   Classification breakdown:")
    for cls, count in sorted(stats.classifications.items(), key=lambda x: -x[1]):
        p = str(int(count / max(1, stats.total_iterations) * 100))
        print("     " + cls.ljust(20) + " " + str(count).rjust(4) + "  (" + p + "%)")

    if stats.op_interesting_count:
        print()
        print("   Top interesting mutations:")
        ranked = sorted(stats.op_interesting_count.keys(),
                        key=lambda k: -stats.op_interesting_count[k] / max(1, stats.op_hit_count.get(k, 1)))
        for op in ranked[:5]:
            rate = stats.op_interesting_count[op] / max(1, stats.op_hit_count[op]) * 100
            line = ("     " + op.ljust(25) + " " + str(stats.op_interesting_count[op]).rjust(3)
                     + "/" + str(stats.op_hit_count[op]).rjust(3) + " (" + str(int(rate)) + "%)")
            print(line)

    if args.export:
        path = Path(args.export)
        if path.suffix == ".json":
            path.write_text(json.dumps(_build_json_report(outcomes, stats), indent=2), encoding="utf-8")
        else:
            path.write_text(_generate_html(outcomes, stats), encoding="utf-8")
        print()
        print("📄 Report exported to " + str(path))


if __name__ == "__main__":
    asyncio.run(main())
