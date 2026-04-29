"""Consensus Adversarial Trainer — autonomous progressive hardening.

An evolutionary training loop that progressively escalates adversarial
conditions against the mBFT consensus protocol.  The trainer starts with
easy scenarios and autonomously increases difficulty as the system proves
resilient.  It tracks generations of training, identifies weaknesses, and
produces an interactive HTML report with fitness curves, heatmaps, and
prescriptive hardening recommendations.

Usage::

    python -m src.adversarial_trainer                        # default 10 generations
    python -m src.adversarial_trainer --generations 20
    python -m src.adversarial_trainer --population 12
    python -m src.adversarial_trainer --export report.html
    python -m src.adversarial_trainer --export results.json
    python -m src.adversarial_trainer --autopilot            # auto-stop on convergence
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.agents.base import BaseAgent
from src.core.protocol import MBFTEngine
from src.core.state import Proposal, Vote

# ── Attack strategies ────────────────────────────────────────────────────

ATTACK_CATALOGUE = [
    "confidence_inflation",   # Byzantine agents report inflated confidence
    "vote_inversion",         # Byzantines vote opposite of their assessment
    "sybil_flooding",         # Extra Byzantine agents overwhelm honest ones
    "selective_veto",         # Byzantine agent vetoes only the strongest leader
    "slowloris_delay",        # Byzantine agents stall (simulated low confidence)
    "coalition_block",        # Coordinated negative votes from a bloc
    "chameleon_flip",         # Byzantine agent switches from honest to hostile mid-round
    "proof_spoofing",         # Byzantine agent supplies fake counter-proofs
]

TASKS = [
    "What is 2 + 2?",
    "Is P = NP?",
    "What colour is the sky?",
    "Compute the integral of x^2 from 0 to 1.",
    "Name the largest planet in the solar system.",
    "What is the capital of France?",
    "Simplify sqrt(144).",
    "Is water wet?",
]


@dataclass
class AttackScenario:
    """A specific adversarial configuration to test the protocol against."""
    attack_type: str
    byzantine_ratio: float       # 0.0–0.6
    agent_count: int             # 3–12
    confidence_noise: float      # extra noise on confidence [0, 0.3]
    threshold: float             # consensus threshold
    difficulty: float = 0.0      # computed composite difficulty

    def compute_difficulty(self) -> float:
        self.difficulty = (
            self.byzantine_ratio * 3.0
            + self.confidence_noise * 2.0
            + max(0, self.agent_count - 5) * 0.1
            + (1.0 - self.threshold) * 1.5
        )
        return self.difficulty


@dataclass
class GenerationResult:
    generation: int
    scenarios: List[AttackScenario]
    commit_rates: List[float]
    avg_rounds: List[float]
    reputation_damage: List[float]
    overall_fitness: float        # 0–1, how well protocol survived
    weakest_attack: Optional[str]
    strongest_defense: Optional[str]


@dataclass
class TrainingHistory:
    generations: List[GenerationResult] = field(default_factory=list)
    convergence_gen: Optional[int] = None
    recommendations: List[str] = field(default_factory=list)


# ── Mock agents with adversarial behaviors ───────────────────────────────

class AdversarialMockAgent(BaseAgent):
    """Agent with configurable adversarial behavior for training."""

    def __init__(
        self,
        agent_id: str,
        is_byzantine: bool = False,
        attack_type: str = "vote_inversion",
        confidence_noise: float = 0.0,
        correct_answer: str = "4",
    ) -> None:
        super().__init__(agent_id)
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.confidence_noise = confidence_noise
        self.correct_answer = correct_answer

    async def generate_proposal(self, task: str) -> Proposal:
        base_conf = random.uniform(0.5, 0.9)
        if self.is_byzantine:
            if self.attack_type == "confidence_inflation":
                base_conf = min(1.0, base_conf + 0.3)
            elif self.attack_type == "slowloris_delay":
                base_conf = max(0.05, base_conf - 0.4)
            elif self.attack_type == "chameleon_flip":
                # Start honest-looking
                base_conf = random.uniform(0.6, 0.85)

        base_conf = max(0.01, min(1.0, base_conf + random.uniform(
            -self.confidence_noise, self.confidence_noise)))

        solution = self.correct_answer if not self.is_byzantine else f"wrong_{random.randint(0,99)}"
        if self.is_byzantine and self.attack_type == "chameleon_flip":
            solution = self.correct_answer  # look honest in proposal

        return Proposal(
            agent_id=self.id,
            solution=solution,
            proof=f"proof_by_{self.id}",
            confidence=base_conf,
        )

    async def verify_proposal(self, leader_proposal: Proposal) -> Vote:
        if not self.is_byzantine:
            agree = leader_proposal.solution == self.correct_answer
            weight = random.uniform(0.4, 0.9) if agree else random.uniform(-0.8, -0.2)
            return Vote(
                voter_id=self.id,
                target_proposal_id=leader_proposal.proposal_id,
                weight=max(-1.0, min(1.0, weight)),
                counter_proof="counter" if not agree else None,
            )

        # Byzantine voting strategies
        if self.attack_type == "vote_inversion":
            weight = random.uniform(-0.9, -0.3)
        elif self.attack_type == "selective_veto":
            weight = -0.95 if leader_proposal.confidence > 0.7 else random.uniform(0.3, 0.7)
        elif self.attack_type == "coalition_block":
            weight = -0.8
        elif self.attack_type == "chameleon_flip":
            weight = random.uniform(-0.9, -0.5)  # flip hostile at vote time
        elif self.attack_type == "proof_spoofing":
            weight = random.uniform(-0.7, -0.2)
        else:
            weight = random.uniform(-0.6, 0.2)

        return Vote(
            voter_id=self.id,
            target_proposal_id=leader_proposal.proposal_id,
            weight=max(-1.0, min(1.0, weight)),
            counter_proof="spoofed_proof" if self.attack_type == "proof_spoofing" else None,
        )


# ── Trainer engine ───────────────────────────────────────────────────────

def _make_population(size: int, difficulty_level: float) -> List[AttackScenario]:
    """Generate a population of attack scenarios at the given difficulty."""
    scenarios = []
    for _ in range(size):
        attack = random.choice(ATTACK_CATALOGUE)
        # Scale parameters with difficulty
        byz_ratio = min(0.6, 0.05 + difficulty_level * random.uniform(0.05, 0.15))
        agent_count = random.randint(3, min(12, 4 + int(difficulty_level * 2)))
        conf_noise = min(0.3, difficulty_level * random.uniform(0.02, 0.08))
        threshold = max(0.3, 1.5 - difficulty_level * random.uniform(0.1, 0.2))
        s = AttackScenario(
            attack_type=attack,
            byzantine_ratio=byz_ratio,
            agent_count=agent_count,
            confidence_noise=conf_noise,
            threshold=threshold,
        )
        s.compute_difficulty()
        scenarios.append(s)
    return scenarios


async def _evaluate_scenario(
    scenario: AttackScenario, trials: int = 5
) -> Tuple[float, float, float]:
    """Run several trials and return (commit_rate, avg_rounds, rep_damage)."""
    commits = 0
    total_rounds = 0
    total_rep_damage = 0.0

    for _ in range(trials):
        n_agents = scenario.agent_count
        n_byz = max(0, int(n_agents * scenario.byzantine_ratio))
        agents = []
        for i in range(n_agents):
            is_byz = i < n_byz
            agents.append(AdversarialMockAgent(
                agent_id=f"a{i}",
                is_byzantine=is_byz,
                attack_type=scenario.attack_type,
                confidence_noise=scenario.confidence_noise,
                correct_answer="4",
            ))

        engine = MBFTEngine(agents, threshold=scenario.threshold, max_rounds=4)
        task = random.choice(TASKS)
        result = await engine.run(task)

        if result and result.committed:
            commits += 1
        total_rounds += len(engine.history)

        initial_rep = sum(1.0 for _ in agents)
        final_rep = sum(engine.reputation.values())
        total_rep_damage += (initial_rep - final_rep) / initial_rep

    return commits / trials, total_rounds / trials, total_rep_damage / trials


async def run_training(
    generations: int = 10,
    population_size: int = 8,
    autopilot: bool = False,
    on_progress=None,
) -> TrainingHistory:
    """Run the adversarial training loop."""
    history = TrainingHistory()
    prev_fitness = 0.0
    plateau_count = 0

    for gen in range(generations):
        difficulty = (gen + 1) / generations * 5.0  # scale 0→5
        scenarios = _make_population(population_size, difficulty)

        commit_rates = []
        avg_rounds_list = []
        rep_damage_list = []

        for sc in scenarios:
            cr, ar, rd = await _evaluate_scenario(sc)
            commit_rates.append(cr)
            avg_rounds_list.append(ar)
            rep_damage_list.append(rd)

        # Fitness: high commit rate + low round count + low rep damage
        mean_cr = sum(commit_rates) / len(commit_rates)
        mean_ar = sum(avg_rounds_list) / len(avg_rounds_list)
        mean_rd = sum(rep_damage_list) / len(rep_damage_list)
        fitness = mean_cr * 0.6 + (1.0 - mean_ar / 4.0) * 0.2 + (1.0 - mean_rd) * 0.2
        fitness = max(0.0, min(1.0, fitness))

        # Find weakest/strongest
        weakest_idx = commit_rates.index(min(commit_rates))
        strongest_idx = commit_rates.index(max(commit_rates))

        gr = GenerationResult(
            generation=gen + 1,
            scenarios=scenarios,
            commit_rates=commit_rates,
            avg_rounds=avg_rounds_list,
            reputation_damage=rep_damage_list,
            overall_fitness=fitness,
            weakest_attack=scenarios[weakest_idx].attack_type,
            strongest_defense=scenarios[strongest_idx].attack_type if commit_rates[strongest_idx] > 0.5 else None,
        )
        history.generations.append(gr)

        if on_progress:
            on_progress(gr)

        # Convergence check
        if abs(fitness - prev_fitness) < 0.02:
            plateau_count += 1
        else:
            plateau_count = 0
        prev_fitness = fitness

        if autopilot and plateau_count >= 3:
            history.convergence_gen = gen + 1
            break

    # Generate recommendations
    history.recommendations = _generate_recommendations(history)
    return history


def _generate_recommendations(history: TrainingHistory) -> List[str]:
    recs = []
    if not history.generations:
        return recs

    last = history.generations[-1]
    if last.overall_fitness < 0.4:
        recs.append("🔴 Protocol is vulnerable at high difficulty — consider raising the commitment threshold.")

    # Identify recurring weak attacks
    weak_attacks: Dict[str, int] = {}
    for g in history.generations:
        if g.weakest_attack:
            weak_attacks[g.weakest_attack] = weak_attacks.get(g.weakest_attack, 0) + 1
    for atk, cnt in sorted(weak_attacks.items(), key=lambda x: -x[1]):
        if cnt >= 2:
            recs.append(f"⚠️  '{atk}' was the weakest attack in {cnt} generations — prioritize defense against this pattern.")
            break

    # Fitness trend
    if len(history.generations) >= 3:
        early = sum(g.overall_fitness for g in history.generations[:3]) / 3
        late = sum(g.overall_fitness for g in history.generations[-3:]) / 3
        if late > early + 0.1:
            recs.append("📈 Fitness improved over generations — protocol shows adaptive resilience.")
        elif late < early - 0.1:
            recs.append("📉 Fitness degraded as difficulty increased — protocol needs hardening.")
        else:
            recs.append("➡️  Fitness remained stable — protocol handles escalation but may be near its limit.")

    if history.convergence_gen:
        recs.append(f"🎯 Training converged at generation {history.convergence_gen}.")

    # Rep damage check
    avg_rd = sum(g.reputation_damage[-1] for g in history.generations if g.reputation_damage) / max(1, len(history.generations))
    if avg_rd > 0.3:
        recs.append("⚡ High reputation damage across generations — slash_factor may be too aggressive.")

    return recs


# ── HTML report ──────────────────────────────────────────────────────────

def _generate_html(history: TrainingHistory) -> str:
    gens = history.generations
    fitness_data = [g.overall_fitness for g in gens]
    gen_labels = [g.generation for g in gens]

    # Attack type heatmap data
    attack_types = sorted(set(ATTACK_CATALOGUE))
    heatmap_rows = []
    for g in gens:
        row = {at: [] for at in attack_types}
        for sc, cr in zip(g.scenarios, g.commit_rates):
            row[sc.attack_type].append(cr)
        heatmap_rows.append({at: (sum(v)/len(v) if v else -1) for at, v in row.items()})

    heatmap_json = json.dumps(heatmap_rows)
    fitness_json = json.dumps(fitness_data)
    labels_json = json.dumps(gen_labels)
    attacks_json = json.dumps(attack_types)
    recs_html = "".join(f"<li>{r}</li>" for r in history.recommendations)

    # Summary stats
    total_scenarios = sum(len(g.scenarios) for g in gens)
    final_fitness = fitness_data[-1] if fitness_data else 0
    best_fitness = max(fitness_data) if fitness_data else 0

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>mBFT Adversarial Trainer Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e0e0e0;padding:24px}}
h1{{text-align:center;font-size:1.8rem;margin-bottom:8px;color:#a78bfa}}
.subtitle{{text-align:center;color:#888;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}}
.card{{background:#16161e;border:1px solid #2a2a3a;border-radius:12px;padding:20px}}
.card h2{{font-size:1rem;color:#a78bfa;margin-bottom:12px}}
.stat{{font-size:2.2rem;font-weight:700;color:#34d399}}
.stat.warn{{color:#f59e0b}}
.stat.danger{{color:#ef4444}}
canvas{{width:100%!important;max-height:300px}}
.heatmap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:0.85rem}}
th,td{{padding:6px 8px;border:1px solid #2a2a3a;text-align:center}}
th{{background:#1e1e2e;color:#a78bfa}}
.recs{{margin-top:16px}}
.recs li{{padding:6px 0;border-bottom:1px solid #1e1e2e}}
.badge{{display:inline-block;padding:2px 8px;border-radius:8px;font-size:0.75rem;font-weight:600}}
.badge-green{{background:#064e3b;color:#34d399}}
.badge-red{{background:#450a0a;color:#ef4444}}
.badge-yellow{{background:#422006;color:#f59e0b}}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
</head><body>
<h1>🥊 mBFT Adversarial Trainer</h1>
<p class="subtitle">Progressive hardening • {len(gens)} generations • {total_scenarios} scenarios tested</p>
<div class="grid">
  <div class="card"><h2>Final Fitness</h2>
    <div class="stat {'danger' if final_fitness<0.4 else 'warn' if final_fitness<0.7 else ''}">{final_fitness:.1%}</div></div>
  <div class="card"><h2>Best Fitness</h2><div class="stat">{best_fitness:.1%}</div></div>
  <div class="card"><h2>Generations</h2><div class="stat">{len(gens)}</div></div>
  <div class="card"><h2>Convergence</h2><div class="stat">{'Gen ' + str(history.convergence_gen) if history.convergence_gen else 'N/A'}</div></div>
</div>
<div class="grid">
  <div class="card" style="grid-column:1/-1"><h2>Fitness Curve</h2>
    <canvas id="fitnessChart"></canvas></div>
</div>
<div class="grid">
  <div class="card" style="grid-column:1/-1"><h2>Attack Resilience Heatmap</h2>
    <p style="color:#888;font-size:0.8rem;margin-bottom:8px">Commit rate by attack type per generation (green=resilient, red=vulnerable, gray=not tested)</p>
    <div class="heatmap"><table><thead><tr><th>Gen</th>{''.join(f'<th>{a}</th>' for a in attack_types)}</tr></thead>
    <tbody id="heatmapBody"></tbody></table></div></div>
</div>
<div class="card"><h2>Recommendations</h2><ul class="recs">{recs_html if recs_html else '<li>No recommendations — protocol appears robust.</li>'}</ul></div>
<script>
const FITNESS={fitness_json}, LABELS={labels_json}, ATTACKS={attacks_json}, HEATMAP={heatmap_json};
new Chart(document.getElementById('fitnessChart'),{{type:'line',data:{{labels:LABELS,datasets:[{{label:'Fitness',data:FITNESS,borderColor:'#a78bfa',backgroundColor:'rgba(167,139,250,0.15)',fill:true,tension:0.3,pointRadius:4}}]}},options:{{responsive:true,scales:{{y:{{min:0,max:1,ticks:{{color:'#888'}},grid:{{color:'#1e1e2e'}}}},x:{{title:{{display:true,text:'Generation',color:'#888'}},ticks:{{color:'#888'}},grid:{{color:'#1e1e2e'}}}}}},plugins:{{legend:{{labels:{{color:'#e0e0e0'}}}}}}}}}});
const hb=document.getElementById('heatmapBody');
HEATMAP.forEach((row,i)=>{{let tr='<tr><td>'+(i+1)+'</td>';ATTACKS.forEach(a=>{{const v=row[a];if(v<0)tr+='<td style="background:#1a1a2a;color:#555">—</td>';else{{const r=Math.round(255*(1-v)),g=Math.round(255*v);tr+='<td style="background:rgba('+r+','+g+',80,0.3);color:#e0e0e0">'+Math.round(v*100)+'%</td>';}}}});tr+='</tr>';hb.innerHTML+=tr;}});
</script></body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="mBFT Adversarial Trainer")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--autopilot", action="store_true",
                        help="Auto-stop when fitness converges")
    parser.add_argument("--export", type=str, default=None,
                        help="Export to HTML or JSON file")
    args = parser.parse_args()

    def on_progress(gr: GenerationResult):
        bar = "█" * int(gr.overall_fitness * 20) + "░" * (20 - int(gr.overall_fitness * 20))
        weak = gr.weakest_attack or "none"
        print(f"  Gen {gr.generation:>3}  [{bar}] {gr.overall_fitness:.1%}  weakest: {weak}")

    print(f"\n🥊 mBFT Adversarial Trainer")
    print(f"   {args.generations} generations × {args.population} scenarios/gen")
    if args.autopilot:
        print(f"   Autopilot: ON (stop on convergence)")
    print()

    history = asyncio.run(run_training(
        generations=args.generations,
        population_size=args.population,
        autopilot=args.autopilot,
        on_progress=on_progress,
    ))

    print(f"\n{'─'*60}")
    print(f"Training complete — {len(history.generations)} generations")
    if history.convergence_gen:
        print(f"Converged at generation {history.convergence_gen}")
    print(f"\nRecommendations:")
    for r in history.recommendations:
        print(f"  {r}")

    if args.export:
        path = Path(args.export)
        if path.suffix == ".json":
            data = {
                "generations": [
                    {
                        "generation": g.generation,
                        "fitness": g.overall_fitness,
                        "weakest_attack": g.weakest_attack,
                        "commit_rates": g.commit_rates,
                        "avg_rounds": g.avg_rounds,
                        "reputation_damage": g.reputation_damage,
                        "scenarios": [
                            {"attack": s.attack_type, "byzantine_ratio": s.byzantine_ratio,
                             "agents": s.agent_count, "difficulty": s.difficulty}
                            for s in g.scenarios
                        ],
                    }
                    for g in history.generations
                ],
                "recommendations": history.recommendations,
                "convergence_generation": history.convergence_gen,
            }
            path.write_text(json.dumps(data, indent=2))
        else:
            path.write_text(_generate_html(history), encoding="utf-8")
        print(f"\nExported → {path}")


if __name__ == "__main__":
    _cli()
