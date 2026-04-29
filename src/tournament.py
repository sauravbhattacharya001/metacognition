"""Consensus Tournament Arena — round-robin team competitions with ELO ratings.

Pit different agent team configurations against each other across multiple
tasks.  Each match runs a full mBFT consensus round and scores teams on
whether they reach commitment, how quickly, and solution quality.

Usage::

    python -m src.tournament                    # default 6-team tournament
    python -m src.tournament --teams 8 --rounds 3
    python -m src.tournament --export report.html
    python -m src.tournament --export results.json
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine

# ── Tournament data ──────────────────────────────────────────────────────

TASKS = [
    "What is 2 + 2?",
    "Is P = NP?",
    "What colour is the sky?",
    "Compute the integral of x^2 from 0 to 1.",
    "Name the largest planet in the solar system.",
    "Is the Goldbach conjecture true?",
    "What is the capital of France?",
    "Simplify sqrt(144).",
]


@dataclass
class TeamConfig:
    name: str
    agent_count: int
    confidence_range: Tuple[float, float]
    byzantine_ratio: float  # fraction of agents that are Byzantine
    answer_diversity: int   # how many distinct answers the team may produce
    description: str = ""


DEFAULT_TEAMS: List[TeamConfig] = [
    TeamConfig("Precision Squad", 5, (0.8, 0.95), 0.0, 1,
               "High-confidence, zero Byzantine, unanimous answers"),
    TeamConfig("Diverse Council", 5, (0.5, 0.9), 0.0, 3,
               "Moderate confidence, honest but diverse opinions"),
    TeamConfig("Saboteur Cell", 5, (0.6, 0.85), 0.4, 2,
               "Contains Byzantine agents that undermine consensus"),
    TeamConfig("Cautious Panel", 5, (0.3, 0.55), 0.0, 1,
               "Low confidence, unanimous — struggles to reach threshold"),
    TeamConfig("Chaos Swarm", 7, (0.4, 0.9), 0.28, 4,
               "Large, noisy, partially Byzantine, very diverse"),
    TeamConfig("Elite Duo", 2, (0.9, 0.99), 0.0, 1,
               "Tiny but extremely confident and aligned"),
]

POSSIBLE_ANSWERS = [
    "4", "yes", "blue", "1/3", "Jupiter", "unknown", "Paris", "12",
    "42", "no", "green", "0.333", "Saturn", "true", "London", "144",
]


@dataclass
class MatchResult:
    team_a: str
    team_b: str
    task: str
    winner: Optional[str]       # None = draw
    a_committed: bool
    b_committed: bool
    a_rounds: int
    b_rounds: int
    a_aggregate: float
    b_aggregate: float


@dataclass
class TeamStats:
    name: str
    elo: float = 1500.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    commits: int = 0
    total_matches: int = 0
    total_aggregate: float = 0.0
    history: List[float] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_agents(cfg: TeamConfig, seed: int) -> List[MockAgent]:
    rng = random.Random(seed)
    answers = rng.sample(POSSIBLE_ANSWERS, min(cfg.answer_diversity, len(POSSIBLE_ANSWERS)))
    agents: List[MockAgent] = []
    n_byz = int(cfg.agent_count * cfg.byzantine_ratio)
    for i in range(cfg.agent_count):
        ans = rng.choice(answers)
        conf = round(rng.uniform(*cfg.confidence_range), 3)
        agents.append(MockAgent(
            agent_id=f"{cfg.name}_{i}",
            answer=ans,
            confidence=conf,
            byzantine=(i < n_byz),
            accept_set=set(answers) if cfg.answer_diversity > 1 else {ans},
        ))
    return agents


def _elo_update(ra: float, rb: float, sa: float, k: float = 32.0) -> Tuple[float, float]:
    """Update ELO ratings. sa=1 if A wins, 0 if B wins, 0.5 draw."""
    ea = 1.0 / (1.0 + math.pow(10, (rb - ra) / 400.0))
    eb = 1.0 - ea
    return ra + k * (sa - ea), rb + k * ((1 - sa) - eb)


async def _run_match(
    team_a: TeamConfig,
    team_b: TeamConfig,
    task: str,
    threshold: float,
    seed: int,
) -> MatchResult:
    agents_a = _build_agents(team_a, seed)
    agents_b = _build_agents(team_b, seed + 1)

    engine_a = MBFTEngine(agents_a, threshold=threshold, max_rounds=4)
    engine_b = MBFTEngine(agents_b, threshold=threshold, max_rounds=4)

    result_a, result_b = await asyncio.gather(
        engine_a.run(task), engine_b.run(task)
    )

    a_committed = result_a is not None and result_a.committed
    b_committed = result_b is not None and result_b.committed
    a_rounds = len(engine_a.history)
    b_rounds = len(engine_b.history)
    a_agg = result_a.aggregate_weight if result_a else 0.0
    b_agg = result_b.aggregate_weight if result_b else 0.0

    # Winner logic: commit beats no-commit; if both commit, fewer rounds wins;
    # if tied on rounds, higher aggregate wins; else draw.
    winner: Optional[str] = None
    if a_committed and not b_committed:
        winner = team_a.name
    elif b_committed and not a_committed:
        winner = team_b.name
    elif a_committed and b_committed:
        if a_rounds < b_rounds:
            winner = team_a.name
        elif b_rounds < a_rounds:
            winner = team_b.name
        elif a_agg > b_agg + 0.01:
            winner = team_a.name
        elif b_agg > a_agg + 0.01:
            winner = team_b.name
    # else: both failed or perfectly tied → draw

    return MatchResult(
        team_a=team_a.name, team_b=team_b.name, task=task,
        winner=winner,
        a_committed=a_committed, b_committed=b_committed,
        a_rounds=a_rounds, b_rounds=b_rounds,
        a_aggregate=round(a_agg, 4), b_aggregate=round(b_agg, 4),
    )


# ── Tournament runner ────────────────────────────────────────────────────

async def run_tournament(
    teams: Optional[List[TeamConfig]] = None,
    n_rounds: int = 2,
    threshold: float = 2.0,
    seed: int = 42,
) -> Tuple[Dict[str, TeamStats], List[MatchResult]]:
    teams = teams or DEFAULT_TEAMS
    stats: Dict[str, TeamStats] = {t.name: TeamStats(name=t.name) for t in teams}
    all_matches: List[MatchResult] = []
    rng = random.Random(seed)

    for rd in range(n_rounds):
        pairs = list(itertools.combinations(teams, 2))
        rng.shuffle(pairs)
        task = TASKS[rd % len(TASKS)]

        for ta, tb in pairs:
            m = await _run_match(ta, tb, task, threshold, rng.randint(0, 10**6))
            all_matches.append(m)

            sa = stats[ta.name]
            sb = stats[tb.name]
            sa.total_matches += 1
            sb.total_matches += 1
            sa.total_aggregate += m.a_aggregate
            sb.total_aggregate += m.b_aggregate
            if m.a_committed:
                sa.commits += 1
            if m.b_committed:
                sb.commits += 1

            if m.winner == ta.name:
                score_a = 1.0
                sa.wins += 1
                sb.losses += 1
            elif m.winner == tb.name:
                score_a = 0.0
                sa.losses += 1
                sb.wins += 1
            else:
                score_a = 0.5
                sa.draws += 1
                sb.draws += 1

            sa.elo, sb.elo = _elo_update(sa.elo, sb.elo, score_a)
            sa.history.append(round(sa.elo, 1))
            sb.history.append(round(sb.elo, 1))

    return stats, all_matches


# ── Reports ──────────────────────────────────────────────────────────────

def _leaderboard(stats: Dict[str, TeamStats]) -> List[TeamStats]:
    return sorted(stats.values(), key=lambda s: s.elo, reverse=True)


def print_report(stats: Dict[str, TeamStats], matches: List[MatchResult]) -> None:
    lb = _leaderboard(stats)
    print("\n+==============================================================+")
    print("|             CONSENSUS TOURNAMENT ARENA                        |")
    print("+==============================================================+\n")

    print("-- LEADERBOARD ----------------------------------------------------")
    print(f"{'#':<3} {'Team':<20} {'ELO':>7} {'W':>4} {'L':>4} {'D':>4} {'Commits':>8} {'Avg Agg':>8}")
    print("-" * 62)
    for i, s in enumerate(lb, 1):
        avg_agg = s.total_aggregate / max(s.total_matches, 1)
        print(f"{i:<3} {s.name:<20} {s.elo:>7.1f} {s.wins:>4} {s.losses:>4} "
              f"{s.draws:>4} {s.commits:>8} {avg_agg:>8.2f}")

    print(f"\n-- MATCH LOG ({len(matches)} matches) ---------------------------------")
    for m in matches:
        w = m.winner or "DRAW"
        print(f"  {m.team_a} vs {m.team_b} -> {w}  "
              f"[{m.a_aggregate:.2f}/{m.b_aggregate:.2f}]  task={m.task[:30]}")


def export_json(stats: Dict[str, TeamStats], matches: List[MatchResult], path: str) -> None:
    data = {
        "leaderboard": [
            {
                "rank": i + 1, "team": s.name, "elo": round(s.elo, 1),
                "wins": s.wins, "losses": s.losses, "draws": s.draws,
                "commits": s.commits, "total_matches": s.total_matches,
                "avg_aggregate": round(s.total_aggregate / max(s.total_matches, 1), 4),
                "elo_history": s.history,
            }
            for i, s in enumerate(_leaderboard(stats))
        ],
        "matches": [
            {
                "team_a": m.team_a, "team_b": m.team_b, "task": m.task,
                "winner": m.winner, "a_committed": m.a_committed,
                "b_committed": m.b_committed, "a_rounds": m.a_rounds,
                "b_rounds": m.b_rounds, "a_aggregate": m.a_aggregate,
                "b_aggregate": m.b_aggregate,
            }
            for m in matches
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"JSON exported -> {path}")


def export_html(stats: Dict[str, TeamStats], matches: List[MatchResult], path: str) -> None:
    lb = _leaderboard(stats)
    rows = ""
    for i, s in enumerate(lb, 1):
        avg_agg = round(s.total_aggregate / max(s.total_matches, 1), 2)
        bar_w = max(5, int((s.elo - 1300) / 4))
        rows += f"""<tr>
          <td>{i}</td><td><b>{s.name}</b></td>
          <td><div class="bar" style="width:{bar_w}px">{s.elo:.1f}</div></td>
          <td>{s.wins}</td><td>{s.losses}</td><td>{s.draws}</td>
          <td>{s.commits}</td><td>{avg_agg}</td></tr>\n"""

    match_rows = ""
    for m in matches:
        w = m.winner or "DRAW"
        cls = "draw" if not m.winner else ""
        match_rows += (f'<tr class="{cls}"><td>{m.team_a}</td><td>{m.team_b}</td>'
                       f'<td>{m.task[:40]}</td><td><b>{w}</b></td>'
                       f'<td>{m.a_aggregate:.2f}</td><td>{m.b_aggregate:.2f}</td></tr>\n')

    # ELO history chart data
    chart_data = json.dumps({s.name: s.history for s in lb})

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>🏆 Consensus Tournament Arena</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px}}
h1{{text-align:center;font-size:1.8em;margin-bottom:6px;color:#58a6ff}}
.sub{{text-align:center;color:#8b949e;margin-bottom:24px}}
table{{width:100%;border-collapse:collapse;margin-bottom:24px}}
th{{background:#161b22;color:#58a6ff;padding:8px 12px;text-align:left;border-bottom:2px solid #30363d}}
td{{padding:6px 12px;border-bottom:1px solid #21262d}}
tr:hover{{background:#161b22}}
tr.draw td{{opacity:0.7}}
.bar{{background:linear-gradient(90deg,#238636,#58a6ff);color:#fff;padding:2px 8px;
      border-radius:4px;font-size:0.85em;display:inline-block;min-width:40px;text-align:right}}
.section{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:24px}}
canvas{{width:100%;max-height:300px;margin-top:12px}}
</style></head><body>
<h1>🏆 Consensus Tournament Arena</h1>
<p class="sub">mBFT agent team round-robin with ELO ratings</p>

<div class="section">
<h2 style="margin-bottom:12px">📊 Leaderboard</h2>
<table><tr><th>#</th><th>Team</th><th>ELO</th><th>W</th><th>L</th><th>D</th><th>Commits</th><th>Avg Agg</th></tr>
{rows}</table></div>

<div class="section">
<h2 style="margin-bottom:12px">📈 ELO History</h2>
<canvas id="eloChart"></canvas>
</div>

<div class="section">
<h2 style="margin-bottom:12px">⚔️ Match Log</h2>
<table><tr><th>Team A</th><th>Team B</th><th>Task</th><th>Winner</th><th>A Agg</th><th>B Agg</th></tr>
{match_rows}</table></div>

<script>
const data={chart_data};
const canvas=document.getElementById('eloChart');
const ctx=canvas.getContext('2d');
const colors=['#58a6ff','#f0883e','#238636','#bc8cff','#f778ba','#3fb950','#79c0ff','#d29922'];
function draw(){{
  canvas.width=canvas.clientWidth;canvas.height=300;
  const teams=Object.keys(data);
  let maxLen=0;teams.forEach(t=>{{if(data[t].length>maxLen)maxLen=data[t].length}});
  if(maxLen<2)return;
  let allVals=[];teams.forEach(t=>allVals.push(...data[t]));
  const minV=Math.min(...allVals)-20,maxV=Math.max(...allVals)+20;
  const W=canvas.width-60,H=canvas.height-40,ox=40,oy=10;
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  for(let v=Math.ceil(minV/50)*50;v<=maxV;v+=50){{
    const y=oy+H-(v-minV)/(maxV-minV)*H;
    ctx.beginPath();ctx.moveTo(ox,y);ctx.lineTo(ox+W,y);ctx.stroke();
    ctx.fillStyle='#8b949e';ctx.font='11px system-ui';ctx.fillText(v,2,y+4);
  }}
  teams.forEach((t,ti)=>{{
    const pts=data[t];ctx.strokeStyle=colors[ti%colors.length];ctx.lineWidth=2;
    ctx.beginPath();
    pts.forEach((v,i)=>{{
      const x=ox+i/(maxLen-1)*W,y=oy+H-(v-minV)/(maxV-minV)*H;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }});
    ctx.stroke();
    // label
    const ly=oy+H-(pts[pts.length-1]-minV)/(maxV-minV)*H;
    ctx.fillStyle=colors[ti%colors.length];ctx.font='11px system-ui';
    ctx.fillText(t,ox+W+4,ly+4);
  }});
}}
draw();window.addEventListener('resize',draw);
</script></body></html>"""
    Path(path).write_text(html, encoding="utf-8")
    print(f"HTML exported -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consensus Tournament Arena — round-robin mBFT team competition"
    )
    parser.add_argument("--teams", type=int, default=len(DEFAULT_TEAMS),
                        help="Number of teams (uses first N defaults)")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Tournament rounds (each = full round-robin)")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="mBFT commit threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--export", type=str, default=None,
                        help="Export path (.html or .json)")
    args = parser.parse_args()

    teams = DEFAULT_TEAMS[:args.teams]
    stats, matches = asyncio.run(
        run_tournament(teams, n_rounds=args.rounds, threshold=args.threshold, seed=args.seed)
    )

    print_report(stats, matches)

    if args.export:
        if args.export.endswith(".json"):
            export_json(stats, matches, args.export)
        else:
            export_html(stats, matches, args.export)


if __name__ == "__main__":
    main()
