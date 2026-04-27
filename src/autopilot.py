"""Consensus Autopilot — autonomous self-governing task queue processor.

The Autopilot runs a continuous loop of consensus rounds over a task queue,
autonomously adapting the protocol's behaviour based on observed outcomes:

- **Adaptive threshold** — raises θ after false commits, lowers it after
  stalls, keeping the sweet-spot between safety and liveness.
- **Agent quarantine** — benches agents whose reputation drops below a floor
  for a configurable cool-down period, then reinstates them on probation.
- **Health dashboard** — tracks commit rate, average rounds, stall streaks,
  and quarantine events in a live summary.
- **Pluggable task source** — accepts an async iterator so tasks can come
  from a queue, file, API, or stdin.

Usage::

    python -m src.autopilot                           # demo with mock tasks
    python -m src.autopilot --tasks tasks.txt         # one task per line
    python -m src.autopilot --export html -o dash.html  # HTML dashboard
    python -m src.autopilot --agents 7 --cycles 20    # larger demo

The Autopilot embodies the "agency" direction for mBFT: it doesn't just run
consensus — it *governs* the swarm, learning from each round and adjusting
policy without human intervention.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import RoundResult


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QuarantineRecord:
    """Tracks a quarantined agent."""
    agent_id: str
    quarantined_at: float
    release_at: float
    reason: str
    reputation_at_quarantine: float


@dataclass
class CycleOutcome:
    """Result of one autopilot cycle (one task through consensus)."""
    cycle_index: int
    task: str
    committed: bool
    solution: Optional[str]
    rounds_used: int
    aggregate_weight: float
    threshold_used: float
    threshold_after: float
    quarantined: List[str]
    reinstated: List[str]
    active_agents: int
    elapsed_s: float


@dataclass
class AutopilotHealth:
    """Running health metrics for the swarm."""
    total_cycles: int = 0
    commits: int = 0
    stalls: int = 0
    consecutive_stalls: int = 0
    max_stall_streak: int = 0
    total_rounds: int = 0
    quarantine_events: int = 0
    threshold_adjustments: int = 0
    history: List[CycleOutcome] = field(default_factory=list)

    @property
    def commit_rate(self) -> float:
        return self.commits / self.total_cycles if self.total_cycles else 0.0

    @property
    def avg_rounds(self) -> float:
        return self.total_rounds / self.total_cycles if self.total_cycles else 0.0


# ---------------------------------------------------------------------------
# Autopilot engine
# ---------------------------------------------------------------------------

class ConsensusAutopilot:
    """Autonomous swarm governor.

    Wraps MBFTEngine and adds adaptive threshold tuning, agent quarantine,
    and continuous health monitoring.
    """

    def __init__(
        self,
        agents: List[MockAgent],
        *,
        initial_threshold: float = 1.5,
        max_rounds: int = 4,
        quarantine_floor: float = 0.3,
        quarantine_cooldown_s: float = 60.0,
        threshold_up_step: float = 0.1,
        threshold_down_step: float = 0.05,
        threshold_min: float = 0.5,
        threshold_max: float = 4.0,
        stall_streak_limit: int = 3,
    ) -> None:
        self.all_agents = list(agents)
        self.threshold = initial_threshold
        self.max_rounds = max_rounds
        self.quarantine_floor = quarantine_floor
        self.quarantine_cooldown_s = quarantine_cooldown_s
        self.threshold_up_step = threshold_up_step
        self.threshold_down_step = threshold_down_step
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.stall_streak_limit = stall_streak_limit

        self._quarantine: Dict[str, QuarantineRecord] = {}
        self._reputation_overrides: Dict[str, float] = {}
        self.health = AutopilotHealth()

    # -- public API --------------------------------------------------------

    async def run_queue(self, tasks: AsyncIterator[str]) -> AutopilotHealth:
        """Process every task from the async iterator."""
        cycle = 0
        async for task in tasks:
            outcome = await self.run_one(task, cycle)
            self.health.history.append(outcome)
            cycle += 1
        return self.health

    async def run_one(self, task: str, cycle_index: int = 0) -> CycleOutcome:
        """Run a single task through consensus with adaptive governance."""
        t0 = time.monotonic()

        # 1. Reinstate any agents whose cooldown has expired
        reinstated = self._reinstate_agents()

        # 2. Build active agent list (exclude quarantined)
        active = [a for a in self.all_agents if a.id not in self._quarantine]
        if len(active) < 2:
            # Not enough agents — force-reinstate everyone
            self._quarantine.clear()
            active = list(self.all_agents)
            reinstated = [a.id for a in active]

        # 3. Run consensus
        engine = MBFTEngine(
            agents=active, threshold=self.threshold, max_rounds=self.max_rounds
        )
        result = await engine.run(task)
        committed = result is not None and result.committed

        # 4. Quarantine agents whose reputation dropped below floor
        quarantined: List[str] = []
        now = time.monotonic()
        for aid, rep in engine.reputation.items():
            if rep < self.quarantine_floor and aid not in self._quarantine:
                self._quarantine[aid] = QuarantineRecord(
                    agent_id=aid,
                    quarantined_at=now,
                    release_at=now + self.quarantine_cooldown_s,
                    reason=f"reputation {rep:.3f} < floor {self.quarantine_floor}",
                    reputation_at_quarantine=rep,
                )
                quarantined.append(aid)
                self.health.quarantine_events += 1

        # 5. Adaptive threshold
        old_threshold = self.threshold
        if committed:
            self.health.commits += 1
            self.health.consecutive_stalls = 0
        else:
            self.health.stalls += 1
            self.health.consecutive_stalls += 1
            self.health.max_stall_streak = max(
                self.health.max_stall_streak, self.health.consecutive_stalls
            )

        self._adapt_threshold(committed)
        if self.threshold != old_threshold:
            self.health.threshold_adjustments += 1

        # 6. Record metrics
        self.health.total_cycles += 1
        rounds_used = len(engine.history)
        self.health.total_rounds += rounds_used
        elapsed = time.monotonic() - t0

        return CycleOutcome(
            cycle_index=cycle_index,
            task=task,
            committed=committed,
            solution=result.committed_solution if result else None,
            rounds_used=rounds_used,
            aggregate_weight=result.aggregate_weight if result else 0.0,
            threshold_used=old_threshold,
            threshold_after=self.threshold,
            quarantined=quarantined,
            reinstated=reinstated,
            active_agents=len(active),
            elapsed_s=elapsed,
        )

    # -- internals ---------------------------------------------------------

    def _reinstate_agents(self) -> List[str]:
        now = time.monotonic()
        reinstated = []
        expired = [
            aid for aid, rec in self._quarantine.items()
            if now >= rec.release_at
        ]
        for aid in expired:
            del self._quarantine[aid]
            reinstated.append(aid)
        return reinstated

    def _adapt_threshold(self, committed: bool) -> None:
        if not committed and self.health.consecutive_stalls >= self.stall_streak_limit:
            # Swarm is stalling — ease the bar
            self.threshold = max(
                self.threshold_min,
                self.threshold - self.threshold_down_step,
            )
        elif committed and self.threshold < self.threshold_max:
            # Successful commit — tighten slightly for safety
            self.threshold = min(
                self.threshold_max,
                self.threshold + self.threshold_up_step,
            )

    @property
    def quarantine_list(self) -> List[QuarantineRecord]:
        return list(self._quarantine.values())

    def status_summary(self) -> str:
        h = self.health
        lines = [
            "=" * 60,
            "  CONSENSUS AUTOPILOT - STATUS",
            "=" * 60,
            f"  Cycles completed:    {h.total_cycles}",
            f"  Commit rate:         {h.commit_rate:.1%}",
            f"  Avg rounds/cycle:    {h.avg_rounds:.2f}",
            f"  Current threshold:   {self.threshold:.3f}",
            f"  Stall streak:        {h.consecutive_stalls} (max {h.max_stall_streak})",
            f"  Quarantine events:   {h.quarantine_events}",
            f"  Threshold adjusts:   {h.threshold_adjustments}",
            f"  Agents active:       {len(self.all_agents) - len(self._quarantine)}/{len(self.all_agents)}",
        ]
        if self._quarantine:
            lines.append("  Quarantined:")
            for rec in self._quarantine.values():
                lines.append(f"    - {rec.agent_id}: {rec.reason}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task sources
# ---------------------------------------------------------------------------

async def tasks_from_list(items: List[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


async def tasks_from_file(path: str) -> AsyncIterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield stripped


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def render_dashboard(autopilot: ConsensusAutopilot) -> str:
    h = autopilot.health
    cycles_json = json.dumps([
        {
            "cycle": c.cycle_index,
            "task": c.task[:50],
            "committed": c.committed,
            "solution": c.solution,
            "rounds": c.rounds_used,
            "weight": round(c.aggregate_weight, 3),
            "threshold": round(c.threshold_used, 3),
            "threshold_after": round(c.threshold_after, 3),
            "active": c.active_agents,
            "quarantined": c.quarantined,
            "reinstated": c.reinstated,
            "elapsed": round(c.elapsed_s, 4),
        }
        for c in h.history
    ])

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>mBFT Autopilot Dashboard</title>
<style>
  :root {{ --bg:#0d1117; --fg:#c9d1d9; --accent:#58a6ff; --red:#f85149;
           --green:#3fb950; --yellow:#d29922; --card:#161b22; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,sans-serif;
          background:var(--bg); color:var(--fg); padding:2rem; }}
  h1 {{ color:var(--accent); margin-bottom:.3rem; }}
  .sub {{ color:#8b949e; margin-bottom:2rem; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
           gap:1rem; margin-bottom:2rem; }}
  .card {{ background:var(--card); border-radius:8px; padding:1.2rem; }}
  .card h3 {{ font-size:.8rem; color:#8b949e; text-transform:uppercase; }}
  .card .v {{ font-size:1.8rem; font-weight:700; margin-top:.2rem; }}
  .g {{ color:var(--green); }} .r {{ color:var(--red); }} .y {{ color:var(--yellow); }}
  canvas {{ background:var(--card); border-radius:8px; margin-bottom:2rem; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card);
           border-radius:8px; overflow:hidden; }}
  th,td {{ padding:.5rem .8rem; text-align:left; border-bottom:1px solid #21262d; }}
  th {{ background:#21262d; font-size:.8rem; text-transform:uppercase; color:#8b949e; }}
  .ok {{ color:var(--green); }} .fail {{ color:var(--red); }}
</style></head><body>
<h1>🤖 mBFT Consensus Autopilot</h1>
<p class="sub">Self-governing swarm task processor — adaptive threshold, agent quarantine, health monitoring</p>

<div class="grid">
  <div class="card"><h3>Cycles</h3><div class="v">{h.total_cycles}</div></div>
  <div class="card"><h3>Commit Rate</h3>
    <div class="v {'g' if h.commit_rate >= .7 else 'y' if h.commit_rate >= .4 else 'r'}">{h.commit_rate:.0%}</div></div>
  <div class="card"><h3>Avg Rounds</h3><div class="v">{h.avg_rounds:.1f}</div></div>
  <div class="card"><h3>Threshold</h3><div class="v">{autopilot.threshold:.2f}</div></div>
  <div class="card"><h3>Quarantines</h3>
    <div class="v {'r' if h.quarantine_events else 'g'}">{h.quarantine_events}</div></div>
  <div class="card"><h3>Threshold Adj</h3><div class="v y">{h.threshold_adjustments}</div></div>
</div>

<h2 style="margin-bottom:1rem">📈 Threshold Adaptation</h2>
<canvas id="tChart" width="800" height="250"></canvas>

<h2 style="margin-bottom:1rem">📋 Cycle Log</h2>
<table>
  <thead><tr><th>#</th><th>Task</th><th>Result</th><th>Solution</th><th>Rounds</th>
    <th>Weight</th><th>θ Used</th><th>θ After</th><th>Active</th><th>Events</th></tr></thead>
  <tbody id="tb"></tbody>
</table>

<script>
const C = {cycles_json};
const tb = document.getElementById('tb');
C.forEach(c => {{
  const events = [];
  if (c.quarantined.length) events.push('🚫 ' + c.quarantined.join(','));
  if (c.reinstated.length) events.push('♻️ ' + c.reinstated.join(','));
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{c.cycle}}</td><td>${{c.task}}</td>
    <td class="${{c.committed?'ok':'fail'}}">${{c.committed?'✅':'❌'}}</td>
    <td>${{c.solution||'—'}}</td><td>${{c.rounds}}</td><td>${{c.weight}}</td>
    <td>${{c.threshold}}</td><td>${{c.threshold_after}}</td>
    <td>${{c.active}}</td><td>${{events.join(' ')||'—'}}</td>`;
  tb.appendChild(tr);
}});

// Threshold chart
const cv = document.getElementById('tChart');
const ctx = cv.getContext('2d');
const W = cv.width, H = cv.height;
const p = {{l:50,r:20,t:15,b:35}};
const pw = W-p.l-p.r, ph = H-p.t-p.b;
if (C.length > 1) {{
  const vals = C.map(c => c.threshold_after);
  const mn = Math.min(...vals) - 0.1, mx = Math.max(...vals) + 0.1;
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2;
  ctx.beginPath();
  C.forEach((c,i) => {{
    const x = p.l + (i/(C.length-1))*pw;
    const y = p.t + ph - ((c.threshold_after-mn)/(mx-mn))*ph;
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    ctx.fillStyle = c.committed ? '#3fb950' : '#f85149';
    ctx.fillRect(x-3, y-3, 6, 6);
  }});
  ctx.stroke();
  ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Cycle', W/2, H-5);
  ctx.save(); ctx.translate(12,H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Threshold (θ)',0,0); ctx.restore();
}}
</script></body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_demo_agents(n: int) -> List[MockAgent]:
    """Build a mixed swarm for the demo.

    Composition: ~70% honest, ~15% noisy (low confidence), ~15% byzantine.
    This produces a realistic mix where most tasks commit but some stall,
    demonstrating the adaptive threshold and quarantine systems.
    """
    import random
    random.seed(42)
    agents = []
    honest_count = max(1, int(n * 0.7))
    noisy_count = max(0, int(n * 0.15))
    byz_count = n - honest_count - noisy_count

    for i in range(honest_count):
        agents.append(MockAgent(
            f"a{i+1}", answer="correct",
            confidence=round(random.uniform(0.6, 0.95), 2),
        ))
    for i in range(noisy_count):
        idx = honest_count + i
        agents.append(MockAgent(
            f"a{idx+1}", answer=f"noise-{idx}",
            confidence=round(random.uniform(0.2, 0.4), 2),
        ))
    for i in range(byz_count):
        idx = honest_count + noisy_count + i
        agents.append(MockAgent(
            f"a{idx+1}", answer=f"byz-{idx}",
            confidence=round(random.uniform(0.7, 0.99), 2),
            byzantine=True,
        ))
    random.shuffle(agents)
    return agents


_DEMO_TASKS = [
    "What is 2 + 2?",
    "Summarize the theory of relativity in one sentence.",
    "Is P = NP?",
    "What caused the 2008 financial crisis?",
    "Translate 'hello world' to Mandarin.",
    "What is the capital of Australia?",
    "Explain quantum entanglement simply.",
    "What year did the Berlin Wall fall?",
    "Largest prime under 100?",
    "Define 'metacognition' in one line.",
]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Autopilot — autonomous swarm governor"
    )
    parser.add_argument(
        "--agents", "-n", type=int, default=7,
        help="Number of agents in the swarm (default: 7)",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=1.5,
        help="Initial consensus threshold θ (default: 1.5)",
    )
    parser.add_argument(
        "--cycles", "-c", type=int, default=10,
        help="Number of demo cycles (ignored with --tasks)",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Path to a task file (one task per line)",
    )
    parser.add_argument(
        "--quarantine-floor", type=float, default=0.3,
        help="Reputation floor for quarantine (default: 0.3)",
    )
    parser.add_argument(
        "--quarantine-cooldown", type=float, default=30.0,
        help="Quarantine cooldown in seconds (default: 30)",
    )
    parser.add_argument(
        "--export", choices=["json", "html"],
        help="Export report as JSON or HTML dashboard",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file path",
    )
    args = parser.parse_args()

    agents = _build_demo_agents(args.agents)
    pilot = ConsensusAutopilot(
        agents,
        initial_threshold=args.threshold,
        quarantine_floor=args.quarantine_floor,
        quarantine_cooldown_s=args.quarantine_cooldown,
    )

    # Build task source
    if args.tasks:
        task_iter = tasks_from_file(args.tasks)
    else:
        task_iter = tasks_from_list(_DEMO_TASKS[:args.cycles])

    # Run
    await pilot.run_queue(task_iter)

    # Output
    if args.export == "json":
        data = {
            "total_cycles": pilot.health.total_cycles,
            "commit_rate": round(pilot.health.commit_rate, 3),
            "avg_rounds": round(pilot.health.avg_rounds, 3),
            "final_threshold": round(pilot.threshold, 3),
            "quarantine_events": pilot.health.quarantine_events,
            "threshold_adjustments": pilot.health.threshold_adjustments,
            "cycles": [
                {
                    "cycle": c.cycle_index,
                    "task": c.task,
                    "committed": c.committed,
                    "solution": c.solution,
                    "rounds": c.rounds_used,
                    "threshold_after": round(c.threshold_after, 3),
                    "quarantined": c.quarantined,
                    "reinstated": c.reinstated,
                }
                for c in pilot.health.history
            ],
        }
        out = json.dumps(data, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(out)
            print(f"JSON report written to {args.output}")
        else:
            print(out)
    elif args.export == "html":
        html = render_dashboard(pilot)
        path = args.output or "autopilot_dashboard.html"
        with open(path, "w") as f:
            f.write(html)
        print(f"Dashboard written to {path}")
    else:
        import io, sys as _sys
        out = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
        _p = lambda *a, **kw: print(*a, **kw, file=out)
        _p(pilot.status_summary())
        _p()
        for c in pilot.health.history:
            status = "COMMIT" if c.committed else "STALL"
            events = ""
            if c.quarantined:
                events += f" quarantined={c.quarantined}"
            if c.reinstated:
                events += f" reinstated={c.reinstated}"
            _p(
                f"  [{c.cycle_index:>2}] {status}  t={c.threshold_used:.2f}->{c.threshold_after:.2f}  "
                f"agents={c.active_agents}  rounds={c.rounds_used}  "
                f"{c.task[:40]}{events}"
            )
        out.flush()


if __name__ == "__main__":
    asyncio.run(main())
