"""Network Partition Simulator for mBFT.

Simulates network partitions during consensus to test how the protocol handles
split-brain scenarios. Agents in different partitions cannot see each other's
votes, leading to partial quorums and potential consensus failures.

Features:
- Random and targeted partition strategies
- Partition healing with configurable timing
- Split-brain detection and analysis
- Interactive HTML report with partition timeline

Usage::

    python -m src.partition                        # default demo
    python -m src.partition --agents 7 --partitions 3
    python -m src.partition --strategy targeted --byzantine 2
    python -m src.partition --heal-after 2 --report partition_report.html
"""
from __future__ import annotations

import argparse
import asyncio
import html
import itertools
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import RoundResult


@dataclass
class Partition:
    """A network partition: a subset of agents that can communicate."""
    partition_id: int
    agent_ids: Set[str]
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"P{self.partition_id}"


@dataclass
class PartitionEvent:
    """Records a partition event for timeline visualization."""
    round_index: int
    event_type: str  # "split", "heal", "detect"
    partitions: List[Set[str]]
    description: str


@dataclass
class PartitionResult:
    """Results from running mBFT under a specific partition configuration."""
    partition_config: List[Partition]
    sub_results: Dict[int, Optional[RoundResult]]  # partition_id -> result
    split_brain: bool
    conflicting_solutions: List[str]
    quorum_achieved: Dict[int, bool]
    events: List[PartitionEvent] = field(default_factory=list)


class PartitionStrategy:
    """Strategies for creating network partitions."""

    @staticmethod
    def random_split(agent_ids: List[str], num_partitions: int = 2,
                     seed: Optional[int] = None) -> List[Partition]:
        """Randomly assign agents to partitions."""
        rng = random.Random(seed)
        shuffled = list(agent_ids)
        rng.shuffle(shuffled)
        partitions = []
        chunk_size = max(1, len(shuffled) // num_partitions)
        for i in range(num_partitions):
            start = i * chunk_size
            end = start + chunk_size if i < num_partitions - 1 else len(shuffled)
            ids = set(shuffled[start:end])
            if ids:
                partitions.append(Partition(partition_id=i, agent_ids=ids))
        return partitions

    @staticmethod
    def isolate_leader(agents: List[MockAgent], threshold: float) -> List[Partition]:
        """Isolate the likely leader (highest confidence) from the rest."""
        sorted_agents = sorted(agents, key=lambda a: a.confidence, reverse=True)
        leader_id = sorted_agents[0].id
        return [
            Partition(0, {leader_id}, label="Leader-Isolated"),
            Partition(1, {a.id for a in sorted_agents[1:]}, label="Majority"),
        ]

    @staticmethod
    def isolate_byzantine(agents: List[MockAgent]) -> List[Partition]:
        """Put Byzantine agents in their own partition."""
        byz = {a.id for a in agents if getattr(a, 'byzantine', False)}
        honest = {a.id for a in agents if not getattr(a, 'byzantine', False)}
        partitions = []
        if byz:
            partitions.append(Partition(0, byz, label="Byzantine"))
        if honest:
            partitions.append(Partition(1, honest, label="Honest"))
        return partitions if partitions else [Partition(0, {a.id for a in agents})]

    @staticmethod
    def minority_split(agents: List[MockAgent]) -> List[Partition]:
        """Split into minority (1/3) and majority (2/3) partitions."""
        ids = [a.id for a in agents]
        split_point = max(1, len(ids) // 3)
        return [
            Partition(0, set(ids[:split_point]), label="Minority"),
            Partition(1, set(ids[split_point:]), label="Majority"),
        ]


class NetworkPartitionSimulator:
    """Runs mBFT consensus under simulated network partitions."""

    def __init__(self, agents: List[MockAgent], threshold: float = 1.5,
                 max_rounds: int = 4, heal_after: Optional[int] = None):
        self.agents = agents
        self.threshold = threshold
        self.max_rounds = max_rounds
        self.heal_after = heal_after
        self._agent_map = {a.id: a for a in agents}

    async def simulate(self, partitions: List[Partition],
                       task: str = "What is the answer?") -> PartitionResult:
        """Run consensus independently in each partition, then analyze."""
        events: List[PartitionEvent] = []
        events.append(PartitionEvent(
            round_index=0, event_type="split",
            partitions=[p.agent_ids for p in partitions],
            description=f"Network split into {len(partitions)} partitions: "
                        + ", ".join(f"{p.label}({len(p.agent_ids)})" for p in partitions)
        ))

        sub_results: Dict[int, Optional[RoundResult]] = {}
        quorum_achieved: Dict[int, bool] = {}

        for p in partitions:
            partition_agents = [self._agent_map[aid] for aid in p.agent_ids
                                if aid in self._agent_map]
            if not partition_agents:
                sub_results[p.partition_id] = None
                quorum_achieved[p.partition_id] = False
                continue

            engine = MBFTEngine(
                agents=partition_agents,
                threshold=self.threshold,
                max_rounds=self.max_rounds,
            )
            result = await engine.run(task)
            sub_results[p.partition_id] = result
            quorum_achieved[p.partition_id] = result is not None and result.committed

        # Detect split-brain
        committed_solutions = []
        for pid, res in sub_results.items():
            if res and res.committed:
                committed_solutions.append(res.committed_solution)

        split_brain = len(set(committed_solutions)) > 1
        if split_brain:
            events.append(PartitionEvent(
                round_index=0, event_type="detect",
                partitions=[p.agent_ids for p in partitions],
                description=f"SPLIT-BRAIN DETECTED: conflicting solutions {committed_solutions}"
            ))

        # Simulate healing if configured
        if self.heal_after is not None:
            events.append(PartitionEvent(
                round_index=self.heal_after, event_type="heal",
                partitions=[{a.id for a in self.agents}],
                description="Network healed — all agents reconnected"
            ))
            # Re-run with full network
            full_engine = MBFTEngine(
                agents=self.agents,
                threshold=self.threshold,
                max_rounds=self.max_rounds,
            )
            healed_result = await full_engine.run(task)
            sub_results[-1] = healed_result  # -1 = healed network
            quorum_achieved[-1] = healed_result is not None and healed_result.committed

        return PartitionResult(
            partition_config=partitions,
            sub_results=sub_results,
            split_brain=split_brain,
            conflicting_solutions=committed_solutions,
            quorum_achieved=quorum_achieved,
            events=events,
        )

    async def sweep_partitions(self, task: str = "What is the answer?",
                               strategies: Optional[List[str]] = None
                               ) -> List[Tuple[str, PartitionResult]]:
        """Run multiple partition strategies and compare results."""
        if strategies is None:
            strategies = ["random", "isolate_leader", "isolate_byzantine", "minority"]

        results = []
        for strategy in strategies:
            partitions = self._get_partitions(strategy)
            result = await self.simulate(partitions, task)
            results.append((strategy, result))
        return results

    def _get_partitions(self, strategy: str) -> List[Partition]:
        if strategy == "random":
            return PartitionStrategy.random_split([a.id for a in self.agents], seed=42)
        elif strategy == "isolate_leader":
            return PartitionStrategy.isolate_leader(self.agents, self.threshold)
        elif strategy == "isolate_byzantine":
            return PartitionStrategy.isolate_byzantine(self.agents)
        elif strategy == "minority":
            return PartitionStrategy.minority_split(self.agents)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def generate_html_report(sweep_results: List[Tuple[str, PartitionResult]],
                         agents: List[MockAgent]) -> str:
    """Generate an interactive HTML report with partition analysis."""
    rows = []
    for strategy, result in sweep_results:
        committed_parts = sum(1 for v in result.quorum_achieved.values() if v)
        total_parts = len(result.partition_config)
        solutions = ", ".join(result.conflicting_solutions) if result.conflicting_solutions else "None"
        brain_badge = ('<span style="color:#e74c3c;font-weight:bold">⚠ SPLIT-BRAIN</span>'
                       if result.split_brain
                       else '<span style="color:#27ae60">✓ Safe</span>')

        partition_detail = ""
        for p in result.partition_config:
            r = result.sub_results.get(p.partition_id)
            status = "Committed" if r and r.committed else "No consensus"
            sol = r.committed_solution if r and r.committed else "—"
            partition_detail += (
                f'<div style="margin:4px 0;padding:6px 10px;background:#f8f9fa;border-radius:4px;">'
                f'<strong>{html.escape(p.label)}</strong> '
                f'({", ".join(sorted(p.agent_ids))}) → '
                f'{status} <code>{html.escape(str(sol))}</code></div>'
            )

        events_html = ""
        for ev in result.events:
            color = {"split": "#e67e22", "heal": "#27ae60", "detect": "#e74c3c"}.get(ev.event_type, "#666")
            events_html += (
                f'<div style="border-left:3px solid {color};padding:4px 8px;margin:4px 0;">'
                f'<small>Round {ev.round_index}</small> '
                f'<strong style="color:{color}">{ev.event_type.upper()}</strong> '
                f'{html.escape(ev.description)}</div>'
            )

        rows.append(f"""
        <div class="card" style="margin:16px 0;padding:16px;border:1px solid #ddd;border-radius:8px;">
            <h3 style="margin-top:0;">{html.escape(strategy)} {brain_badge}</h3>
            <p>Partitions with quorum: {committed_parts}/{total_parts} | Solutions: <code>{html.escape(solutions)}</code></p>
            <div style="margin:8px 0;">{partition_detail}</div>
            <details><summary>Timeline Events</summary>{events_html}</details>
        </div>""")

    agent_rows = "".join(
        f'<tr><td>{html.escape(a.id)}</td><td>{a.confidence:.2f}</td>'
        f'<td>{"⚠ Byzantine" if getattr(a, "byzantine", False) else "Honest"}</td>'
        f'<td><code>{html.escape(a.answer)}</code></td></tr>'
        for a in agents
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>mBFT Network Partition Analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; background: #fafbfc; color: #24292e; }}
h1 {{ border-bottom: 2px solid #e1e4e8; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #d1d5da; padding: 8px 12px; text-align: left; }}
th {{ background: #f1f3f5; }}
.card {{ transition: box-shadow 0.2s; }}
.card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
details {{ margin-top: 8px; }}
summary {{ cursor: pointer; color: #0366d6; }}
</style></head><body>
<h1>🔀 mBFT Network Partition Analysis</h1>
<p>Testing consensus resilience under {len(sweep_results)} partition strategies with {len(agents)} agents.</p>

<h2>Agent Fleet</h2>
<table><tr><th>ID</th><th>Confidence</th><th>Type</th><th>Answer</th></tr>{agent_rows}</table>

<h2>Partition Strategies</h2>
{"".join(rows)}

<h2>Key Findings</h2>
<ul>
{"".join(f'<li><strong>{s}</strong>: {"⚠ Split-brain — conflicting commits across partitions" if r.split_brain else "✓ No split-brain" + (f" — {sum(1 for v in r.quorum_achieved.values() if v)} partition(s) reached consensus" if any(r.quorum_achieved.values()) else " — no partition reached consensus")}</li>' for s, r in sweep_results)}
</ul>

<footer style="margin-top:2em;color:#6a737d;font-size:0.85em;">
Generated by <code>python -m src.partition</code> — mBFT Network Partition Simulator
</footer></body></html>"""


def build_agents(num_agents: int = 5, num_byzantine: int = 1,
                 seed: int = 42) -> List[MockAgent]:
    """Build a mixed swarm of honest and Byzantine agents."""
    rng = random.Random(seed)
    agents = []
    answers = ["42", "42", "42", "41", "999"]  # majority agree on 42
    for i in range(num_agents):
        is_byz = i >= num_agents - num_byzantine
        ans = answers[i] if i < len(answers) else "42"
        conf = round(rng.uniform(0.5, 0.95), 2) if not is_byz else round(rng.uniform(0.8, 0.99), 2)
        agents.append(MockAgent(
            agent_id=f"a{i+1}", answer=ans, confidence=conf,
            byzantine=is_byz,
        ))
    return agents


async def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Network Partition Simulator — test consensus under network splits"
    )
    parser.add_argument("--agents", type=int, default=5, help="Number of agents (default: 5)")
    parser.add_argument("--byzantine", type=int, default=1, help="Number of Byzantine agents (default: 1)")
    parser.add_argument("--partitions", type=int, default=2, help="Number of partitions for random split")
    parser.add_argument("--strategy", choices=["random", "isolate_leader", "isolate_byzantine", "minority", "sweep"],
                        default="sweep", help="Partition strategy (default: sweep all)")
    parser.add_argument("--threshold", type=float, default=1.5, help="Consensus threshold (default: 1.5)")
    parser.add_argument("--heal-after", type=int, default=None, help="Heal partition after N rounds")
    parser.add_argument("--task", default="What is the answer to life?", help="Task prompt")
    parser.add_argument("--report", type=str, default=None, help="Write HTML report to file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    opts = parser.parse_args(args)

    agents = build_agents(opts.agents, opts.byzantine, opts.seed)
    sim = NetworkPartitionSimulator(
        agents=agents, threshold=opts.threshold,
        max_rounds=4, heal_after=opts.heal_after,
    )

    if opts.strategy == "sweep":
        results = await sim.sweep_partitions(opts.task)
    else:
        partitions = sim._get_partitions(opts.strategy)
        result = await sim.simulate(partitions, opts.task)
        results = [(opts.strategy, result)]

    if opts.json:
        out = []
        for strategy, result in results:
            out.append({
                "strategy": strategy,
                "split_brain": result.split_brain,
                "conflicting_solutions": result.conflicting_solutions,
                "partitions": [
                    {"id": p.partition_id, "label": p.label,
                     "agents": sorted(p.agent_ids),
                     "quorum": result.quorum_achieved.get(p.partition_id, False)}
                    for p in result.partition_config
                ],
                "events": [
                    {"round": e.round_index, "type": e.event_type, "desc": e.description}
                    for e in result.events
                ],
            })
        print(json.dumps(out, indent=2))
    else:
        print("=" * 60)
        print("mBFT Network Partition Simulator")
        print(f"Agents: {len(agents)} ({opts.byzantine} Byzantine) | Threshold: {opts.threshold}")
        print("=" * 60)
        for strategy, result in results:
            print(f"\n> Strategy: {strategy}")
            brain_flag = "!! SPLIT-BRAIN" if result.split_brain else "OK Safe"
            print(f"  Status: {brain_flag}")
            for p in result.partition_config:
                r = result.sub_results.get(p.partition_id)
                status = "Committed" if r and r.committed else "No consensus"
                sol = r.committed_solution if r and r.committed else "-"
                print(f"  {p.label} ({', '.join(sorted(p.agent_ids))}): {status} -> {sol}")
            for ev in result.events:
                print(f"  [{ev.event_type.upper()}] Round {ev.round_index}: {ev.description}")

        # Healed result if present
        for _, result in results:
            if -1 in result.sub_results:
                healed = result.sub_results[-1]
                if healed and healed.committed:
                    print(f"\n[OK] Healed network reached consensus: {healed.committed_solution}")
                else:
                    print("\n[FAIL] Healed network still failed to reach consensus")

    if opts.report:
        report_html = generate_html_report(results, agents)
        with open(opts.report, "w", encoding="utf-8") as f:
            f.write(report_html)
        print(f"\n📄 HTML report written to {opts.report}")


if __name__ == "__main__":
    asyncio.run(main())
