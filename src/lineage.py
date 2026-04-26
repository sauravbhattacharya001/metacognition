"""Consensus Lineage Tracker — causal ancestry of mBFT decisions.

Traces how proposals evolve across rounds, identifies which agent ideas
influenced the final committed solution, builds an ancestry graph, and
detects innovation vs. convergence dynamics.

Run standalone::

    python -m src.lineage --agents 5 --byzantine 1 --output lineage.html
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import math
import random
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import Proposal, RoundResult


# ---------------------------------------------------------------------------
# Instrumented engine to capture all proposals per round
# ---------------------------------------------------------------------------

class InstrumentedEngine(MBFTEngine):
    """MBFTEngine subclass that records every proposal from every round."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.all_proposals: List[List[Proposal]] = []

    async def _gather_proposals(self, task_prompt: str) -> List[Proposal]:
        proposals = await super()._gather_proposals(task_prompt)
        self.all_proposals.append(list(proposals))
        return proposals


# ---------------------------------------------------------------------------
# Lineage analysis
# ---------------------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    """Text similarity via SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()


class LineageNode:
    """A proposal in the lineage graph."""
    __slots__ = ("proposal", "round_idx", "parents", "children",
                 "influence_score", "is_innovation", "is_on_winning_chain")

    def __init__(self, proposal: Proposal, round_idx: int) -> None:
        self.proposal = proposal
        self.round_idx = round_idx
        self.parents: List[LineageNode] = []
        self.children: List[LineageNode] = []
        self.influence_score: float = 0.0
        self.is_innovation: bool = False
        self.is_on_winning_chain: bool = False


class ConsensusLineageTracker:
    """Analyse the causal lineage of mBFT consensus runs."""

    def __init__(self, similarity_threshold: float = 0.4) -> None:
        self.sim_thresh = similarity_threshold
        self.nodes: List[LineageNode] = []
        self.nodes_by_round: Dict[int, List[LineageNode]] = defaultdict(list)
        self.committed_node: Optional[LineageNode] = None

    # -- build ---------------------------------------------------------------

    def build(
        self,
        engine: InstrumentedEngine,
        results: List[RoundResult],
    ) -> None:
        """Construct lineage graph from an instrumented engine run."""
        # Create nodes
        for ri, proposals in enumerate(engine.all_proposals):
            for p in proposals:
                node = LineageNode(p, ri)
                self.nodes.append(node)
                self.nodes_by_round[ri].append(node)

        # Build edges based on similarity across consecutive rounds
        for ri in sorted(self.nodes_by_round.keys()):
            if ri == 0:
                continue
            for child in self.nodes_by_round[ri]:
                for parent in self.nodes_by_round[ri - 1]:
                    sim = _sim(parent.proposal.solution, child.proposal.solution)
                    if sim >= self.sim_thresh:
                        child.parents.append(parent)
                        parent.children.append(child)

        # Identify committed node
        for r in reversed(results):
            if r.committed:
                for node in self.nodes_by_round.get(r.round_index, []):
                    if node.proposal.agent_id == r.leader_id:
                        self.committed_node = node
                        break
                if self.committed_node:
                    break

        self._compute_influence(engine)
        self._detect_innovations()
        self._trace_winning_chain()

    # -- metrics -------------------------------------------------------------

    def _compute_influence(self, engine: InstrumentedEngine) -> None:
        if not self.committed_node:
            return
        final_sol = self.committed_node.proposal.solution
        rep = engine.reputation
        for node in self.nodes:
            sim = _sim(node.proposal.solution, final_sol)
            r = rep.get(node.proposal.agent_id, 1.0)
            node.influence_score = sim * node.proposal.confidence * r

    def _detect_innovations(self) -> None:
        for ri in sorted(self.nodes_by_round.keys()):
            for node in self.nodes_by_round[ri]:
                if ri == 0:
                    # First-round proposals are all innovations by definition
                    node.is_innovation = True
                    continue
                max_sim = max(
                    (_sim(node.proposal.solution, prev.proposal.solution)
                     for prev in self.nodes_by_round[ri - 1]),
                    default=0.0,
                )
                node.is_innovation = max_sim < self.sim_thresh

    def _trace_winning_chain(self) -> None:
        """Walk backwards from the committed node along strongest parent."""
        cur = self.committed_node
        while cur:
            cur.is_on_winning_chain = True
            if not cur.parents:
                break
            cur = max(cur.parents,
                      key=lambda p: _sim(p.proposal.solution,
                                         cur.proposal.solution))  # type: ignore[arg-type]

    # -- convergence ---------------------------------------------------------

    def convergence_per_round(self) -> List[Tuple[int, float]]:
        """Average pairwise similarity within each round."""
        result = []
        for ri in sorted(self.nodes_by_round.keys()):
            nodes = self.nodes_by_round[ri]
            if len(nodes) < 2:
                result.append((ri, 1.0))
                continue
            pairs = 0
            total = 0.0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    total += _sim(nodes[i].proposal.solution,
                                  nodes[j].proposal.solution)
                    pairs += 1
            result.append((ri, total / pairs if pairs else 0.0))
        return result

    # -- agent influence ranking ---------------------------------------------

    def agent_influence(self) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = defaultdict(float)
        for node in self.nodes:
            scores[node.proposal.agent_id] = max(
                scores[node.proposal.agent_id], node.influence_score)
        return sorted(scores.items(), key=lambda x: -x[1])

    # -- winning chain -------------------------------------------------------

    def winning_chain(self) -> List[LineageNode]:
        chain = [n for n in self.nodes if n.is_on_winning_chain]
        chain.sort(key=lambda n: n.round_idx)
        return chain

    # -- summary -------------------------------------------------------------

    def text_summary(self) -> str:
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  CONSENSUS LINEAGE TRACKER")
        lines.append("=" * 60)

        # Winning chain
        chain = self.winning_chain()
        if chain:
            lines.append("\n📜 Winning Lineage Chain:")
            for n in chain:
                marker = "✅" if n == self.committed_node else "→"
                lines.append(
                    f"  {marker} Round {n.round_idx} | {n.proposal.agent_id} | "
                    f"conf={n.proposal.confidence:.2f} | "
                    f"\"{n.proposal.solution[:60]}\"")
        else:
            lines.append("\n⚠️  No committed solution — lineage chain empty.")

        # Agent influence
        lines.append("\n🏆 Agent Influence Ranking:")
        for rank, (aid, score) in enumerate(self.agent_influence(), 1):
            bar = "█" * int(score * 20)
            lines.append(f"  {rank}. {aid:>12s}  {score:.3f}  {bar}")

        # Innovation count
        lines.append("\n💡 Innovation Count per Round:")
        for ri in sorted(self.nodes_by_round.keys()):
            innov = sum(1 for n in self.nodes_by_round[ri] if n.is_innovation)
            lines.append(f"  Round {ri}: {innov} novel proposal(s)")

        # Convergence
        conv = self.convergence_per_round()
        lines.append("\n📈 Convergence Trend:")
        for ri, sim in conv:
            bar = "▓" * int(sim * 30)
            lines.append(f"  Round {ri}: {sim:.3f} {bar}")
        if len(conv) >= 2:
            delta = conv[-1][1] - conv[0][1]
            trend = "↗ Increasing" if delta > 0.05 else "↘ Decreasing" if delta < -0.05 else "→ Stable"
            lines.append(f"  Trend: {trend} (Δ={delta:+.3f})")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    # -- JSON export ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        nodes_data = []
        for n in self.nodes:
            nodes_data.append({
                "round": n.round_idx,
                "agent": n.proposal.agent_id,
                "solution": n.proposal.solution,
                "confidence": n.proposal.confidence,
                "influence": round(n.influence_score, 4),
                "innovation": n.is_innovation,
                "winning_chain": n.is_on_winning_chain,
                "proposal_id": n.proposal.proposal_id,
            })
        edges = []
        for n in self.nodes:
            for p in n.parents:
                edges.append({
                    "from": p.proposal.proposal_id,
                    "to": n.proposal.proposal_id,
                    "similarity": round(_sim(p.proposal.solution,
                                             n.proposal.solution), 3),
                })
        return {
            "nodes": nodes_data,
            "edges": edges,
            "agent_influence": self.agent_influence(),
            "convergence": self.convergence_per_round(),
            "committed": self.committed_node.proposal.proposal_id if self.committed_node else None,
        }

    # -- HTML report ---------------------------------------------------------

    def html_report(self) -> str:
        data = self.to_dict()
        conv = data["convergence"]
        influence = data["agent_influence"]
        chain_ids = {n.proposal.proposal_id for n in self.winning_chain()}
        innovations = [n for n in data["nodes"] if n["innovation"]]

        # Assign colors per agent
        agents = sorted({n["agent"] for n in data["nodes"]})
        palette = ["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8",
                    "#ff922b", "#20c997", "#748ffc", "#f06595", "#ced4da"]
        agent_colors = {a: palette[i % len(palette)] for i, a in enumerate(agents)}

        nodes_js = json.dumps(data["nodes"])
        edges_js = json.dumps(data["edges"])
        influence_js = json.dumps(influence)
        conv_js = json.dumps(conv)
        colors_js = json.dumps(agent_colors)
        chain_js = json.dumps(list(chain_ids))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Consensus Lineage Tracker</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f0f13;color:#e0e0e0;font-family:system-ui,-apple-system,sans-serif;padding:20px}}
h1{{text-align:center;font-size:1.8rem;margin-bottom:8px;color:#00d4ff}}
.subtitle{{text-align:center;color:#888;margin-bottom:24px;font-size:.9rem}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1400px;margin:0 auto}}
.card{{background:#1a1a24;border:1px solid #2a2a3a;border-radius:12px;padding:20px;overflow:hidden}}
.card h2{{font-size:1.1rem;color:#00d4ff;margin-bottom:12px;border-bottom:1px solid #2a2a3a;padding-bottom:8px}}
.full{{grid-column:1/-1}}
canvas{{width:100%;background:#12121a;border-radius:8px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #2a2a3a}}
th{{color:#00d4ff;font-weight:600}}
.bar{{height:14px;border-radius:4px;background:linear-gradient(90deg,#00d4ff,#cc5de8);min-width:2px}}
.innov-card{{background:#1e1e2e;border-left:3px solid #ffd43b;padding:12px;margin:8px 0;border-radius:0 8px 8px 0}}
.innov-card .agent{{color:#ffd43b;font-weight:600}}
.innov-card .sol{{color:#ccc;font-size:.85rem;margin-top:4px}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}}
.legend-item{{display:flex;align-items:center;gap:6px;font-size:.8rem}}
.legend-dot{{width:12px;height:12px;border-radius:50%}}
svg text{{fill:#e0e0e0;font-family:system-ui;font-size:11px}}
</style>
</head>
<body>
<h1>📜 Consensus Lineage Tracker</h1>
<p class="subtitle">Causal ancestry of mBFT consensus decisions</p>
<div class="grid">
  <div class="card full" id="graph-card">
    <h2>🔗 Lineage Graph</h2>
    <div class="legend" id="legend"></div>
    <svg id="lineage-svg" width="100%" height="400"></svg>
  </div>
  <div class="card">
    <h2>📈 Convergence</h2>
    <canvas id="conv-chart" height="200"></canvas>
  </div>
  <div class="card">
    <h2>🏆 Agent Influence</h2>
    <table id="inf-table"><thead><tr><th>#</th><th>Agent</th><th>Score</th><th></th></tr></thead><tbody></tbody></table>
  </div>
  <div class="card full">
    <h2>💡 Innovations</h2>
    <div id="innovations"></div>
  </div>
</div>
<script>
const nodes={nodes_js};
const edges={edges_js};
const influence={influence_js};
const convergence={conv_js};
const agentColors={colors_js};
const chainIds=new Set({chain_js});

// Legend
const legend=document.getElementById('legend');
Object.entries(agentColors).forEach(([a,c])=>{{
  const d=document.createElement('div');d.className='legend-item';
  d.innerHTML=`<span class="legend-dot" style="background:${{c}}"></span>${{a}}`;
  legend.appendChild(d);
}});

// Lineage SVG
(function(){{
  const svg=document.getElementById('lineage-svg');
  const rounds=[...new Set(nodes.map(n=>n.round))].sort((a,b)=>a-b);
  const W=svg.clientWidth||900, H=400;
  svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);
  const rSpacing=W/(rounds.length+1);
  const posMap={{}};
  rounds.forEach((r,ri)=>{{
    const rNodes=nodes.filter(n=>n.round===r);
    const ySpacing=H/(rNodes.length+1);
    rNodes.forEach((n,ni)=>{{
      const x=rSpacing*(ri+1);
      const y=ySpacing*(ni+1);
      posMap[n.proposal_id]={{x,y,node:n}};
    }});
  }});
  // Edges
  edges.forEach(e=>{{
    const f=posMap[e.from],t=posMap[e.to];
    if(!f||!t)return;
    const onChain=chainIds.has(e.from)&&chainIds.has(e.to);
    const line=document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',f.x);line.setAttribute('y1',f.y);
    line.setAttribute('x2',t.x);line.setAttribute('y2',t.y);
    line.setAttribute('stroke',onChain?'#ffd43b':'#3a3a4a');
    line.setAttribute('stroke-width',onChain?3:1);
    line.setAttribute('stroke-opacity',onChain?1:.5);
    svg.appendChild(line);
  }});
  // Nodes
  Object.values(posMap).forEach(({{x,y,node:n}})=>{{
    const g=document.createElementNS('http://www.w3.org/2000/svg','g');
    const r=chainIds.has(n.proposal_id)?16:11;
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',x);c.setAttribute('cy',y);c.setAttribute('r',r);
    c.setAttribute('fill',agentColors[n.agent]||'#888');
    c.setAttribute('stroke',chainIds.has(n.proposal_id)?'#ffd43b':'none');
    c.setAttribute('stroke-width',3);
    c.setAttribute('opacity',n.winning_chain?1:.6);
    g.appendChild(c);
    const t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',x);t.setAttribute('y',y-r-4);t.setAttribute('text-anchor','middle');
    t.textContent=n.agent.replace('agent-','A');
    g.appendChild(t);
    const title=document.createElementNS('http://www.w3.org/2000/svg','title');
    title.textContent=`${{n.agent}} R${{n.round}}\\nconf=${{n.confidence.toFixed(2)}}\\ninfl=${{n.influence.toFixed(3)}}\\n"${{n.solution.slice(0,80)}}"`;
    g.appendChild(title);
    svg.appendChild(g);
  }});
  // Round labels
  rounds.forEach((r,ri)=>{{
    const t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',rSpacing*(ri+1));t.setAttribute('y',H-8);
    t.setAttribute('text-anchor','middle');t.setAttribute('fill','#888');
    t.textContent=`Round ${{r}}`;
    svg.appendChild(t);
  }});
}})();

// Convergence chart
(function(){{
  const canvas=document.getElementById('conv-chart');
  const ctx=canvas.getContext('2d');
  canvas.width=canvas.offsetWidth*2;canvas.height=400;
  const W=canvas.width,H=canvas.height,pad=50;
  ctx.fillStyle='#12121a';ctx.fillRect(0,0,W,H);
  if(convergence.length<1)return;
  const maxR=convergence[convergence.length-1][0];
  const xScale=(W-2*pad)/(maxR||1);
  ctx.strokeStyle='#2a2a3a';ctx.lineWidth=1;
  for(let i=0;i<=10;i++){{
    const y=pad+(H-2*pad)*(1-i/10);
    ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();
  }}
  ctx.strokeStyle='#00d4ff';ctx.lineWidth=3;ctx.beginPath();
  convergence.forEach(([r,s],i)=>{{
    const x=pad+r*xScale;const y=pad+(H-2*pad)*(1-s);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});
  ctx.stroke();
  convergence.forEach(([r,s])=>{{
    const x=pad+r*xScale;const y=pad+(H-2*pad)*(1-s);
    ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fillStyle='#00d4ff';ctx.fill();
    ctx.fillStyle='#888';ctx.font='11px system-ui';ctx.textAlign='center';
    ctx.fillText(`R${{r}}: ${{s.toFixed(2)}}`,x,y-12);
  }});
}})();

// Influence table
(function(){{
  const tb=document.querySelector('#inf-table tbody');
  const maxScore=Math.max(...influence.map(([,s])=>s),0.01);
  influence.forEach(([a,s],i)=>{{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${{i+1}}</td><td style="color:${{agentColors[a]||'#ccc'}}">${{a}}</td><td>${{s.toFixed(3)}}</td><td><div class="bar" style="width:${{(s/maxScore*100).toFixed(0)}}%"></div></td>`;
    tb.appendChild(tr);
  }});
}})();

// Innovations
(function(){{
  const div=document.getElementById('innovations');
  const innovs=nodes.filter(n=>n.innovation);
  if(!innovs.length){{div.innerHTML='<p style="color:#888">No novel proposals detected.</p>';return;}}
  innovs.forEach(n=>{{
    const d=document.createElement('div');d.className='innov-card';
    d.innerHTML=`<span class="agent" style="color:${{agentColors[n.agent]||'#ffd43b'}}">${{n.agent}}</span> — Round ${{n.round}} (conf ${{n.confidence.toFixed(2)}})<div class="sol">"${{n.solution.slice(0,120)}}"</div>`;
    div.appendChild(d);
  }});
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_agents(n_agents: int, n_byzantine: int) -> list[MockAgent]:
    """Create a mix of honest and Byzantine mock agents."""
    solutions = [
        "Apply leader rotation with reputation weighting",
        "Use confidence-weighted voting with slash penalties",
        "Implement epistemic leader election via metacognitive scoring",
        "Deploy defeasible reasoning with counter-proof verification",
        "Combine Bayesian updating with social choice aggregation",
        "Leverage proof-carrying proposals with threshold finality",
        "Use iterative refinement with reputation-gated veto power",
    ]
    agents: list[MockAgent] = []
    honest_answer = solutions[0]
    for i in range(n_agents):
        is_byz = i >= (n_agents - n_byzantine)
        aid = f"agent-{i}"
        if is_byz:
            agents.append(MockAgent(
                aid,
                answer=random.choice(solutions[2:]),
                confidence=round(random.uniform(0.3, 0.9), 2),
                byzantine=True,
            ))
        else:
            # Honest agents: varied confidence, mostly agree
            ans = solutions[i % 3]
            conf = round(random.uniform(0.6, 0.95), 2)
            accept = {solutions[j % 3] for j in range(n_agents)}
            agents.append(MockAgent(aid, answer=ans, confidence=conf,
                                    accept_set=accept))
    return agents


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Consensus Lineage Tracker — causal ancestry of mBFT decisions")
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--byzantine", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--task", default="Solve the Byzantine agreement problem")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--similarity", type=float, default=0.4)
    parser.add_argument("--output", help="Save HTML report")
    parser.add_argument("--json", dest="json_path", help="Save JSON data")
    args = parser.parse_args(argv)

    agents = _build_agents(args.agents, args.byzantine)
    engine = InstrumentedEngine(agents, args.threshold, max_rounds=args.rounds)
    result = asyncio.run(engine.run(args.task))

    tracker = ConsensusLineageTracker(similarity_threshold=args.similarity)
    tracker.build(engine, engine.history)

    print(tracker.text_summary())

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(tracker.html_report())
        print(f"\n✅ HTML report saved to {args.output}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(tracker.to_dict(), f, indent=2)
        print(f"✅ JSON data saved to {args.json_path}")


if __name__ == "__main__":
    main()
