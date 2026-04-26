"""Consensus Epidemic Simulator — models belief/misinformation spread through agent networks.

Uses SIR-like epidemic dynamics to show how Byzantine agents can "infect" honest
ones with faulty beliefs, and how mBFT's reputation slashing acts as immunity.

Features:
  - SIR (Susceptible → Infected → Recovered) belief propagation model
  - Byzantine "super-spreader" injection with configurable virulence
  - Reputation-based immunity: higher-reputation agents resist infection
  - Vaccination via mBFT slashing: detected Byzantine agents get quarantined
  - Multiple network topologies: complete, ring, small-world, scale-free
  - Auto-pilot mode: runs parameter sweeps to find epidemic thresholds
  - Interactive HTML report with epidemic curves, network visualization, R0 estimation

Usage:
    python -m src.epidemic                          # default demo
    python -m src.epidemic --agents 20 --byzantine 3
    python -m src.epidemic --topology small-world --rewire 0.3
    python -m src.epidemic --autopilot              # sweep β/γ space
    python -m src.epidemic --output report.html
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class AgentState(Enum):
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    RECOVERED = "recovered"
    QUARANTINED = "quarantined"


class Topology(Enum):
    COMPLETE = "complete"
    RING = "ring"
    SMALL_WORLD = "small-world"
    SCALE_FREE = "scale-free"


@dataclass
class EpidemicAgent:
    id: str
    state: AgentState = AgentState.SUSCEPTIBLE
    reputation: float = 1.0
    is_byzantine: bool = False
    infected_tick: int = -1
    recovered_tick: int = -1
    infection_source: Optional[str] = None
    infections_caused: int = 0


@dataclass
class EpidemicTick:
    tick: int
    susceptible: int
    infected: int
    recovered: int
    quarantined: int
    new_infections: int
    new_recoveries: int
    new_quarantines: int
    r_effective: float
    events: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    ticks: List[EpidemicTick]
    agents: List[EpidemicAgent]
    edges: List[Tuple[str, str]]
    topology: str
    params: Dict
    peak_infected: int
    peak_tick: int
    total_infected: int
    r0_estimate: float
    epidemic_threshold: float
    herd_immunity_pct: float
    containment_success: bool


class EpidemicSimulator:
    """SIR-based belief epidemic simulator for mBFT agent networks."""

    def __init__(
        self,
        n_agents: int = 12,
        n_byzantine: int = 2,
        beta: float = 0.3,       # infection probability per contact
        gamma: float = 0.1,      # recovery probability per tick
        virulence: float = 0.8,  # byzantine super-spreader multiplier
        slash_rate: float = 0.2, # probability of detecting & quarantining per tick
        immunity_factor: float = 0.5,  # reputation-based resistance
        topology: Topology = Topology.SMALL_WORLD,
        rewire_prob: float = 0.3,
        max_ticks: int = 100,
        seed: Optional[int] = None,
    ):
        self.n_agents = n_agents
        self.n_byzantine = n_byzantine
        self.beta = beta
        self.gamma = gamma
        self.virulence = virulence
        self.slash_rate = slash_rate
        self.immunity_factor = immunity_factor
        self.topology = topology
        self.rewire_prob = rewire_prob
        self.max_ticks = max_ticks
        self.rng = random.Random(seed)

        self.agents: List[EpidemicAgent] = []
        self.edges: List[Tuple[str, str]] = []
        self.adjacency: Dict[str, List[str]] = {}
        self.ticks: List[EpidemicTick] = []

    def build_network(self) -> None:
        self.agents = [
            EpidemicAgent(id=f"agent-{i}") for i in range(self.n_agents)
        ]
        byz_indices = self.rng.sample(range(self.n_agents), min(self.n_byzantine, self.n_agents))
        for i in byz_indices:
            self.agents[i].is_byzantine = True
            self.agents[i].state = AgentState.INFECTED
            self.agents[i].infected_tick = 0

        self.edges = []
        self.adjacency = {a.id: [] for a in self.agents}

        if self.topology == Topology.COMPLETE:
            self._build_complete()
        elif self.topology == Topology.RING:
            self._build_ring()
        elif self.topology == Topology.SMALL_WORLD:
            self._build_small_world()
        elif self.topology == Topology.SCALE_FREE:
            self._build_scale_free()

    def _add_edge(self, a: str, b: str) -> None:
        if a != b and b not in self.adjacency[a]:
            self.edges.append((a, b))
            self.adjacency[a].append(b)
            self.adjacency[b].append(a)

    def _build_complete(self) -> None:
        ids = [a.id for a in self.agents]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                self._add_edge(a, b)

    def _build_ring(self) -> None:
        ids = [a.id for a in self.agents]
        n = len(ids)
        for i in range(n):
            self._add_edge(ids[i], ids[(i + 1) % n])
            self._add_edge(ids[i], ids[(i + 2) % n])

    def _build_small_world(self) -> None:
        self._build_ring()
        ids = [a.id for a in self.agents]
        n = len(ids)
        new_edges = []
        for a_id, b_id in list(self.edges):
            if self.rng.random() < self.rewire_prob:
                target = ids[self.rng.randint(0, n - 1)]
                if target != a_id and target not in self.adjacency[a_id]:
                    self.adjacency[a_id].remove(b_id)
                    self.adjacency[b_id].remove(a_id)
                    new_edges.append((a_id, target))
        self.edges = [(a, b) for a, b in self.edges
                      if b in self.adjacency[a]]
        for a, b in new_edges:
            self._add_edge(a, b)

    def _build_scale_free(self) -> None:
        ids = [a.id for a in self.agents]
        if len(ids) < 3:
            self._build_complete()
            return
        for i in range(min(3, len(ids))):
            for j in range(i + 1, min(3, len(ids))):
                self._add_edge(ids[i], ids[j])
        for i in range(3, len(ids)):
            degrees = [len(self.adjacency[a_id]) + 1 for a_id in ids[:i]]
            total = sum(degrees)
            probs = [d / total for d in degrees]
            targets = set()
            while len(targets) < min(2, i):
                r = self.rng.random()
                cumul = 0.0
                for j, p in enumerate(probs):
                    cumul += p
                    if r <= cumul:
                        targets.add(ids[j])
                        break
            for t in targets:
                self._add_edge(ids[i], t)

    def run(self) -> SimulationResult:
        self.build_network()
        self.ticks = []
        generation_infections: Dict[int, int] = {0: self.n_byzantine}

        for tick in range(self.max_ticks):
            counts = self._count_states()
            if counts[AgentState.INFECTED] == 0:
                self.ticks.append(EpidemicTick(
                    tick=tick, susceptible=counts[AgentState.SUSCEPTIBLE],
                    infected=0, recovered=counts[AgentState.RECOVERED],
                    quarantined=counts[AgentState.QUARANTINED],
                    new_infections=0, new_recoveries=0, new_quarantines=0,
                    r_effective=0.0,
                ))
                break

            new_inf, new_rec, new_quar, events = self._step(tick)
            generation_infections[tick] = new_inf
            counts = self._count_states()

            r_eff = new_inf / max(counts[AgentState.INFECTED], 1)
            self.ticks.append(EpidemicTick(
                tick=tick, susceptible=counts[AgentState.SUSCEPTIBLE],
                infected=counts[AgentState.INFECTED],
                recovered=counts[AgentState.RECOVERED],
                quarantined=counts[AgentState.QUARANTINED],
                new_infections=new_inf, new_recoveries=new_rec,
                new_quarantines=new_quar, r_effective=r_eff, events=events,
            ))

        peak_tick_data = max(self.ticks, key=lambda t: t.infected) if self.ticks else None
        total_infected = sum(
            1 for a in self.agents
            if a.state != AgentState.SUSCEPTIBLE
        )
        avg_degree = (2 * len(self.edges) / max(self.n_agents, 1))
        r0 = self.beta * avg_degree / max(self.gamma, 0.001)
        threshold = 1.0 / max(r0, 0.001)
        herd = max(0.0, 1.0 - threshold)

        final_counts = self._count_states()
        containment = (final_counts[AgentState.SUSCEPTIBLE] / max(self.n_agents, 1)) > 0.5

        return SimulationResult(
            ticks=self.ticks,
            agents=self.agents,
            edges=self.edges,
            topology=self.topology.value,
            params={
                "n_agents": self.n_agents, "n_byzantine": self.n_byzantine,
                "beta": self.beta, "gamma": self.gamma,
                "virulence": self.virulence, "slash_rate": self.slash_rate,
                "immunity_factor": self.immunity_factor,
            },
            peak_infected=peak_tick_data.infected if peak_tick_data else 0,
            peak_tick=peak_tick_data.tick if peak_tick_data else 0,
            total_infected=total_infected,
            r0_estimate=round(r0, 2),
            epidemic_threshold=round(threshold, 3),
            herd_immunity_pct=round(herd * 100, 1),
            containment_success=containment,
        )

    def _step(self, tick: int) -> Tuple[int, int, int, List[str]]:
        new_infections = 0
        new_recoveries = 0
        new_quarantines = 0
        events: List[str] = []
        to_infect: List[Tuple[EpidemicAgent, str]] = []
        to_recover: List[EpidemicAgent] = []
        to_quarantine: List[EpidemicAgent] = []

        for agent in self.agents:
            if agent.state != AgentState.INFECTED:
                continue

            # Quarantine check (mBFT slash detection)
            if agent.is_byzantine and self.rng.random() < self.slash_rate:
                to_quarantine.append(agent)
                events.append(f"🛡️ {agent.id} quarantined (Byzantine detected via slashing)")
                continue

            # Recovery
            if self.rng.random() < self.gamma:
                to_recover.append(agent)
                events.append(f"💚 {agent.id} recovered")
                continue

            # Spread
            for neighbor_id in self.adjacency[agent.id]:
                neighbor = next(a for a in self.agents if a.id == neighbor_id)
                if neighbor.state != AgentState.SUSCEPTIBLE:
                    continue
                eff_beta = self.beta
                if agent.is_byzantine:
                    eff_beta *= self.virulence + 1.0
                resistance = neighbor.reputation * self.immunity_factor
                infection_prob = eff_beta * (1.0 - resistance)
                infection_prob = max(0.0, min(1.0, infection_prob))
                if self.rng.random() < infection_prob:
                    to_infect.append((neighbor, agent.id))

        for agent, source in to_infect:
            if agent.state == AgentState.SUSCEPTIBLE:
                agent.state = AgentState.INFECTED
                agent.infected_tick = tick
                agent.infection_source = source
                src_agent = next(a for a in self.agents if a.id == source)
                src_agent.infections_caused += 1
                new_infections += 1
                events.append(f"🦠 {agent.id} infected by {source}")

        for agent in to_recover:
            agent.state = AgentState.RECOVERED
            agent.recovered_tick = tick
            agent.reputation = min(1.0, agent.reputation + 0.1)
            new_recoveries += 1

        for agent in to_quarantine:
            agent.state = AgentState.QUARANTINED
            agent.reputation *= 0.3
            new_quarantines += 1

        return new_infections, new_recoveries, new_quarantines, events

    def _count_states(self) -> Dict[AgentState, int]:
        counts = {s: 0 for s in AgentState}
        for a in self.agents:
            counts[a.state] += 1
        return counts


def run_autopilot(n_agents: int = 12, n_byzantine: int = 2,
                  topology: Topology = Topology.SMALL_WORLD,
                  seed: Optional[int] = None) -> List[Dict]:
    """Sweep β/γ space to find epidemic thresholds and optimal slash rates."""
    results = []
    betas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    gammas = [0.05, 0.1, 0.2, 0.3]
    slash_rates = [0.0, 0.1, 0.2, 0.4]

    for beta in betas:
        for gamma in gammas:
            for sr in slash_rates:
                sim = EpidemicSimulator(
                    n_agents=n_agents, n_byzantine=n_byzantine,
                    beta=beta, gamma=gamma, slash_rate=sr,
                    topology=topology, seed=seed,
                )
                res = sim.run()
                results.append({
                    "beta": beta, "gamma": gamma, "slash_rate": sr,
                    "r0": res.r0_estimate,
                    "peak_infected": res.peak_infected,
                    "total_infected": res.total_infected,
                    "containment": res.containment_success,
                    "peak_tick": res.peak_tick,
                })
    return results


def generate_html_report(result: SimulationResult,
                         autopilot_data: Optional[List[Dict]] = None) -> str:
    """Generate interactive HTML epidemic report."""
    tick_data = json.dumps([{
        "tick": t.tick, "s": t.susceptible, "i": t.infected,
        "r": t.recovered, "q": t.quarantined, "r_eff": round(t.r_effective, 2),
        "new_inf": t.new_infections,
    } for t in result.ticks])

    agent_data = json.dumps([{
        "id": a.id, "state": a.state.value, "byzantine": a.is_byzantine,
        "reputation": round(a.reputation, 2),
        "infected_tick": a.infected_tick, "recovered_tick": a.recovered_tick,
        "source": a.infection_source, "caused": a.infections_caused,
    } for a in result.agents])

    edge_data = json.dumps(result.edges)
    autopilot_json = json.dumps(autopilot_data) if autopilot_data else "null"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>mBFT Epidemic Simulator</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a1a;color:#e0e0e0;padding:20px}}
h1{{text-align:center;color:#00e5ff;margin-bottom:5px;font-size:1.8em}}
.subtitle{{text-align:center;color:#888;margin-bottom:20px;font-size:0.9em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:20px}}
.card{{background:#141428;border:1px solid #2a2a4a;border-radius:12px;padding:16px}}
.card h3{{color:#00e5ff;margin-bottom:10px;font-size:1em}}
.metric{{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1a1a3a}}
.metric .label{{color:#999}}.metric .value{{font-weight:bold}}
.good{{color:#4caf50}}.warn{{color:#ff9800}}.bad{{color:#f44336}}.info{{color:#42a5f5}}
canvas{{width:100%;height:300px;display:block;background:#0d0d20;border-radius:8px;margin-top:8px}}
.net-canvas{{height:350px}}
.events{{max-height:200px;overflow-y:auto;font-size:0.85em;padding:8px;background:#0d0d20;border-radius:8px;margin-top:8px}}
.events div{{padding:2px 0;border-bottom:1px solid #1a1a3a}}
.legend{{display:flex;gap:16px;justify-content:center;margin:8px 0;font-size:0.85em}}
.legend span{{display:flex;align-items:center;gap:4px}}
.legend .dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
table{{width:100%;border-collapse:collapse;font-size:0.85em;margin-top:8px}}
th,td{{padding:6px 8px;text-align:left;border-bottom:1px solid #1a1a3a}}
th{{color:#00e5ff;font-weight:600}}
.auto-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px;margin-top:8px}}
.auto-cell{{background:#0d0d20;padding:8px;border-radius:6px;font-size:0.8em}}
</style></head><body>
<h1>🦠 mBFT Consensus Epidemic Simulator</h1>
<p class="subtitle">Belief propagation &amp; Byzantine containment via reputation-based immunity</p>

<div class="grid">
  <div class="card"><h3>📊 Epidemic Overview</h3>
    <div class="metric"><span class="label">Topology</span><span class="value info">{result.topology}</span></div>
    <div class="metric"><span class="label">Agents</span><span class="value">{result.params['n_agents']}</span></div>
    <div class="metric"><span class="label">Byzantine (Patient Zero)</span><span class="value bad">{result.params['n_byzantine']}</span></div>
    <div class="metric"><span class="label">R₀ Estimate</span><span class="value {'bad' if result.r0_estimate>1 else 'good'}">{result.r0_estimate}</span></div>
    <div class="metric"><span class="label">Epidemic Threshold</span><span class="value info">{result.epidemic_threshold}</span></div>
    <div class="metric"><span class="label">Herd Immunity Needed</span><span class="value warn">{result.herd_immunity_pct}%</span></div>
  </div>
  <div class="card"><h3>📈 Outcomes</h3>
    <div class="metric"><span class="label">Peak Infected</span><span class="value bad">{result.peak_infected}</span></div>
    <div class="metric"><span class="label">Peak Tick</span><span class="value">{result.peak_tick}</span></div>
    <div class="metric"><span class="label">Total Ever Infected</span><span class="value warn">{result.total_infected} / {result.params['n_agents']}</span></div>
    <div class="metric"><span class="label">Duration</span><span class="value">{len(result.ticks)} ticks</span></div>
    <div class="metric"><span class="label">Containment</span><span class="value {'good' if result.containment_success else 'bad'}">{'✅ Success' if result.containment_success else '❌ Failed'}</span></div>
  </div>
  <div class="card"><h3>⚙️ Parameters</h3>
    <div class="metric"><span class="label">β (infection rate)</span><span class="value">{result.params['beta']}</span></div>
    <div class="metric"><span class="label">γ (recovery rate)</span><span class="value">{result.params['gamma']}</span></div>
    <div class="metric"><span class="label">Virulence multiplier</span><span class="value">{result.params['virulence']}</span></div>
    <div class="metric"><span class="label">Slash/quarantine rate</span><span class="value">{result.params['slash_rate']}</span></div>
    <div class="metric"><span class="label">Immunity factor</span><span class="value">{result.params['immunity_factor']}</span></div>
  </div>
</div>

<div class="grid" style="grid-template-columns:1fr 1fr">
  <div class="card"><h3>🦠 SIR Epidemic Curve</h3>
    <div class="legend">
      <span><span class="dot" style="background:#42a5f5"></span> Susceptible</span>
      <span><span class="dot" style="background:#f44336"></span> Infected</span>
      <span><span class="dot" style="background:#4caf50"></span> Recovered</span>
      <span><span class="dot" style="background:#ff9800"></span> Quarantined</span>
    </div>
    <canvas id="sirChart"></canvas>
  </div>
  <div class="card"><h3>📡 R(t) Effective Reproduction</h3>
    <div class="legend">
      <span><span class="dot" style="background:#e040fb"></span> R(t)</span>
      <span style="color:#f44336">--- R=1 threshold</span>
    </div>
    <canvas id="rChart"></canvas>
  </div>
</div>

<div class="grid" style="grid-template-columns:1fr 1fr">
  <div class="card"><h3>🌐 Network Topology</h3>
    <canvas id="netCanvas" class="net-canvas"></canvas>
  </div>
  <div class="card"><h3>🕵️ Agent Status</h3>
    <table><thead><tr><th>Agent</th><th>State</th><th>Rep</th><th>Byz?</th><th>Infected@</th><th>Caused</th></tr></thead>
    <tbody id="agentTable"></tbody></table>
  </div>
</div>

<div class="card" style="margin-bottom:20px"><h3>📋 Event Log</h3>
  <div class="events" id="eventLog"></div>
</div>

<div id="autopilotSection"></div>

<script>
const ticks={tick_data};
const agents={agent_data};
const edges={edge_data};
const autopilotData={autopilot_json};

// SIR Chart
(function(){{
  const c=document.getElementById('sirChart'),ctx=c.getContext('2d');
  function draw(){{
    c.width=c.offsetWidth;c.height=c.offsetHeight;
    if(!ticks.length)return;
    const w=c.width,h=c.height,p=40,maxT=ticks.length,maxV={result.params['n_agents']};
    ctx.fillStyle='#0d0d20';ctx.fillRect(0,0,w,h);
    // Axes
    ctx.strokeStyle='#333';ctx.lineWidth=1;ctx.beginPath();
    ctx.moveTo(p,p);ctx.lineTo(p,h-p);ctx.lineTo(w-p,h-p);ctx.stroke();
    // Grid
    for(let i=0;i<=4;i++){{
      const y=p+(h-2*p)*(i/4);
      ctx.strokeStyle='#1a1a3a';ctx.beginPath();ctx.moveTo(p,y);ctx.lineTo(w-p,y);ctx.stroke();
      ctx.fillStyle='#666';ctx.font='11px sans-serif';ctx.textAlign='right';
      ctx.fillText(Math.round(maxV*(1-i/4)),p-5,y+4);
    }}
    function line(data,color){{
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
      data.forEach((v,i)=>{{
        const x=p+i*(w-2*p)/(maxT-1||1),y=h-p-v*(h-2*p)/maxV;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
      }});ctx.stroke();
    }}
    line(ticks.map(t=>t.s),'#42a5f5');
    line(ticks.map(t=>t.i),'#f44336');
    line(ticks.map(t=>t.r),'#4caf50');
    line(ticks.map(t=>t.q),'#ff9800');
    ctx.fillStyle='#666';ctx.textAlign='center';ctx.fillText('Tick',w/2,h-5);
  }}
  draw();window.addEventListener('resize',draw);
}})();

// R(t) Chart
(function(){{
  const c=document.getElementById('rChart'),ctx=c.getContext('2d');
  function draw(){{
    c.width=c.offsetWidth;c.height=c.offsetHeight;
    if(!ticks.length)return;
    const w=c.width,h=c.height,p=40,maxT=ticks.length;
    const maxR=Math.max(3,Math.max(...ticks.map(t=>t.r_eff))*1.2);
    ctx.fillStyle='#0d0d20';ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='#333';ctx.lineWidth=1;ctx.beginPath();
    ctx.moveTo(p,p);ctx.lineTo(p,h-p);ctx.lineTo(w-p,h-p);ctx.stroke();
    // R=1 threshold
    const r1y=h-p-1*(h-2*p)/maxR;
    ctx.strokeStyle='#f4433666';ctx.setLineDash([5,5]);ctx.beginPath();
    ctx.moveTo(p,r1y);ctx.lineTo(w-p,r1y);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#f44336';ctx.font='11px sans-serif';ctx.fillText('R=1',w-p+5,r1y+4);
    // Line
    ctx.strokeStyle='#e040fb';ctx.lineWidth=2;ctx.beginPath();
    ticks.forEach((t,i)=>{{
      const x=p+i*(w-2*p)/(maxT-1||1),y=h-p-t.r_eff*(h-2*p)/maxR;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }});ctx.stroke();
  }}
  draw();window.addEventListener('resize',draw);
}})();

// Network
(function(){{
  const c=document.getElementById('netCanvas'),ctx=c.getContext('2d');
  function draw(){{
    c.width=c.offsetWidth;c.height=c.offsetHeight;
    const w=c.width,h=c.height,cx=w/2,cy=h/2,rad=Math.min(w,h)*0.38;
    ctx.fillStyle='#0d0d20';ctx.fillRect(0,0,w,h);
    const pos={{}};
    agents.forEach((a,i)=>{{
      const angle=2*Math.PI*i/agents.length-Math.PI/2;
      pos[a.id]={{x:cx+rad*Math.cos(angle),y:cy+rad*Math.sin(angle)}};
    }});
    // Edges
    ctx.strokeStyle='#2a2a4a';ctx.lineWidth=1;
    edges.forEach(([a,b])=>{{
      if(pos[a]&&pos[b]){{ctx.beginPath();ctx.moveTo(pos[a].x,pos[a].y);ctx.lineTo(pos[b].x,pos[b].y);ctx.stroke();}}
    }});
    // Nodes
    const colors={{susceptible:'#42a5f5',infected:'#f44336',recovered:'#4caf50',quarantined:'#ff9800'}};
    agents.forEach(a=>{{
      const p=pos[a.id];if(!p)return;
      ctx.beginPath();ctx.arc(p.x,p.y,a.byzantine?12:9,0,Math.PI*2);
      ctx.fillStyle=colors[a.state]||'#666';ctx.fill();
      if(a.byzantine){{ctx.strokeStyle='#ff0';ctx.lineWidth=2;ctx.stroke();}}
      ctx.fillStyle='#fff';ctx.font='9px sans-serif';ctx.textAlign='center';
      ctx.fillText(a.id.replace('agent-',''),p.x,p.y+3);
    }});
  }}
  draw();window.addEventListener('resize',draw);
}})();

// Agent Table
(function(){{
  const tb=document.getElementById('agentTable');
  const colors={{susceptible:'info',infected:'bad',recovered:'good',quarantined:'warn'}};
  agents.forEach(a=>{{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${{a.id}}</td><td class="${{colors[a.state]||''}}">${{a.state}}</td>`
      +`<td>${{a.reputation}}</td><td>${{a.byzantine?'⚠️ Yes':'No'}}</td>`
      +`<td>${{a.infected_tick>=0?a.infected_tick:'—'}}</td><td>${{a.caused}}</td>`;
    tb.appendChild(tr);
  }});
}})();

// Event Log
(function(){{
  const el=document.getElementById('eventLog');
  ticks.forEach(t=>{{
    const raw=ticks[t.tick];
    if(!raw)return;
  }});
  // Flatten all events
  const allEvents={json.dumps([e for t in result.ticks for e in t.events])};
  allEvents.forEach(e=>{{
    const d=document.createElement('div');d.textContent=e;el.appendChild(d);
  }});
}})();

// Autopilot
if(autopilotData){{
  const sec=document.getElementById('autopilotSection');
  sec.innerHTML='<div class="card"><h3>🤖 Autopilot: β/γ/Slash Sweep</h3>'
    +'<p style="color:#999;margin:8px 0;font-size:0.85em">Parameter sweep showing containment outcomes across infection/recovery/slash rates</p>'
    +'<div class="auto-grid" id="autoGrid"></div></div>';
  const grid=document.getElementById('autoGrid');
  autopilotData.forEach(d=>{{
    const cell=document.createElement('div');
    cell.className='auto-cell';
    cell.style.borderLeft=`3px solid ${{d.containment?'#4caf50':'#f44336'}}`;
    cell.innerHTML=`<b>β=${{d.beta}} γ=${{d.gamma}} slash=${{d.slash_rate}}</b><br>`
      +`R₀=${{d.r0}} | Peak=${{d.peak_infected}} | Total=${{d.total_infected}}<br>`
      +`${{d.containment?'✅ Contained':'❌ Spread'}}`;
    grid.appendChild(cell);
  }});
}}
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="mBFT Consensus Epidemic Simulator")
    parser.add_argument("--agents", type=int, default=12, help="Number of agents")
    parser.add_argument("--byzantine", type=int, default=2, help="Number of Byzantine agents (patient zero)")
    parser.add_argument("--beta", type=float, default=0.3, help="Infection probability per contact")
    parser.add_argument("--gamma", type=float, default=0.1, help="Recovery probability per tick")
    parser.add_argument("--virulence", type=float, default=0.8, help="Byzantine virulence multiplier")
    parser.add_argument("--slash-rate", type=float, default=0.2, help="Quarantine detection rate")
    parser.add_argument("--immunity", type=float, default=0.5, help="Reputation immunity factor")
    parser.add_argument("--topology", choices=["complete", "ring", "small-world", "scale-free"],
                        default="small-world", help="Network topology")
    parser.add_argument("--rewire", type=float, default=0.3, help="Small-world rewire probability")
    parser.add_argument("--ticks", type=int, default=100, help="Max simulation ticks")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--autopilot", action="store_true", help="Run parameter sweep")
    parser.add_argument("--output", type=str, default=None, help="Output HTML file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of HTML")
    args = parser.parse_args()

    topo = Topology(args.topology)

    sim = EpidemicSimulator(
        n_agents=args.agents, n_byzantine=args.byzantine,
        beta=args.beta, gamma=args.gamma, virulence=args.virulence,
        slash_rate=args.slash_rate, immunity_factor=args.immunity,
        topology=topo, rewire_prob=args.rewire,
        max_ticks=args.ticks, seed=args.seed,
    )
    result = sim.run()

    autopilot_data = None
    if args.autopilot:
        autopilot_data = run_autopilot(
            n_agents=args.agents, n_byzantine=args.byzantine,
            topology=topo, seed=args.seed,
        )

    # Console summary
    print("=" * 60)
    print("  🦠 mBFT Consensus Epidemic Simulator")
    print("=" * 60)
    print(f"  Topology:      {result.topology}")
    print(f"  Agents:        {result.params['n_agents']} ({result.params['n_byzantine']} Byzantine)")
    print(f"  R₀:            {result.r0_estimate}")
    print(f"  Peak infected: {result.peak_infected} at tick {result.peak_tick}")
    print(f"  Total infected:{result.total_infected} / {result.params['n_agents']}")
    print(f"  Duration:      {len(result.ticks)} ticks")
    print(f"  Containment:   {'✅ Success' if result.containment_success else '❌ Failed'}")
    print(f"  Herd immunity: {result.herd_immunity_pct}% needed")
    print("=" * 60)

    if args.json:
        output = json.dumps({
            "params": result.params, "topology": result.topology,
            "r0": result.r0_estimate, "peak_infected": result.peak_infected,
            "total_infected": result.total_infected,
            "containment": result.containment_success,
            "ticks": len(result.ticks),
        }, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"\nJSON saved to {args.output}")
        else:
            print(output)
    else:
        html = generate_html_report(result, autopilot_data)
        out_path = args.output or "epidemic_report.html"
        Path(out_path).write_text(html, encoding="utf-8")
        print(f"\n📊 Report saved to {out_path}")

    if autopilot_data:
        contained = sum(1 for d in autopilot_data if d["containment"])
        print(f"\n🤖 Autopilot: {contained}/{len(autopilot_data)} configurations contained the epidemic")
        best = min((d for d in autopilot_data if d["containment"]),
                    key=lambda d: d["total_infected"], default=None)
        if best:
            print(f"   Best: β={best['beta']} γ={best['gamma']} slash={best['slash_rate']} → {best['total_infected']} infected")


if __name__ == "__main__":
    main()
