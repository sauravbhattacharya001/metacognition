"""Consensus Replay Animator.

Records every mBFT protocol event (proposals, votes, leader elections,
slashing, commits) and generates an interactive HTML animation where users
can step through rounds, see vote flows between agents, watch reputation
bars change, and understand exactly how consensus was (or wasn't) reached.

Usage::

    python -m src.replay                              # default demo
    python -m src.replay --agents 7 --byzantine 2     # custom swarm
    python -m src.replay --threshold 2.0              # custom threshold
    python -m src.replay --output replay.html         # custom output
    python -m src.replay --speed fast                 # animation speed
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


@dataclass
class AgentSnapshot:
    agent_id: str
    reputation: float
    is_byzantine: bool
    confidence: float
    answer: str


@dataclass
class VoteEvent:
    voter_id: str
    weight: float
    effective_weight: float
    is_rejection: bool
    counter_proof: Optional[str]


@dataclass
class RoundEvent:
    round_index: int
    leader_id: str
    leader_solution: str
    leader_confidence: float
    votes: List[VoteEvent]
    aggregate_weight: float
    threshold: float
    committed: bool
    slashed: List[str]
    reputations_after: dict


@dataclass
class ReplayData:
    swarm_size: int
    threshold: float
    byzantine_count: int
    agents: List[AgentSnapshot]
    rounds: List[RoundEvent] = field(default_factory=list)
    final_committed: bool = False
    final_solution: Optional[str] = None


def _build_swarm(
    n: int,
    byzantine_count: int,
    honest_conf: float = 0.80,
    byz_conf: float = 0.95,
) -> list[MockAgent]:
    agents = []
    for i in range(n):
        is_byz = i >= (n - byzantine_count)
        agents.append(
            MockAgent(
                agent_id=f"a{i+1}",
                answer="correct" if not is_byz else f"byz-{i}",
                confidence=honest_conf if not is_byz else byz_conf,
                byzantine=is_byz,
            )
        )
    return agents


async def record_replay(
    swarm_size: int = 5,
    byzantine_count: int = 1,
    threshold: float = 1.5,
) -> ReplayData:
    """Run mBFT and capture every event for replay."""
    agents = _build_swarm(swarm_size, byzantine_count)
    engine = MBFTEngine(agents=agents, threshold=threshold, max_rounds=4)

    agent_snapshots = [
        AgentSnapshot(
            agent_id=a.id,
            reputation=1.0,
            is_byzantine=a.byzantine,
            confidence=a.confidence,
            answer=a.answer,
        )
        for a in agents
    ]

    replay = ReplayData(
        swarm_size=swarm_size,
        threshold=threshold,
        byzantine_count=byzantine_count,
        agents=agent_snapshots,
    )

    result = await engine.run("What is the answer to life?")

    # Extract round-by-round events from engine history
    rep_tracker = {a.id: 1.0 for a in agents}
    for rr in engine.history:
        vote_events = []
        for v in rr.votes:
            eff = v.weight * rep_tracker.get(v.voter_id, 1.0)
            vote_events.append(VoteEvent(
                voter_id=v.voter_id,
                weight=v.weight,
                effective_weight=round(eff, 4),
                is_rejection=v.is_rejection,
                counter_proof=v.counter_proof,
            ))

        # Update reputation tracking
        for s in rr.slashed:
            rep_tracker[s] = rep_tracker.get(s, 1.0) * engine.slash_factor

        replay.rounds.append(RoundEvent(
            round_index=rr.round_index,
            leader_id=rr.leader_id,
            leader_solution=rr.committed_solution or "(no commit)",
            leader_confidence=next(
                (a.confidence for a in agents if a.id == rr.leader_id), 0
            ),
            votes=vote_events,
            aggregate_weight=round(rr.aggregate_weight, 4),
            threshold=rr.threshold,
            committed=rr.committed,
            slashed=list(rr.slashed),
            reputations_after=dict(rep_tracker),
        ))

    replay.final_committed = result is not None and result.committed
    replay.final_solution = result.committed_solution if result else None
    return replay


def _serialize(replay: ReplayData) -> str:
    """Convert to JSON for embedding in HTML."""

    def _convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return json.dumps(_convert(replay), indent=2)


def render_html(replay: ReplayData, speed: str = "normal") -> str:
    data_json = _serialize(replay)
    speed_ms = {"slow": 1200, "normal": 700, "fast": 300}.get(speed, 700)

    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>mBFT Consensus Replay</title>
<style>
:root { --bg:#0d1117; --fg:#c9d1d9; --accent:#58a6ff; --red:#f85149;
  --green:#3fb950; --yellow:#d29922; --card:#161b22; --border:#21262d; }
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--fg);overflow-x:hidden}
.header{padding:1.5rem 2rem;border-bottom:1px solid var(--border)}
.header h1{color:var(--accent);font-size:1.5rem}
.header .sub{color:#8b949e;font-size:0.9rem;margin-top:0.3rem}
.controls{display:flex;align-items:center;gap:1rem;padding:1rem 2rem;border-bottom:1px solid var(--border);background:var(--card)}
.controls button{background:var(--accent);color:#000;border:none;padding:0.5rem 1.2rem;border-radius:6px;cursor:pointer;font-weight:600;font-size:0.9rem}
.controls button:hover{opacity:0.85}
.controls button:disabled{opacity:0.4;cursor:default}
.controls .round-label{font-size:1rem;font-weight:600;min-width:120px}
.controls .status-pill{padding:0.3rem 0.8rem;border-radius:12px;font-size:0.8rem;font-weight:600}
.status-committed{background:var(--green);color:#000}
.status-failed{background:var(--red);color:#fff}
.status-pending{background:var(--yellow);color:#000}
.main{display:grid;grid-template-columns:1fr 320px;gap:0;min-height:calc(100vh - 140px)}
.arena{padding:2rem;position:relative}
.sidebar{border-left:1px solid var(--border);padding:1.5rem;overflow-y:auto}
.sidebar h3{font-size:0.85rem;color:#8b949e;text-transform:uppercase;margin-bottom:0.8rem}

/* Agent circles */
.agent-ring{position:relative;width:100%;max-width:500px;height:400px;margin:0 auto}
.agent-node{position:absolute;width:64px;height:64px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:0.85rem;transition:all 0.5s ease;border:3px solid var(--border)}
.agent-node.honest{background:#1a3a2a;border-color:var(--green)}
.agent-node.byzantine{background:#3a1a1a;border-color:var(--red)}
.agent-node.leader{box-shadow:0 0 20px var(--accent);border-color:var(--accent);transform:scale(1.15)}
.agent-node.slashed{opacity:0.5;border-style:dashed}

/* Vote arrows */
.vote-layer{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.vote-line{stroke-width:2;opacity:0;transition:opacity 0.4s}
.vote-line.show{opacity:0.8}
.vote-line.accept{stroke:var(--green)}
.vote-line.reject{stroke:var(--red)}

/* Reputation bars */
.rep-bar-wrap{margin-bottom:0.6rem}
.rep-bar-label{display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:0.2rem}
.rep-bar{height:8px;background:var(--border);border-radius:4px;overflow:hidden}
.rep-bar-fill{height:100%;border-radius:4px;transition:width 0.6s ease}
.rep-bar-fill.full{background:var(--green)}
.rep-bar-fill.slashed{background:var(--yellow)}
.rep-bar-fill.low{background:var(--red)}

/* Aggregate gauge */
.gauge{margin:1.5rem 0}
.gauge-track{height:20px;background:var(--border);border-radius:10px;position:relative;overflow:visible}
.gauge-fill{height:100%;border-radius:10px;transition:width 0.8s ease;position:relative}
.gauge-fill.pass{background:var(--green)}
.gauge-fill.fail{background:var(--red)}
.gauge-threshold{position:absolute;top:-4px;height:28px;width:2px;background:var(--yellow)}
.gauge-label{display:flex;justify-content:space-between;font-size:0.75rem;color:#8b949e;margin-top:0.3rem}

/* Vote log */
.vote-log{margin-top:1rem}
.vote-entry{padding:0.4rem 0;border-bottom:1px solid var(--border);font-size:0.8rem;display:flex;gap:0.5rem;align-items:center}
.vote-entry .badge{padding:0.15rem 0.5rem;border-radius:4px;font-size:0.7rem;font-weight:600}
.badge-accept{background:#1a3a2a;color:var(--green)}
.badge-reject{background:#3a1a1a;color:var(--red)}

/* Timeline */
.timeline{display:flex;gap:0.5rem;margin-top:1rem}
.timeline-dot{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:0.7rem;font-weight:700;cursor:pointer;border:2px solid var(--border);transition:all 0.3s}
.timeline-dot.active{border-color:var(--accent);box-shadow:0 0 8px var(--accent)}
.timeline-dot.committed{background:#1a3a2a;border-color:var(--green)}
.timeline-dot.failed{background:#3a1a1a;border-color:var(--red)}

.final-banner{text-align:center;padding:1.5rem;margin-top:1rem;border-radius:8px;font-size:1.1rem;font-weight:600}
.final-banner.win{background:#1a3a2a;color:var(--green)}
.final-banner.lose{background:#3a1a1a;color:var(--red)}
</style></head><body>

<div class="header">
  <h1>🎬 mBFT Consensus Replay</h1>
  <div class="sub" id="subtitle"></div>
</div>

<div class="controls">
  <button id="btnPrev" onclick="prevRound()">◀ Prev</button>
  <button id="btnPlay" onclick="togglePlay()">▶ Play</button>
  <button id="btnNext" onclick="nextRound()">Next ▶</button>
  <div class="round-label" id="roundLabel">Round 0</div>
  <div id="statusPill"></div>
</div>

<div class="main">
  <div class="arena">
    <div class="agent-ring" id="ring">
      <svg class="vote-layer" id="voteSvg"></svg>
    </div>
    <div class="gauge" id="gaugeWrap">
      <div style="font-size:0.85rem;margin-bottom:0.3rem;font-weight:600">Aggregate Weight vs Threshold</div>
      <div class="gauge-track" id="gaugeTrack">
        <div class="gauge-fill" id="gaugeFill"></div>
        <div class="gauge-threshold" id="gaugeThreshold"></div>
      </div>
      <div class="gauge-label"><span id="gaugeVal">0</span><span id="gaugeMax"></span></div>
    </div>
    <div class="timeline" id="timeline"></div>
  </div>
  <div class="sidebar">
    <h3>Reputation</h3>
    <div id="repBars"></div>
    <h3 style="margin-top:1.5rem">Vote Log</h3>
    <div class="vote-log" id="voteLog"></div>
    <div id="finalBanner"></div>
  </div>
</div>

<script>
const DATA = """ + data_json + """;
const SPEED = """ + str(speed_ms) + """;
let currentRound = -1;
let playing = false;
let playTimer = null;

// Init
document.getElementById('subtitle').textContent =
  `${DATA.swarm_size} agents | ${DATA.byzantine_count} Byzantine | θ = ${DATA.threshold}`;

// Position agents in a circle
const ring = document.getElementById('ring');
const agents = DATA.agents;
const cx = 250, cy = 200, radius = 150;

agents.forEach((a, i) => {
  const angle = (2 * Math.PI * i / agents.length) - Math.PI / 2;
  const x = cx + radius * Math.cos(angle) - 32;
  const y = cy + radius * Math.sin(angle) - 32;
  const node = document.createElement('div');
  node.className = `agent-node ${a.is_byzantine ? 'byzantine' : 'honest'}`;
  node.id = `node-${a.agent_id}`;
  node.textContent = a.agent_id;
  node.style.left = x + 'px';
  node.style.top = y + 'px';
  node.title = `${a.agent_id} | conf=${a.confidence} | ${a.is_byzantine ? 'BYZANTINE' : 'honest'} | answer="${a.answer}"`;
  ring.appendChild(node);
});

// Build timeline dots
const timeline = document.getElementById('timeline');
DATA.rounds.forEach((r, i) => {
  const dot = document.createElement('div');
  dot.className = `timeline-dot ${r.committed ? 'committed' : 'failed'}`;
  dot.textContent = i;
  dot.onclick = () => showRound(i);
  dot.id = `tdot-${i}`;
  timeline.appendChild(dot);
});

// Gauge setup
const maxAgg = Math.max(...DATA.rounds.map(r => r.aggregate_weight), DATA.threshold) * 1.2;
document.getElementById('gaugeMax').textContent = maxAgg.toFixed(1);
const threshPct = (DATA.threshold / maxAgg) * 100;
document.getElementById('gaugeThreshold').style.left = threshPct + '%';

function showRound(idx) {
  if (idx < 0 || idx >= DATA.rounds.length) return;
  currentRound = idx;
  const r = DATA.rounds[idx];

  // Round label
  document.getElementById('roundLabel').textContent = `Round ${r.round_index}`;

  // Status pill
  const pill = document.getElementById('statusPill');
  if (r.committed) {
    pill.innerHTML = `<span class="status-pill status-committed">✅ COMMITTED</span>`;
  } else {
    pill.innerHTML = `<span class="status-pill status-failed">❌ NO COMMIT</span>`;
  }

  // Agent highlights
  agents.forEach(a => {
    const node = document.getElementById(`node-${a.agent_id}`);
    node.classList.remove('leader', 'slashed');
    if (a.agent_id === r.leader_id) node.classList.add('leader');
    if (r.slashed.includes(a.agent_id)) node.classList.add('slashed');
  });

  // Vote SVG arrows
  const svg = document.getElementById('voteSvg');
  svg.innerHTML = '';
  const leaderNode = document.getElementById(`node-${r.leader_id}`);
  const lx = parseInt(leaderNode.style.left) + 32;
  const ly = parseInt(leaderNode.style.top) + 32;

  r.votes.forEach((v, vi) => {
    const voterNode = document.getElementById(`node-${v.voter_id}`);
    const vx = parseInt(voterNode.style.left) + 32;
    const vy = parseInt(voterNode.style.top) + 32;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', vx); line.setAttribute('y1', vy);
    line.setAttribute('x2', lx); line.setAttribute('y2', ly);
    line.classList.add('vote-line', v.is_rejection ? 'reject' : 'accept');
    svg.appendChild(line);
    setTimeout(() => line.classList.add('show'), vi * 150);
  });

  // Gauge
  const pct = Math.min((r.aggregate_weight / maxAgg) * 100, 100);
  const gaugeFill = document.getElementById('gaugeFill');
  gaugeFill.style.width = pct + '%';
  gaugeFill.className = `gauge-fill ${r.committed ? 'pass' : 'fail'}`;
  document.getElementById('gaugeVal').textContent = r.aggregate_weight.toFixed(3);

  // Reputation bars
  const repDiv = document.getElementById('repBars');
  repDiv.innerHTML = '';
  agents.forEach(a => {
    const rep = r.reputations_after[a.agent_id] || 0;
    const cls = rep >= 1.0 ? 'full' : rep >= 0.5 ? 'slashed' : 'low';
    repDiv.innerHTML += `<div class="rep-bar-wrap">
      <div class="rep-bar-label"><span>${a.agent_id}${a.is_byzantine ? ' 🏴' : ''}</span><span>${(rep*100).toFixed(0)}%</span></div>
      <div class="rep-bar"><div class="rep-bar-fill ${cls}" style="width:${rep*100}%"></div></div></div>`;
  });

  // Vote log
  const logDiv = document.getElementById('voteLog');
  logDiv.innerHTML = `<div class="vote-entry"><strong>👑 Leader: ${r.leader_id}</strong> (conf=${r.leader_confidence})</div>`;
  r.votes.forEach(v => {
    const badge = v.is_rejection ? 'badge-reject' : 'badge-accept';
    const label = v.is_rejection ? 'REJECT' : 'ACCEPT';
    logDiv.innerHTML += `<div class="vote-entry">
      <span class="badge ${badge}">${label}</span>
      <span>${v.voter_id} → w=${v.weight.toFixed(2)} (eff=${v.effective_weight.toFixed(3)})</span>
    </div>`;
    if (v.counter_proof) {
      logDiv.innerHTML += `<div class="vote-entry" style="padding-left:1.5rem;color:#8b949e;font-style:italic">${v.counter_proof}</div>`;
    }
  });

  // Timeline active
  DATA.rounds.forEach((_, i) => {
    document.getElementById(`tdot-${i}`).classList.toggle('active', i === idx);
  });

  // Final banner
  const banner = document.getElementById('finalBanner');
  if (idx === DATA.rounds.length - 1) {
    if (DATA.final_committed) {
      banner.innerHTML = `<div class="final-banner win">🎉 Consensus Achieved: "${DATA.final_solution}"</div>`;
    } else {
      banner.innerHTML = `<div class="final-banner lose">💀 No Consensus Reached</div>`;
    }
  } else {
    banner.innerHTML = '';
  }

  // Button states
  document.getElementById('btnPrev').disabled = idx <= 0;
  document.getElementById('btnNext').disabled = idx >= DATA.rounds.length - 1;
}

function prevRound() { showRound(currentRound - 1); }
function nextRound() { showRound(currentRound + 1); }
function togglePlay() {
  playing = !playing;
  document.getElementById('btnPlay').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {
    if (currentRound >= DATA.rounds.length - 1) currentRound = -1;
    playTimer = setInterval(() => {
      if (currentRound >= DATA.rounds.length - 1) { togglePlay(); return; }
      nextRound();
    }, SPEED);
  } else {
    clearInterval(playTimer);
  }
}

// Start at round 0
if (DATA.rounds.length > 0) showRound(0);
</script></body></html>"""


async def main() -> None:
    parser = argparse.ArgumentParser(description="mBFT Consensus Replay Animator")
    parser.add_argument("--agents", "-n", type=int, default=5,
                        help="Number of agents (default: 5)")
    parser.add_argument("--byzantine", "-b", type=int, default=1,
                        help="Number of Byzantine agents (default: 1)")
    parser.add_argument("--threshold", "-t", type=float, default=1.5,
                        help="Consensus threshold θ (default: 1.5)")
    parser.add_argument("--speed", choices=["slow", "normal", "fast"],
                        default="normal", help="Animation speed")
    parser.add_argument("--output", "-o", type=str, default="consensus_replay.html",
                        help="Output HTML file (default: consensus_replay.html)")
    parser.add_argument("--export", choices=["json", "html"], default="html",
                        help="Export format (default: html)")
    args = parser.parse_args()

    if args.byzantine >= args.agents:
        print("Error: Byzantine count must be less than total agents.")
        return

    replay = await record_replay(
        swarm_size=args.agents,
        byzantine_count=args.byzantine,
        threshold=args.threshold,
    )

    if args.export == "json":
        print(_serialize(replay))
    else:
        html = render_html(replay, speed=args.speed)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Replay written to {args.output}")
        print(f"  Rounds: {len(replay.rounds)}")
        print(f"  Committed: {replay.final_committed}")
        if replay.final_solution:
            print(f"  Solution: {replay.final_solution}")


if __name__ == "__main__":
    asyncio.run(main())
