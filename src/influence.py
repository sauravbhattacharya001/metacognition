"""Consensus Influence Mapper — vote influence propagation, kingmaker
detection, power-asymmetry analysis for mBFT consensus rounds.

CLI usage::

    python -m src.influence --agents 7 --rounds 20 --output influence_report.html
"""
from __future__ import annotations

import argparse
import asyncio
import html as _html
import json
import random
from typing import Dict, List, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import RoundResult
from src.stats_utils import gini as _gini, pearson as _pearson


# ── helpers ──────────────────────────────────────────────────────────────




# ── simulation ───────────────────────────────────────────────────────────

async def _run_simulation(
    n_agents: int,
    byzantine_ratio: float,
    n_rounds: int,
    threshold: float,
) -> Tuple[List[str], List[RoundResult]]:
    """Run *n_rounds* independent consensus tasks and return agent ids + results."""
    n_byz = max(1, int(n_agents * byzantine_ratio))
    agents = []
    for i in range(n_agents):
        byz = i < n_byz
        conf = round(random.uniform(0.3, 0.95), 2)
        answer = random.choice(["A", "B"]) if byz else "A"
        agents.append(MockAgent(
            agent_id=f"agent-{i}",
            answer=answer,
            confidence=conf,
            byzantine=byz,
        ))

    ids = [a.id for a in agents]
    results: List[RoundResult] = []

    for r in range(n_rounds):
        engine = MBFTEngine(agents, threshold=threshold, max_rounds=4)
        task = f"consensus-task-{r}"
        res = await engine.run(task)
        if res is not None:
            results.append(res)

    return ids, results


# ── metrics ──────────────────────────────────────────────────────────────

def _compute_metrics(
    agent_ids: List[str],
    results: List[RoundResult],
    threshold: float,
) -> Dict:
    n = len(results)
    if n == 0:
        return {"agents": {}, "gini": 0.0, "coalitions": [], "timeline": []}

    # Build vote matrix: agent → list of weights per round
    vote_matrix: Dict[str, List[float]] = {aid: [] for aid in agent_ids}
    outcomes: List[int] = []  # 1=committed, 0=not

    for res in results:
        vote_map = {v.voter_id: v.weight for v in res.votes}
        for aid in agent_ids:
            vote_matrix[aid].append(vote_map.get(aid, 0.0))
        outcomes.append(1 if res.committed else 0)

    # Per-agent metrics
    agent_metrics: Dict[str, Dict] = {}
    for aid in agent_ids:
        weights = vote_matrix[aid]

        # Influence Radius: correlation with outcome
        influence_radius = _pearson(weights, [float(o) for o in outcomes])

        # Swing Power & Kingmaker
        swing = 0
        kingmaker = 0
        for i, res in enumerate(results):
            agg = res.aggregate_weight
            w = weights[i]
            # Would removing this agent's vote change the outcome?
            agg_without = agg - w
            committed_with = agg >= threshold
            committed_without = agg_without >= threshold
            if committed_with != committed_without:
                swing += 1
                # Kingmaker: the single decisive flip
                kingmaker += 1

        agent_metrics[aid] = {
            "swing_power": swing / n if n else 0,
            "kingmaker_score": kingmaker / n if n else 0,
            "influence_radius": round(influence_radius, 4),
            "avg_weight": round(sum(weights) / len(weights), 4) if weights else 0,
        }

    # Power Asymmetry (Gini)
    swing_values = [m["swing_power"] for m in agent_metrics.values()]
    gini = round(_gini(swing_values), 4)

    # Coalition Detection (Pearson > 0.6)
    coalitions = []
    ids_list = list(agent_ids)
    for i in range(len(ids_list)):
        for j in range(i + 1, len(ids_list)):
            corr = _pearson(vote_matrix[ids_list[i]], vote_matrix[ids_list[j]])
            if abs(corr) > 0.6:
                coalitions.append({
                    "agents": [ids_list[i], ids_list[j]],
                    "correlation": round(corr, 4),
                })

    # Timeline
    timeline = []
    for i, res in enumerate(results):
        timeline.append({
            "round": i,
            "committed": res.committed,
            "aggregate": round(res.aggregate_weight, 4),
            "leader": res.leader_id,
        })

    return {
        "agents": agent_metrics,
        "gini": gini,
        "coalitions": coalitions,
        "timeline": timeline,
    }


# ── recommendations ──────────────────────────────────────────────────────

def _recommendations(metrics: Dict) -> List[str]:
    recs = []
    agents = metrics["agents"]
    gini = metrics["gini"]

    if gini > 0.5:
        recs.append(f"⚠️ High power asymmetry (Gini={gini:.2f}) — influence is concentrated. Consider reputation rebalancing.")
    elif gini > 0.3:
        recs.append(f"Moderate power asymmetry (Gini={gini:.2f}) — monitor for increasing concentration.")

    for aid, m in agents.items():
        if m["kingmaker_score"] > 0.4:
            recs.append(f"🎯 {aid} is a kingmaker (score={m['kingmaker_score']:.2f}) — diversify voting power to reduce single-agent dependency.")
        if m["influence_radius"] < -0.3:
            recs.append(f"🔍 {aid} is a consistent contrarian (influence={m['influence_radius']:.2f}) — may be Byzantine or providing healthy dissent.")

    n_coalitions = len(metrics["coalitions"])
    if n_coalitions > 3:
        recs.append(f"🤝 {n_coalitions} coalitions detected — check for collusion risk.")

    if not recs:
        recs.append("✅ Power distribution looks healthy. No immediate concerns.")

    return recs


# ── HTML report ──────────────────────────────────────────────────────────

def _generate_html(metrics: Dict, recs: List[str], cfg: Dict) -> str:
    agents_json = json.dumps(metrics["agents"])
    timeline_json = json.dumps(metrics["timeline"])
    coalitions_json = json.dumps(metrics["coalitions"])
    recs_html = "".join(f"<li>{_html.escape(r)}</li>" for r in recs)

    # Agent table rows
    rows = ""
    for aid, m in sorted(metrics["agents"].items()):
        rows += f"""<tr>
            <td>{_html.escape(aid)}</td>
            <td>{m['swing_power']:.2%}</td>
            <td>{m['kingmaker_score']:.2%}</td>
            <td>{m['influence_radius']:.4f}</td>
            <td>{m['avg_weight']:.4f}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Consensus Influence Map</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px}}
h1{{color:#58a6ff;margin-bottom:8px}}
h2{{color:#79c0ff;margin:24px 0 12px}}
.subtitle{{color:#8b949e;margin-bottom:24px}}
table{{border-collapse:collapse;width:100%;margin-bottom:16px}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #21262d}}
th{{background:#161b22;color:#58a6ff}}
tr:hover{{background:#161b22}}
canvas{{border:1px solid #30363d;border-radius:8px;margin:12px 0}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:12px 0}}
.gauge-wrap{{display:flex;align-items:center;gap:16px}}
.recs li{{margin:8px 0;line-height:1.5}}
.cfg{{color:#8b949e;font-size:0.85em}}
</style></head><body>
<h1>🗺️ Consensus Influence Map</h1>
<p class="subtitle">mBFT Vote Influence Analysis</p>
<p class="cfg">Agents: {cfg['agents']} | Rounds: {cfg['rounds']} | Byzantine: {cfg['byzantine']:.0%} | Threshold: {cfg['threshold']}</p>

<h2>📊 Agent Influence Table</h2>
<table><thead><tr><th>Agent</th><th>Swing Power</th><th>Kingmaker</th><th>Influence Radius</th><th>Avg Weight</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>⚖️ Power Asymmetry</h2>
<div class="card gauge-wrap">
<canvas id="giniGauge" width="200" height="120"></canvas>
<div><strong>Gini Coefficient:</strong> {metrics['gini']:.4f}<br>
<span style="color:#8b949e">0 = perfect equality, 1 = total concentration</span></div>
</div>

<h2>📈 Power Distribution</h2>
<canvas id="powerChart" width="700" height="250"></canvas>

<h2>🔗 Coalition Network</h2>
<canvas id="coalitionGraph" width="700" height="400"></canvas>

<h2>📉 Swing Timeline</h2>
<canvas id="timeline" width="700" height="200"></canvas>

<h2>💡 Proactive Recommendations</h2>
<div class="card"><ul class="recs">{recs_html}</ul></div>

<script>
const agents = {agents_json};
const timeline = {timeline_json};
const coalitions = {coalitions_json};

// Gini gauge
(function(){{
  const c=document.getElementById('giniGauge'),ctx=c.getContext('2d');
  const g={metrics['gini']};
  ctx.beginPath();ctx.arc(100,100,80,Math.PI,2*Math.PI);
  ctx.strokeStyle='#21262d';ctx.lineWidth=16;ctx.stroke();
  ctx.beginPath();ctx.arc(100,100,80,Math.PI,Math.PI+g*Math.PI);
  ctx.strokeStyle=g>0.5?'#f85149':g>0.3?'#d29922':'#3fb950';ctx.lineWidth=16;ctx.stroke();
  ctx.fillStyle='#c9d1d9';ctx.font='bold 20px system-ui';ctx.textAlign='center';
  ctx.fillText(g.toFixed(3),100,90);
}})();

// Power distribution bar chart
(function(){{
  const c=document.getElementById('powerChart'),ctx=c.getContext('2d');
  const ids=Object.keys(agents),n=ids.length;
  const bw=Math.min(60,600/n),gap=10;
  const startX=50;
  ids.forEach((id,i)=>{{
    const m=agents[id];
    const x=startX+i*(bw+gap);
    // Swing bar
    const h1=m.swing_power*180;
    ctx.fillStyle='#58a6ff';ctx.fillRect(x,220-h1,bw/2-1,h1);
    // Kingmaker bar
    const h2=m.kingmaker_score*180;
    ctx.fillStyle='#f0883e';ctx.fillRect(x+bw/2+1,220-h2,bw/2-1,h2);
    // Label
    ctx.fillStyle='#8b949e';ctx.font='10px system-ui';ctx.textAlign='center';
    ctx.fillText(id.replace('agent-','A'),x+bw/2,238);
  }});
  ctx.fillStyle='#58a6ff';ctx.fillRect(startX,8,12,12);
  ctx.fillStyle='#c9d1d9';ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillText('Swing',startX+16,18);
  ctx.fillStyle='#f0883e';ctx.fillRect(startX+70,8,12,12);
  ctx.fillStyle='#c9d1d9';ctx.fillText('Kingmaker',startX+86,18);
}})();

// Coalition network graph
(function(){{
  const c=document.getElementById('coalitionGraph'),ctx=c.getContext('2d');
  const ids=Object.keys(agents),n=ids.length;
  const cx=350,cy=200,r=150;
  const pos={{}};
  ids.forEach((id,i)=>{{
    const a=-Math.PI/2+2*Math.PI*i/n;
    pos[id]={{x:cx+r*Math.cos(a),y:cy+r*Math.sin(a)}};
  }});
  // Edges
  coalitions.forEach(co=>{{
    const a=pos[co.agents[0]],b=pos[co.agents[1]];
    if(!a||!b)return;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
    ctx.strokeStyle=co.correlation>0?'rgba(63,185,80,0.5)':'rgba(248,81,73,0.5)';
    ctx.lineWidth=Math.abs(co.correlation)*4;ctx.stroke();
  }});
  // Nodes
  ids.forEach(id=>{{
    const p=pos[id],m=agents[id];
    const sz=8+m.swing_power*30;
    ctx.beginPath();ctx.arc(p.x,p.y,sz,0,2*Math.PI);
    ctx.fillStyle=m.kingmaker_score>0.3?'#f0883e':'#58a6ff';ctx.fill();
    ctx.strokeStyle='#c9d1d9';ctx.lineWidth=1;ctx.stroke();
    ctx.fillStyle='#c9d1d9';ctx.font='11px system-ui';ctx.textAlign='center';
    ctx.fillText(id.replace('agent-','A'),p.x,p.y+sz+14);
  }});
}})();

// Timeline
(function(){{
  const c=document.getElementById('timeline'),ctx=c.getContext('2d');
  const n=timeline.length;if(!n)return;
  const sx=50,w=600,h=160;
  const step=w/Math.max(n-1,1);
  ctx.strokeStyle='#30363d';ctx.beginPath();ctx.moveTo(sx,h);ctx.lineTo(sx+w,h);ctx.stroke();
  // Aggregate line
  ctx.beginPath();
  timeline.forEach((t,i)=>{{
    const x=sx+i*step,y=h-t.aggregate/2*h;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});
  ctx.strokeStyle='#58a6ff';ctx.lineWidth=2;ctx.stroke();
  // Dots
  timeline.forEach((t,i)=>{{
    const x=sx+i*step,y=h-t.aggregate/2*h;
    ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);
    ctx.fillStyle=t.committed?'#3fb950':'#f85149';ctx.fill();
  }});
  ctx.fillStyle='#8b949e';ctx.font='10px system-ui';ctx.textAlign='left';
  ctx.fillText('Aggregate weight over rounds (green=committed, red=failed)',sx,12);
}})();
</script></body></html>"""


# ── main ─────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Consensus Influence Mapper")
    ap.add_argument("--agents", type=int, default=7, help="Number of agents")
    ap.add_argument("--byzantine", type=float, default=0.2, help="Byzantine fraction")
    ap.add_argument("--rounds", type=int, default=20, help="Consensus tasks")
    ap.add_argument("--threshold", type=float, default=0.6, help="Commit threshold")
    ap.add_argument("--output", default="influence_report.html", help="HTML output")
    ap.add_argument("--json", action="store_true", help="Also export JSON")
    args = ap.parse_args(argv)

    print("Consensus Influence Mapper")
    print(f"   {args.agents} agents | {args.rounds} rounds | byzantine={args.byzantine:.0%} | threshold={args.threshold}")

    ids, results = asyncio.run(_run_simulation(args.agents, args.byzantine, args.rounds, args.threshold))
    metrics = _compute_metrics(ids, results, args.threshold)
    recs = _recommendations(metrics)

    cfg = {"agents": args.agents, "rounds": args.rounds, "byzantine": args.byzantine, "threshold": args.threshold}
    html = _generate_html(metrics, recs, cfg)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   HTML report -> {args.output}")

    if args.json:
        jpath = args.output.replace(".html", ".json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "metrics": metrics, "recommendations": recs}, f, indent=2)
        print(f"   JSON export -> {jpath}")

    # Summary
    print(f"\n   Gini coefficient: {metrics['gini']:.4f}")
    print(f"   Coalitions found: {len(metrics['coalitions'])}")
    for r in recs:
        clean = r.encode('ascii', 'replace').decode('ascii')
        print(f"   {clean}")


if __name__ == "__main__":
    main()
