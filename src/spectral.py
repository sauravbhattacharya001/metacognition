"""Consensus Spectral Analyzer — frequency-domain analysis of voting dynamics.

Applies FFT to voting weight time-series from mBFT consensus runs to detect
periodic behaviors, oscillations, dominant frequencies, and phase relationships.
Produces interactive HTML reports with frequency spectra, spectrograms, and
phase coherence analysis.

Usage::

    python -m src.spectral --agents 7 --byzantine 2 --rounds 64 --tasks 20
    python -m src.spectral --agents 10 --rounds 128 --harmonic-report
    python -m src.spectral --help

Features:
- Per-agent FFT power spectra of voting weight time-series
- Fleet-wide spectrogram (agents × frequency)
- Dominant frequency extraction with period interpretation
- Phase coherence analysis across agent pairs
- Oscillation detection (agents stuck in flip-flop patterns)
- Resonance detection (synchronized voting behavior)
- Auto-monitor mode for continuous spectral surveillance
- Interactive HTML report with charts and tables
"""
from __future__ import annotations

import argparse
import asyncio
import html as html_mod
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Minimal FFT (no numpy dependency)
# ---------------------------------------------------------------------------

def _fft(x: List[complex]) -> List[complex]:
    """Radix-2 Cooley-Tukey FFT. Pads to next power of 2."""
    n = len(x)
    if n <= 1:
        return list(x)
    p = 1
    while p < n:
        p <<= 1
    x = list(x) + [0j] * (p - n)
    n = p
    if n == 1:
        return x
    even = _fft(x[0::2])
    odd = _fft(x[1::2])
    T = [math.e ** (-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + \
           [even[k] - T[k] for k in range(n // 2)]


def power_spectrum(signal: List[float]) -> Tuple[List[float], List[float]]:
    """Return (frequencies_normalized, power) for a real signal."""
    n = len(signal)
    if n < 2:
        return [], []
    mean = sum(signal) / n
    centered = [complex(v - mean) for v in signal]
    F = _fft(centered)
    N = len(F)
    half = N // 2
    freqs = [k / N for k in range(half)]
    power = [(F[k].real ** 2 + F[k].imag ** 2) / N for k in range(half)]
    return freqs, power


def phase_angles(signal: List[float]) -> List[float]:
    """Return phase angle at dominant frequency for a signal."""
    n = len(signal)
    if n < 2:
        return []
    mean = sum(signal) / n
    centered = [complex(v - mean) for v in signal]
    F = _fft(centered)
    return [math.atan2(c.imag, c.real) for c in F[:len(F) // 2]]


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

async def _run_spectral_sim(
    num_agents: int = 7,
    num_byzantine: int = 2,
    num_rounds: int = 64,
    num_tasks: int = 20,
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """Run consensus simulations and collect voting time-series."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.agents.metacognitive import MockAgent
    from src.core.protocol import MBFTEngine

    import random
    rng = random.Random(42)

    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    byz_ids = set(agent_ids[:num_byzantine])
    answers = ["alpha", "beta", "gamma", "delta"]

    voting_series: Dict[str, List[float]] = {a: [] for a in agent_ids}
    commitment_series: List[int] = []
    aggregate_series: List[float] = []

    for t in range(num_tasks):
        correct = rng.choice(answers)
        agents = []
        for aid in agent_ids:
            is_byz = aid in byz_ids
            if is_byz:
                ans = rng.choice(answers)
                conf = rng.uniform(0.1, 0.9)
            else:
                ans = correct if rng.random() > 0.15 else rng.choice(answers)
                conf = rng.uniform(0.5, 0.95)
            a = MockAgent(
                agent_id=aid,
                answer=ans,
                confidence=round(conf, 3),
                byzantine=is_byz,
                accept_set={correct, ans},
            )
            agents.append(a)

        engine = MBFTEngine(agents, threshold=threshold, max_rounds=num_rounds)
        task_prompt = f"spectral_task_{t}_{rng.randint(0,9999)}"

        result = await engine.run(task_prompt)

        for rr in engine.history:
            vote_map = {v.voter_id: v.weight for v in rr.votes}
            for aid in agent_ids:
                voting_series[aid].append(vote_map.get(aid, 0.0))
            aggregate_series.append(rr.aggregate_weight)
            commitment_series.append(1 if rr.committed else 0)

    return {
        "agent_ids": agent_ids,
        "byzantine_ids": list(byz_ids),
        "voting_series": voting_series,
        "aggregate_series": aggregate_series,
        "commitment_series": commitment_series,
        "num_tasks": num_tasks,
        "num_rounds": num_rounds,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_spectral(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform spectral analysis on collected voting data."""
    agent_ids = data["agent_ids"]
    voting_series = data["voting_series"]
    aggregate_series = data["aggregate_series"]
    byz_ids = set(data["byzantine_ids"])

    results: Dict[str, Any] = {
        "agents": {},
        "fleet_spectrogram": {},
        "phase_coherence": [],
        "oscillators": [],
        "resonance_groups": [],
        "aggregate_spectrum": {},
        "recommendations": [],
    }

    for aid in agent_ids:
        series = voting_series[aid]
        freqs, pwr = power_spectrum(series)
        if not freqs:
            continue

        max_idx = max(range(len(pwr)), key=lambda i: pwr[i]) if pwr else 0
        dom_freq = freqs[max_idx] if freqs else 0
        dom_power = pwr[max_idx] if pwr else 0
        total_power = sum(pwr) or 1e-9
        spectral_concentration = dom_power / total_power
        period = (1.0 / dom_freq) if dom_freq > 0 else float("inf")

        results["agents"][aid] = {
            "dominant_freq": round(dom_freq, 4),
            "dominant_power": round(dom_power, 4),
            "period": round(period, 2),
            "spectral_concentration": round(spectral_concentration, 4),
            "total_power": round(total_power, 4),
            "is_byzantine": aid in byz_ids,
            "top_freqs": [
                {"freq": round(freqs[i], 4), "power": round(pwr[i], 4)}
                for i in sorted(range(len(pwr)), key=lambda i: -pwr[i])[:5]
            ],
        }

        if spectral_concentration > 0.5 and period < 10:
            results["oscillators"].append({
                "agent": aid,
                "frequency": round(dom_freq, 4),
                "period": round(period, 2),
                "concentration": round(spectral_concentration, 4),
                "is_byzantine": aid in byz_ids,
            })

    # Phase coherence
    phases_at_dom: Dict[str, float] = {}
    for aid in agent_ids:
        series = voting_series[aid]
        ph = phase_angles(series)
        if ph:
            ag = results["agents"].get(aid)
            if ag:
                dom_idx = 0
                freqs, pwr = power_spectrum(series)
                if pwr:
                    dom_idx = max(range(len(pwr)), key=lambda i: pwr[i])
                if dom_idx < len(ph):
                    phases_at_dom[aid] = ph[dom_idx]

    aids_with_phase = list(phases_at_dom.keys())
    for i in range(len(aids_with_phase)):
        for j in range(i + 1, len(aids_with_phase)):
            a1, a2 = aids_with_phase[i], aids_with_phase[j]
            diff = abs(phases_at_dom[a1] - phases_at_dom[a2])
            diff = diff % (2 * math.pi)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            coherence = 1.0 - (diff / math.pi)
            results["phase_coherence"].append({
                "agent_a": a1,
                "agent_b": a2,
                "phase_diff": round(diff, 4),
                "coherence": round(coherence, 4),
            })

    # Resonance groups via union-find
    high_coherence = [pc for pc in results["phase_coherence"] if pc["coherence"] > 0.8]
    groups: Dict[str, set] = {}
    for pc in high_coherence:
        a, b = pc["agent_a"], pc["agent_b"]
        ga = groups.get(a)
        gb = groups.get(b)
        if ga and gb:
            if ga is not gb:
                ga.update(gb)
                for m in gb:
                    groups[m] = ga
        elif ga:
            ga.add(b)
            groups[b] = ga
        elif gb:
            gb.add(a)
            groups[a] = gb
        else:
            new_g = {a, b}
            groups[a] = new_g
            groups[b] = new_g

    seen_groups: List[frozenset] = []
    for g in groups.values():
        fg = frozenset(g)
        if fg not in seen_groups and len(fg) > 1:
            seen_groups.append(fg)
            results["resonance_groups"].append({
                "members": sorted(fg),
                "size": len(fg),
                "has_byzantine": bool(fg & byz_ids),
            })

    # Aggregate spectrum
    freqs, pwr = power_spectrum(aggregate_series)
    if freqs:
        max_idx = max(range(len(pwr)), key=lambda i: pwr[i])
        results["aggregate_spectrum"] = {
            "dominant_freq": round(freqs[max_idx], 4),
            "dominant_power": round(pwr[max_idx], 4),
            "period": round(1.0 / freqs[max_idx], 2) if freqs[max_idx] > 0 else None,
            "top_freqs": [
                {"freq": round(freqs[i], 4), "power": round(pwr[i], 4)}
                for i in sorted(range(len(pwr)), key=lambda i: -pwr[i])[:5]
            ],
        }

    # Fleet spectrogram
    all_freqs = None
    spec_data = {}
    for aid in agent_ids:
        freqs, pwr = power_spectrum(voting_series[aid])
        if all_freqs is None:
            all_freqs = freqs
        spec_data[aid] = pwr
    results["fleet_spectrogram"] = {
        "frequencies": [round(f, 4) for f in (all_freqs or [])],
        "agents": {aid: [round(p, 4) for p in pwr] for aid, pwr in spec_data.items()},
    }

    # Recommendations
    recs = results["recommendations"]
    if results["oscillators"]:
        byz_osc = [o for o in results["oscillators"] if o["is_byzantine"]]
        honest_osc = [o for o in results["oscillators"] if not o["is_byzantine"]]
        if byz_osc:
            recs.append(f"[!] {len(byz_osc)} Byzantine agent(s) show strong oscillation -- possible flip-flop attack detected")
        if honest_osc:
            recs.append(f"[!] {len(honest_osc)} honest agent(s) oscillating -- may indicate unstable confidence calibration")

    if results["resonance_groups"]:
        byz_groups = [g for g in results["resonance_groups"] if g["has_byzantine"]]
        if byz_groups:
            recs.append(f"[!] {len(byz_groups)} resonance group(s) include Byzantine agents -- possible coordinated voting")
        clean = [g for g in results["resonance_groups"] if not g["has_byzantine"]]
        if clean:
            recs.append(f"[OK] {len(clean)} honest resonance group(s) detected -- healthy consensus convergence")

    agg = results["aggregate_spectrum"]
    if agg and agg.get("period") and agg["period"] < 8:
        recs.append(f"[!] Aggregate consensus oscillates with period ~{agg['period']:.1f} rounds -- consider adjusting threshold or slash factor")

    if not recs:
        recs.append("[OK] No significant spectral anomalies detected -- consensus dynamics appear stable")

    return results


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Generate interactive HTML report."""
    agents_json = json.dumps(analysis["agents"], indent=2)
    spectrogram_json = json.dumps(analysis["fleet_spectrogram"])
    coherence_json = json.dumps(analysis["phase_coherence"])
    oscillators_json = json.dumps(analysis["oscillators"])
    resonance_json = json.dumps(analysis["resonance_groups"])
    aggregate_json = json.dumps(analysis["aggregate_spectrum"])
    recs_html = "".join(f"<li>{html_mod.escape(r)}</li>" for r in analysis["recommendations"])

    agent_rows = ""
    for aid in sorted(analysis["agents"].keys()):
        ag = analysis["agents"][aid]
        byz = "Byzantine" if ag["is_byzantine"] else "Honest"
        agent_rows += f"""<tr>
            <td>{html_mod.escape(aid)}</td><td>{byz}</td>
            <td>{ag['dominant_freq']}</td><td>{ag['period']}</td>
            <td>{ag['spectral_concentration']}</td><td>{ag['total_power']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Consensus Spectral Analyzer</title>
<style>
:root {{ --bg: #0a0a0f; --card: #12121a; --border: #2a2a3a; --text: #e0e0e0; --accent: #7c5cff; --warn: #ff6b6b; --ok: #4ecdc4; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:20px; }}
h1 {{ color:var(--accent); margin-bottom:8px; }} h2 {{ color:var(--accent); margin:20px 0 10px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:8px 12px; text-align:left; border-bottom:1px solid var(--border); }}
th {{ color:var(--accent); }} tr:hover {{ background:#1a1a2a; }}
canvas {{ width:100%; height:300px; border-radius:8px; }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }}
.tabs {{ display:flex; gap:8px; margin-bottom:12px; }}
.tab {{ padding:8px 16px; border-radius:6px; cursor:pointer; background:var(--border); color:var(--text); border:none; }}
.tab.active {{ background:var(--accent); color:#fff; }}
.panel {{ display:none; }} .panel.active {{ display:block; }}
</style></head><body>
<h1>Consensus Spectral Analyzer</h1>
<p>Frequency-domain analysis of mBFT voting dynamics</p>
<p style="color:#888;">Agents: {len(data['agent_ids'])} | Byzantine: {len(data['byzantine_ids'])} | Tasks: {data['num_tasks']} | Threshold: {data['threshold']}</p>

<div class="tabs">
  <button class="tab active" onclick="showTab('overview')">Overview</button>
  <button class="tab" onclick="showTab('spectra')">Agent Spectra</button>
  <button class="tab" onclick="showTab('spectrogram')">Spectrogram</button>
  <button class="tab" onclick="showTab('phase')">Phase Coherence</button>
  <button class="tab" onclick="showTab('detection')">Detection</button>
</div>

<div id="overview" class="panel active card">
  <h2>Recommendations</h2>
  <ul>{recs_html}</ul>
  <h2 style="margin-top:16px;">Aggregate Consensus Spectrum</h2>
  <canvas id="aggChart"></canvas>
</div>

<div id="spectra" class="panel card">
  <h2>Per-Agent Spectral Analysis</h2>
  <table><thead><tr>
    <th>Agent</th><th>Type</th><th>Dom. Freq</th><th>Period</th><th>Concentration</th><th>Total Power</th>
  </tr></thead><tbody>{agent_rows}</tbody></table>
  <h2 style="margin-top:16px;">Power Spectra</h2>
  <canvas id="spectraChart"></canvas>
</div>

<div id="spectrogram" class="panel card">
  <h2>Fleet Spectrogram</h2>
  <canvas id="spectrogramCanvas" style="height:400px;"></canvas>
  <p style="color:#888;margin-top:8px;">X-axis: frequency | Y-axis: agent | Color: power intensity</p>
</div>

<div id="phase" class="panel card">
  <h2>Phase Coherence Matrix</h2>
  <canvas id="phaseCanvas" style="height:400px;"></canvas>
</div>

<div id="detection" class="panel card">
  <h2>Oscillators Detected: {len(analysis['oscillators'])}</h2>
  <table><thead><tr><th>Agent</th><th>Frequency</th><th>Period</th><th>Concentration</th><th>Byzantine</th></tr></thead><tbody>
  {"".join(f'<tr><td>{o["agent"]}</td><td>{o["frequency"]}</td><td>{o["period"]}</td><td>{o["concentration"]}</td><td>{"Yes" if o["is_byzantine"] else "No"}</td></tr>' for o in analysis['oscillators'])}
  </tbody></table>
  <h2 style="margin-top:16px;">Resonance Groups: {len(analysis['resonance_groups'])}</h2>
  {"".join(f'<div class="card"><b>Group ({g["size"]} agents)</b>: {", ".join(g["members"])} {"[Has Byzantine]" if g["has_byzantine"] else "[Clean]"}</div>' for g in analysis['resonance_groups'])}
</div>

<script>
const agents = {agents_json};
const spectrogram = {spectrogram_json};
const coherence = {coherence_json};
const aggregate = {aggregate_json};

function showTab(id) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}

function drawBarChart(canvasId, labels, datasets) {{
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  c.width = c.offsetWidth * 2; c.height = c.offsetHeight * 2;
  ctx.scale(2, 2);
  const w = c.offsetWidth, h = c.offsetHeight;
  const pad = {{l:50,r:20,t:20,b:30}};
  const pw = w-pad.l-pad.r, ph = h-pad.t-pad.b;
  let maxV = 0;
  datasets.forEach(ds => ds.data.forEach(v => {{ if(v>maxV) maxV=v; }}));
  if(maxV === 0) maxV = 1;
  const colors = ['#7c5cff','#ff6b6b','#4ecdc4','#ffd93d','#6bcb77','#ff8e72','#a78bfa'];
  ctx.fillStyle = '#888'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  for(let i=0;i<=4;i++) {{
    const y = pad.t + ph - (i/4)*ph;
    ctx.fillText((maxV*i/4).toFixed(2), pad.l-5, y+3);
    ctx.strokeStyle = '#2a2a3a'; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
  }}
  datasets.forEach((ds, di) => {{
    ctx.strokeStyle = colors[di % colors.length]; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ds.data.forEach((v, i) => {{
      const x = pad.l + (i / (ds.data.length-1||1)) * pw;
      const y = pad.t + ph - (v/maxV)*ph;
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }});
}}

function drawSpectrogram() {{
  const c = document.getElementById('spectrogramCanvas');
  if(!c) return;
  const ctx = c.getContext('2d');
  c.width = c.offsetWidth*2; c.height = c.offsetHeight*2;
  ctx.scale(2,2);
  const w=c.offsetWidth, h=c.offsetHeight;
  const aids = Object.keys(spectrogram.agents);
  const freqs = spectrogram.frequencies;
  if(!aids.length||!freqs.length) return;
  let maxP = 0;
  aids.forEach(a => spectrogram.agents[a].forEach(v => {{ if(v>maxP) maxP=v; }}));
  if(maxP===0) maxP=1;
  const pad={{l:80,r:20,t:20,b:30}};
  const cw=(w-pad.l-pad.r)/freqs.length;
  const ch=(h-pad.t-pad.b)/aids.length;
  aids.forEach((aid,yi) => {{
    const pwr = spectrogram.agents[aid];
    pwr.forEach((v,xi) => {{
      const intensity = Math.min(v/maxP, 1);
      const r = Math.floor(124*intensity + 10);
      const g = Math.floor(92*intensity*(1-intensity*0.5));
      const b = Math.floor(255*intensity);
      ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
      ctx.fillRect(pad.l+xi*cw, pad.t+yi*ch, cw+1, ch+1);
    }});
    ctx.fillStyle='#888'; ctx.font='9px sans-serif'; ctx.textAlign='right';
    ctx.fillText(aid, pad.l-4, pad.t+yi*ch+ch/2+3);
  }});
}}

function drawPhaseMatrix() {{
  const c = document.getElementById('phaseCanvas');
  if(!c||!coherence.length) return;
  const ctx = c.getContext('2d');
  c.width=c.offsetWidth*2; c.height=c.offsetHeight*2;
  ctx.scale(2,2);
  const w=c.offsetWidth, h=c.offsetHeight;
  const aidSet = new Set();
  coherence.forEach(pc => {{ aidSet.add(pc.agent_a); aidSet.add(pc.agent_b); }});
  const aids = [...aidSet].sort();
  const n = aids.length;
  if(!n) return;
  const pad={{l:80,r:20,t:80,b:20}};
  const cs = Math.min((w-pad.l-pad.r)/n, (h-pad.t-pad.b)/n);
  const mat = {{}};
  coherence.forEach(pc => {{
    if(!mat[pc.agent_a]) mat[pc.agent_a]={{}};
    if(!mat[pc.agent_b]) mat[pc.agent_b]={{}};
    mat[pc.agent_a][pc.agent_b] = pc.coherence;
    mat[pc.agent_b][pc.agent_a] = pc.coherence;
  }});
  aids.forEach((a,yi) => {{
    aids.forEach((b,xi) => {{
      const v = (a===b) ? 1.0 : ((mat[a]||{{}})[b]||0);
      const r = Math.floor(78 + 177*v);
      const g = Math.floor(92*(1-v) + 205*v);
      const bl = Math.floor(255*(1-v) + 196*v);
      ctx.fillStyle = `rgb(${{r}},${{g}},${{bl}})`;
      ctx.fillRect(pad.l+xi*cs, pad.t+yi*cs, cs, cs);
    }});
    ctx.fillStyle='#888'; ctx.font='9px sans-serif'; ctx.textAlign='right';
    ctx.fillText(a, pad.l-4, pad.t+yi*cs+cs/2+3);
    ctx.save(); ctx.translate(pad.l+yi*cs+cs/2, pad.t-4); ctx.rotate(-Math.PI/4);
    ctx.textAlign='left'; ctx.fillText(a, 0, 0); ctx.restore();
  }});
}}

window.addEventListener('load', () => {{
  if(aggregate && aggregate.top_freqs) {{
    drawBarChart('aggChart', aggregate.top_freqs.map(f=>f.freq.toFixed(3)), [{{data: aggregate.top_freqs.map(f=>f.power)}}]);
  }}
  const ds = [];
  Object.keys(agents).sort().forEach(aid => {{
    const ag = agents[aid];
    if(ag.top_freqs) ds.push({{label:aid, data:ag.top_freqs.map(f=>f.power)}});
  }});
  if(ds.length) drawBarChart('spectraChart', [], ds);
  drawSpectrogram();
  drawPhaseMatrix();
}});
window.addEventListener('resize', () => {{ drawSpectrogram(); drawPhaseMatrix(); }});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consensus Spectral Analyzer -- frequency-domain voting analysis"
    )
    parser.add_argument("--agents", type=int, default=7, help="Number of agents")
    parser.add_argument("--byzantine", type=int, default=2, help="Number of Byzantine agents")
    parser.add_argument("--rounds", type=int, default=64, help="Max rounds per task")
    parser.add_argument("--tasks", type=int, default=20, help="Number of tasks to simulate")
    parser.add_argument("--threshold", type=float, default=0.6, help="Consensus threshold")
    parser.add_argument("--output", type=str, default=None, help="Output HTML file path")
    parser.add_argument("--json", action="store_true", help="Print JSON analysis to stdout")
    parser.add_argument("--harmonic-report", action="store_true", help="Print text summary")
    args = parser.parse_args()

    print(f"Running spectral analysis: {args.agents} agents, {args.byzantine} Byzantine, {args.tasks} tasks...")
    data = await _run_spectral_sim(
        num_agents=args.agents,
        num_byzantine=args.byzantine,
        num_rounds=args.rounds,
        num_tasks=args.tasks,
        threshold=args.threshold,
    )

    print("Analyzing frequency spectra...")
    analysis = analyze_spectral(data)

    if args.json:
        print(json.dumps(analysis, indent=2))
        return

    if args.harmonic_report:
        print("\n=== SPECTRAL ANALYSIS REPORT ===\n")
        for aid in sorted(analysis["agents"].keys()):
            ag = analysis["agents"][aid]
            byz = " [BYZANTINE]" if ag["is_byzantine"] else ""
            print(f"  {aid}{byz}: dom_freq={ag['dominant_freq']:.4f} period={ag['period']:.1f} concentration={ag['spectral_concentration']:.4f}")

        if analysis["oscillators"]:
            print(f"\nOscillators: {len(analysis['oscillators'])}")
            for o in analysis["oscillators"]:
                print(f"  {o['agent']}: freq={o['frequency']} period={o['period']}")

        if analysis["resonance_groups"]:
            print(f"\nResonance Groups: {len(analysis['resonance_groups'])}")
            for g in analysis["resonance_groups"]:
                print(f"  [{', '.join(g['members'])}] {'(has Byzantine)' if g['has_byzantine'] else '(clean)'}")

        print("\nRecommendations:")
        for r in analysis["recommendations"]:
            print(f"  {r}")
        print()
        return

    out_path = args.output or "spectral_report.html"
    html = generate_html_report(data, analysis)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
