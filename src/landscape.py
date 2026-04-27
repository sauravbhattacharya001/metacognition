"""Consensus Stability Landscape — autonomous phase-transition mapper.

Sweeps the (threshold × byzantine_ratio) parameter space to discover
stability regions, phase boundaries, and tipping points in mBFT consensus.

Generates an interactive HTML heatmap report showing:
- Commit-rate heatmap across the parameter grid
- Phase boundary detection (gradient magnitude)
- Critical tipping-point identification
- Stability region classification
- Proactive configuration recommendations

Usage::

    python -m src.landscape                        # defaults: 10 agents, 15×15 grid
    python -m src.landscape --agents 20 --res 25   # higher resolution
    python -m src.landscape --autopilot             # auto-detect critical zones & zoom
    python -m src.landscape -o report.html          # custom output path
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── project imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.metacognitive import MockAgent  # noqa: E402
from src.core.protocol import MBFTEngine  # noqa: E402


# ── data models ──────────────────────────────────────────────────────────

@dataclass
class CellResult:
    threshold: float
    byzantine_ratio: float
    commit_rate: float
    avg_rounds: float
    avg_aggregate: float
    trials: int


@dataclass
class PhaseRegion:
    name: str
    description: str
    cells: int
    avg_commit_rate: float
    threshold_range: Tuple[float, float]
    byz_range: Tuple[float, float]


@dataclass
class TippingPoint:
    threshold: float
    byzantine_ratio: float
    gradient_magnitude: float
    direction: str  # "threshold" or "byzantine"


@dataclass
class LandscapeReport:
    grid: List[List[CellResult]]
    regions: List[PhaseRegion]
    tipping_points: List[TippingPoint]
    recommendations: List[str]
    elapsed_sec: float
    n_agents: int
    resolution: int
    trials_per_cell: int


# ── core sweep logic ────────────────────────────────────────────────────

async def _run_cell(
    n_agents: int,
    threshold: float,
    byzantine_ratio: float,
    trials: int,
    max_rounds: int,
) -> CellResult:
    """Run *trials* consensus rounds for one (threshold, byz_ratio) cell."""
    n_byz = max(0, min(n_agents, int(round(n_agents * byzantine_ratio))))
    commits = 0
    total_rounds = 0
    total_agg = 0.0

    for _ in range(trials):
        agents = []
        for i in range(n_agents):
            is_byz = i < n_byz
            agents.append(
                MockAgent(
                    agent_id=f"a{i}",
                    answer="correct" if not is_byz else "wrong",
                    confidence=0.8 if not is_byz else 0.6,
                    byzantine=is_byz,
                )
            )
        engine = MBFTEngine(agents, threshold=threshold, max_rounds=max_rounds)
        result = await engine.run("stability-probe")
        if result and result.committed:
            commits += 1
        total_rounds += len(engine.history)
        if engine.history:
            total_agg += engine.history[-1].aggregate_weight

    return CellResult(
        threshold=threshold,
        byzantine_ratio=byzantine_ratio,
        commit_rate=commits / trials if trials else 0.0,
        avg_rounds=total_rounds / trials if trials else 0.0,
        avg_aggregate=total_agg / trials if trials else 0.0,
        trials=trials,
    )


async def sweep(
    n_agents: int = 10,
    resolution: int = 15,
    trials: int = 8,
    max_rounds: int = 4,
    progress_cb=None,
) -> List[List[CellResult]]:
    """Sweep the parameter grid and return a 2-D matrix of CellResults.

    Rows index threshold (ascending), columns index byzantine_ratio (ascending).
    """
    thresholds = [round(i / (resolution - 1), 4) for i in range(resolution)]
    byz_ratios = [round(i / (resolution - 1), 4) for i in range(resolution)]

    # cap threshold at practical range
    thresholds = [max(0.1, t * (n_agents * 1.0)) for t in _linspace(0.1, n_agents * 0.95, resolution)]

    # re-normalise to nice round numbers
    thresholds = _linspace(0.5, n_agents * 0.9, resolution)
    byz_ratios = _linspace(0.0, 0.6, resolution)

    grid: List[List[CellResult]] = []
    total = resolution * resolution
    done = 0

    for ti, thr in enumerate(thresholds):
        row: List[CellResult] = []
        for bi, byz in enumerate(byz_ratios):
            cell = await _run_cell(n_agents, thr, byz, trials, max_rounds)
            row.append(cell)
            done += 1
            if progress_cb:
                progress_cb(done, total)
        grid.append(row)
    return grid


def _linspace(start: float, stop: float, n: int) -> List[float]:
    if n <= 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [round(start + i * step, 4) for i in range(n)]


# ── analysis ─────────────────────────────────────────────────────────────

def _detect_tipping_points(grid: List[List[CellResult]], top_k: int = 10) -> List[TippingPoint]:
    """Find cells with the largest commit-rate gradient."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    points: List[TippingPoint] = []

    for r in range(rows):
        for c in range(cols):
            g_thr = 0.0
            g_byz = 0.0
            if r > 0:
                g_thr += abs(grid[r][c].commit_rate - grid[r - 1][c].commit_rate)
            if r < rows - 1:
                g_thr += abs(grid[r + 1][c].commit_rate - grid[r][c].commit_rate)
            if c > 0:
                g_byz += abs(grid[r][c].commit_rate - grid[r][c - 1].commit_rate)
            if c < cols - 1:
                g_byz += abs(grid[r][c + 1].commit_rate - grid[r][c].commit_rate)
            mag = math.sqrt(g_thr ** 2 + g_byz ** 2)
            direction = "threshold" if g_thr >= g_byz else "byzantine"
            if mag > 0.1:
                points.append(TippingPoint(
                    threshold=grid[r][c].threshold,
                    byzantine_ratio=grid[r][c].byzantine_ratio,
                    gradient_magnitude=round(mag, 4),
                    direction=direction,
                ))

    points.sort(key=lambda p: p.gradient_magnitude, reverse=True)
    return points[:top_k]


def _classify_regions(grid: List[List[CellResult]]) -> List[PhaseRegion]:
    """Classify grid cells into stability regions."""
    stable: List[CellResult] = []
    unstable: List[CellResult] = []
    transition: List[CellResult] = []
    collapsed: List[CellResult] = []

    for row in grid:
        for c in row:
            if c.commit_rate >= 0.8:
                stable.append(c)
            elif c.commit_rate >= 0.4:
                transition.append(c)
            elif c.commit_rate >= 0.1:
                unstable.append(c)
            else:
                collapsed.append(c)

    def _region(name: str, desc: str, cells: List[CellResult]) -> Optional[PhaseRegion]:
        if not cells:
            return None
        return PhaseRegion(
            name=name,
            description=desc,
            cells=len(cells),
            avg_commit_rate=round(sum(c.commit_rate for c in cells) / len(cells), 3),
            threshold_range=(min(c.threshold for c in cells), max(c.threshold for c in cells)),
            byz_range=(min(c.byzantine_ratio for c in cells), max(c.byzantine_ratio for c in cells)),
        )

    regions = [
        _region("Stable Consensus", "High commit rate — reliable agreement zone", stable),
        _region("Transition Zone", "Moderate commit rate — sensitive to perturbations", transition),
        _region("Unstable", "Low commit rate — consensus rarely achieved", unstable),
        _region("Collapsed", "Near-zero commits — system failure region", collapsed),
    ]
    return [r for r in regions if r is not None]


def _generate_recommendations(
    regions: List[PhaseRegion],
    tipping: List[TippingPoint],
    n_agents: int,
) -> List[str]:
    """Produce actionable advice from the landscape analysis."""
    recs: List[str] = []

    stable = next((r for r in regions if r.name == "Stable Consensus"), None)
    if stable:
        recs.append(
            f"✅ Safe operating zone: threshold {stable.threshold_range[0]:.2f}–"
            f"{stable.threshold_range[1]:.2f} with Byzantine ratio "
            f"{stable.byz_range[0]:.0%}–{stable.byz_range[1]:.0%} "
            f"(avg commit rate {stable.avg_commit_rate:.0%})."
        )
    else:
        recs.append("⚠️ No stable consensus region found — consider adding more honest agents.")

    if tipping:
        tp = tipping[0]
        recs.append(
            f"🔴 Sharpest phase transition at threshold={tp.threshold:.2f}, "
            f"byz_ratio={tp.byzantine_ratio:.0%} (gradient={tp.gradient_magnitude:.3f}, "
            f"axis={tp.direction}). Avoid operating near this boundary."
        )

    transition = next((r for r in regions if r.name == "Transition Zone"), None)
    if transition:
        recs.append(
            f"⚡ Transition zone spans threshold {transition.threshold_range[0]:.2f}–"
            f"{transition.threshold_range[1]:.2f}. Add monitoring/alerting when "
            f"operating in this range."
        )

    collapsed = next((r for r in regions if r.name == "Collapsed"), None)
    if collapsed and collapsed.cells > 0:
        recs.append(
            f"💀 {collapsed.cells} parameter combinations lead to total consensus "
            f"failure. Ensure Byzantine ratio stays below {collapsed.byz_range[0]:.0%} "
            f"for low thresholds."
        )

    # BFT theoretical limit
    bft_limit = (n_agents - 1) / 3 / n_agents if n_agents > 1 else 0.33
    recs.append(
        f"📐 Classical BFT limit: f < n/3 → max Byzantine ratio ≈ {bft_limit:.0%} "
        f"for {n_agents} agents. mBFT may tolerate higher ratios with reputation slashing."
    )

    return recs


# ── autopilot zoom ───────────────────────────────────────────────────────

async def autopilot_zoom(
    grid: List[List[CellResult]],
    n_agents: int,
    trials: int,
    max_rounds: int,
) -> List[List[CellResult]]:
    """Auto-zoom into the sharpest transition boundary for higher detail."""
    tipping = _detect_tipping_points(grid, top_k=1)
    if not tipping:
        return []

    tp = tipping[0]
    # zoom into ±10% around the tipping point
    t_lo = max(0.1, tp.threshold - 0.5)
    t_hi = tp.threshold + 0.5
    b_lo = max(0.0, tp.byzantine_ratio - 0.05)
    b_hi = min(1.0, tp.byzantine_ratio + 0.05)

    zoom_res = 10
    thresholds = _linspace(t_lo, t_hi, zoom_res)
    byz_ratios = _linspace(b_lo, b_hi, zoom_res)

    zoom_grid: List[List[CellResult]] = []
    for thr in thresholds:
        row = []
        for byz in byz_ratios:
            cell = await _run_cell(n_agents, thr, byz, trials * 2, max_rounds)
            row.append(cell)
        zoom_grid.append(row)
    return zoom_grid


# ── HTML report ──────────────────────────────────────────────────────────

def _render_html(report: LandscapeReport, zoom_grid: Optional[List[List[CellResult]]] = None) -> str:
    """Render a self-contained interactive HTML report."""
    grid_json = json.dumps([
        [{"t": c.threshold, "b": c.byzantine_ratio, "cr": c.commit_rate,
          "ar": c.avg_rounds, "aa": c.avg_aggregate}
         for c in row]
        for row in report.grid
    ])
    zoom_json = "null"
    if zoom_grid:
        zoom_json = json.dumps([
            [{"t": c.threshold, "b": c.byzantine_ratio, "cr": c.commit_rate,
              "ar": c.avg_rounds, "aa": c.avg_aggregate}
             for c in row]
            for row in zoom_grid
        ])

    regions_json = json.dumps([
        {"name": r.name, "cells": r.cells, "avg_cr": r.avg_commit_rate,
         "t_range": list(r.threshold_range), "b_range": list(r.byz_range)}
        for r in report.regions
    ])
    tipping_json = json.dumps([
        {"t": tp.threshold, "b": tp.byzantine_ratio,
         "grad": tp.gradient_magnitude, "dir": tp.direction}
        for tp in report.tipping_points
    ])
    recs_json = json.dumps(report.recommendations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mBFT Consensus Stability Landscape</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}}
h1{{font-size:1.6rem;margin-bottom:4px;color:#58a6ff}}
.subtitle{{color:#8b949e;margin-bottom:20px;font-size:.9rem}}
.grid-container{{display:flex;gap:30px;flex-wrap:wrap;margin-bottom:30px}}
.heatmap-wrap{{position:relative}}
canvas{{border:1px solid #30363d;border-radius:8px;cursor:crosshair}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px}}
.panel h2{{font-size:1.1rem;color:#58a6ff;margin-bottom:10px}}
.tooltip{{position:fixed;background:#1c2128;border:1px solid #58a6ff;border-radius:6px;padding:10px;font-size:.85rem;pointer-events:none;display:none;z-index:100;color:#c9d1d9}}
.region{{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:6px;padding:8px 12px;margin:4px;font-size:.85rem}}
.region .name{{font-weight:600;color:#f0f6fc}}
.tp{{background:#21262d;border-left:3px solid #f85149;padding:8px 12px;margin:4px 0;border-radius:0 6px 6px 0;font-size:.85rem}}
.rec{{padding:6px 0;border-bottom:1px solid #21262d;font-size:.9rem}}
.rec:last-child{{border-bottom:none}}
.legend{{display:flex;align-items:center;gap:4px;margin:10px 0;font-size:.8rem}}
.legend-bar{{width:200px;height:14px;border-radius:3px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-bottom:16px}}
.stat{{background:#21262d;border-radius:6px;padding:10px;text-align:center}}
.stat .val{{font-size:1.3rem;font-weight:700;color:#58a6ff}}
.stat .lbl{{font-size:.75rem;color:#8b949e}}
.tabs{{display:flex;gap:8px;margin-bottom:12px}}
.tab{{padding:6px 14px;border-radius:6px;background:#21262d;border:1px solid #30363d;cursor:pointer;font-size:.85rem;color:#8b949e}}
.tab.active{{background:#58a6ff;color:#0d1117;border-color:#58a6ff;font-weight:600}}
</style>
</head>
<body>
<h1>🗺️ mBFT Consensus Stability Landscape</h1>
<p class="subtitle">Phase-transition mapping across threshold × Byzantine ratio parameter space
 — {report.n_agents} agents, {report.resolution}×{report.resolution} grid, {report.trials_per_cell} trials/cell
 — completed in {report.elapsed_sec:.1f}s</p>

<div class="stats" id="stats"></div>

<div class="tabs" id="tabs">
  <div class="tab active" data-view="commit">Commit Rate</div>
  <div class="tab" data-view="rounds">Avg Rounds</div>
  <div class="tab" data-view="gradient">Phase Gradient</div>
</div>

<div class="grid-container">
  <div class="heatmap-wrap">
    <canvas id="main" width="500" height="500"></canvas>
    <div class="legend">
      <span>0%</span>
      <canvas id="legendBar" width="200" height="14" class="legend-bar"></canvas>
      <span>100%</span>
    </div>
    <p style="text-align:center;font-size:.8rem;color:#8b949e;margin-top:4px">
      X: Byzantine Ratio → &nbsp; Y: Threshold ↑
    </p>
  </div>
  <div id="zoomWrap" class="heatmap-wrap" style="display:none">
    <h3 style="color:#f0883e;margin-bottom:6px;font-size:.95rem">🔍 Autopilot Zoom — Transition Boundary</h3>
    <canvas id="zoom" width="350" height="350"></canvas>
  </div>
</div>

<div class="tooltip" id="tip"></div>

<div class="panel" id="regionsPanel"><h2>🌍 Stability Regions</h2><div id="regions"></div></div>
<div class="panel" id="tippingPanel"><h2>⚡ Tipping Points</h2><div id="tipping"></div></div>
<div class="panel"><h2>💡 Recommendations</h2><div id="recs"></div></div>

<script>
const grid = {grid_json};
const zoomGrid = {zoom_json};
const regions = {regions_json};
const tipping = {tipping_json};
const recs = {recs_json};

// stats
const statsEl = document.getElementById('stats');
const allCells = grid.flat();
const avgCR = allCells.reduce((s,c)=>s+c.cr,0)/allCells.length;
const maxCR = Math.max(...allCells.map(c=>c.cr));
const stableCount = allCells.filter(c=>c.cr>=0.8).length;
statsEl.innerHTML = [
  {{val:(avgCR*100).toFixed(1)+'%', lbl:'Avg Commit Rate'}},
  {{val:stableCount, lbl:'Stable Cells'}},
  {{val:allCells.length-stableCount, lbl:'Non-Stable Cells'}},
  {{val:tipping.length, lbl:'Tipping Points'}},
  {{val:regions.length, lbl:'Phase Regions'}},
].map(s=>`<div class="stat"><div class="val">${{s.val}}</div><div class="lbl">${{s.lbl}}</div></div>`).join('');

// color scales
function commitColor(v) {{
  if(v>=0.8) return `rgb(${{Math.round(40+v*60)}}, ${{Math.round(160+v*80)}}, ${{Math.round(80+v*40)}})`;
  if(v>=0.4) return `rgb(${{Math.round(200+v*55)}}, ${{Math.round(160+v*60)}}, 60)`;
  return `rgb(${{Math.round(180+v*75)}}, ${{Math.round(60+v*80)}}, ${{Math.round(60+v*40)}})`;
}}
function roundsColor(v, mx) {{
  const t = v/mx;
  return `rgb(${{Math.round(30+t*200)}}, ${{Math.round(80+t*80)}}, ${{Math.round(200-t*140)}})`;
}}
function gradColor(v, mx) {{
  const t = Math.min(v/mx, 1);
  return `rgb(${{Math.round(255*t)}}, ${{Math.round(100*(1-t))}}, ${{Math.round(200*(1-t))}})`;
}}

let currentView = 'commit';

function drawHeatmap(canvasId, data) {{
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const rows = data.length, cols = data[0].length;
  const cw = canvas.width/cols, ch = canvas.height/rows;

  // compute gradient magnitudes
  const grads = data.map((row,r) => row.map((c,ci) => {{
    let gt=0, gb=0;
    if(r>0) gt += Math.abs(c.cr - data[r-1][ci].cr);
    if(r<rows-1) gt += Math.abs(data[r+1][ci].cr - c.cr);
    if(ci>0) gb += Math.abs(c.cr - data[r][ci-1].cr);
    if(ci<cols-1) gb += Math.abs(data[r][ci+1].cr - c.cr);
    return Math.sqrt(gt*gt + gb*gb);
  }}));
  const maxGrad = Math.max(...grads.flat(), 0.01);
  const maxRounds = Math.max(...data.flat().map(c=>c.ar), 0.01);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for(let r=0; r<rows; r++) {{
    for(let c=0; c<cols; c++) {{
      const cell = data[r][c];
      const x = c*cw, y = (rows-1-r)*ch;
      if(currentView==='commit') ctx.fillStyle = commitColor(cell.cr);
      else if(currentView==='rounds') ctx.fillStyle = roundsColor(cell.ar, maxRounds);
      else ctx.fillStyle = gradColor(grads[r][c], maxGrad);
      ctx.fillRect(x, y, cw+1, ch+1);

      // mark tipping points
      if(currentView !== 'rounds') {{
        const isTP = tipping.some(tp => Math.abs(tp.t-cell.t)<0.01 && Math.abs(tp.b-cell.b)<0.005);
        if(isTP) {{
          ctx.strokeStyle = '#f85149';
          ctx.lineWidth = 2;
          ctx.strokeRect(x+2, y+2, cw-4, ch-4);
        }}
      }}
    }}
  }}
}}

function drawLegend() {{
  const c = document.getElementById('legendBar');
  const ctx = c.getContext('2d');
  for(let i=0;i<200;i++) {{
    const v = i/200;
    ctx.fillStyle = currentView==='commit' ? commitColor(v) :
                    currentView==='rounds' ? roundsColor(v*4, 4) :
                    gradColor(v, 1);
    ctx.fillRect(i, 0, 1, 14);
  }}
}}

// tabs
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    currentView = tab.dataset.view;
    drawHeatmap('main', grid);
    if(zoomGrid) drawHeatmap('zoom', zoomGrid);
    drawLegend();
  }});
}});

// tooltip
const tip = document.getElementById('tip');
function setupTooltip(canvasId, data) {{
  const canvas = document.getElementById(canvasId);
  const rows = data.length, cols = data[0].length;
  canvas.addEventListener('mousemove', e => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const ci = Math.floor(mx / (canvas.width/cols));
    const ri = rows - 1 - Math.floor(my / (canvas.height/rows));
    if(ri>=0 && ri<rows && ci>=0 && ci<cols) {{
      const c = data[ri][ci];
      tip.style.display = 'block';
      tip.style.left = (e.clientX+12)+'px';
      tip.style.top = (e.clientY+12)+'px';
      tip.innerHTML = `<b>Threshold:</b> ${{c.t.toFixed(2)}}<br>
        <b>Byzantine:</b> ${{(c.b*100).toFixed(1)}}%<br>
        <b>Commit Rate:</b> ${{(c.cr*100).toFixed(1)}}%<br>
        <b>Avg Rounds:</b> ${{c.ar.toFixed(2)}}<br>
        <b>Avg Aggregate:</b> ${{c.aa.toFixed(2)}}`;
    }}
  }});
  canvas.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
}}

// regions
const regEl = document.getElementById('regions');
regEl.innerHTML = regions.map(r =>
  `<div class="region"><span class="name">${{r.name}}</span><br>
   ${{r.cells}} cells · avg ${{(r.avg_cr*100).toFixed(0)}}% commit<br>
   threshold ${{r.t_range[0].toFixed(2)}}–${{r.t_range[1].toFixed(2)}} ·
   byz ${{(r.b_range[0]*100).toFixed(0)}}–${{(r.b_range[1]*100).toFixed(0)}}%</div>`
).join('');

// tipping
const tpEl = document.getElementById('tipping');
tpEl.innerHTML = tipping.length ? tipping.map(tp =>
  `<div class="tp">threshold=${{tp.t.toFixed(2)}} byz=${{(tp.b*100).toFixed(1)}}%
   — gradient ${{tp.grad.toFixed(3)}} (${{tp.dir}} axis)</div>`
).join('') : '<p style="color:#8b949e">No sharp phase transitions detected.</p>';

// recs
const recEl = document.getElementById('recs');
recEl.innerHTML = recs.map(r => `<div class="rec">${{r}}</div>`).join('');

// render
drawHeatmap('main', grid);
setupTooltip('main', grid);
drawLegend();

if(zoomGrid) {{
  document.getElementById('zoomWrap').style.display = 'block';
  drawHeatmap('zoom', zoomGrid);
  setupTooltip('zoom', zoomGrid);
}}
</script>
</body>
</html>"""


# ── CLI entry ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Stability Landscape — phase-transition mapper",
    )
    parser.add_argument("--agents", "-n", type=int, default=10, help="Number of agents (default 10)")
    parser.add_argument("--res", "-r", type=int, default=15, help="Grid resolution NxN (default 15)")
    parser.add_argument("--trials", "-t", type=int, default=8, help="Trials per cell (default 8)")
    parser.add_argument("--max-rounds", type=int, default=4, help="Max consensus rounds per trial")
    parser.add_argument("--autopilot", action="store_true", help="Auto-zoom into critical transition zone")
    parser.add_argument("-o", "--output", default="landscape_report.html", help="Output HTML path")
    parser.add_argument("--json", action="store_true", help="Also emit JSON data")
    args = parser.parse_args()

    print(f"🗺️  Consensus Stability Landscape")
    print(f"   {args.agents} agents · {args.res}×{args.res} grid · {args.trials} trials/cell")
    print()

    start = time.time()

    def progress(done, total):
        pct = done / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r   [{bar}] {pct:5.1f}%  ({done}/{total} cells)", end="", flush=True)

    grid = asyncio.run(sweep(args.agents, args.res, args.trials, args.max_rounds, progress))
    print()

    regions = _classify_regions(grid)
    tipping = _detect_tipping_points(grid)
    recs = _generate_recommendations(regions, tipping, args.agents)
    elapsed = time.time() - start

    zoom_grid = None
    if args.autopilot:
        print("   🔍 Autopilot: zooming into transition boundary...")
        zoom_grid = asyncio.run(autopilot_zoom(grid, args.agents, args.trials, args.max_rounds))
        elapsed = time.time() - start

    report = LandscapeReport(
        grid=grid,
        regions=regions,
        tipping_points=tipping,
        recommendations=recs,
        elapsed_sec=round(elapsed, 2),
        n_agents=args.agents,
        resolution=args.res,
        trials_per_cell=args.trials,
    )

    html = _render_html(report, zoom_grid)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n   ✅ Report written to {args.output}")

    if args.json:
        json_path = args.output.replace(".html", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "agents": args.agents,
                "resolution": args.res,
                "trials": args.trials,
                "elapsed_sec": report.elapsed_sec,
                "regions": [{"name": r.name, "cells": r.cells, "avg_commit_rate": r.avg_commit_rate}
                            for r in regions],
                "tipping_points": [{"threshold": tp.threshold, "byzantine_ratio": tp.byzantine_ratio,
                                     "gradient": tp.gradient_magnitude, "direction": tp.direction}
                                    for tp in tipping],
                "recommendations": recs,
            }, f, indent=2)
        print(f"   📊 JSON data written to {json_path}")

    # summary
    print(f"\n   📊 Summary ({elapsed:.1f}s):")
    for r in regions:
        print(f"      {r.name}: {r.cells} cells (avg {r.avg_commit_rate:.0%})")
    print(f"      Tipping points: {len(tipping)}")
    print()
    for rec in recs:
        print(f"   {rec}")


if __name__ == "__main__":
    main()
