"""Swarm Task Decomposer — autonomous hierarchical task decomposition for mBFT.

Breaks complex tasks into subtask DAGs with dependency tracking, complexity
estimation, critical-path analysis, and parallel execution scheduling.
Instead of running a single consensus round on a monolithic task, the
decomposer splits it into atomic subtasks, identifies dependencies, finds the
critical path, and produces an optimal parallel execution schedule that assigns
subtasks to available agents.

Capabilities:

- **Heuristic Decomposition** — keyword/pattern-based splitting: conjunctions,
  enumerated steps, implicit dependency markers.
- **Category Classification** — maps subtasks to categories (reasoning,
  retrieval, synthesis, verification, general) via keyword matching.
- **Complexity Estimation** — scores each subtask 0–1 based on word count,
  qualifier presence, and category weight.
- **DAG Validation** — cycle detection, orphan detection, missing-dep checks.
- **Critical Path** — longest weighted path via topological DP.
- **Wave Scheduler** — Kahn's-algorithm-based parallel wave scheduling.
- **Agent Assignment** — strength-aware or round-robin load-balanced assignment.
- **Result Merging** — reassembles subtask outputs into coherent final answer.
- **HTML Dashboard** — interactive single-file visualization (DAG, Gantt,
  assignments, stats).
- **JSON Export** — full serialization.

Usage::

    python -m src.decomposer                          # demo
    python -m src.decomposer --task "Build a web app and deploy it"
    python -m src.decomposer --agents 7 --export html -o dag.html
    python -m src.decomposer --export json -o dag.json

The Decomposer embodies the "agency" direction for mBFT: the swarm doesn't
just solve tasks — it **plans** how to solve them, distributing work
autonomously across agents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Subtask(BaseModel):
    """A single atomic work unit in the decomposition DAG."""
    id: str
    description: str
    complexity: float = Field(ge=0.0, le=1.0, default=0.5)
    dependencies: List[str] = Field(default_factory=list)
    category: str = "general"
    estimated_rounds: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecompositionResult(BaseModel):
    """Full result of decomposing a task."""
    original_task: str
    subtasks: List[Subtask]
    edges: List[Tuple[str, str]] = Field(default_factory=list)
    critical_path: List[str] = Field(default_factory=list)
    critical_path_cost: float = 0.0
    parallelism_factor: float = 0.0
    depth: int = 0
    schedule: List[List[str]] = Field(default_factory=list)
    agent_assignments: Dict[str, List[str]] = Field(default_factory=dict)


class DecompositionStrategy(BaseModel):
    """Configurable decomposition parameters."""
    max_subtasks: int = 20
    min_complexity_threshold: float = 0.1
    prefer_parallel: bool = True
    category_weights: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Category keywords
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "reasoning": [
        "compare", "analyze", "evaluate", "decide", "reason", "assess",
        "judge", "critique", "debate", "infer", "deduce", "think",
    ],
    "retrieval": [
        "find", "search", "look up", "fetch", "gather", "collect",
        "retrieve", "get", "locate", "obtain", "query",
    ],
    "synthesis": [
        "combine", "merge", "integrate", "summarize", "compile",
        "synthesize", "assemble", "consolidate", "compose", "create",
        "build", "generate", "write", "produce", "design", "develop",
    ],
    "verification": [
        "check", "verify", "validate", "test", "confirm", "audit",
        "review", "inspect", "ensure", "assert",
    ],
}

_COMPLEXITY_QUALIFIERS: Dict[str, float] = {
    "simple": -0.15,
    "easy": -0.15,
    "basic": -0.10,
    "trivial": -0.20,
    "complex": 0.15,
    "difficult": 0.15,
    "hard": 0.10,
    "advanced": 0.15,
    "detailed": 0.10,
    "thorough": 0.10,
    "comprehensive": 0.15,
}

# Patterns that split tasks into sub-components
_ENUM_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+[\.\)]\s+|[-*•]\s+|"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
    r"(?:ly)?[,:]?\s+)",
    re.IGNORECASE,
)
_CONJUNCTION_SPLIT = re.compile(
    r"\s+(?:and then|then|and also|also|additionally|moreover|furthermore|"
    r"after that|next|finally|lastly|subsequently)\s+",
    re.IGNORECASE,
)
_DEPENDENCY_MARKERS = re.compile(
    r"(?:using the result|based on|after|once .+ (?:is|are) done|"
    r"with the output|depending on|requires)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core decomposer
# ---------------------------------------------------------------------------

class SwarmTaskDecomposer:
    """Autonomous task decomposer for mBFT swarms."""

    def __init__(self, strategy: Optional[DecompositionStrategy] = None) -> None:
        self.strategy = strategy or DecompositionStrategy()

    # -- public API ---------------------------------------------------------

    def decompose(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecompositionResult:
        """Decompose *task* into a subtask DAG with schedule & stats."""
        task = task.strip()
        if not task:
            return DecompositionResult(original_task=task, subtasks=[])

        raw_parts = self._split_task(task)
        subtasks = self._build_subtasks(raw_parts, context)

        # Enforce max_subtasks
        subtasks = subtasks[: self.strategy.max_subtasks]

        # Filter out below-threshold complexity
        if len(subtasks) > 1:
            subtasks = [
                s for s in subtasks
                if s.complexity >= self.strategy.min_complexity_threshold
            ] or subtasks[:1]

        # Fix dangling dependency refs after trimming
        ids = {s.id for s in subtasks}
        for s in subtasks:
            s.dependencies = [d for d in s.dependencies if d in ids]

        # Validate & potentially break cycles
        valid, problems = self.validate_dag(subtasks)
        if not valid:
            subtasks = self._break_cycles(subtasks)

        edges = []
        for s in subtasks:
            for dep in s.dependencies:
                edges.append((dep, s.id))

        cp, cp_cost = self.critical_path(subtasks)
        waves = self.schedule(subtasks)
        max_width = max((len(w) for w in waves), default=0)
        n = len(subtasks)
        parallelism = max_width / n if n else 0.0

        return DecompositionResult(
            original_task=task,
            subtasks=subtasks,
            edges=edges,
            critical_path=cp,
            critical_path_cost=cp_cost,
            parallelism_factor=round(parallelism, 3),
            depth=len(waves),
            schedule=waves,
        )

    def assign_agents(
        self,
        result: DecompositionResult,
        agent_ids: List[str],
        agent_strengths: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> DecompositionResult:
        """Assign subtasks to agents, respecting wave order & strengths."""
        if not agent_ids or not result.subtasks:
            return result

        assignments: Dict[str, List[str]] = {a: [] for a in agent_ids}
        load: Dict[str, float] = {a: 0.0 for a in agent_ids}
        task_map = {s.id: s for s in result.subtasks}

        for wave in result.schedule:
            for sid in wave:
                st = task_map[sid]
                best_agent = self._pick_agent(
                    st, agent_ids, agent_strengths, load,
                )
                assignments[best_agent].append(sid)
                load[best_agent] += st.complexity

        result.agent_assignments = assignments
        return result

    def validate_dag(
        self, subtasks: List[Subtask],
    ) -> Tuple[bool, List[str]]:
        """Return ``(is_valid, list_of_problems)``."""
        problems: List[str] = []
        ids = {s.id for s in subtasks}
        adj: Dict[str, List[str]] = defaultdict(list)
        in_deg: Dict[str, int] = {s.id: 0 for s in subtasks}

        for s in subtasks:
            for d in s.dependencies:
                if d not in ids:
                    problems.append(f"subtask {s.id!r} depends on missing {d!r}")
                else:
                    adj[d].append(s.id)
                    in_deg[s.id] += 1

        # Kahn's for cycle check
        q: deque[str] = deque(sid for sid, deg in in_deg.items() if deg == 0)
        visited = 0
        while q:
            node = q.popleft()
            visited += 1
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    q.append(nb)
        if visited < len(subtasks):
            problems.append("cycle detected in dependency graph")

        # Orphan check — nodes with deps not reachable from roots
        roots = {s.id for s in subtasks if not s.dependencies}
        if not roots and subtasks:
            problems.append("no root subtasks (all have dependencies)")

        return (len(problems) == 0, problems)

    def critical_path(
        self, subtasks: List[Subtask],
    ) -> Tuple[List[str], float]:
        """Longest weighted path through the DAG."""
        if not subtasks:
            return [], 0.0

        task_map = {s.id: s for s in subtasks}
        topo = self._topo_sort(subtasks)
        if not topo:
            return [subtasks[0].id], subtasks[0].complexity

        cost: Dict[str, float] = {}
        pred: Dict[str, Optional[str]] = {}

        for sid in topo:
            st = task_map[sid]
            if not st.dependencies:
                cost[sid] = st.complexity
                pred[sid] = None
            else:
                best_dep = max(
                    (d for d in st.dependencies if d in cost),
                    key=lambda d: cost[d],
                    default=None,
                )
                if best_dep is not None:
                    cost[sid] = cost[best_dep] + st.complexity
                    pred[sid] = best_dep
                else:
                    cost[sid] = st.complexity
                    pred[sid] = None

        if not cost:
            return [], 0.0

        end = max(cost, key=lambda k: cost[k])
        path: List[str] = []
        cur: Optional[str] = end
        while cur is not None:
            path.append(cur)
            cur = pred.get(cur)
        path.reverse()
        return path, round(cost[end], 4)

    def schedule(self, subtasks: List[Subtask]) -> List[List[str]]:
        """Wave-based parallel scheduler (Kahn's algorithm)."""
        if not subtasks:
            return []

        ids = {s.id for s in subtasks}
        adj: Dict[str, List[str]] = defaultdict(list)
        in_deg: Dict[str, int] = {s.id: 0 for s in subtasks}

        for s in subtasks:
            for d in s.dependencies:
                if d in ids:
                    adj[d].append(s.id)
                    in_deg[s.id] += 1

        waves: List[List[str]] = []
        current = sorted(sid for sid, deg in in_deg.items() if deg == 0)

        while current:
            waves.append(current)
            nxt: List[str] = []
            for sid in current:
                for nb in adj[sid]:
                    in_deg[nb] -= 1
                    if in_deg[nb] == 0:
                        nxt.append(nb)
            current = sorted(nxt)

        return waves

    def merge_results(self, subtask_results: Dict[str, str]) -> str:
        """Merge completed subtask outputs into a coherent answer."""
        if not subtask_results:
            return ""
        parts = []
        for sid in sorted(subtask_results.keys()):
            parts.append(f"[{sid}] {subtask_results[sid]}")
        return "\n\n".join(parts)

    def export_json(self, result: DecompositionResult) -> str:
        """JSON export."""
        return result.model_dump_json(indent=2)

    def export_html(self, result: DecompositionResult) -> str:
        """Interactive single-file HTML dashboard."""
        task_map = {s.id: s for s in result.subtasks}
        cp_set = set(result.critical_path)

        # Build SVG DAG
        wave_map: Dict[str, int] = {}
        for wi, wave in enumerate(result.schedule):
            for sid in wave:
                wave_map[sid] = wi

        svg_nodes = []
        svg_edges = []
        node_pos: Dict[str, Tuple[float, float]] = {}

        for wi, wave in enumerate(result.schedule):
            for ni, sid in enumerate(wave):
                x = 80 + ni * 180
                y = 60 + wi * 100
                node_pos[sid] = (x, y)
                st = task_map[sid]
                color = "#e74c3c" if sid in cp_set else _complexity_color(st.complexity)
                label = _truncate(st.description, 20)
                svg_nodes.append(
                    f'<g transform="translate({x},{y})">'
                    f'<rect x="-70" y="-25" width="140" height="50" rx="8" '
                    f'fill="{color}" opacity="0.85"/>'
                    f'<text text-anchor="middle" dy="0" fill="#fff" '
                    f'font-size="11" font-family="sans-serif">{_html_esc(label)}</text>'
                    f'<text text-anchor="middle" dy="16" fill="#ffffffcc" '
                    f'font-size="9" font-family="sans-serif">'
                    f'c={st.complexity:.2f} | {st.category}</text>'
                    f'</g>'
                )

        for src, dst in result.edges:
            if src in node_pos and dst in node_pos:
                x1, y1 = node_pos[src]
                x2, y2 = node_pos[dst]
                ec = "#e74c3c" if src in cp_set and dst in cp_set else "#888"
                svg_edges.append(
                    f'<line x1="{x1}" y1="{y1+25}" x2="{x2}" y2="{y2-25}" '
                    f'stroke="{ec}" stroke-width="2" marker-end="url(#arrow)"/>'
                )

        max_x = max((p[0] for p in node_pos.values()), default=200) + 100
        max_y = max((p[1] for p in node_pos.values()), default=200) + 80

        # Agent assignment table
        assign_rows = ""
        for agent, tasks in result.agent_assignments.items():
            load = sum(task_map[t].complexity for t in tasks if t in task_map)
            bar_w = min(load * 200, 300)
            assign_rows += (
                f"<tr><td>{_html_esc(agent)}</td>"
                f"<td>{len(tasks)}</td>"
                f"<td>{load:.2f}</td>"
                f'<td><div style="background:#3498db;height:18px;'
                f'width:{bar_w}px;border-radius:4px"></div></td></tr>'
            )

        # Schedule Gantt
        gantt_rows = ""
        for wi, wave in enumerate(result.schedule):
            cells = ", ".join(wave)
            gantt_rows += f"<tr><td>Wave {wi}</td><td>{_html_esc(cells)}</td></tr>"

        html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en"><head><meta charset="utf-8"/>
        <title>Swarm Task Decomposer — Dashboard</title>
        <style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;padding:24px}}
        h1{{font-size:1.6rem;margin-bottom:8px;color:#fff}}
        h2{{font-size:1.2rem;margin:20px 0 8px;color:#aaa}}
        .stats{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}}
        .stat{{background:#1a1d27;border-radius:10px;padding:14px 20px;min-width:140px}}
        .stat .val{{font-size:1.5rem;font-weight:700;color:#3498db}}
        .stat .lbl{{font-size:.75rem;color:#888;margin-top:2px}}
        table{{border-collapse:collapse;width:100%;margin-bottom:16px}}
        th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid #222}}
        th{{color:#888;font-size:.8rem;text-transform:uppercase}}
        svg{{background:#181b24;border-radius:10px;display:block;margin-bottom:16px}}
        .task-orig{{background:#1a1d27;border-radius:8px;padding:12px;margin-bottom:16px;
                    font-style:italic;color:#aaa}}
        </style></head><body>
        <h1>🔀 Swarm Task Decomposer</h1>
        <div class="task-orig">{_html_esc(result.original_task)}</div>
        <div class="stats">
          <div class="stat"><div class="val">{len(result.subtasks)}</div><div class="lbl">Subtasks</div></div>
          <div class="stat"><div class="val">{result.depth}</div><div class="lbl">Depth (waves)</div></div>
          <div class="stat"><div class="val">{result.parallelism_factor:.2f}</div><div class="lbl">Parallelism</div></div>
          <div class="stat"><div class="val">{result.critical_path_cost:.2f}</div><div class="lbl">Critical Path Cost</div></div>
        </div>
        <h2>Dependency DAG</h2>
        <svg width="{max_x}" height="{max_y}" viewBox="0 0 {max_x} {max_y}">
          <defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#888"/></marker></defs>
          {"".join(svg_edges)}
          {"".join(svg_nodes)}
        </svg>
        <h2>Execution Schedule</h2>
        <table><tr><th>Wave</th><th>Subtasks</th></tr>{gantt_rows}</table>
        <h2>Agent Assignments</h2>
        <table><tr><th>Agent</th><th>Tasks</th><th>Load</th><th>Bar</th></tr>
        {assign_rows if assign_rows else "<tr><td colspan='4' style='color:#666'>No agents assigned yet</td></tr>"}
        </table>
        <h2>Critical Path</h2>
        <p style="color:#e74c3c;font-weight:600">{" → ".join(result.critical_path) or "—"}</p>
        </body></html>
        """)
        return html

    # -- internal helpers ---------------------------------------------------

    def _split_task(self, task: str) -> List[str]:
        """Split a task string into candidate sub-parts."""
        # Try enumerated list first
        parts = _ENUM_PATTERN.split(task)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

        # Try conjunction splitting
        parts = _CONJUNCTION_SPLIT.split(task)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

        # Try sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', task)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        if len(sentences) > 1:
            return sentences

        return [task]

    def _build_subtasks(
        self,
        parts: List[str],
        context: Optional[Dict[str, Any]],
    ) -> List[Subtask]:
        """Convert raw text parts into Subtask objects with deps."""
        subtasks: List[Subtask] = []
        for i, part in enumerate(parts):
            sid = f"t{i}"
            cat = self._classify_category(part)
            cplx = self._estimate_complexity(part, cat)
            deps = self._detect_dependencies(part, i, subtasks)
            st = Subtask(
                id=sid,
                description=part,
                complexity=round(cplx, 3),
                dependencies=deps,
                category=cat,
                estimated_rounds=max(1, round(cplx * 3)),
                metadata={"source_index": i},
            )
            subtasks.append(st)

        # If prefer_parallel and no explicit deps, keep them independent
        # Otherwise, for fully independent tasks, add sequential deps
        if not self.strategy.prefer_parallel:
            for i in range(1, len(subtasks)):
                if not subtasks[i].dependencies:
                    subtasks[i].dependencies.append(subtasks[i - 1].id)

        return subtasks

    def _classify_category(self, text: str) -> str:
        """Keyword-match to assign a category."""
        lower = text.lower()
        scores: Dict[str, int] = defaultdict(int)
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    scores[cat] += 1
        if scores:
            return max(scores, key=lambda k: scores[k])
        return "general"

    def _estimate_complexity(self, text: str, category: str) -> float:
        """Heuristic complexity scorer."""
        words = text.split()
        # Base: word-count driven (more words = more complex)
        base = min(0.3 + len(words) * 0.02, 0.9)

        # Qualifier adjustments
        lower = text.lower()
        for qualifier, adj in _COMPLEXITY_QUALIFIERS.items():
            if qualifier in lower:
                base += adj

        # Category weight
        cat_w = self.strategy.category_weights.get(category, 1.0)
        base *= cat_w

        return max(0.05, min(base, 1.0))

    def _detect_dependencies(
        self,
        text: str,
        index: int,
        existing: List[Subtask],
    ) -> List[str]:
        """Detect implicit dependency markers pointing to earlier subtasks."""
        if not existing:
            return []
        if _DEPENDENCY_MARKERS.search(text):
            # Link to immediately preceding subtask
            return [existing[-1].id]
        return []

    def _topo_sort(self, subtasks: List[Subtask]) -> List[str]:
        """Kahn's topological sort; returns [] if cycle detected."""
        ids = {s.id for s in subtasks}
        adj: Dict[str, List[str]] = defaultdict(list)
        in_deg: Dict[str, int] = {s.id: 0 for s in subtasks}
        for s in subtasks:
            for d in s.dependencies:
                if d in ids:
                    adj[d].append(s.id)
                    in_deg[s.id] += 1
        q: deque[str] = deque(sorted(sid for sid, deg in in_deg.items() if deg == 0))
        order: List[str] = []
        while q:
            node = q.popleft()
            order.append(node)
            for nb in sorted(adj[node]):
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    q.append(nb)
        return order if len(order) == len(subtasks) else []

    def _break_cycles(self, subtasks: List[Subtask]) -> List[Subtask]:
        """Remove back-edges until the graph is a DAG."""
        # Simple approach: repeatedly drop the last dep that causes a cycle
        for _ in range(50):
            topo = self._topo_sort(subtasks)
            if topo:
                return subtasks
            # Find a cycle member and drop one dep
            for s in reversed(subtasks):
                if s.dependencies:
                    s.dependencies.pop()
                    break
        return subtasks

    @staticmethod
    def _pick_agent(
        subtask: Subtask,
        agent_ids: List[str],
        strengths: Optional[Dict[str, Dict[str, float]]],
        load: Dict[str, float],
    ) -> str:
        """Pick the best agent for a subtask."""
        if strengths:
            scored = []
            for a in agent_ids:
                s = strengths.get(a, {})
                affinity = s.get(subtask.category, 0.5)
                # Prefer high affinity, low load
                score = affinity - load[a] * 0.3
                scored.append((score, a))
            scored.sort(key=lambda x: -x[0])
            return scored[0][1]
        # Round-robin by load
        return min(agent_ids, key=lambda a: load[a])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _complexity_color(c: float) -> str:
    """Map complexity 0–1 to a color."""
    if c < 0.3:
        return "#27ae60"
    if c < 0.6:
        return "#f39c12"
    return "#e74c3c"


def _truncate(s: str, n: int) -> str:
    return s[:n] + "…" if len(s) > n else s


def _html_esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEMO_TASK = (
    "1. Research the current state of quantum error correction. "
    "2. Analyze the three most promising approaches and compare their "
    "fault-tolerance thresholds. "
    "3. Using the result of the analysis, design a hybrid protocol "
    "combining surface codes and color codes. "
    "4. Verify the protocol against known noise models. "
    "5. Write a comprehensive summary with recommendations."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Task Decomposer — break complex tasks into subtask DAGs",
    )
    parser.add_argument("--task", default=_DEMO_TASK, help="Task to decompose")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument(
        "--strategy", default=None,
        help='JSON strategy config, e.g. \'{"max_subtasks": 10}\'',
    )
    parser.add_argument("--export", choices=["json", "html"], default=None)
    parser.add_argument("-o", "--output", default=None, help="Output file")
    args = parser.parse_args()

    strategy = DecompositionStrategy()
    if args.strategy:
        strategy = DecompositionStrategy(**json.loads(args.strategy))

    decomposer = SwarmTaskDecomposer(strategy=strategy)
    result = decomposer.decompose(args.task)
    agent_ids = [f"agent-{i}" for i in range(args.agents)]
    result = decomposer.assign_agents(result, agent_ids)

    if args.export == "json":
        output = decomposer.export_json(result)
    elif args.export == "html":
        output = decomposer.export_html(result)
    else:
        # Pretty-print summary
        print("=" * 60)
        print("Swarm Task Decomposer")
        print("=" * 60)
        print(f"\nTask: {result.original_task[:80]}...")
        print(f"Subtasks: {len(result.subtasks)}")
        print(f"Depth: {result.depth} waves")
        print(f"Parallelism: {result.parallelism_factor:.2f}")
        print(f"Critical path cost: {result.critical_path_cost:.2f}")
        print(f"\n--- Subtasks ---")
        for s in result.subtasks:
            deps = f" deps=[{', '.join(s.dependencies)}]" if s.dependencies else ""
            print(f"  [{s.id}] {s.description[:50]:50s} "
                  f"c={s.complexity:.2f} cat={s.category}{deps}")
        print(f"\n--- Schedule ---")
        for wi, wave in enumerate(result.schedule):
            print(f"  Wave {wi}: {', '.join(wave)}")
        print(f"\n--- Critical Path ---")
        print(f"  {' -> '.join(result.critical_path)}")
        print(f"\n--- Agent Assignments ---")
        for agent, tasks in result.agent_assignments.items():
            print(f"  {agent}: {', '.join(tasks)}")
        print()
        return

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote {args.export} to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
