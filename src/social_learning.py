"""Swarm Social Learning Engine — autonomous cultural evolution.

Models how multi-agent swarms develop cumulative culture through social
learning: observation, imitation, teaching, and innovation.  Agents acquire
skills from each other, build on existing knowledge, and occasionally
innovate novel skill combinations.  Over time, cultural lineages form and
complexity accumulates — just as in biological cultural evolution.

Capabilities:

- **Skill Ecology** — library of skills with complexity levels,
  prerequisites, fitness values, and lineage tracking.
- **Observation** — agents detect skill demonstrations by neighbors;
  probability modulated by complexity and attention.
- **Imitation** — agents attempt to replicate observed skills with lossy
  copying that introduces cultural drift.
- **Teaching** — high-proficiency agents actively transmit knowledge to
  learners; more reliable but costs energy.
- **Innovation** — occasional combination of existing skills into novel
  composite skills; rate depends on repertoire diversity.
- **Cultural Health Score** — composite 0-100 metric from skill diversity
  (Shannon entropy), learning rate, innovation rate, complexity depth,
  knowledge inequality (Gini), and stagnation detection.
- **Insight Generator** — autonomous observations: bottlenecks, monopolies,
  hotspots, dying skills, emerging traditions.
- **Interactive HTML Dashboard** — skill heatmap, complexity timeline,
  innovation tree, Gini trend, leaderboard, insights.

Usage (Python API)::

    from src.social_learning import SocialLearningEngine

    engine = SocialLearningEngine(num_agents=20, num_initial_skills=10)
    report = engine.simulate(steps=200)

    print(report.health.score)
    print(report.health.skill_diversity)
    print(report.insights)

    engine.export_html("social_learning_report.html")
    engine.export_json("social_learning_data.json")

CLI::

    python -m src.social_learning                         # demo
    python -m src.social_learning --agents 30 --steps 500
    python -m src.social_learning --innovation-rate 0.05
    python -m src.social_learning --out report.html --json social_learning.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------

class LearningMode(str, Enum):
    """Modes of skill acquisition."""
    OBSERVATION = "observation"
    IMITATION = "imitation"
    TEACHING = "teaching"
    INNOVATION = "innovation"
    INDEPENDENT = "independent"


class SkillComplexity(str, Enum):
    """Complexity tiers for skills."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MASTERWORK = "masterwork"


COMPLEXITY_LEVELS: Dict[SkillComplexity, int] = {
    SkillComplexity.TRIVIAL: 0,
    SkillComplexity.SIMPLE: 1,
    SkillComplexity.MODERATE: 2,
    SkillComplexity.COMPLEX: 3,
    SkillComplexity.MASTERWORK: 4,
}


@dataclass
class Skill:
    """A learnable skill in the cultural ecosystem."""
    skill_id: str
    name: str
    complexity: SkillComplexity
    prerequisites: List[str] = field(default_factory=list)
    fitness_value: float = 0.5
    generation_introduced: int = 0
    inventor_id: Optional[str] = None
    parent_skills: List[str] = field(default_factory=list)


@dataclass
class AgentProfile:
    """An agent's cultural state."""
    agent_id: str
    skills: Dict[str, float] = field(default_factory=dict)
    learning_events: int = 0
    teaching_events: int = 0
    innovations: int = 0
    generation: int = 0
    teacher_reputation: float = 0.5
    energy: float = 1.0


@dataclass
class LearningEvent:
    """A recorded learning event."""
    tick: int
    learner_id: str
    skill_id: str
    mode: LearningMode
    success: bool
    teacher_id: Optional[str] = None
    proficiency_gained: float = 0.0
    attempt_number: int = 1


@dataclass
class CulturalLineage:
    """Transmission history of a skill."""
    skill_id: str
    transmission_chain: List[Tuple[str, str, int]] = field(default_factory=list)
    total_transmissions: int = 0
    fidelity: float = 1.0
    geographic_spread: int = 0


@dataclass
class Insight:
    """An autonomous cultural observation."""
    category: str
    message: str
    severity: str
    related_skills: List[str] = field(default_factory=list)
    related_agents: List[str] = field(default_factory=list)


@dataclass
class CulturalHealthReport:
    """Composite cultural health assessment."""
    score: float = 0.0
    skill_diversity: float = 0.0
    learning_rate: float = 0.0
    innovation_rate: float = 0.0
    max_complexity_depth: int = 0
    knowledge_gini: float = 0.0
    stagnation_risk: float = 0.0
    cultural_momentum: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SimulationReport:
    """Full simulation output."""
    ticks: int = 0
    num_agents: int = 0
    total_skills: int = 0
    total_events: int = 0
    health: CulturalHealthReport = field(default_factory=CulturalHealthReport)
    insights: List[Insight] = field(default_factory=list)
    complexity_timeline: List[int] = field(default_factory=list)
    gini_timeline: List[float] = field(default_factory=list)
    learning_mode_counts: Dict[str, int] = field(default_factory=dict)
    top_teachers: List[Tuple[str, int]] = field(default_factory=list)
    innovation_count: int = 0


# ---------------------------------------------------------------------------
# Skill Name Generation
# ---------------------------------------------------------------------------

_SKILL_PREFIXES = [
    "fire", "stone", "water", "wind", "metal", "wood", "silk", "clay",
    "bone", "herb", "crystal", "shadow", "light", "frost", "thunder",
]
_SKILL_SUFFIXES = [
    "craft", "weaving", "forging", "shaping", "binding", "calling",
    "reading", "singing", "dancing", "brewing", "carving", "painting",
]


def _generate_skill_name(rng: random.Random) -> str:
    return f"{rng.choice(_SKILL_PREFIXES)}-{rng.choice(_SKILL_SUFFIXES)}"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SocialLearningEngine:
    """Autonomous cultural evolution engine for multi-agent swarms."""

    def __init__(
        self,
        num_agents: int = 20,
        num_initial_skills: int = 10,
        observation_range: int = 3,
        teaching_cost: float = 0.1,
        innovation_rate: float = 0.02,
        mutation_rate: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.num_agents = max(2, num_agents)
        self.num_initial_skills = max(1, num_initial_skills)
        self.observation_range = max(1, observation_range)
        self.teaching_cost = max(0.0, teaching_cost)
        self.innovation_rate = max(0.0, min(1.0, innovation_rate))
        self.mutation_rate = max(0.0, min(1.0, mutation_rate))
        self.rng = random.Random(seed)

        self.current_tick: int = 0
        self.skills: Dict[str, Skill] = {}
        self.agents: Dict[str, AgentProfile] = {}
        self.events: List[LearningEvent] = []
        self.lineages: Dict[str, CulturalLineage] = {}
        self._skill_counter: int = 0
        self._complexity_timeline: List[int] = []
        self._gini_timeline: List[float] = []

        self._initialize()

    def _initialize(self) -> None:
        """Set up initial skills and agents."""
        for i in range(self.num_initial_skills):
            skill = self._create_initial_skill(i)
            self.skills[skill.skill_id] = skill
            self.lineages[skill.skill_id] = CulturalLineage(skill_id=skill.skill_id)

        for i in range(self.num_agents):
            agent_id = f"agent-{i:03d}"
            agent = AgentProfile(agent_id=agent_id, generation=0)
            starter_skills = [
                s for s in self.skills.values()
                if s.complexity in (SkillComplexity.TRIVIAL, SkillComplexity.SIMPLE)
            ]
            num_start = min(len(starter_skills), self.rng.randint(1, 3))
            for skill in self.rng.sample(starter_skills, num_start):
                agent.skills[skill.skill_id] = self.rng.uniform(0.3, 0.8)
            self.agents[agent_id] = agent

    def _create_initial_skill(self, index: int) -> Skill:
        """Create a skill for the initial ecology."""
        self._skill_counter += 1
        skill_id = f"skill-{self._skill_counter:04d}"
        name = _generate_skill_name(self.rng)

        if index < 4:
            complexity = SkillComplexity.TRIVIAL
            prereqs: List[str] = []
        elif index < 7:
            complexity = SkillComplexity.SIMPLE
            trivials = [s.skill_id for s in self.skills.values()
                        if s.complexity == SkillComplexity.TRIVIAL]
            prereqs = self.rng.sample(trivials, min(1, len(trivials)))
        else:
            complexity = SkillComplexity.MODERATE
            simples = [s.skill_id for s in self.skills.values()
                       if s.complexity in (SkillComplexity.TRIVIAL, SkillComplexity.SIMPLE)]
            prereqs = self.rng.sample(simples, min(2, len(simples)))

        return Skill(
            skill_id=skill_id,
            name=name,
            complexity=complexity,
            prerequisites=prereqs,
            fitness_value=self.rng.uniform(0.3, 1.0),
            generation_introduced=0,
        )

    def _has_prerequisites(self, agent: AgentProfile, skill: Skill) -> bool:
        """Check if agent has all prerequisites for a skill."""
        for prereq_id in skill.prerequisites:
            if prereq_id not in agent.skills or agent.skills[prereq_id] < 0.2:
                return False
        return True

    def _get_neighbors(self, agent_id: str) -> List[str]:
        """Get agents within observation range (index-based proximity)."""
        all_ids = list(self.agents.keys())
        idx = all_ids.index(agent_id)
        neighbors = []
        for i in range(max(0, idx - self.observation_range),
                       min(len(all_ids), idx + self.observation_range + 1)):
            if all_ids[i] != agent_id:
                neighbors.append(all_ids[i])
        return neighbors

    def _observation_probability(self, skill: Skill) -> float:
        """Base probability of successfully observing a skill demonstration."""
        level = COMPLEXITY_LEVELS[skill.complexity]
        return max(0.1, 1.0 - level * 0.2)

    def _imitation_probability(self, agent: AgentProfile, skill: Skill) -> float:
        """Probability of successful imitation."""
        if not self._has_prerequisites(agent, skill):
            return 0.0
        level = COMPLEXITY_LEVELS[skill.complexity]
        base = max(0.05, 0.8 - level * 0.15)
        if skill.prerequisites:
            avg_prof = statistics.mean(
                agent.skills.get(p, 0.0) for p in skill.prerequisites
            )
            base += avg_prof * 0.2
        return min(0.95, base)

    def _teaching_probability(self, teacher: AgentProfile, skill: Skill) -> float:
        """Probability of successful teaching."""
        teacher_prof = teacher.skills.get(skill.skill_id, 0.0)
        return min(0.95, 0.5 + teacher_prof * 0.4 + teacher.teacher_reputation * 0.1)

    def tick(self) -> None:
        """Advance one simulation tick."""
        self.current_tick += 1

        agent_ids = list(self.agents.keys())
        self.rng.shuffle(agent_ids)

        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            agent.energy = min(1.0, agent.energy + 0.05)

            neighbors = self._get_neighbors(agent_id)
            if not neighbors:
                continue

            self._do_observation(agent, neighbors)
            self._do_imitation(agent, neighbors)
            self._do_teaching(agent, neighbors)

        self._do_innovation(agent_ids)
        self._do_decay()

        self._complexity_timeline.append(self._max_complexity_depth())
        self._gini_timeline.append(self._compute_gini())

    def _do_observation(self, agent: AgentProfile, neighbors: List[str]) -> None:
        """Agent observes neighbors demonstrating skills."""
        neighbor_id = self.rng.choice(neighbors)
        neighbor = self.agents[neighbor_id]
        if not neighbor.skills:
            return

        demonstrable = [
            sid for sid in neighbor.skills
            if sid not in agent.skills and neighbor.skills[sid] >= 0.4
        ]
        if not demonstrable:
            return

        skill_id = self.rng.choice(demonstrable)
        skill = self.skills[skill_id]
        prob = self._observation_probability(skill)

        if self.rng.random() < prob:
            event = LearningEvent(
                tick=self.current_tick,
                learner_id=agent.agent_id,
                skill_id=skill_id,
                mode=LearningMode.OBSERVATION,
                success=True,
                teacher_id=neighbor_id,
                proficiency_gained=0.0,
            )
            self.events.append(event)
            agent.learning_events += 1

    def _do_imitation(self, agent: AgentProfile, neighbors: List[str]) -> None:
        """Agent attempts to imitate observed skills."""
        neighbor_skills: Set[str] = set()
        for nid in neighbors:
            neighbor = self.agents[nid]
            for sid, prof in neighbor.skills.items():
                if prof >= 0.3 and sid not in agent.skills:
                    neighbor_skills.add(sid)

        if not neighbor_skills:
            return

        skill_id = self.rng.choice(list(neighbor_skills))
        skill = self.skills.get(skill_id)
        if skill is None:
            return

        prob = self._imitation_probability(agent, skill)
        success = self.rng.random() < prob

        if success:
            base_prof = self.rng.uniform(0.2, 0.6)
            mutation = self.rng.gauss(0, self.mutation_rate)
            proficiency = max(0.1, min(0.8, base_prof + mutation))
            agent.skills[skill_id] = proficiency

            lineage = self.lineages.get(skill_id)
            if lineage:
                demonstrator = None
                for nid in neighbors:
                    if skill_id in self.agents[nid].skills:
                        demonstrator = nid
                        break
                if demonstrator:
                    lineage.transmission_chain.append(
                        (demonstrator, agent.agent_id, self.current_tick)
                    )
                    lineage.total_transmissions += 1
                    lineage.fidelity = lineage.fidelity * 0.9 + proficiency * 0.1
                holders = sum(
                    1 for a in self.agents.values() if skill_id in a.skills
                )
                lineage.geographic_spread = holders

        event = LearningEvent(
            tick=self.current_tick,
            learner_id=agent.agent_id,
            skill_id=skill_id,
            mode=LearningMode.IMITATION,
            success=success,
            proficiency_gained=agent.skills.get(skill_id, 0.0) if success else 0.0,
        )
        self.events.append(event)
        agent.learning_events += 1

    def _do_teaching(self, agent: AgentProfile, neighbors: List[str]) -> None:
        """High-proficiency agent teaches a neighbor."""
        if agent.energy < self.teaching_cost:
            return

        teachable = [
            (sid, prof) for sid, prof in agent.skills.items() if prof >= 0.6
        ]
        if not teachable:
            return

        if self.rng.random() > 0.3:
            return

        skill_id, _ = self.rng.choice(teachable)
        skill = self.skills.get(skill_id)
        if skill is None:
            return

        potential_students = [
            nid for nid in neighbors
            if skill_id not in self.agents[nid].skills
            and self._has_prerequisites(self.agents[nid], skill)
        ]
        if not potential_students:
            return

        student_id = self.rng.choice(potential_students)
        student = self.agents[student_id]

        prob = self._teaching_probability(agent, skill)
        success = self.rng.random() < prob

        if success:
            proficiency = self.rng.uniform(0.4, 0.8)
            student.skills[skill_id] = proficiency
            agent.teacher_reputation = min(1.0, agent.teacher_reputation + 0.05)

            lineage = self.lineages.get(skill_id)
            if lineage:
                lineage.transmission_chain.append(
                    (agent.agent_id, student_id, self.current_tick)
                )
                lineage.total_transmissions += 1
                lineage.fidelity = lineage.fidelity * 0.9 + proficiency * 0.1
                holders = sum(
                    1 for a in self.agents.values() if skill_id in a.skills
                )
                lineage.geographic_spread = holders

        agent.energy -= self.teaching_cost
        agent.teaching_events += 1
        student.learning_events += 1

        event = LearningEvent(
            tick=self.current_tick,
            learner_id=student_id,
            skill_id=skill_id,
            mode=LearningMode.TEACHING,
            success=success,
            teacher_id=agent.agent_id,
            proficiency_gained=student.skills.get(skill_id, 0.0) if success else 0.0,
        )
        self.events.append(event)

    def _do_innovation(self, agent_ids: List[str]) -> None:
        """Rare innovation: agents combine skills into new ones."""
        for agent_id in agent_ids:
            if self.rng.random() > self.innovation_rate:
                continue

            agent = self.agents[agent_id]
            if len(agent.skills) < 2:
                continue

            parent_ids = self.rng.sample(list(agent.skills.keys()), 2)
            parents = [self.skills[pid] for pid in parent_ids if pid in self.skills]
            if len(parents) < 2:
                continue

            max_parent_level = max(COMPLEXITY_LEVELS[p.complexity] for p in parents)
            new_level = min(4, max_parent_level + 1)
            complexity_map = {v: k for k, v in COMPLEXITY_LEVELS.items()}
            new_complexity = complexity_map[new_level]

            self._skill_counter += 1
            new_id = f"skill-{self._skill_counter:04d}"
            new_name = _generate_skill_name(self.rng)

            new_skill = Skill(
                skill_id=new_id,
                name=new_name,
                complexity=new_complexity,
                prerequisites=parent_ids,
                fitness_value=self.rng.uniform(0.4, 1.0),
                generation_introduced=self.current_tick,
                inventor_id=agent_id,
                parent_skills=parent_ids,
            )
            self.skills[new_id] = new_skill
            self.lineages[new_id] = CulturalLineage(skill_id=new_id)

            agent.skills[new_id] = self.rng.uniform(0.5, 0.9)
            agent.innovations += 1

            event = LearningEvent(
                tick=self.current_tick,
                learner_id=agent_id,
                skill_id=new_id,
                mode=LearningMode.INNOVATION,
                success=True,
                proficiency_gained=agent.skills[new_id],
            )
            self.events.append(event)

    def _do_decay(self) -> None:
        """Skills unused for long periods decay slightly."""
        if self.current_tick % 10 != 0:
            return
        for agent in self.agents.values():
            to_remove = []
            for skill_id, prof in agent.skills.items():
                if self.rng.random() < 0.1:
                    new_prof = prof - 0.02
                    if new_prof <= 0.05:
                        to_remove.append(skill_id)
                    else:
                        agent.skills[skill_id] = new_prof
            for sid in to_remove:
                del agent.skills[sid]

    def _max_complexity_depth(self) -> int:
        """Compute max prerequisite chain depth."""
        cache: Dict[str, int] = {}

        def depth(skill_id: str) -> int:
            if skill_id in cache:
                return cache[skill_id]
            skill = self.skills.get(skill_id)
            if not skill or not skill.prerequisites:
                cache[skill_id] = 0
                return 0
            d = 1 + max(depth(p) for p in skill.prerequisites if p in self.skills)
            cache[skill_id] = d
            return d

        if not self.skills:
            return 0
        return max(depth(sid) for sid in self.skills)

    def _compute_gini(self) -> float:
        """Compute Gini coefficient of knowledge distribution."""
        counts = [len(a.skills) for a in self.agents.values()]
        if not counts or max(counts) == 0:
            return 0.0
        n = len(counts)
        counts_sorted = sorted(counts)
        cumulative = sum((2 * (i + 1) - n - 1) * counts_sorted[i] for i in range(n))
        total = sum(counts)
        if total == 0:
            return 0.0
        return cumulative / (n * total)

    def simulate(self, steps: int = 100) -> SimulationReport:
        """Run full simulation."""
        for _ in range(steps):
            self.tick()

        health = self.analyze()
        insights = self.get_insights()

        mode_counts: Dict[str, int] = defaultdict(int)
        for ev in self.events:
            mode_counts[ev.mode.value] += 1

        teachers = sorted(
            [(a.agent_id, a.teaching_events) for a in self.agents.values()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        innovation_count = sum(1 for e in self.events if e.mode == LearningMode.INNOVATION)

        return SimulationReport(
            ticks=self.current_tick,
            num_agents=len(self.agents),
            total_skills=len(self.skills),
            total_events=len(self.events),
            health=health,
            insights=insights,
            complexity_timeline=self._complexity_timeline,
            gini_timeline=self._gini_timeline,
            learning_mode_counts=dict(mode_counts),
            top_teachers=teachers,
            innovation_count=innovation_count,
        )

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent's profile."""
        return self.agents.get(agent_id)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill definition."""
        return self.skills.get(skill_id)

    def get_lineage(self, skill_id: str) -> Optional[CulturalLineage]:
        """Get transmission lineage for a skill."""
        return self.lineages.get(skill_id)

    def analyze(self) -> CulturalHealthReport:
        """Compute cultural health metrics."""
        skill_counts: Dict[str, int] = defaultdict(int)
        for agent in self.agents.values():
            for sid in agent.skills:
                skill_counts[sid] += 1
        total_holdings = sum(skill_counts.values())
        if total_holdings > 0:
            probs = [c / total_holdings for c in skill_counts.values()]
            diversity = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(skill_counts)) if len(skill_counts) > 1 else 1.0
            norm_diversity = diversity / max_entropy if max_entropy > 0 else 0.0
        else:
            norm_diversity = 0.0

        recent_events = [e for e in self.events if e.tick > self.current_tick - 20]
        learning_rate = len(recent_events) / 20.0 if self.current_tick >= 20 else (
            len(self.events) / max(1, self.current_tick)
        )

        innovations = sum(1 for e in self.events if e.mode == LearningMode.INNOVATION)
        innovation_rate = (innovations / max(1, self.current_tick)) * 100

        max_depth = self._max_complexity_depth()
        gini = self._compute_gini()

        if self.current_tick > 20:
            recent_innovations = sum(
                1 for e in self.events
                if e.mode == LearningMode.INNOVATION and e.tick > self.current_tick - 50
            )
            stagnation = max(0.0, 1.0 - recent_innovations / 3.0)
        else:
            stagnation = 0.0

        recent_success = sum(
            1 for e in recent_events if e.success and e.mode != LearningMode.OBSERVATION
        )
        momentum = min(1.0, recent_success / max(1, len(self.agents)) * 2)

        score = (
            norm_diversity * 25
            + min(1.0, learning_rate / 5.0) * 20
            + min(1.0, innovation_rate / 5.0) * 15
            + min(1.0, max_depth / 4.0) * 20
            + (1.0 - gini) * 10
            + (1.0 - stagnation) * 10
        )
        score = max(0.0, min(100.0, score))

        recs: List[str] = []
        if norm_diversity < 0.4:
            recs.append("Low skill diversity — encourage exploration of underrepresented skills")
        if gini > 0.6:
            recs.append("High knowledge inequality — promote teaching between agents")
        if stagnation > 0.7:
            recs.append("Cultural stagnation detected — increase innovation incentives")
        if max_depth < 2 and self.current_tick > 50:
            recs.append("Low cumulative complexity — skills not building on each other")
        if momentum < 0.2:
            recs.append("Low cultural momentum — transmission rate is declining")

        return CulturalHealthReport(
            score=score,
            skill_diversity=norm_diversity,
            learning_rate=learning_rate,
            innovation_rate=innovation_rate,
            max_complexity_depth=max_depth,
            knowledge_gini=gini,
            stagnation_risk=stagnation,
            cultural_momentum=momentum,
            recommendations=recs,
        )

    def get_insights(self) -> List[Insight]:
        """Generate autonomous cultural insights."""
        insights: List[Insight] = []

        for skill_id, lineage in self.lineages.items():
            holders = [
                aid for aid, a in self.agents.items() if skill_id in a.skills
            ]
            if len(holders) == 1 and self.skills[skill_id].fitness_value > 0.6:
                insights.append(Insight(
                    category="monopoly",
                    message=f"Skill '{self.skills[skill_id].name}' held only by {holders[0]} "
                            f"— high value at risk of loss",
                    severity="warning",
                    related_skills=[skill_id],
                    related_agents=holders,
                ))

        if self.current_tick > 50:
            for skill_id, lineage in self.lineages.items():
                if lineage.total_transmissions == 0:
                    holders = [
                        aid for aid, a in self.agents.items() if skill_id in a.skills
                    ]
                    if 0 < len(holders) <= 2:
                        insights.append(Insight(
                            category="dying_skill",
                            message=f"Skill '{self.skills[skill_id].name}' has never been "
                                    f"transmitted and has only {len(holders)} holder(s)",
                            severity="warning",
                            related_skills=[skill_id],
                            related_agents=holders,
                        ))

        innovators = [(a.agent_id, a.innovations) for a in self.agents.values()
                      if a.innovations >= 2]
        if innovators:
            top = max(innovators, key=lambda x: x[1])
            insights.append(Insight(
                category="hotspot",
                message=f"Innovation hotspot: {top[0]} has created {top[1]} new skills",
                severity="info",
                related_agents=[top[0]],
            ))

        prereq_demand: Dict[str, int] = defaultdict(int)
        for skill in self.skills.values():
            for p in skill.prerequisites:
                prereq_demand[p] += 1
        for skill_id, demand in prereq_demand.items():
            if demand >= 3:
                holders = sum(1 for a in self.agents.values() if skill_id in a.skills)
                if holders < self.num_agents * 0.3:
                    skill = self.skills.get(skill_id)
                    if skill:
                        insights.append(Insight(
                            category="bottleneck",
                            message=f"Skill '{skill.name}' is a prerequisite for {demand} "
                                    f"skills but only held by {holders} agents",
                            severity="critical",
                            related_skills=[skill_id],
                        ))

        traditions = [
            (sid, lin) for sid, lin in self.lineages.items()
            if lin.total_transmissions >= 5
        ]
        for sid, lin in sorted(traditions, key=lambda x: x[1].total_transmissions, reverse=True)[:3]:
            skill = self.skills.get(sid)
            if skill:
                insights.append(Insight(
                    category="tradition",
                    message=f"Skill '{skill.name}' is becoming a cultural tradition "
                            f"({lin.total_transmissions} transmissions, "
                            f"{lin.geographic_spread} holders)",
                    severity="info",
                    related_skills=[sid],
                ))

        return insights

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        report = SimulationReport(
            ticks=self.current_tick,
            num_agents=len(self.agents),
            total_skills=len(self.skills),
            total_events=len(self.events),
            health=self.analyze(),
            insights=self.get_insights(),
            complexity_timeline=self._complexity_timeline,
            gini_timeline=self._gini_timeline,
        )
        html_content = self._render_html(report)
        Path(path).write_text(html_content, encoding="utf-8")

    def export_json(self, path: str) -> None:
        """Export full state as JSON."""
        data = {
            "tick": self.current_tick,
            "num_agents": len(self.agents),
            "num_skills": len(self.skills),
            "num_events": len(self.events),
            "health": asdict(self.analyze()),
            "skills": {sid: asdict(s) for sid, s in self.skills.items()},
            "agents": {
                aid: {
                    "agent_id": a.agent_id,
                    "num_skills": len(a.skills),
                    "learning_events": a.learning_events,
                    "teaching_events": a.teaching_events,
                    "innovations": a.innovations,
                    "teacher_reputation": round(a.teacher_reputation, 3),
                }
                for aid, a in self.agents.items()
            },
            "lineages": {
                sid: {
                    "total_transmissions": lin.total_transmissions,
                    "fidelity": round(lin.fidelity, 3),
                    "geographic_spread": lin.geographic_spread,
                }
                for sid, lin in self.lineages.items()
            },
            "complexity_timeline": self._complexity_timeline,
            "gini_timeline": [round(g, 3) for g in self._gini_timeline],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _render_html(self, report: SimulationReport) -> str:
        """Render interactive HTML dashboard."""
        health = report.health
        insights = report.insights

        score_color = "#22c55e" if health.score >= 70 else "#eab308" if health.score >= 40 else "#ef4444"

        insights_html = ""
        for ins in insights[:15]:
            sev_color = {"info": "#3b82f6", "warning": "#eab308", "critical": "#ef4444"}.get(
                ins.severity, "#6b7280"
            )
            insights_html += (
                f'<div style="border-left:4px solid {sev_color};padding:8px 12px;'
                f'margin:6px 0;background:#1e293b;border-radius:4px">'
                f'<span style="color:{sev_color};font-weight:bold">[{ins.category.upper()}]</span> '
                f'{html_mod.escape(ins.message)}</div>'
            )

        teachers = sorted(
            [(a.agent_id, a.teaching_events) for a in self.agents.values()],
            key=lambda x: x[1], reverse=True,
        )[:5]
        teacher_rows = "".join(
            f"<tr><td>{html_mod.escape(t[0])}</td><td>{t[1]}</td></tr>"
            for t in teachers if t[1] > 0
        )

        timeline_js = json.dumps(self._complexity_timeline[-100:])
        gini_js = json.dumps([round(g, 3) for g in self._gini_timeline[-100:]])

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Swarm Social Learning — Cultural Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}}
h1{{color:#f8fafc;margin-bottom:8px}}
.subtitle{{color:#94a3b8;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:24px}}
.card{{background:#1e293b;border-radius:12px;padding:20px;border:1px solid #334155}}
.card h3{{color:#f1f5f9;margin-bottom:12px;font-size:14px;text-transform:uppercase;letter-spacing:0.5px}}
.score{{font-size:48px;font-weight:bold;text-align:center;margin:12px 0}}
.metric{{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #334155}}
.metric:last-child{{border:none}}
.metric .label{{color:#94a3b8}}
.metric .value{{color:#f1f5f9;font-weight:600}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-size:12px;text-transform:uppercase}}
canvas{{width:100%;height:200px}}
</style></head><body>
<h1>🧬 Swarm Social Learning Engine</h1>
<p class="subtitle">Cultural Evolution Dashboard — {report.ticks} ticks, {report.num_agents} agents, {report.total_skills} skills</p>

<div class="grid">
  <div class="card">
    <h3>Cultural Health Score</h3>
    <div class="score" style="color:{score_color}">{health.score:.0f}</div>
    <div class="metric"><span class="label">Skill Diversity</span><span class="value">{health.skill_diversity:.2f}</span></div>
    <div class="metric"><span class="label">Learning Rate</span><span class="value">{health.learning_rate:.2f}/tick</span></div>
    <div class="metric"><span class="label">Innovation Rate</span><span class="value">{health.innovation_rate:.1f}/100t</span></div>
    <div class="metric"><span class="label">Max Complexity</span><span class="value">{health.max_complexity_depth}</span></div>
    <div class="metric"><span class="label">Knowledge Gini</span><span class="value">{health.knowledge_gini:.3f}</span></div>
    <div class="metric"><span class="label">Stagnation Risk</span><span class="value">{health.stagnation_risk:.2f}</span></div>
    <div class="metric"><span class="label">Momentum</span><span class="value">{health.cultural_momentum:.2f}</span></div>
  </div>

  <div class="card">
    <h3>Complexity Timeline</h3>
    <canvas id="complexityChart"></canvas>
  </div>

  <div class="card">
    <h3>Knowledge Inequality (Gini)</h3>
    <canvas id="giniChart"></canvas>
  </div>

  <div class="card">
    <h3>Top Teachers</h3>
    <table><thead><tr><th>Agent</th><th>Sessions</th></tr></thead>
    <tbody>{teacher_rows}</tbody></table>
  </div>
</div>

<div class="card" style="margin-bottom:24px">
  <h3>Autonomous Insights</h3>
  {insights_html if insights_html else '<p style="color:#64748b">No insights generated yet</p>'}
</div>

<div class="card">
  <h3>Recommendations</h3>
  {"".join(f'<p style="padding:4px 0">• {html_mod.escape(r)}</p>' for r in health.recommendations) or '<p style="color:#64748b">Culture is healthy</p>'}
</div>

<script>
function drawLine(canvasId, data, color) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 400;
  ctx.scale(2, 2);
  const w = canvas.offsetWidth, h = 200;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((v, i) => {{
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 20) - 10;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();
}}
drawLine('complexityChart', {timeline_js}, '#22c55e');
drawLine('giniChart', {gini_js}, '#eab308');
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Social Learning Engine — cultural evolution simulator"
    )
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--skills", type=int, default=10, help="Initial skill count")
    parser.add_argument("--steps", type=int, default=200, help="Simulation ticks")
    parser.add_argument("--innovation-rate", type=float, default=0.02,
                        help="Per-agent innovation probability per tick")
    parser.add_argument("--teaching-cost", type=float, default=0.1,
                        help="Energy cost of teaching")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="HTML report output path")
    parser.add_argument("--json", type=str, default=None, help="JSON export path")
    args = parser.parse_args()

    print("🧬 Swarm Social Learning Engine")
    print("=" * 50)
    print(f"Agents: {args.agents} | Initial Skills: {args.skills} | Steps: {args.steps}")
    print(f"Innovation Rate: {args.innovation_rate} | Teaching Cost: {args.teaching_cost}")
    print()

    engine = SocialLearningEngine(
        num_agents=args.agents,
        num_initial_skills=args.skills,
        innovation_rate=args.innovation_rate,
        teaching_cost=args.teaching_cost,
        seed=args.seed,
    )
    report = engine.simulate(steps=args.steps)
    health = report.health

    score_icon = "✅" if health.score >= 70 else "⚠️" if health.score >= 40 else "❌"
    print(f"{score_icon} Cultural Health: {health.score:.0f}/100")
    print(f"   Skill Diversity: {health.skill_diversity:.2f}")
    print(f"   Learning Rate: {health.learning_rate:.2f}/tick")
    print(f"   Innovation Rate: {health.innovation_rate:.1f}/100 ticks")
    print(f"   Max Complexity: {health.max_complexity_depth}")
    print(f"   Knowledge Gini: {health.knowledge_gini:.3f}")
    print(f"   Stagnation Risk: {health.stagnation_risk:.2f}")
    print(f"   Momentum: {health.cultural_momentum:.2f}")
    print()

    print(f"📊 Summary:")
    print(f"   Total Skills: {report.total_skills}")
    print(f"   Innovations: {report.innovation_count}")
    print(f"   Learning Events: {report.total_events}")
    print()

    if report.top_teachers:
        print("🎓 Top Teachers:")
        for aid, count in report.top_teachers[:5]:
            if count > 0:
                print(f"   {aid}: {count} sessions")
        print()

    print("💡 Insights:")
    for ins in report.insights[:8]:
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(ins.severity, "•")
        print(f"   {icon} [{ins.category}] {ins.message}")
    print()

    if health.recommendations:
        print("📋 Recommendations:")
        for rec in health.recommendations:
            print(f"   • {rec}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n📄 HTML report: {args.out}")

    if args.json:
        engine.export_json(args.json)
        print(f"📄 JSON report: {args.json}")


if __name__ == "__main__":
    main()
