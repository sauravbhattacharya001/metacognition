"""Tests for Swarm Chemotaxis Engine."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chemotaxis import (
    SwarmChemotaxisEngine,
    ChemicalGrid,
    ChemicalType,
    ChemicalSource,
    ChemotacticAgent,
    SourceLocalizer,
    HealthAssessor,
    ReceptorState,
    AgentMotorState,
    GradientMeasurement,
    CHEMICAL_POLARITY,
    DEFAULT_DIFFUSION,
    DEFAULT_DECAY,
)


# ---------------------------------------------------------------------------
# ChemicalGrid Tests
# ---------------------------------------------------------------------------

def test_grid_creation():
    grid = ChemicalGrid(10)
    assert grid.size == 10
    assert grid.coverage() == 0.0


def test_grid_emit_and_read():
    grid = ChemicalGrid(10)
    grid.emit(3, 4, ChemicalType.ATTRACTANT, 5.0)
    assert grid.read(3, 4, ChemicalType.ATTRACTANT) == 5.0
    assert grid.read(3, 4, ChemicalType.REPELLENT) == 0.0


def test_grid_read_all():
    grid = ChemicalGrid(10)
    grid.emit(5, 5, ChemicalType.ATTRACTANT, 2.0)
    grid.emit(5, 5, ChemicalType.NUTRIENT, 3.0)
    all_chems = grid.read_all(5, 5)
    assert all_chems[ChemicalType.ATTRACTANT] == 2.0
    assert all_chems[ChemicalType.NUTRIENT] == 3.0


def test_grid_total_at():
    grid = ChemicalGrid(10)
    grid.emit(1, 1, ChemicalType.ATTRACTANT, 2.0)
    grid.emit(1, 1, ChemicalType.TOXIN, 3.0)
    assert grid.total_at(1, 1) == 5.0


def test_grid_coverage():
    grid = ChemicalGrid(5)
    assert grid.coverage() == 0.0
    for i in range(5):
        grid.emit(i, 0, ChemicalType.ATTRACTANT, 1.0)
    assert grid.coverage() == 5 / 25


def test_grid_wrapping():
    grid = ChemicalGrid(10)
    grid.emit(15, 12, ChemicalType.NUTRIENT, 1.0)
    assert grid.read(5, 2, ChemicalType.NUTRIENT) == 1.0


def test_grid_diffusion():
    grid = ChemicalGrid(10)
    grid.emit(5, 5, ChemicalType.ATTRACTANT, 10.0)
    initial = grid.read(5, 5, ChemicalType.ATTRACTANT)
    grid.step([])  # one step of diffusion + decay
    after = grid.read(5, 5, ChemicalType.ATTRACTANT)
    assert after < initial
    # Neighbors should have some concentration
    neighbors = [grid.read(6, 5, ChemicalType.ATTRACTANT),
                 grid.read(4, 5, ChemicalType.ATTRACTANT)]
    assert any(n > 0 for n in neighbors)


def test_grid_decay():
    grid = ChemicalGrid(10)
    grid.emit(3, 3, ChemicalType.TOXIN, 5.0)
    for _ in range(50):
        grid.step([])
    # After many steps, should be very low
    assert grid.read(3, 3, ChemicalType.TOXIN) < 0.1


def test_grid_source_emission():
    grid = ChemicalGrid(10)
    src = ChemicalSource(x=5, y=5, chemical=ChemicalType.BEACON, strength=2.0)
    grid.step([src])
    assert grid.read(5, 5, ChemicalType.BEACON) > 0


def test_grid_gradient():
    grid = ChemicalGrid(20)
    # Strong source on right side
    grid.emit(15, 10, ChemicalType.ATTRACTANT, 20.0)
    grid.step([])  # diffuse once
    dx, dy, mag = grid.gradient_at(13, 10, ChemicalType.ATTRACTANT)
    assert dx > 0  # gradient should point toward source (right)
    assert mag > 0


def test_grid_snapshot():
    grid = ChemicalGrid(10)
    grid.emit(5, 5, ChemicalType.ATTRACTANT, 3.0)
    snap = grid.snapshot()
    assert snap["total_concentration"] == 3.0
    assert snap["coverage_pct"] > 0


# ---------------------------------------------------------------------------
# ChemotacticAgent Tests
# ---------------------------------------------------------------------------

def test_agent_creation():
    agent = ChemotacticAgent("a1", 5.0, 5.0, 20)
    assert agent.agent_id == "a1"
    assert agent.x == 5.0
    assert agent.y == 5.0
    assert len(agent.receptors) == len(ChemicalType)


def test_agent_has_all_receptors():
    agent = ChemotacticAgent("a1", 0.0, 0.0, 10)
    for chem in ChemicalType:
        assert chem in agent.receptors
        assert agent.receptors[chem].sensitivity == 1.0


def test_agent_moves():
    grid = ChemicalGrid(20)
    agent = ChemotacticAgent("a1", 10.0, 10.0, 20)
    agent.move(grid, 1)
    # Agent should have moved
    assert len(agent.positions) == 2
    pos = agent.positions[-1]
    assert pos[2] == 1  # tick


def test_agent_stays_in_bounds():
    grid = ChemicalGrid(10)
    agent = ChemotacticAgent("a1", 9.5, 9.5, 10)
    for t in range(50):
        agent.move(grid, t + 1)
    for px, py, _ in agent.positions:
        assert 0 <= px < 10
        assert 0 <= py < 10


def test_agent_deposits_trail():
    grid = ChemicalGrid(20)
    agent = ChemotacticAgent("a1", 10.0, 10.0, 20)
    for t in range(5):
        agent.move(grid, t + 1)
    # Some cells should have trail pheromone
    has_trail = False
    for y in range(20):
        for x in range(20):
            if grid.read(x, y, ChemicalType.TRAIL) > 0:
                has_trail = True
                break
    assert has_trail


def test_agent_receptor_adaptation():
    grid = ChemicalGrid(20)
    # Heavy concentration at agent position
    for _ in range(10):
        grid.emit(10, 10, ChemicalType.ATTRACTANT, 20.0)
    agent = ChemotacticAgent("a1", 10.0, 10.0, 20)
    initial_sensitivity = agent.receptors[ChemicalType.ATTRACTANT].sensitivity
    for t in range(20):
        agent.sense(grid, t + 1)
    # Sensitivity should have decreased (desensitization)
    assert agent.receptors[ChemicalType.ATTRACTANT].sensitivity < initial_sensitivity


def test_agent_tumble_suppression():
    """Favorable gradient should suppress tumbling."""
    grid = ChemicalGrid(30)
    # Create strong gradient
    grid.emit(20, 15, ChemicalType.ATTRACTANT, 50.0)
    for _ in range(5):
        grid.step([])

    agent = ChemotacticAgent("a1", 15.0, 15.0, 30)
    agent.motor.direction = 0  # pointing right (toward source)

    # Run many steps and count tumbles
    tumble_count = 0
    for t in range(100):
        prev_tumbles = agent.motor.total_tumbles
        agent.move(grid, t + 1)
        if agent.motor.total_tumbles > prev_tumbles:
            tumble_count += 1

    # With gradient, should tumble less than base rate (20%)
    # Give generous margin for randomness
    assert tumble_count < 40  # less than 40% tumble rate


def test_agent_source_detection():
    src = ChemicalSource(x=5, y=5, chemical=ChemicalType.NUTRIENT, strength=3.0)
    agent = ChemotacticAgent("a1", 5.0, 5.0, 20)
    agent.check_sources([src], radius=2.0)
    traj = agent.trajectory()
    assert traj.sources_reached == 1


def test_agent_trajectory():
    grid = ChemicalGrid(20)
    agent = ChemotacticAgent("a1", 10.0, 10.0, 20)
    for t in range(10):
        agent.move(grid, t + 1)
    traj = agent.trajectory()
    assert traj.agent_id == "a1"
    assert traj.total_distance > 0
    assert 0 <= traj.chemotactic_index <= 1.0
    assert 0 <= traj.run_tumble_ratio <= 1.0


def test_agent_receptor_history():
    grid = ChemicalGrid(20)
    agent = ChemotacticAgent("a1", 10.0, 10.0, 20)
    for t in range(5):
        agent.move(grid, t + 1)
    traj = agent.trajectory()
    for chem in ChemicalType:
        assert chem in traj.receptor_history
        assert len(traj.receptor_history[chem]) > 1


# ---------------------------------------------------------------------------
# ReceptorState Tests
# ---------------------------------------------------------------------------

def test_receptor_defaults():
    r = ReceptorState(chemical=ChemicalType.ATTRACTANT)
    assert r.sensitivity == 1.0
    assert r.methylation == 0.5
    assert r.adaptation_rate == 0.05


def test_receptor_saturation_threshold():
    r = ReceptorState(chemical=ChemicalType.TOXIN, saturation_threshold=5.0)
    assert r.saturation_threshold == 5.0


# ---------------------------------------------------------------------------
# AgentMotorState Tests
# ---------------------------------------------------------------------------

def test_motor_defaults():
    m = AgentMotorState()
    assert m.running is True
    assert m.tumble_rate_base == 0.2
    assert m.total_runs == 0
    assert m.total_tumbles == 0


# ---------------------------------------------------------------------------
# SourceLocalizer Tests
# ---------------------------------------------------------------------------

def test_localizer_no_measurements():
    loc = SourceLocalizer(20)
    report = loc.localize([], [])
    assert len(report.sources) == 0
    assert report.convergence_score == 0.0


def test_localizer_few_measurements():
    loc = SourceLocalizer(20)
    meas = [
        GradientMeasurement("a1", 10, 10, 0.5, 0.5, 0.7, ChemicalType.ATTRACTANT, 1),
        GradientMeasurement("a2", 8, 10, 0.6, 0.3, 0.65, ChemicalType.ATTRACTANT, 2),
    ]
    report = loc.localize(meas, [])
    # Not enough measurements (need >=3)
    assert len(report.sources) == 0


def test_localizer_sufficient_measurements():
    loc = SourceLocalizer(20)
    src = ChemicalSource(x=15, y=15, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    meas = [
        GradientMeasurement("a1", 10, 15, 0.8, 0.0, 0.8, ChemicalType.ATTRACTANT, 1),
        GradientMeasurement("a2", 15, 10, 0.0, 0.8, 0.8, ChemicalType.ATTRACTANT, 2),
        GradientMeasurement("a3", 12, 12, 0.5, 0.5, 0.7, ChemicalType.ATTRACTANT, 3),
    ]
    report = loc.localize(meas, [src])
    assert len(report.sources) >= 1
    det = report.sources[0]
    assert det.chemical == ChemicalType.ATTRACTANT
    assert det.confidence > 0
    assert det.contributing_agents >= 2


def test_localizer_multiple_chemicals():
    loc = SourceLocalizer(20)
    src_a = ChemicalSource(x=15, y=15, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    src_n = ChemicalSource(x=5, y=5, chemical=ChemicalType.NUTRIENT, strength=3.0)
    meas = [
        GradientMeasurement("a1", 10, 15, 0.8, 0.0, 0.8, ChemicalType.ATTRACTANT, 1),
        GradientMeasurement("a2", 15, 10, 0.0, 0.8, 0.8, ChemicalType.ATTRACTANT, 2),
        GradientMeasurement("a3", 12, 12, 0.5, 0.5, 0.7, ChemicalType.ATTRACTANT, 3),
        GradientMeasurement("a1", 8, 5, -0.7, 0.0, 0.7, ChemicalType.NUTRIENT, 1),
        GradientMeasurement("a2", 5, 8, 0.0, -0.7, 0.7, ChemicalType.NUTRIENT, 2),
        GradientMeasurement("a4", 6, 6, -0.4, -0.4, 0.56, ChemicalType.NUTRIENT, 3),
    ]
    report = loc.localize(meas, [src_a, src_n])
    chems_found = {d.chemical for d in report.sources}
    assert ChemicalType.ATTRACTANT in chems_found
    assert ChemicalType.NUTRIENT in chems_found


# ---------------------------------------------------------------------------
# HealthAssessor Tests
# ---------------------------------------------------------------------------

def test_health_no_data():
    assessor = HealthAssessor()
    from src.chemotaxis import LocalizationReport
    loc = LocalizationReport(sources=[], convergence_score=0.0, triangulation_accuracy=20.0)
    grid = ChemicalGrid(10)
    health = assessor.assess([], loc, grid, [])
    assert 0 <= health.score <= 100
    assert len(health.recommendations) > 0


def test_health_with_good_data():
    from src.chemotaxis import LocalizationReport, DetectedSource, AgentTrajectory
    assessor = HealthAssessor()
    loc = LocalizationReport(
        sources=[DetectedSource(15, 15, ChemicalType.ATTRACTANT, 5.0, 0.8, 3, 1.0)],
        convergence_score=0.8,
        triangulation_accuracy=2.0,
    )
    traj = AgentTrajectory(
        agent_id="a1",
        positions=[(5, 5, 0), (15, 15, 100)],
        chemotactic_index=0.7,
        total_distance=20.0,
        net_displacement=14.1,
        receptor_history={ct: [0.8] for ct in ChemicalType},
        run_tumble_ratio=0.75,
        sources_reached=1,
    )
    grid = ChemicalGrid(20)
    src = ChemicalSource(x=15, y=15, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    health = assessor.assess([traj], loc, grid, [src])
    assert health.score > 30  # should be decent with good metrics


def test_health_recommendations_low_ci():
    from src.chemotaxis import LocalizationReport, AgentTrajectory
    assessor = HealthAssessor()
    loc = LocalizationReport(sources=[], convergence_score=0.1, triangulation_accuracy=20.0)
    traj = AgentTrajectory(
        agent_id="a1", positions=[(5, 5, 0)], chemotactic_index=0.05,
        total_distance=100.0, net_displacement=5.0,
        receptor_history={ct: [0.9] for ct in ChemicalType},
        run_tumble_ratio=0.5, sources_reached=0,
    )
    grid = ChemicalGrid(10)
    health = assessor.assess([traj], loc, grid, [])
    recs = " ".join(health.recommendations)
    assert "chemotactic index" in recs.lower() or "random" in recs.lower()


# ---------------------------------------------------------------------------
# SwarmChemotaxisEngine Tests
# ---------------------------------------------------------------------------

def test_engine_creation():
    engine = SwarmChemotaxisEngine(grid_size=20, num_agents=5)
    assert engine.grid_size == 20
    assert engine.num_agents == 5
    assert len(engine.agents) == 5


def test_engine_add_source():
    engine = SwarmChemotaxisEngine(grid_size=20, num_agents=3)
    src = engine.add_source(x=10, y=10, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    assert len(engine.sources) == 1
    assert src.chemical == ChemicalType.ATTRACTANT


def test_engine_tick():
    engine = SwarmChemotaxisEngine(grid_size=20, num_agents=3)
    engine.add_source(x=15, y=15, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    engine.tick()
    assert engine._tick == 1


def test_engine_simulate():
    engine = SwarmChemotaxisEngine(grid_size=20, num_agents=4)
    engine.add_source(x=15, y=15, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    report = engine.simulate(steps=50)
    assert report.total_ticks == 50
    assert len(report.trajectories) == 4
    assert report.health.score >= 0
    assert report.fleet_ci >= 0


def test_engine_multiple_sources():
    engine = SwarmChemotaxisEngine(grid_size=30, num_agents=6)
    engine.add_source(x=25, y=25, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    engine.add_source(x=5, y=5, chemical=ChemicalType.REPELLENT, strength=3.0)
    engine.add_source(x=15, y=20, chemical=ChemicalType.NUTRIENT, strength=4.0)
    report = engine.simulate(steps=100)
    assert report.num_sources == 3
    assert len(report.trajectories) == 6


def test_engine_localization():
    engine = SwarmChemotaxisEngine(grid_size=30, num_agents=8)
    engine.add_source(x=25, y=25, chemical=ChemicalType.ATTRACTANT, strength=8.0)
    report = engine.simulate(steps=150)
    # Should detect at least the strong source
    assert report.localization is not None


def test_engine_export_html(tmp_path):
    engine = SwarmChemotaxisEngine(grid_size=15, num_agents=3)
    engine.add_source(x=10, y=10, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    engine.simulate(steps=30)
    out = str(tmp_path / "test_chemotaxis.html")
    engine.export_html(out)
    content = Path(out).read_text()
    assert "Swarm Chemotaxis" in content
    assert "Navigation Health" in content


def test_engine_export_json(tmp_path):
    engine = SwarmChemotaxisEngine(grid_size=15, num_agents=3)
    engine.add_source(x=10, y=10, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    engine.simulate(steps=30)
    out = str(tmp_path / "test_chemotaxis.json")
    engine.export_json(out)
    import json
    data = json.loads(Path(out).read_text())
    assert "health" in data
    assert "trajectories" in data


# ---------------------------------------------------------------------------
# Chemical Constants Tests
# ---------------------------------------------------------------------------

def test_chemical_polarity():
    assert CHEMICAL_POLARITY[ChemicalType.ATTRACTANT] > 0
    assert CHEMICAL_POLARITY[ChemicalType.REPELLENT] < 0
    assert CHEMICAL_POLARITY[ChemicalType.NUTRIENT] > 0
    assert CHEMICAL_POLARITY[ChemicalType.TOXIN] < 0


def test_diffusion_rates():
    for chem in ChemicalType:
        assert chem in DEFAULT_DIFFUSION
        assert 0 < DEFAULT_DIFFUSION[chem] < 1


def test_decay_rates():
    for chem in ChemicalType:
        assert chem in DEFAULT_DECAY
        assert 0 < DEFAULT_DECAY[chem] < 1


# ---------------------------------------------------------------------------
# Demo Tests
# ---------------------------------------------------------------------------

def test_run_demo():
    from src.chemotaxis import run_demo
    report = run_demo(grid_size=15, num_agents=4, steps=50)
    assert report.total_ticks == 50
    assert report.health.score >= 0
    assert len(report.trajectories) == 4


def test_demo_health_bounded():
    from src.chemotaxis import run_demo
    report = run_demo(grid_size=20, num_agents=6, steps=80)
    assert 0 <= report.health.score <= 100
    assert 0 <= report.health.avg_chemotactic_index <= 1.0
    assert 0 <= report.health.receptor_health <= 1.0
    assert 0 <= report.health.source_coverage <= 1.0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

def test_agents_move_toward_attractant():
    """Over many steps, agents should generally end up closer to attractants."""
    engine = SwarmChemotaxisEngine(grid_size=40, num_agents=10)
    src_x, src_y = 35, 35
    engine.add_source(x=src_x, y=src_y,
                      chemical=ChemicalType.ATTRACTANT, strength=10.0)
    report = engine.simulate(steps=200)

    # At least some agents should be closer to source than their start
    closer_count = 0
    for traj in report.trajectories:
        start_x, start_y, _ = traj.positions[0]
        end_x, end_y, _ = traj.positions[-1]
        start_dist = math.sqrt((start_x - src_x)**2 + (start_y - src_y)**2)
        end_dist = math.sqrt((end_x - src_x)**2 + (end_y - src_y)**2)
        if end_dist < start_dist:
            closer_count += 1
    # At least 30% should be closer (accounting for randomness)
    assert closer_count >= 3


def test_full_pipeline():
    """End-to-end test: create, simulate, analyze, export."""
    engine = SwarmChemotaxisEngine(grid_size=25, num_agents=6)
    engine.add_source(x=20, y=20, chemical=ChemicalType.ATTRACTANT, strength=6.0)
    engine.add_source(x=5, y=5, chemical=ChemicalType.REPELLENT, strength=4.0)

    report = engine.simulate(steps=100)

    assert report.health is not None
    assert report.localization is not None
    assert report.fleet_ci >= 0
    assert report.grid_size == 25
    assert report.total_ticks == 100
    assert report.num_sources == 2
    assert len(report.trajectories) == 6

    # All trajectories should have valid data
    for traj in report.trajectories:
        assert len(traj.positions) > 1
        assert traj.total_distance >= 0
        assert 0 <= traj.chemotactic_index <= 1.0
