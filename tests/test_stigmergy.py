"""Tests for Swarm Stigmergy Engine."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stigmergy import (
    StigmergyEngine,
    PheromoneGrid,
    PheromoneType,
    PheromoneDeposit,
    GradientComputer,
    GradientVector,
    TraceArchaeologist,
    HealthAssessor,
    StigmergicAgent,
    DEFAULT_HALF_LIVES,
)


# ---------------------------------------------------------------------------
# PheromoneGrid Tests
# ---------------------------------------------------------------------------

def test_grid_creation():
    grid = PheromoneGrid(10)
    assert grid.size == 10
    assert grid.coverage() == 0.0


def test_grid_deposit():
    grid = PheromoneGrid(10)
    dep = PheromoneDeposit(
        agent_id="a1", ptype=PheromoneType.ATTRACTION,
        intensity=1.0, x=3, y=4, tick_deposited=1
    )
    grid.deposit(dep)
    assert grid.read_type(3, 4, PheromoneType.ATTRACTION) == 1.0
    assert grid.total_intensity_at(3, 4) == 1.0


def test_grid_reinforcement():
    grid = PheromoneGrid(10)
    dep1 = PheromoneDeposit(agent_id="a1", ptype=PheromoneType.ATTRACTION,
                            intensity=1.0, x=5, y=5, tick_deposited=1)
    dep2 = PheromoneDeposit(agent_id="a2", ptype=PheromoneType.ATTRACTION,
                            intensity=1.0, x=5, y=5, tick_deposited=2)
    grid.deposit(dep1)
    grid.deposit(dep2)
    # Second deposit should be superlinearly amplified
    val = grid.read_type(5, 5, PheromoneType.ATTRACTION)
    assert val > 2.0  # More than simple addition due to reinforcement


def test_grid_evaporation():
    grid = PheromoneGrid(10)
    dep = PheromoneDeposit(agent_id="a1", ptype=PheromoneType.ATTRACTION,
                           intensity=5.0, x=2, y=2, tick_deposited=0)
    grid.deposit(dep)
    initial = grid.read_type(2, 2, PheromoneType.ATTRACTION)
    grid.evaporate(10)
    after = grid.read_type(2, 2, PheromoneType.ATTRACTION)
    assert after < initial


def test_grid_full_evaporation():
    grid = PheromoneGrid(10)
    dep = PheromoneDeposit(agent_id="a1", ptype=PheromoneType.ATTRACTION,
                           intensity=0.05, x=1, y=1, tick_deposited=0)
    grid.deposit(dep)
    grid.evaporate(100)
    assert grid.read_type(1, 1, PheromoneType.ATTRACTION) == 0.0
    assert grid.coverage() == 0.0


def test_grid_coverage():
    grid = PheromoneGrid(5)
    for i in range(5):
        dep = PheromoneDeposit(agent_id="a1", ptype=PheromoneType.ATTRACTION,
                               intensity=1.0, x=i, y=0, tick_deposited=0)
        grid.deposit(dep)
    assert grid.coverage() == 5 / 25


def test_grid_type_distribution():
    grid = PheromoneGrid(10)
    grid.deposit(PheromoneDeposit("a1", PheromoneType.ATTRACTION, 2.0, 0, 0, 0))
    grid.deposit(PheromoneDeposit("a1", PheromoneType.DANGER, 3.0, 1, 1, 0))
    dist = grid.type_distribution()
    assert dist[PheromoneType.ATTRACTION] == 2.0
    assert dist[PheromoneType.DANGER] == 3.0


def test_grid_wrapping():
    grid = PheromoneGrid(10)
    dep = PheromoneDeposit("a1", PheromoneType.SUCCESS, 1.0, 15, 12, 0)
    grid.deposit(dep)
    # 15 % 10 = 5, 12 % 10 = 2
    assert grid.read_type(5, 2, PheromoneType.SUCCESS) == 1.0


# ---------------------------------------------------------------------------
# Gradient Tests
# ---------------------------------------------------------------------------

def test_gradient_empty():
    grid = PheromoneGrid(10)
    gc = GradientComputer(grid)
    g = gc.compute(5, 5)
    assert g.magnitude == 0.0
    assert g.attractors == 0
    assert g.repulsors == 0


def test_gradient_attraction():
    grid = PheromoneGrid(20)
    grid.deposit(PheromoneDeposit("a1", PheromoneType.ATTRACTION, 5.0, 7, 5, 0))
    gc = GradientComputer(grid)
    g = gc.compute(5, 5, sense_radius=3)
    # Gradient should point toward x=7 (positive dx)
    assert g.dx > 0
    assert g.magnitude > 0


def test_gradient_repulsion():
    grid = PheromoneGrid(20)
    grid.deposit(PheromoneDeposit("a1", PheromoneType.DANGER, 5.0, 7, 5, 0))
    gc = GradientComputer(grid)
    g = gc.compute(5, 5, sense_radius=3)
    # Gradient should point away from x=7 (negative dx)
    assert g.dx < 0
    assert g.repulsors > 0


def test_gradient_direction_degrees():
    grid = PheromoneGrid(20)
    grid.deposit(PheromoneDeposit("a1", PheromoneType.ATTRACTION, 5.0, 5, 8, 0))
    gc = GradientComputer(grid)
    g = gc.compute(5, 5, sense_radius=4)
    # Should point roughly toward y=8 (positive dy → ~90 degrees)
    assert 45 < g.direction_degrees < 135


# ---------------------------------------------------------------------------
# Agent Tests
# ---------------------------------------------------------------------------

def test_agent_creation():
    agent = StigmergicAgent("test-1", 5, 5, 20)
    assert agent.agent_id == "test-1"
    assert agent.x == 5
    assert agent.y == 5


def test_agent_move():
    agent = StigmergicAgent("test-1", 5, 5, 20)
    gradient = GradientVector(dx=1.0, dy=0.0, magnitude=2.0,
                              dominant_type=PheromoneType.ATTRACTION,
                              attractors=1, repulsors=0)
    # Force non-exploration by setting explore_bias very low
    agent.explore_bias = 0.0
    agent.move(gradient, tick=1)
    assert agent.steps_taken == 1
    assert len(agent.trail) == 1


def test_agent_trail_export():
    agent = StigmergicAgent("test-1", 3, 3, 10)
    gradient = GradientVector(dx=0, dy=0, magnitude=0,
                              dominant_type=PheromoneType.ATTRACTION,
                              attractors=0, repulsors=0)
    for i in range(5):
        agent.move(gradient, tick=i)
    trail = agent.to_trail()
    assert trail.agent_id == "test-1"
    assert len(trail.positions) == 5


def test_agent_deposit_decision():
    agent = StigmergicAgent("test-1", 5, 5, 20)
    # Call many times; at least some should produce deposits
    deposits = []
    for i in range(100):
        dep = agent.decide_deposit(tick=i)
        if dep:
            deposits.append(dep)
    assert len(deposits) > 0  # probabilistic but should have some


# ---------------------------------------------------------------------------
# Archaeology Tests
# ---------------------------------------------------------------------------

def test_archaeology_empty():
    grid = PheromoneGrid(10)
    arch = TraceArchaeologist(grid)
    report = arch.full_report(current_tick=0)
    assert report.total_deposits == 0
    assert report.coverage_pct == 0.0


def test_find_highways():
    grid = PheromoneGrid(10)
    # Create a line of high-intensity deposits
    for x in range(5):
        grid.deposit(PheromoneDeposit("a1", PheromoneType.ATTRACTION, 5.0, x, 3, 0))
    arch = TraceArchaeologist(grid)
    highways = arch.find_highways(threshold=3.0)
    assert len(highways) >= 1
    assert highways[0].total_intensity > 10


def test_find_dead_zones():
    grid = PheromoneGrid(10)
    # Deposit only in one corner
    grid.deposit(PheromoneDeposit("a1", PheromoneType.ATTRACTION, 5.0, 0, 0, 0))
    arch = TraceArchaeologist(grid)
    dead_zones = arch.find_dead_zones(min_cells=4)
    assert len(dead_zones) >= 1


def test_detect_oscillations():
    grid = PheromoneGrid(10)
    # Manually inject oscillating history
    hist = grid.history
    hist[5][5][PheromoneType.ATTRACTION] = [
        1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0
    ]
    arch = TraceArchaeologist(grid)
    oscillations = arch.detect_oscillations(min_cycles=2)
    assert len(oscillations) >= 1
    assert oscillations[0].x == 5
    assert oscillations[0].y == 5


# ---------------------------------------------------------------------------
# Health Assessment Tests
# ---------------------------------------------------------------------------

def test_health_healthy():
    grid = PheromoneGrid(10)
    # Create moderate, diverse coverage
    for i in range(10):
        for ptype in list(PheromoneType)[:4]:
            grid.deposit(PheromoneDeposit(
                f"a{i}", ptype, 1.0,
                (i * 3) % 10, (i * 7) % 10, 5
            ))
    arch = TraceArchaeologist(grid).full_report(10)
    assessor = HealthAssessor()
    health = assessor.assess(arch, grid)
    assert health.score > 0
    assert len(health.recommendations) > 0


def test_health_stagnant():
    grid = PheromoneGrid(20)
    # Nothing deposited → stagnant
    arch = TraceArchaeologist(grid).full_report(100)
    assessor = HealthAssessor()
    health = assessor.assess(arch, grid)
    assert health.stagnation_risk > 0.5
    assert health.coverage_pct == 0.0


# ---------------------------------------------------------------------------
# Engine Integration Tests
# ---------------------------------------------------------------------------

def test_engine_creation():
    engine = StigmergyEngine(grid_size=15, evaporation_rate=0.1)
    assert engine.grid_size == 15
    assert engine.current_tick == 0


def test_engine_deposit():
    engine = StigmergyEngine(grid_size=10)
    dep = engine.deposit("a1", 3, 4, PheromoneType.SUCCESS, 2.0)
    assert dep.agent_id == "a1"
    assert engine.grid.read_type(3, 4, PheromoneType.SUCCESS) == 2.0


def test_engine_tick():
    engine = StigmergyEngine(grid_size=10)
    engine.deposit("a1", 5, 5, PheromoneType.ATTRACTION, 3.0)
    before = engine.grid.read_type(5, 5, PheromoneType.ATTRACTION)
    engine.tick(steps=5)
    after = engine.grid.read_type(5, 5, PheromoneType.ATTRACTION)
    assert after < before
    assert engine.current_tick == 5


def test_engine_sense_gradient():
    engine = StigmergyEngine(grid_size=20)
    engine.deposit("a1", 10, 10, PheromoneType.ATTRACTION, 5.0)
    gradient = engine.sense_gradient("a2", 8, 10)
    assert gradient.magnitude > 0


def test_engine_add_agent():
    engine = StigmergyEngine(grid_size=10)
    agent = engine.add_agent("scout-1")
    assert agent.agent_id == "scout-1"
    assert len(engine.agents) == 1


def test_engine_simulate():
    engine = StigmergyEngine(grid_size=10, evaporation_rate=0.05)
    report = engine.simulate(steps=50, num_agents=5)
    assert report.current_tick == 50
    assert len(report.trails) == 5
    assert report.health.score >= 0


def test_engine_simulate_produces_deposits():
    engine = StigmergyEngine(grid_size=10)
    report = engine.simulate(steps=100, num_agents=8)
    assert report.archaeology.total_deposits > 0


def test_engine_export_json(tmp_path):
    engine = StigmergyEngine(grid_size=8)
    engine.simulate(steps=30, num_agents=4)
    out = str(tmp_path / "test.json") if hasattr(tmp_path, '__truediv__') else "test_stigmergy.json"
    engine.export_json(out)
    import json
    data = json.loads(Path(out).read_text())
    assert "health" in data
    assert "archaeology" in data


def test_engine_export_html(tmp_path):
    engine = StigmergyEngine(grid_size=8)
    engine.simulate(steps=30, num_agents=4)
    out = str(tmp_path / "test.html") if hasattr(tmp_path, '__truediv__') else "test_stigmergy.html"
    engine.export_html(out)
    content = Path(out).read_text(encoding="utf-8")
    assert "Swarm Stigmergy" in content
    assert "Environment Health" in content


def test_engine_analyze():
    engine = StigmergyEngine(grid_size=12)
    engine.deposit("a1", 3, 3, PheromoneType.ATTRACTION, 2.0)
    engine.deposit("a2", 6, 6, PheromoneType.DANGER, 1.5)
    report = engine.analyze()
    assert report.grid_size == 12
    assert report.archaeology.total_deposits == 2


# ---------------------------------------------------------------------------
# Pheromone Type Tests
# ---------------------------------------------------------------------------

def test_pheromone_types():
    assert len(PheromoneType) == 7
    assert PheromoneType.MONUMENT.value == "monument"


def test_default_half_lives():
    assert DEFAULT_HALF_LIVES[PheromoneType.MONUMENT] > DEFAULT_HALF_LIVES[PheromoneType.ATTRACTION]


def test_monument_persists_longer():
    grid = PheromoneGrid(10)
    grid.deposit(PheromoneDeposit("a1", PheromoneType.ATTRACTION, 1.0, 0, 0, 0))
    grid.deposit(PheromoneDeposit("a1", PheromoneType.MONUMENT, 1.0, 1, 0, 0))
    grid.evaporate(50)
    attr = grid.read_type(0, 0, PheromoneType.ATTRACTION)
    mon = grid.read_type(1, 0, PheromoneType.MONUMENT)
    assert mon > attr  # Monument should persist longer


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_zero_intensity_deposit():
    grid = PheromoneGrid(10)
    dep = PheromoneDeposit("a1", PheromoneType.ATTRACTION, 0.0, 5, 5, 0)
    grid.deposit(dep)
    assert grid.read_type(5, 5, PheromoneType.ATTRACTION) == 0.0


def test_large_grid():
    engine = StigmergyEngine(grid_size=50)
    engine.deposit("a1", 25, 25, PheromoneType.RESOURCE, 3.0)
    g = engine.sense_gradient("a2", 24, 25)
    assert g.magnitude >= 0


def test_many_agents_simulation():
    engine = StigmergyEngine(grid_size=15)
    report = engine.simulate(steps=30, num_agents=20)
    assert len(report.trails) == 20
    assert report.archaeology.total_deposits > 0


def test_multiple_types_same_cell():
    grid = PheromoneGrid(10)
    grid.deposit(PheromoneDeposit("a1", PheromoneType.ATTRACTION, 2.0, 3, 3, 0))
    grid.deposit(PheromoneDeposit("a1", PheromoneType.DANGER, 1.0, 3, 3, 0))
    grid.deposit(PheromoneDeposit("a1", PheromoneType.RESOURCE, 0.5, 3, 3, 0))
    cell = grid.read(3, 3)
    assert len(cell) == 3
    assert cell[PheromoneType.ATTRACTION] == 2.0
    assert cell[PheromoneType.DANGER] == 1.0


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    tmp_path = Path(tempfile.mkdtemp())

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            import inspect
            if "tmp_path" in inspect.signature(test_fn).parameters:
                test_fn(tmp_path)
            else:
                test_fn()
            passed += 1
            print(f"  PASS {test_fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {test_fn.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed!")
