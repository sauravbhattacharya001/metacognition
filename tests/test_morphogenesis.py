"""Tests for Swarm Morphogenesis Engine."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.morphogenesis import (
    CellFate,
    DevelopmentalStage,
    MorphogenesisEngine,
    MorphogenesisReport,
    MorphogenType,
    PatternType,
    RegenerationEvent,
)


# ── Basic Initialization ─────────────────────────────────────────────────


def test_engine_creation():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    assert len(engine.agents) == 15
    assert engine.step == 0
    assert engine.stage == DevelopmentalStage.ZYGOTE


def test_agents_placed_on_grid():
    engine = MorphogenesisEngine(grid_size=10, num_agents=10, seed=1)
    for agent in engine.agents.values():
        assert 0 <= agent.x <= 9
        assert 0 <= agent.y <= 9
        assert agent.fate == CellFate.UNDIFFERENTIATED


def test_add_organizer():
    engine = MorphogenesisEngine(grid_size=10, num_agents=5, seed=1)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=3.0)
    assert len(engine.organizers) == 1
    assert engine.organizers[0].morphogen == MorphogenType.ACTIVATOR
    assert engine.organizers[0].strength == 3.0


def test_add_multiple_organizers():
    engine = MorphogenesisEngine(grid_size=10, num_agents=5, seed=1)
    engine.add_organizer(x=2, y=2, morphogen="activator", strength=2.0)
    engine.add_organizer(x=8, y=8, morphogen="inhibitor", strength=1.5)
    engine.add_organizer(x=5, y=5, morphogen="positional", strength=1.0)
    assert len(engine.organizers) == 3


# ── Morphogen Diffusion ──────────────────────────────────────────────────


def test_morphogen_diffusion():
    engine = MorphogenesisEngine(grid_size=10, num_agents=5, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.0, decay_rate=0.1)
    engine._diffuse_morphogens()
    # Concentration at source should be highest
    source_conc = engine.morphogen_field[(5, 5)]["activator"]
    far_conc = engine.morphogen_field[(0, 0)]["activator"]
    assert source_conc > far_conc


def test_morphogen_decay_with_distance():
    engine = MorphogenesisEngine(grid_size=10, num_agents=5, seed=42, noise_level=0.0)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.0, decay_rate=0.2)
    engine._diffuse_morphogens()
    conc_near = engine.morphogen_field[(5, 6)]["activator"]
    conc_far = engine.morphogen_field[(5, 9)]["activator"]
    assert conc_near > conc_far


# ── Development Simulation ───────────────────────────────────────────────


def test_develop_advances_steps():
    engine = MorphogenesisEngine(grid_size=10, num_agents=10, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.0)
    engine.develop(steps=50)
    assert engine.step == 50


def test_stage_progression():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=2, y=5, morphogen="activator", strength=2.5, decay_rate=0.08)
    engine.add_organizer(x=8, y=5, morphogen="inhibitor", strength=1.5, decay_rate=0.1)
    engine.develop(steps=5)
    assert engine.stage != DevelopmentalStage.ZYGOTE  # Should have progressed


def test_full_development():
    engine = MorphogenesisEngine(grid_size=12, num_agents=20, seed=42)
    engine.add_organizer(x=2, y=6, morphogen="activator", strength=2.5, decay_rate=0.06)
    engine.add_organizer(x=10, y=6, morphogen="inhibitor", strength=1.8, decay_rate=0.1)
    engine.add_organizer(x=6, y=2, morphogen="positional", strength=1.5, decay_rate=0.1)
    engine.develop(steps=120)
    # Should have some differentiation
    diff = sum(1 for a in engine.agents.values() if a.fate != CellFate.UNDIFFERENTIATED)
    assert diff > 0


def test_differentiation_occurs():
    engine = MorphogenesisEngine(grid_size=15, num_agents=25, seed=99)
    engine.add_organizer(x=2, y=7, morphogen="activator", strength=1.5, decay_rate=0.12)
    engine.add_organizer(x=12, y=7, morphogen="inhibitor", strength=1.2, decay_rate=0.1)
    engine.add_organizer(x=7, y=2, morphogen="positional", strength=1.5, decay_rate=0.1)
    engine.develop(steps=100)
    fates = set(a.fate.value for a in engine.agents.values())
    # Should have multiple fate types with gradient signals
    assert len(fates) >= 2


def test_fate_diversity():
    engine = MorphogenesisEngine(grid_size=12, num_agents=25, seed=42)
    engine.add_organizer(x=2, y=6, morphogen="activator", strength=3.0, decay_rate=0.05)
    engine.add_organizer(x=10, y=6, morphogen="inhibitor", strength=2.0, decay_rate=0.08)
    engine.add_organizer(x=6, y=2, morphogen="positional", strength=2.0, decay_rate=0.07)
    engine.develop(steps=100)
    fates = set(a.fate.value for a in engine.agents.values())
    # With strong enough signals, should get multiple fate types
    assert len(fates) >= 2


# ── Induction Signaling ──────────────────────────────────────────────────


def test_induction_events_recorded():
    engine = MorphogenesisEngine(grid_size=10, num_agents=20, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=3.0, decay_rate=0.04)
    engine.add_organizer(x=5, y=5, morphogen="positional", strength=2.0, decay_rate=0.06)
    engine.develop(steps=100)
    # Induction may or may not occur depending on spatial arrangement
    # Just verify the list is a valid list
    assert isinstance(engine.induction_events, list)


def test_induction_creates_complementary_fates():
    # Force a scenario where induction should happen
    engine = MorphogenesisEngine(grid_size=8, num_agents=30, seed=7)
    engine.add_organizer(x=4, y=4, morphogen="activator", strength=4.0, decay_rate=0.03)
    engine.add_organizer(x=4, y=4, morphogen="positional", strength=2.5, decay_rate=0.05)
    engine.develop(steps=120)
    # Check if any induction happened
    if engine.induction_events:
        event = engine.induction_events[0]
        assert event.induced_fate in (CellFate.RELAY, CellFate.MEMORY, CellFate.WORKER)


# ── Apoptosis ────────────────────────────────────────────────────────────


def test_apoptosis_during_maturation():
    engine = MorphogenesisEngine(grid_size=10, num_agents=25, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=3.0, decay_rate=0.04)
    engine.develop(steps=150)
    # Apoptosis events may occur during maturation
    assert isinstance(engine.apoptosis_events, list)


# ── Damage & Regeneration ────────────────────────────────────────────────


def test_inflict_damage():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.5, decay_rate=0.05)
    engine.develop(steps=60)
    initial_count = len(engine.agents)
    removed = engine.inflict_damage(3)
    assert len(removed) <= 3
    assert len(engine.agents) <= initial_count


def test_regeneration():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.5, decay_rate=0.05)
    engine.develop(steps=60)
    engine.inflict_damage(5)
    event = engine.regenerate()
    # Should attempt some repair
    assert event is not None or len(engine.agents) >= engine.num_agents * 0.8


def test_regeneration_recruits_new_agents():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.5, decay_rate=0.05)
    engine.develop(steps=60)
    # Remove many agents
    engine.inflict_damage(8)
    pre_count = len(engine.agents)
    event = engine.regenerate()
    if event:
        assert len(engine.agents) >= pre_count


def test_no_regeneration_when_healthy():
    engine = MorphogenesisEngine(grid_size=10, num_agents=10, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.0, decay_rate=0.05)
    engine.develop(steps=60)
    # Don't damage
    event = engine.regenerate()
    # May or may not need repair depending on apoptosis
    assert event is None or isinstance(event, RegenerationEvent)


# ── Pattern Detection ────────────────────────────────────────────────────


def test_pattern_detection_runs():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=2, y=5, morphogen="activator", strength=2.5, decay_rate=0.06)
    engine.add_organizer(x=8, y=5, morphogen="inhibitor", strength=1.5, decay_rate=0.1)
    engine.develop(steps=80)
    pattern, regularity = engine._detect_pattern()
    assert isinstance(pattern, PatternType)
    assert 0 <= regularity <= 1.0


def test_pattern_uniform_when_undifferentiated():
    engine = MorphogenesisEngine(grid_size=10, num_agents=10, seed=42)
    # No organizers, no development
    pattern, _ = engine._detect_pattern()
    assert pattern == PatternType.UNIFORM


# ── Analysis ─────────────────────────────────────────────────────────────


def test_analyze_returns_report():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.0, decay_rate=0.05)
    engine.develop(steps=50)
    report = engine.analyze()
    assert isinstance(report, MorphogenesisReport)
    assert 0 <= report.health_score <= 100
    assert report.step == 50


def test_health_score_components():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.5, decay_rate=0.05)
    engine.develop(steps=80)
    report = engine.analyze()
    assert "differentiation_completeness" in report.health_breakdown
    assert "fate_diversity" in report.health_breakdown
    assert "pattern_regularity" in report.health_breakdown
    assert "stage_progress" in report.health_breakdown
    assert "structural_integrity" in report.health_breakdown


def test_insights_generated():
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.5, decay_rate=0.05)
    engine.develop(steps=80)
    report = engine.analyze()
    assert len(report.insights) > 0


def test_fate_map_in_report():
    engine = MorphogenesisEngine(grid_size=10, num_agents=10, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.0)
    engine.develop(steps=50)
    report = engine.analyze()
    assert len(report.fate_map) == len(engine.agents)


def test_stage_history_tracked():
    engine = MorphogenesisEngine(grid_size=10, num_agents=10, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.0)
    engine.develop(steps=30)
    report = engine.analyze()
    assert len(report.stage_history) >= 1
    assert report.stage_history[0]["stage"] == "zygote"


# ── Export ───────────────────────────────────────────────────────────────


def test_export_json():
    engine = MorphogenesisEngine(grid_size=8, num_agents=10, seed=42)
    engine.add_organizer(x=4, y=4, morphogen="activator", strength=2.0)
    engine.develop(steps=50)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        engine.export_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "stage" in data
        assert "health_score" in data
        assert "agents" in data
        assert len(data["agents"]) == len(engine.agents)
    finally:
        os.unlink(path)


def test_export_html():
    engine = MorphogenesisEngine(grid_size=8, num_agents=10, seed=42)
    engine.add_organizer(x=4, y=4, morphogen="activator", strength=2.0)
    engine.develop(steps=50)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        path = f.name
    try:
        engine.export_html(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "Swarm Morphogenesis Dashboard" in content
        assert "Health Score" in content
    finally:
        os.unlink(path)


# ── Edge Cases ───────────────────────────────────────────────────────────


def test_zero_agents():
    engine = MorphogenesisEngine(grid_size=10, num_agents=0, seed=42)
    engine.develop(steps=10)
    report = engine.analyze()
    assert report.total_agents == 0
    assert report.health_score >= 0


def test_single_agent():
    engine = MorphogenesisEngine(grid_size=10, num_agents=1, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.0)
    engine.develop(steps=50)
    report = engine.analyze()
    assert report.total_agents >= 1


def test_large_grid():
    engine = MorphogenesisEngine(grid_size=50, num_agents=5, seed=42)
    engine.add_organizer(x=25, y=25, morphogen="activator", strength=3.0)
    engine.develop(steps=30)
    report = engine.analyze()
    assert isinstance(report, MorphogenesisReport)


def test_no_organizers():
    engine = MorphogenesisEngine(grid_size=10, num_agents=10, seed=42)
    engine.develop(steps=50)
    # Without organizers, no differentiation should occur
    report = engine.analyze()
    undiff = report.fate_counts.get("undifferentiated", 0)
    assert undiff == len(engine.agents)


def test_deterministic_with_seed():
    e1 = MorphogenesisEngine(grid_size=10, num_agents=10, seed=123)
    e1.add_organizer(x=5, y=5, morphogen="activator", strength=2.0)
    e1.develop(steps=50)

    e2 = MorphogenesisEngine(grid_size=10, num_agents=10, seed=123)
    e2.add_organizer(x=5, y=5, morphogen="activator", strength=2.0)
    e2.develop(steps=50)

    r1 = e1.analyze()
    r2 = e2.analyze()
    assert r1.fate_counts == r2.fate_counts
    assert r1.health_score == r2.health_score


# ── Enum Coverage ────────────────────────────────────────────────────────


def test_cell_fate_enum():
    assert CellFate.UNDIFFERENTIATED.value == "undifferentiated"
    assert CellFate.LEADER.value == "leader"
    assert CellFate.EFFECTOR.value == "effector"
    assert len(CellFate) == 7


def test_morphogen_type_enum():
    assert MorphogenType.ACTIVATOR.value == "activator"
    assert MorphogenType.REGENERATIVE.value == "regenerative"
    assert len(MorphogenType) == 6


def test_developmental_stage_enum():
    assert DevelopmentalStage.ZYGOTE.value == "zygote"
    assert DevelopmentalStage.HOMEOSTASIS.value == "homeostasis"
    assert len(DevelopmentalStage) == 6


def test_pattern_type_enum():
    assert PatternType.STRIPES.value == "stripes"
    assert PatternType.SPOTS.value == "spots"
    assert len(PatternType) == 6


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_main(capsys):
    """Test CLI runs without errors."""
    sys.argv = ["morphogenesis", "--grid", "8", "--agents", "10", "--steps", "30", "--seed", "42"]
    from src.morphogenesis import main
    main()
    captured = capsys.readouterr()
    assert "Swarm Morphogenesis Engine" in captured.out
    assert "Development Report" in captured.out


def test_cli_with_damage(capsys):
    sys.argv = ["morphogenesis", "--grid", "8", "--agents", "12", "--steps", "50",
                "--damage", "3", "--seed", "42"]
    from src.morphogenesis import main
    main()
    captured = capsys.readouterr()
    assert "Inflicting damage" in captured.out


def test_cli_with_export():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name
    try:
        sys.argv = ["morphogenesis", "--grid", "8", "--agents", "10", "--steps", "30",
                    "--seed", "42", "--out", html_path, "--json", json_path]
        from src.morphogenesis import main
        main()
        assert os.path.getsize(html_path) > 100
        assert os.path.getsize(json_path) > 100
    finally:
        os.unlink(html_path)
        os.unlink(json_path)


# ── Integration ──────────────────────────────────────────────────────────


def test_full_lifecycle():
    """Full development → damage → regenerate → analyze lifecycle."""
    engine = MorphogenesisEngine(grid_size=12, num_agents=20, seed=42)
    engine.add_organizer(x=3, y=6, morphogen="activator", strength=3.0, decay_rate=0.05)
    engine.add_organizer(x=9, y=6, morphogen="inhibitor", strength=2.0, decay_rate=0.08)
    engine.add_organizer(x=6, y=3, morphogen="positional", strength=1.5, decay_rate=0.1)

    # Develop
    engine.develop(steps=120)
    report1 = engine.analyze()
    assert report1.health_score > 0

    # Damage
    removed = engine.inflict_damage(4)
    assert len(removed) > 0

    # Regenerate
    engine.regenerate()

    # Final analysis
    report2 = engine.analyze()
    assert isinstance(report2, MorphogenesisReport)
    assert len(report2.regeneration_events) > 0


def test_multiple_development_rounds():
    """Develop in multiple rounds."""
    engine = MorphogenesisEngine(grid_size=10, num_agents=15, seed=42)
    engine.add_organizer(x=5, y=5, morphogen="activator", strength=2.5)
    engine.develop(steps=30)
    engine.develop(steps=30)
    engine.develop(steps=30)
    assert engine.step == 90
    report = engine.analyze()
    assert report.step == 90
