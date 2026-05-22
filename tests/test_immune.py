"""Tests for the Consensus Immune System."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


from src.immune import (
    Antibody,
    ImmuneMemory,
    ImmuneSystem,
    Pathogen,
    _cosine,
    _hash_id,
    _jaccard,
    _simulate,
    generate_immune_report,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_round(
    round_index: int = 0,
    agents: list | None = None,
    adversarial: set | None = None,
    scenario: int = 0,
) -> dict:
    """Create a synthetic round result."""
    agents = agents or [f"a{i}" for i in range(4)]
    adversarial = adversarial or set()
    leader = agents[0]
    proposals = {}
    votes = {}
    for a in agents:
        conf = 0.2 if a in adversarial else 0.8
        proposals[a] = {"text": f"p_{a}", "confidence": conf}
        approve = False if a in adversarial else True
        votes[a] = {"approve": approve, "confidence": conf, "target": leader}
    return {
        "round_index": round_index,
        "scenario": scenario,
        "proposals": proposals,
        "votes": votes,
        "committed": True,
        "leader": leader,
    }


# ---------------------------------------------------------------------------
# Pathogen dataclass
# ---------------------------------------------------------------------------

class TestPathogen:
    def test_create(self):
        p = Pathogen("id1", "byzantine_vote", 0.8, 5, ["a1"], {"key": "val"})
        assert p.pathogen_id == "id1"
        assert p.category == "byzantine_vote"
        assert p.severity == 0.8
        assert not p.neutralized

    def test_categories(self):
        assert len(Pathogen.CATEGORIES) == 6
        assert "collusion_ring" in Pathogen.CATEGORIES


# ---------------------------------------------------------------------------
# Antibody dataclass
# ---------------------------------------------------------------------------

class TestAntibody:
    def test_create(self):
        ab = Antibody("ab1", "p1", "quarantine", ["a1"], 0.9, 10)
        assert ab.antibody_id == "ab1"
        assert ab.activations == 0
        assert ab.effectiveness == 0.5

    def test_rule_types(self):
        assert len(Antibody.RULE_TYPES) == 5
        assert "quarantine" in Antibody.RULE_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_hash_id_deterministic(self):
        assert _hash_id("a", "b") == _hash_id("a", "b")

    def test_hash_id_different(self):
        assert _hash_id("a", "b") != _hash_id("a", "c")

    def test_jaccard_identical(self):
        assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({1, 2}, {3, 4}) == 0.0

    def test_jaccard_empty(self):
        assert _jaccard(set(), set()) == 0.0

    def test_cosine_identical(self):
        assert abs(_cosine([1, 0, 1], [1, 0, 1]) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        assert abs(_cosine([1, 0], [0, 1])) < 1e-6

    def test_cosine_zero(self):
        assert _cosine([0, 0], [1, 1]) == 0.0


# ---------------------------------------------------------------------------
# ImmuneMemory
# ---------------------------------------------------------------------------

class TestImmuneMemory:
    def test_record_and_recall(self):
        mem = ImmuneMemory()
        p = Pathogen("p1", "byzantine_vote", 0.7, 1, ["a1"], {})
        mem.record_pathogen(p)
        assert len(mem.pathogen_history) == 1
        similar = mem.recall_similar("byzantine_vote", [])
        assert len(similar) == 1

    def test_recall_by_agents(self):
        mem = ImmuneMemory()
        p = Pathogen("p1", "collusion_ring", 0.5, 1, ["a1", "a2"], {})
        mem.record_pathogen(p)
        similar = mem.recall_similar("byzantine_vote", ["a1"])
        assert len(similar) == 1  # matched by agent

    def test_active_antibodies(self):
        mem = ImmuneMemory()
        ab = Antibody("ab1", "p1", "quarantine", ["a1"], 0.8, 0)
        mem.antibody_library["ab1"] = ab
        assert len(mem.get_active_antibodies()) == 1

    def test_active_antibodies_filters_weak(self):
        mem = ImmuneMemory()
        ab = Antibody("ab1", "p1", "quarantine", ["a1"], 0.03, 0)
        mem.antibody_library["ab1"] = ab
        assert len(mem.get_active_antibodies()) == 0

    def test_decay_removes_expired(self):
        mem = ImmuneMemory()
        ab = Antibody("ab1", "p1", "quarantine", ["a1"], 0.04, 0)
        mem.antibody_library["ab1"] = ab
        mem.decay_antibodies(rate=0.05)
        assert "ab1" not in mem.antibody_library

    def test_decay_reduces_strength(self):
        mem = ImmuneMemory()
        ab = Antibody("ab1", "p1", "quarantine", ["a1"], 0.8, 0)
        mem.antibody_library["ab1"] = ab
        mem.decay_antibodies(rate=0.1)
        assert abs(ab.strength - 0.7) < 1e-6

    def test_save_load(self):
        mem = ImmuneMemory()
        p = Pathogen("p1", "byzantine_vote", 0.7, 1, ["a1"], {"k": "v"})
        mem.record_pathogen(p)
        ab = Antibody("ab1", "p1", "quarantine", ["a1"], 0.8, 0)
        mem.antibody_library["ab1"] = ab
        mem.vaccination_log.append({"round": 1, "category": "test", "agents": ["a1"], "strength": 0.5})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        mem.save(path)

        mem2 = ImmuneMemory()
        mem2.load(path)
        assert len(mem2.pathogen_history) == 1
        assert mem2.pathogen_history[0].pathogen_id == "p1"
        assert "ab1" in mem2.antibody_library
        assert len(mem2.vaccination_log) == 1
        Path(path).unlink()


# ---------------------------------------------------------------------------
# ImmuneSystem — basic
# ---------------------------------------------------------------------------

class TestImmuneSystem:
    def test_init(self):
        sys = ImmuneSystem(["a1", "a2"], sensitivity=0.5)
        assert len(sys.agents) == 2
        assert sys.sensitivity == 0.5

    def test_sensitivity_clamped(self):
        sys = ImmuneSystem(["a1"], sensitivity=1.5)
        assert sys.sensitivity == 1.0
        sys2 = ImmuneSystem(["a1"], sensitivity=-0.5)
        assert sys2.sensitivity == 0.0

    def test_scan_round_no_threats(self):
        sys = ImmuneSystem(["a1", "a2", "a3"])
        rr = _make_round(agents=["a1", "a2", "a3"])
        pathogens = sys.scan_round(rr)
        # First round — not enough history for detection
        assert isinstance(pathogens, list)

    def test_run_cycle_returns_dict(self):
        sys = ImmuneSystem(["a1", "a2"])
        rr = _make_round(agents=["a1", "a2"])
        result = sys.run_cycle(rr)
        assert "round" in result
        assert "health" in result
        assert "new_pathogens" in result


# ---------------------------------------------------------------------------
# Detection algorithms
# ---------------------------------------------------------------------------

class TestDetection:
    def _run_rounds(self, sys, agents, adversarial, n=8):
        """Run multiple rounds to build history."""
        for i in range(n):
            rr = _make_round(round_index=i, agents=agents, adversarial=adversarial)
            sys.scan_round(rr)

    def test_byzantine_detection(self):
        agents = ["a1", "a2", "a3", "a4"]
        sys = ImmuneSystem(agents, sensitivity=0.5)
        # a1 proposes with high confidence but votes against
        for i in range(8):
            rr = _make_round(round_index=i, agents=agents)
            rr["proposals"]["a1"]["confidence"] = 0.9
            rr["votes"]["a1"]["approve"] = False
            sys.scan_round(rr)

        # After several rounds, should detect byzantine
        all_pathogens = sys.memory.pathogen_history
        byz = [p for p in all_pathogens if p.category == "byzantine_vote"]
        assert len(byz) > 0

    def test_collusion_detection(self):
        agents = ["a1", "a2", "a3", "a4", "a5"]
        sys = ImmuneSystem(agents, sensitivity=0.5)
        for i in range(10):
            rr = _make_round(round_index=i, agents=agents)
            # a1 and a2 always vote the same
            rr["votes"]["a1"]["approve"] = True
            rr["votes"]["a2"]["approve"] = True
            # Others vary
            rr["votes"]["a3"]["approve"] = i % 2 == 0
            rr["votes"]["a4"]["approve"] = i % 3 == 0
            rr["votes"]["a5"]["approve"] = i % 2 != 0
            sys.scan_round(rr)

        collude = [p for p in sys.memory.pathogen_history if p.category == "collusion_ring"]
        assert len(collude) > 0

    def test_flip_flop_detection(self):
        agents = ["a1", "a2", "a3"]
        sys = ImmuneSystem(agents, sensitivity=0.5)
        for i in range(10):
            rr = _make_round(round_index=i, agents=agents)
            # a1 flips every round
            rr["votes"]["a1"]["approve"] = i % 2 == 0
            sys.scan_round(rr)

        flips = [p for p in sys.memory.pathogen_history if p.category == "flip_flopping"]
        assert len(flips) > 0

    def test_free_riding_absence(self):
        agents = ["a1", "a2", "a3"]
        sys = ImmuneSystem(agents, sensitivity=0.5)
        for i in range(8):
            rr = _make_round(round_index=i, agents=agents)
            # a3 never votes
            del rr["votes"]["a3"]
            sys.scan_round(rr)

        free = [p for p in sys.memory.pathogen_history if p.category == "free_riding"]
        assert len(free) > 0

    def test_free_riding_low_effort(self):
        agents = ["a1", "a2", "a3"]
        sys = ImmuneSystem(agents, sensitivity=0.8)
        for i in range(8):
            rr = _make_round(round_index=i, agents=agents)
            rr["votes"]["a3"]["confidence"] = 0.01
            sys.scan_round(rr)

        free = [p for p in sys.memory.pathogen_history if p.category == "free_riding"]
        assert len(free) > 0

    def test_sybil_detection(self):
        agents = ["a1", "a2", "a3", "a4"]
        sys = ImmuneSystem(agents, sensitivity=0.5)
        for i in range(10):
            rr = _make_round(round_index=i, agents=agents)
            # a1 and a2 have identical voting and confidence
            val = i % 3 != 0
            rr["votes"]["a1"]["approve"] = val
            rr["votes"]["a2"]["approve"] = val
            rr["votes"]["a1"]["confidence"] = 0.85
            rr["votes"]["a2"]["confidence"] = 0.85
            # Others vary
            rr["votes"]["a3"]["approve"] = i % 2 == 0
            rr["votes"]["a3"]["confidence"] = 0.3 + (i * 0.05)
            sys.scan_round(rr)

        sybil = [p for p in sys.memory.pathogen_history if p.category == "sybil_cluster"]
        assert len(sybil) > 0

    def test_reputation_manipulation(self):
        agents = ["a1", "a2", "a3"]
        sys = ImmuneSystem(agents, sensitivity=0.5)
        for i in range(10):
            rr = _make_round(round_index=i, agents=agents)
            # a1 high confidence but always wrong
            rr["votes"]["a1"]["confidence"] = 0.95
            rr["votes"]["a1"]["approve"] = not rr["committed"]
            sys.scan_round(rr)

        manip = [p for p in sys.memory.pathogen_history if p.category == "reputation_manipulation"]
        assert len(manip) > 0


# ---------------------------------------------------------------------------
# Antibody generation
# ---------------------------------------------------------------------------

class TestAntibodyGeneration:
    def test_generates_antibodies(self):
        sys = ImmuneSystem(["a1", "a2"])
        p = Pathogen("p1", "byzantine_vote", 0.7, 1, ["a1"], {})
        abs_ = sys.generate_antibodies([p])
        assert len(abs_) == 1
        assert abs_[0].rule_type == "vote_discount"

    def test_rule_type_mapping(self):
        sys = ImmuneSystem(["a1"])
        mappings = {
            "byzantine_vote": "vote_discount",
            "collusion_ring": "coalition_break",
            "flip_flopping": "enhanced_scrutiny",
            "free_riding": "weight_reduction",
            "sybil_cluster": "quarantine",
            "reputation_manipulation": "weight_reduction",
        }
        for cat, expected_rule in mappings.items():
            p = Pathogen(f"p_{cat}", cat, 0.5, 1, ["a1"], {})
            abs_ = sys.generate_antibodies([p])
            assert abs_[0].rule_type == expected_rule

    def test_recidivism_boosts_strength(self):
        mem = ImmuneMemory()
        # Record prior infections
        for i in range(5):
            mem.record_pathogen(Pathogen(f"old_{i}", "byzantine_vote", 0.5, i, ["a1"], {}))
        sys = ImmuneSystem(["a1"], memory=mem)
        p = Pathogen("new1", "byzantine_vote", 0.5, 10, ["a1"], {})
        abs_ = sys.generate_antibodies([p])
        assert abs_[0].strength > 0.5 * 0.8  # boosted by recidivism


# ---------------------------------------------------------------------------
# Antibody application
# ---------------------------------------------------------------------------

class TestAntibodyApplication:
    def test_quarantine_zeros_weight(self):
        sys = ImmuneSystem(["a1", "a2"])
        ab = Antibody("ab1", "p1", "quarantine", ["a1"], 1.0, 0)
        sys.memory.antibody_library["ab1"] = ab
        weights = sys.apply_antibodies({"a1": 1.0, "a2": 1.0})
        assert weights["a1"] == 0.0
        assert weights["a2"] == 1.0

    def test_weight_reduction(self):
        sys = ImmuneSystem(["a1"])
        ab = Antibody("ab1", "p1", "weight_reduction", ["a1"], 0.8, 0)
        sys.memory.antibody_library["ab1"] = ab
        weights = sys.apply_antibodies({"a1": 1.0})
        assert weights["a1"] < 1.0
        assert weights["a1"] > 0.0

    def test_vote_discount(self):
        sys = ImmuneSystem(["a1"])
        ab = Antibody("ab1", "p1", "vote_discount", ["a1"], 0.8, 0)
        sys.memory.antibody_library["ab1"] = ab
        weights = sys.apply_antibodies({"a1": 1.0})
        expected = 1.0 * (1.0 - 0.8 * 0.4)
        assert abs(weights["a1"] - expected) < 1e-6

    def test_unknown_agent_unchanged(self):
        sys = ImmuneSystem(["a1"])
        ab = Antibody("ab1", "p1", "quarantine", ["a99"], 1.0, 0)
        sys.memory.antibody_library["ab1"] = ab
        weights = sys.apply_antibodies({"a1": 1.0})
        assert weights["a1"] == 1.0

    def test_increments_activations(self):
        sys = ImmuneSystem(["a1"])
        ab = Antibody("ab1", "p1", "quarantine", ["a1"], 1.0, 0)
        sys.memory.antibody_library["ab1"] = ab
        sys.apply_antibodies({"a1": 1.0})
        assert ab.activations == 1


# ---------------------------------------------------------------------------
# Vaccination
# ---------------------------------------------------------------------------

class TestVaccination:
    def test_vaccinate_with_history(self):
        mem = ImmuneMemory()
        mem.record_pathogen(Pathogen("p1", "byzantine_vote", 0.6, 1, ["a1"], {}))
        mem.record_pathogen(Pathogen("p2", "byzantine_vote", 0.8, 2, ["a1"], {}))
        sys = ImmuneSystem(["a1", "a2"], memory=mem)
        ab = sys.vaccinate("byzantine_vote")
        assert ab is not None
        assert ab.rule_type == "enhanced_scrutiny"
        assert "a1" in ab.affected_agents
        assert len(mem.vaccination_log) == 1

    def test_vaccinate_no_history(self):
        sys = ImmuneSystem(["a1"])
        ab = sys.vaccinate("byzantine_vote")
        assert ab is None

    def test_auto_vaccinate_in_cycle(self):
        mem = ImmuneMemory()
        for i in range(4):
            mem.record_pathogen(Pathogen(f"p{i}", "free_riding", 0.5, i, ["a2"], {}))
        sys = ImmuneSystem(["a1", "a2"], memory=mem)
        rr = _make_round(agents=["a1", "a2"])
        sys.run_cycle(rr)
        assert len(mem.vaccination_log) >= 1


# ---------------------------------------------------------------------------
# Health report
# ---------------------------------------------------------------------------

class TestHealthReport:
    def test_perfect_health(self):
        sys = ImmuneSystem(["a1", "a2"])
        report = sys.get_health_report()
        assert report["health_score"] == 100
        assert report["total_pathogens"] == 0

    def test_health_degrades_with_pathogens(self):
        mem = ImmuneMemory()
        for i in range(5):
            mem.record_pathogen(Pathogen(f"p{i}", "byzantine_vote", 0.7, i, ["a1"], {}))
        sys = ImmuneSystem(["a1"], memory=mem)
        report = sys.get_health_report()
        assert report["health_score"] < 100

    def test_agent_risk(self):
        mem = ImmuneMemory()
        mem.record_pathogen(Pathogen("p1", "byzantine_vote", 0.7, 1, ["a1"], {}))
        sys = ImmuneSystem(["a1", "a2"], memory=mem)
        report = sys.get_health_report()
        assert report["agent_risk"]["a1"] > report["agent_risk"]["a2"]


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_generates_html(self):
        sys = ImmuneSystem(["a1", "a2"])
        html = generate_immune_report(sys)
        assert "<!DOCTYPE html>" in html
        assert "Consensus Immune System" in html

    def test_html_with_data(self):
        sys, _ = _simulate(6, 10, 2, 0.6, seed=42)
        html = generate_immune_report(sys)
        assert "Pathogen Log" in html
        assert "Antibody Library" in html


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_simulate_runs(self):
        sys, results = _simulate(5, 8, 2, 0.6, seed=123)
        assert len(results) == 16  # 8 rounds * 2 scenarios
        assert sys._total_rounds == 16

    def test_simulate_detects_threats(self):
        sys, _ = _simulate(8, 15, 3, 0.6, seed=42)
        assert sys.memory.pathogen_history  # should find some pathogens

    def test_simulate_seed_reproducible(self):
        _, r1 = _simulate(4, 5, 1, 0.6, seed=99)
        _, r2 = _simulate(4, 5, 1, 0.6, seed=99)
        assert r1 == r2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_runs(self, capsys):
        main(["--agents", "4", "--rounds", "5", "--scenarios", "1", "--seed", "42"])
        captured = capsys.readouterr()
        assert "Consensus Immune System" in captured.out
        assert "Health Score" in captured.out

    def test_cli_json_output(self, tmp_path):
        out = tmp_path / "out.json"
        main(["--agents", "4", "--rounds", "5", "--scenarios", "1", "--seed", "42", "--json", str(out)])
        data = json.loads(out.read_text())
        assert "health" in data
        assert "cycles" in data

    def test_cli_html_output(self, tmp_path):
        out = tmp_path / "report.html"
        main(["--agents", "4", "--rounds", "5", "--scenarios", "1", "--seed", "42", "--output", str(out)])
        html = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
