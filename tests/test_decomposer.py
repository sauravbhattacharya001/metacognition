"""Tests for src.decomposer — Swarm Task Decomposer."""
from __future__ import annotations

import json
import unittest

from src.decomposer import (
    DecompositionResult,
    DecompositionStrategy,
    Subtask,
    SwarmTaskDecomposer,
)


class TestSubtaskModel(unittest.TestCase):
    def test_defaults(self):
        s = Subtask(id="t0", description="do stuff")
        self.assertEqual(s.category, "general")
        self.assertEqual(s.dependencies, [])
        self.assertEqual(s.estimated_rounds, 1)

    def test_complexity_bounds(self):
        s = Subtask(id="t0", description="x", complexity=0.0)
        self.assertEqual(s.complexity, 0.0)
        s2 = Subtask(id="t1", description="x", complexity=1.0)
        self.assertEqual(s2.complexity, 1.0)


class TestDecompose(unittest.TestCase):
    def setUp(self):
        self.d = SwarmTaskDecomposer()

    def test_empty_task(self):
        r = self.d.decompose("")
        self.assertEqual(r.subtasks, [])
        self.assertEqual(r.depth, 0)

    def test_single_simple_task(self):
        r = self.d.decompose("Find the answer to 2+2")
        self.assertGreaterEqual(len(r.subtasks), 1)
        self.assertEqual(r.depth, 1)

    def test_enumerated_steps(self):
        task = "1. Gather data 2. Analyze data 3. Write report"
        r = self.d.decompose(task)
        self.assertGreaterEqual(len(r.subtasks), 3)

    def test_conjunction_split(self):
        task = "Search for papers and then summarize the findings and also write a conclusion"
        r = self.d.decompose(task)
        self.assertGreaterEqual(len(r.subtasks), 2)

    def test_sentence_split(self):
        task = "Check the logs for errors. Review the deployment status. Report findings."
        r = self.d.decompose(task)
        self.assertGreaterEqual(len(r.subtasks), 2)

    def test_dependency_detection(self):
        task = "1. Gather data 2. Using the result of gathering, analyze trends"
        r = self.d.decompose(task)
        has_dep = any(s.dependencies for s in r.subtasks)
        self.assertTrue(has_dep)

    def test_category_classification_retrieval(self):
        r = self.d.decompose("Find and search for relevant papers")
        cats = {s.category for s in r.subtasks}
        self.assertIn("retrieval", cats)

    def test_category_classification_reasoning(self):
        r = self.d.decompose("Compare and analyze the two approaches")
        cats = {s.category for s in r.subtasks}
        self.assertIn("reasoning", cats)

    def test_category_classification_verification(self):
        r = self.d.decompose("Verify the results and validate the model")
        cats = {s.category for s in r.subtasks}
        self.assertIn("verification", cats)

    def test_category_classification_synthesis(self):
        r = self.d.decompose("Write a comprehensive summary and create a report")
        cats = {s.category for s in r.subtasks}
        self.assertIn("synthesis", cats)

    def test_max_subtasks_enforced(self):
        strat = DecompositionStrategy(max_subtasks=3)
        d = SwarmTaskDecomposer(strategy=strat)
        task = " ".join(f"{i}. Step {i}" for i in range(1, 15))
        r = d.decompose(task)
        self.assertLessEqual(len(r.subtasks), 3)

    def test_min_complexity_filter(self):
        strat = DecompositionStrategy(min_complexity_threshold=0.5)
        d = SwarmTaskDecomposer(strategy=strat)
        r = d.decompose("1. Do a simple trivial thing 2. Perform complex thorough analysis")
        # At least one subtask should survive
        self.assertGreaterEqual(len(r.subtasks), 1)

    def test_prefer_parallel_false(self):
        strat = DecompositionStrategy(prefer_parallel=False)
        d = SwarmTaskDecomposer(strategy=strat)
        task = "Find papers. Analyze results. Write report."
        r = d.decompose(task)
        # Should create sequential deps
        if len(r.subtasks) > 1:
            has_dep = any(s.dependencies for s in r.subtasks)
            self.assertTrue(has_dep)


class TestValidateDAG(unittest.TestCase):
    def setUp(self):
        self.d = SwarmTaskDecomposer()

    def test_valid_dag(self):
        subtasks = [
            Subtask(id="a", description="x"),
            Subtask(id="b", description="y", dependencies=["a"]),
        ]
        valid, problems = self.d.validate_dag(subtasks)
        self.assertTrue(valid)
        self.assertEqual(problems, [])

    def test_cycle_detected(self):
        subtasks = [
            Subtask(id="a", description="x", dependencies=["b"]),
            Subtask(id="b", description="y", dependencies=["a"]),
        ]
        valid, problems = self.d.validate_dag(subtasks)
        self.assertFalse(valid)
        self.assertTrue(any("cycle" in p for p in problems))

    def test_missing_dependency(self):
        subtasks = [
            Subtask(id="a", description="x", dependencies=["nonexistent"]),
        ]
        valid, problems = self.d.validate_dag(subtasks)
        self.assertFalse(valid)
        self.assertTrue(any("missing" in p for p in problems))

    def test_all_deps_no_root(self):
        subtasks = [
            Subtask(id="a", description="x", dependencies=["b"]),
            Subtask(id="b", description="y", dependencies=["a"]),
        ]
        valid, problems = self.d.validate_dag(subtasks)
        self.assertFalse(valid)


class TestCriticalPath(unittest.TestCase):
    def setUp(self):
        self.d = SwarmTaskDecomposer()

    def test_single_node(self):
        subtasks = [Subtask(id="a", description="x", complexity=0.5)]
        path, cost = self.d.critical_path(subtasks)
        self.assertEqual(path, ["a"])
        self.assertAlmostEqual(cost, 0.5)

    def test_chain(self):
        subtasks = [
            Subtask(id="a", description="x", complexity=0.3),
            Subtask(id="b", description="y", complexity=0.4, dependencies=["a"]),
            Subtask(id="c", description="z", complexity=0.2, dependencies=["b"]),
        ]
        path, cost = self.d.critical_path(subtasks)
        self.assertEqual(path, ["a", "b", "c"])
        self.assertAlmostEqual(cost, 0.9)

    def test_parallel_picks_longest(self):
        subtasks = [
            Subtask(id="a", description="x", complexity=0.1),
            Subtask(id="b", description="y", complexity=0.8),
            Subtask(id="c", description="z", complexity=0.2, dependencies=["a", "b"]),
        ]
        path, cost = self.d.critical_path(subtasks)
        self.assertIn("b", path)
        self.assertIn("c", path)
        self.assertAlmostEqual(cost, 1.0)

    def test_empty(self):
        path, cost = self.d.critical_path([])
        self.assertEqual(path, [])
        self.assertEqual(cost, 0.0)


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.d = SwarmTaskDecomposer()

    def test_all_parallel(self):
        subtasks = [
            Subtask(id="a", description="x"),
            Subtask(id="b", description="y"),
            Subtask(id="c", description="z"),
        ]
        waves = self.d.schedule(subtasks)
        self.assertEqual(len(waves), 1)
        self.assertEqual(len(waves[0]), 3)

    def test_fully_serial(self):
        subtasks = [
            Subtask(id="a", description="x"),
            Subtask(id="b", description="y", dependencies=["a"]),
            Subtask(id="c", description="z", dependencies=["b"]),
        ]
        waves = self.d.schedule(subtasks)
        self.assertEqual(len(waves), 3)
        for w in waves:
            self.assertEqual(len(w), 1)

    def test_diamond(self):
        subtasks = [
            Subtask(id="a", description="start"),
            Subtask(id="b", description="left", dependencies=["a"]),
            Subtask(id="c", description="right", dependencies=["a"]),
            Subtask(id="d", description="end", dependencies=["b", "c"]),
        ]
        waves = self.d.schedule(subtasks)
        self.assertEqual(len(waves), 3)
        self.assertIn("b", waves[1])
        self.assertIn("c", waves[1])

    def test_empty(self):
        self.assertEqual(self.d.schedule([]), [])


class TestAgentAssignment(unittest.TestCase):
    def setUp(self):
        self.d = SwarmTaskDecomposer()

    def test_round_robin(self):
        result = self.d.decompose("1. Task A 2. Task B 3. Task C 4. Task D")
        result = self.d.assign_agents(result, ["a1", "a2"])
        all_assigned = []
        for tasks in result.agent_assignments.values():
            all_assigned.extend(tasks)
        self.assertEqual(len(all_assigned), len(result.subtasks))

    def test_strength_matching(self):
        subtasks = [
            Subtask(id="t0", description="find stuff", category="retrieval", complexity=0.5),
            Subtask(id="t1", description="analyze it", category="reasoning", complexity=0.5),
        ]
        result = DecompositionResult(
            original_task="test",
            subtasks=subtasks,
            schedule=[["t0", "t1"]],
        )
        strengths = {
            "searcher": {"retrieval": 0.9, "reasoning": 0.1},
            "thinker": {"retrieval": 0.1, "reasoning": 0.9},
        }
        result = self.d.assign_agents(result, ["searcher", "thinker"], strengths)
        self.assertIn("t0", result.agent_assignments["searcher"])
        self.assertIn("t1", result.agent_assignments["thinker"])

    def test_no_agents(self):
        result = self.d.decompose("Find stuff")
        result = self.d.assign_agents(result, [])
        self.assertEqual(result.agent_assignments, {})


class TestMergeResults(unittest.TestCase):
    def test_merge(self):
        d = SwarmTaskDecomposer()
        merged = d.merge_results({"t0": "Hello", "t1": "World"})
        self.assertIn("Hello", merged)
        self.assertIn("World", merged)

    def test_empty_merge(self):
        d = SwarmTaskDecomposer()
        self.assertEqual(d.merge_results({}), "")


class TestExport(unittest.TestCase):
    def setUp(self):
        self.d = SwarmTaskDecomposer()
        self.result = self.d.decompose("1. Find data 2. Analyze it 3. Report")
        self.result = self.d.assign_agents(self.result, ["a1", "a2"])

    def test_json_export(self):
        j = self.d.export_json(self.result)
        data = json.loads(j)
        self.assertIn("original_task", data)
        self.assertIn("subtasks", data)
        self.assertIn("schedule", data)

    def test_html_export(self):
        html = self.d.export_html(self.result)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Swarm Task Decomposer", html)
        self.assertIn("Wave", html)

    def test_html_no_assignments(self):
        result = self.d.decompose("Just one thing")
        html = self.d.export_html(result)
        self.assertIn("No agents assigned", html)


class TestStrategyConfig(unittest.TestCase):
    def test_category_weights(self):
        strat = DecompositionStrategy(
            category_weights={"reasoning": 2.0, "retrieval": 0.5}
        )
        d = SwarmTaskDecomposer(strategy=strat)
        r = d.decompose("1. Find papers 2. Analyze them")
        # reasoning subtask should have higher complexity
        task_map = {s.category: s.complexity for s in r.subtasks}
        if "reasoning" in task_map and "retrieval" in task_map:
            self.assertGreater(task_map["reasoning"], task_map["retrieval"])


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.d = SwarmTaskDecomposer()

    def test_very_long_task(self):
        task = " and then ".join(f"step {i}" for i in range(50))
        r = self.d.decompose(task)
        self.assertLessEqual(len(r.subtasks), 20)

    def test_whitespace_only(self):
        r = self.d.decompose("   \n\t  ")
        self.assertEqual(r.subtasks, [])

    def test_special_characters(self):
        r = self.d.decompose("Analyze <html> & 'quotes' in \"data\"")
        self.assertGreaterEqual(len(r.subtasks), 1)
        html = self.d.export_html(r)
        self.assertNotIn("<html>", html.split("<style>")[1] if "<style>" in html else "")


if __name__ == "__main__":
    unittest.main()
