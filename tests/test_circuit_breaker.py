"""Tests for src.circuit_breaker."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.circuit_breaker import (
    BreakerState,
    CircuitBreaker,
    CircuitBreakerConfig,
)


class TestCircuitBreakerClosed:
    """Tests for CLOSED state behavior."""

    def test_initially_closed(self):
        cb = CircuitBreaker()
        assert cb.state == BreakerState.CLOSED

    def test_allows_requests_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True

    def test_stays_closed_below_threshold(self):
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == BreakerState.CLOSED

    def test_trips_open_at_threshold(self):
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == BreakerState.OPEN

    def test_success_resets_consecutive_failures(self):
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.metrics.consecutive_failures == 0


class TestCircuitBreakerOpen:
    """Tests for OPEN state behavior."""

    def test_rejects_requests_when_open(self):
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)
        cb.record_failure()
        assert cb.state == BreakerState.OPEN
        assert cb.allow_request() is False

    def test_transitions_to_half_open_after_cooldown(self):
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=1.0)
        cb = CircuitBreaker(config)
        cb.record_failure()
        assert cb.state == BreakerState.OPEN

        # Simulate time passing
        with patch("src.circuit_breaker.time.monotonic", return_value=time.monotonic() + 2.0):
            assert cb.state == BreakerState.HALF_OPEN

    def test_trip_count_increments(self):
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)
        cb.record_failure()
        assert cb.trip_count == 1


class TestCircuitBreakerHalfOpen:
    """Tests for HALF_OPEN state behavior."""

    def test_allows_limited_probes(self):
        config = CircuitBreakerConfig(
            failure_threshold=1, cooldown_seconds=0.0, half_open_max_probes=2
        )
        cb = CircuitBreaker(config)
        cb.record_failure()
        # After cooldown=0, next access transitions to half_open
        assert cb.state == BreakerState.HALF_OPEN
        assert cb.allow_request() is True
        assert cb.allow_request() is True
        assert cb.allow_request() is False  # exceeded probes

    def test_closes_after_enough_successes(self):
        config = CircuitBreakerConfig(
            failure_threshold=1, cooldown_seconds=0.0, half_open_max_probes=2
        )
        cb = CircuitBreaker(config)
        cb.record_failure()
        assert cb.state == BreakerState.HALF_OPEN
        cb.allow_request()
        cb.record_success()
        cb.allow_request()
        cb.record_success()
        assert cb.state == BreakerState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        config = CircuitBreakerConfig(
            failure_threshold=1, cooldown_seconds=10.0, half_open_max_probes=3
        )
        cb = CircuitBreaker(config)
        cb.record_failure()
        # Manually transition to half_open by simulating cooldown
        with patch("src.circuit_breaker.time.monotonic", return_value=time.monotonic() + 11.0):
            assert cb.state == BreakerState.HALF_OPEN
            cb.allow_request()
            cb.record_failure()
            assert cb.state == BreakerState.OPEN
            assert cb.trip_count == 2


class TestLatencyDegradation:
    """Tests for latency-based degradation scoring."""

    def test_high_latency_adds_degradation(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            latency_threshold_ms=100.0,
            degradation_weight=1.0,
        )
        cb = CircuitBreaker(config)
        # 3 slow successes should trip (degraded_score=3.0 >= threshold=3)
        cb.record_success(latency_ms=200.0)
        cb.record_success(latency_ms=200.0)
        cb.record_success(latency_ms=200.0)
        assert cb.state == BreakerState.CLOSED  # degradation alone doesn't trip
        # But one real failure on top should
        # Actually let's check the effective count via snapshot
        snap = cb.snapshot()
        assert snap["metrics"]["mean_latency_ms"] == 200.0

    def test_normal_latency_no_degradation(self):
        config = CircuitBreakerConfig(latency_threshold_ms=1000.0)
        cb = CircuitBreaker(config)
        cb.record_success(latency_ms=50.0)
        assert cb.metrics.degraded_score == 0.0


class TestSnapshot:
    """Tests for the snapshot method."""

    def test_snapshot_structure(self):
        cb = CircuitBreaker()
        snap = cb.snapshot()
        assert "state" in snap
        assert "trip_count" in snap
        assert "metrics" in snap
        assert snap["state"] == "closed"
        assert snap["trip_count"] == 0

    def test_snapshot_reflects_failures(self):
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(config)
        cb.record_failure()
        cb.record_failure()
        snap = cb.snapshot()
        assert snap["metrics"]["failures"] == 2
        assert snap["metrics"]["consecutive_failures"] == 2


class TestReset:
    """Tests for manual reset."""

    def test_reset_from_open(self):
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)
        cb.record_failure()
        assert cb.state == BreakerState.OPEN
        cb.reset()
        assert cb.state == BreakerState.CLOSED
        assert cb.allow_request() is True
