"""Circuit Breaker for mBFT Consensus Rounds.

Implements the circuit-breaker pattern (Nygard 2007) adapted for distributed
consensus systems.  When a node experiences repeated failures during consensus
rounds (timeouts, deserialization errors, quorum loss), the breaker *opens*
to shed load and prevent cascading failures across the cluster.

States:
    CLOSED   — normal operation, requests pass through
    OPEN     — failures exceeded threshold, requests are rejected immediately
    HALF_OPEN — after a cooldown, a limited number of probe requests are allowed

Metrics tracked per-window:
    - Total attempts
    - Failure count and rate
    - Consecutive failures
    - Mean latency (to detect degradation before full failure)

Usage:
    from src.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    config = CircuitBreakerConfig(failure_threshold=5, window_seconds=60.0)
    breaker = CircuitBreaker(config)

    if breaker.allow_request():
        try:
            result = await run_consensus_round(...)
            breaker.record_success(latency_ms=elapsed)
        except ConsensusTimeout:
            breaker.record_failure()
    else:
        # Shed load — back off or route to healthy replica
        ...
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for :class:`CircuitBreaker`.

    Attributes:
        failure_threshold: Number of failures in the window to trip open.
        window_seconds: Sliding window duration for failure counting.
        cooldown_seconds: Time to wait in OPEN before transitioning to HALF_OPEN.
        half_open_max_probes: Max requests allowed in HALF_OPEN before deciding.
        latency_threshold_ms: If mean latency exceeds this, count as degraded.
        degradation_weight: Degraded (slow) responses add this fraction toward
            the failure count (0.0 = ignore latency, 1.0 = treat as full failure).
    """

    failure_threshold: int = 5
    window_seconds: float = 60.0
    cooldown_seconds: float = 30.0
    half_open_max_probes: int = 3
    latency_threshold_ms: float = 5000.0
    degradation_weight: float = 0.5


@dataclass
class _WindowMetrics:
    """Metrics for the current sliding window."""

    window_start: float = field(default_factory=time.monotonic)
    attempts: int = 0
    failures: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    latency_sum_ms: float = 0.0
    latency_count: int = 0
    degraded_score: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Failure rate in the current window (0.0 - 1.0)."""
        if self.attempts == 0:
            return 0.0
        return (self.failures + self.degraded_score) / self.attempts

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency of successful requests in ms."""
        if self.latency_count == 0:
            return 0.0
        return self.latency_sum_ms / self.latency_count

    def reset(self) -> None:
        """Reset metrics for a new window."""
        self.window_start = time.monotonic()
        self.attempts = 0
        self.failures = 0
        self.successes = 0
        self.consecutive_failures = 0
        self.latency_sum_ms = 0.0
        self.latency_count = 0
        self.degraded_score = 0.0


class CircuitBreaker:
    """Circuit breaker for mBFT consensus rounds.

    Thread-safety: NOT thread-safe. Use one breaker per async task / node.
    For multi-threaded usage, wrap calls with a lock externally.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = BreakerState.CLOSED
        self._metrics = _WindowMetrics()
        self._opened_at: float = 0.0
        self._half_open_probes: int = 0
        self._half_open_successes: int = 0
        self._trip_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        """Current breaker state (may transition on access)."""
        self._maybe_transition()
        return self._state

    @property
    def trip_count(self) -> int:
        """Total number of times the breaker has tripped open."""
        return self._trip_count

    @property
    def metrics(self) -> _WindowMetrics:
        """Current window metrics (read-only snapshot semantics)."""
        self._roll_window_if_needed()
        return self._metrics

    def allow_request(self) -> bool:
        """Check whether a request should be allowed through.

        Returns True if the request may proceed, False if shed.
        """
        self._maybe_transition()

        if self._state == BreakerState.CLOSED:
            return True

        if self._state == BreakerState.HALF_OPEN:
            if self._half_open_probes < self._config.half_open_max_probes:
                self._half_open_probes += 1
                return True
            return False

        # OPEN
        return False

    def record_success(self, latency_ms: float = 0.0) -> None:
        """Record a successful consensus operation."""
        self._roll_window_if_needed()
        self._metrics.attempts += 1
        self._metrics.successes += 1
        self._metrics.consecutive_failures = 0

        if latency_ms > 0:
            self._metrics.latency_sum_ms += latency_ms
            self._metrics.latency_count += 1

            # Check for latency degradation
            if latency_ms > self._config.latency_threshold_ms:
                self._metrics.degraded_score += self._config.degradation_weight

        if self._state == BreakerState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._config.half_open_max_probes:
                self._close()

    def record_failure(self) -> None:
        """Record a failed consensus operation."""
        self._roll_window_if_needed()
        self._metrics.attempts += 1
        self._metrics.failures += 1
        self._metrics.consecutive_failures += 1

        if self._state == BreakerState.HALF_OPEN:
            # Any failure in half-open immediately re-opens
            self._open()
            return

        # Check if we should trip
        effective_failures = self._metrics.failures + self._metrics.degraded_score
        if effective_failures >= self._config.failure_threshold:
            self._open()

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED state."""
        self._close()

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of breaker state."""
        self._maybe_transition()
        return {
            "state": self._state.value,
            "trip_count": self._trip_count,
            "metrics": {
                "attempts": self._metrics.attempts,
                "failures": self._metrics.failures,
                "successes": self._metrics.successes,
                "failure_rate": round(self._metrics.failure_rate, 4),
                "mean_latency_ms": round(self._metrics.mean_latency_ms, 1),
                "consecutive_failures": self._metrics.consecutive_failures,
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_transition(self) -> None:
        """Check time-based state transitions."""
        if self._state == BreakerState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.cooldown_seconds:
                self._state = BreakerState.HALF_OPEN
                self._half_open_probes = 0
                self._half_open_successes = 0

    def _roll_window_if_needed(self) -> None:
        """Roll the metrics window if it has expired."""
        elapsed = time.monotonic() - self._metrics.window_start
        if elapsed > self._config.window_seconds:
            self._metrics.reset()

    def _open(self) -> None:
        """Transition to OPEN."""
        self._state = BreakerState.OPEN
        self._opened_at = time.monotonic()
        self._trip_count += 1

    def _close(self) -> None:
        """Transition to CLOSED."""
        self._state = BreakerState.CLOSED
        self._metrics.reset()
        self._half_open_probes = 0
        self._half_open_successes = 0
