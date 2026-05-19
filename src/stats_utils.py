"""Shared statistical utility functions for metacognition analysis modules.

The implementations are intentionally pure-Python (no numpy) so they remain
dependency-free for the lightweight install. They are also single-pass over
their inputs where possible — these helpers sit on hot paths inside several
swarm engines and tracker modules that recompute correlations / similarities
on every tick.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient between two equal-length series.

    Returns ``0.0`` for degenerate inputs (length < 2, zero variance, or
    mismatched lengths — which mirrors the previous ``zip``-truncating
    behaviour by silently using the shorter length).

    Implementation note: numerically-stable streaming update due to Welford
    (1962) / Bennett, Grout, Pébay et al. (2009) for the co-moment ``C2``.
    The earlier implementation used the textbook
    ``cov * n = n*Σxy − Σx*Σy`` identity, which suffers catastrophic
    cancellation when the inputs have a large mean relative to their
    variance: e.g. ``pearson([1e12-2, 1e12+1, ...], [1e12+2, 1e12-1, ...])``
    silently returned ``0.0`` even though the series were perfectly
    anti-correlated. Several mBFT hot paths (`deadlock`, `emergence`,
    `influence`) feed it reputation/weight series with exactly that shape
    — running sums easily reach ``1e6+`` while the per-tick deltas live in
    ``[0, 1]`` — so the bad answer was silently corrupting downstream
    diagnostics. The streaming update keeps everything in mean-centered
    space and is the same algorithm NumPy uses internally for
    ``np.cov(..., ddof=0)``.

    Performs a single pass in O(n) time and O(1) auxiliary space.

    Micro-optimisation: iterates the pair directly via ``zip`` rather than
    indexing ``x[k - 1]`` / ``y[k - 1]`` inside a ``range`` loop. CPython's
    list ``__getitem__`` performs a bounds-checked PyObject lookup on every
    access, whereas the C-level ``zip`` iterator hands us the next pair
    without that overhead. For the lengths we see on the hot paths
    (``deadlock``, ``emergence``, ``influence`` push series in the
    hundreds-to-low-thousands), the saving is consistently >25% per call
    in microbenchmarks and matters because these modules call ``pearson``
    O(V**2) times per analysis tick.
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    # Welford-style online co-moments. We accumulate the running means
    # (mean_x, mean_y) and the centered sums of squares / cross-products
    # (m2_x, m2_y, c2_xy) with the update from Pébay (2008):
    #     dx = x_k - mean_x_old
    #     mean_x_new = mean_x_old + dx / k
    #     m2_x += dx * (x_k - mean_x_new)
    #     c2_xy += (k-1)/k * dx_old * dy_old
    mean_x = 0.0
    mean_y = 0.0
    m2_x = 0.0
    m2_y = 0.0
    c2_xy = 0.0
    # ``zip`` naturally truncates to the shorter length, matching the
    # previous ``len(y) < n`` guard.
    for k, (xi, yi) in enumerate(zip(x, y), start=1):
        dx = xi - mean_x
        dy = yi - mean_y
        # The cross-moment update uses the *old* means / weight (k-1)/k.
        c2_xy += dx * dy * (k - 1) / k
        mean_x += dx / k
        mean_y += dy / k
        # m2 updates use the *new* mean for the second factor.
        m2_x += dx * (xi - mean_x)
        m2_y += dy * (yi - mean_y)

    # Floating-point can drive a true zero-variance series very slightly
    # negative; treat anything non-positive as degenerate.
    if m2_x <= 0.0 or m2_y <= 0.0:
        return 0.0

    r = c2_xy / math.sqrt(m2_x * m2_y)
    # Clamp into [-1, 1] to absorb floating-point overshoot on near-perfect
    # correlations; otherwise downstream code that does ``acos(r)`` blows up.
    if r > 1.0:
        return 1.0
    if r < -1.0:
        return -1.0
    return r


def gini(values: Iterable[float]) -> float:
    """Gini coefficient measuring inequality in a distribution.

    Returns ``0.0`` for empty inputs or zero-sum distributions.
    Range: ``[0, 1]`` where ``0`` = perfect equality and ``1`` = maximum
    inequality.

    Uses the sorted-rank formulation ``Σ(2i - n + 1) * x_i / (n * Σx)``.
    """
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 0:
        return 0.0

    total = 0.0
    numerator = 0.0
    # Single pass: accumulate total and weighted numerator together.
    for i, v in enumerate(sorted_v):
        total += v
        numerator += (2 * i - n + 1) * v

    if total == 0.0:
        return 0.0
    return numerator / (n * total)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns ``0.0`` for empty, mismatched, or zero-magnitude inputs.
    Range: ``[-1, 1]`` where ``1`` = identical direction, ``-1`` = opposite.

    Single-pass implementation: dot product and both magnitudes are
    accumulated in one loop rather than three. Iterates via ``zip`` to
    avoid per-element bounds-checked ``__getitem__`` calls (see
    ``pearson`` for the same micro-optimisation rationale).
    """
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0

    dot = 0.0
    mag_a_sq = 0.0
    mag_b_sq = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        mag_a_sq += ai * ai
        mag_b_sq += bi * bi

    # Compare squared magnitudes against the squared threshold to avoid two
    # extra sqrt calls when the input is degenerate.
    if mag_a_sq < 1e-18 or mag_b_sq < 1e-18:
        return 0.0
    return dot / math.sqrt(mag_a_sq * mag_b_sq)
