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

    Implementation note: single pass using the running-sums identity
    ``cov_xy * n = n*Σxy - Σx*Σy`` so the result is computed in O(n) time
    and O(1) auxiliary space instead of the previous four passes.
    """
    n = len(x)
    if len(y) < n:
        n = len(y)
    if n < 2:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    for i in range(n):
        xi = x[i]
        yi = y[i]
        sum_x += xi
        sum_y += yi
        sum_xx += xi * xi
        sum_yy += yi * yi
        sum_xy += xi * yi

    # Variance numerators (n * variance):
    var_x_num = n * sum_xx - sum_x * sum_x
    var_y_num = n * sum_yy - sum_y * sum_y
    # Floating-point can drive a true zero-variance series very slightly
    # negative; treat anything non-positive as degenerate.
    if var_x_num <= 0.0 or var_y_num <= 0.0:
        return 0.0

    cov_num = n * sum_xy - sum_x * sum_y
    return cov_num / math.sqrt(var_x_num * var_y_num)


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
    accumulated in one loop rather than three.
    """
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0

    dot = 0.0
    mag_a_sq = 0.0
    mag_b_sq = 0.0
    for i in range(n):
        ai = a[i]
        bi = b[i]
        dot += ai * bi
        mag_a_sq += ai * ai
        mag_b_sq += bi * bi

    # Compare squared magnitudes against the squared threshold to avoid two
    # extra sqrt calls when the input is degenerate.
    if mag_a_sq < 1e-18 or mag_b_sq < 1e-18:
        return 0.0
    return dot / math.sqrt(mag_a_sq * mag_b_sq)
