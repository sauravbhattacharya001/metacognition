"""Shared statistical utility functions for metacognition analysis modules."""
from __future__ import annotations

import math
from typing import List


def pearson(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient between two equal-length series.

    Returns 0.0 for degenerate inputs (length < 2, zero variance).
    """
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (sx * sy)


def gini(values: List[float]) -> float:
    """Gini coefficient measuring inequality in a distribution.

    Returns 0.0 for empty inputs or zero-sum distributions.
    Range: [0, 1] where 0 = perfect equality, 1 = maximum inequality.
    """
    if not values or sum(values) == 0:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    numerator = sum((2 * i - n + 1) * v for i, v in enumerate(sorted_v))
    return numerator / (n * total)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns 0.0 for empty, mismatched, or zero-magnitude inputs.
    Range: [-1, 1] where 1 = identical direction, -1 = opposite.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return dot / (mag_a * mag_b)
