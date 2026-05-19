"""Tests for shared statistical utilities."""
from __future__ import annotations

import math

import pytest

from src.stats_utils import clamp, clamp01, cosine_similarity, gini, pearson


class TestPearson:
    def test_perfect_positive_correlation(self):
        assert pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        assert pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Symmetric pattern -> zero linear correlation
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 1, 3]
        result = pearson(x, y)
        assert -1.0 <= result <= 1.0

    def test_known_value(self):
        # Hand-computed: r for x=[1,2,3,4], y=[1,3,2,4]
        # mx=2.5, my=2.5, cov*n = 0.5*(-1.5) + (-0.5)*(0.5) ... let's just check bounds
        # mx=my=2.5; deviations x: [-1.5,-0.5,0.5,1.5], y: [-1.5,0.5,-0.5,1.5]
        # cov = (-1.5)(-1.5)+(-0.5)(0.5)+(0.5)(-0.5)+(1.5)(1.5) = 2.25-0.25-0.25+2.25 = 4.0
        # sx=sy=sqrt(5), so r = 4 / 5 = 0.8
        r = pearson([1, 2, 3, 4], [1, 3, 2, 4])
        assert r == pytest.approx(0.8, abs=1e-9)

    def test_single_element_returns_zero(self):
        assert pearson([5], [3]) == 0.0

    def test_empty_returns_zero(self):
        assert pearson([], []) == 0.0

    def test_zero_variance_x_returns_zero(self):
        assert pearson([2, 2, 2, 2], [1, 2, 3, 4]) == 0.0

    def test_zero_variance_y_returns_zero(self):
        assert pearson([1, 2, 3, 4], [7, 7, 7, 7]) == 0.0

    def test_both_zero_variance(self):
        assert pearson([5, 5, 5], [9, 9, 9]) == 0.0

    def test_result_within_bounds(self):
        # Property: |pearson| <= 1 for any input
        import random
        rng = random.Random(42)
        for _ in range(20):
            n = rng.randint(2, 20)
            x = [rng.uniform(-100, 100) for _ in range(n)]
            y = [rng.uniform(-100, 100) for _ in range(n)]
            r = pearson(x, y)
            assert -1.0 - 1e-9 <= r <= 1.0 + 1e-9

    def test_symmetry(self):
        x = [1.0, 4.0, 9.0, 16.0, 25.0]
        y = [2.0, 3.5, 5.1, 7.2, 9.4]
        assert pearson(x, y) == pytest.approx(pearson(y, x))

    def test_scale_invariance(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.5, 6.1, 8.3, 10.0]
        r1 = pearson(x, y)
        r2 = pearson([2 * v for v in x], [3 * v + 5 for v in y])
        assert r1 == pytest.approx(r2, abs=1e-9)

    # ------------------------------------------------------------------
    # Regression tests for the catastrophic-cancellation bug.
    #
    # Prior implementation used the naive ``n*Σxy - Σx*Σy`` identity which
    # silently returned 0.0 (or the wrong sign) for series whose mean was
    # large relative to their variance — exactly the shape produced by
    # mBFT reputation / weight accumulators on long runs. The streaming
    # Welford update should give the correct answer to ~5 decimal places
    # even at 1e12 scale.
    # ------------------------------------------------------------------
    def test_large_mean_perfect_negative_correlation(self):
        # Reflective pattern around a 1e12 mean. Old impl returned ~0.0;
        # numerically-stable impl should return ~ -1.0.
        xs = [1e12 + 1, 1e12 - 1, 1e12 + 2, 1e12 - 2, 1e12]
        ys = [1e12 - 1, 1e12 + 1, 1e12 - 2, 1e12 + 2, 1e12]
        r = pearson(xs, ys)
        assert r == pytest.approx(-1.0, abs=1e-3)

    def test_large_mean_perfect_positive_correlation(self):
        # Strictly linear series at 1e9 offset.
        xs = [1e9 + i for i in range(50)]
        ys = [2e9 + 3 * i for i in range(50)]
        r = pearson(xs, ys)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_large_offset_invariance_against_zero_mean_baseline(self):
        # Adding a constant to both series must not change the correlation
        # by more than floating-point can justify. This is the property the
        # old naive ``n*Σxy − Σx*Σy`` formula violated catastrophically:
        # at an offset of ~1e9 it flipped the sign / collapsed to 0.
        # Tolerance grows with offset because float64 still loses precision
        # near 1e12, just gracefully instead of catastrophically.
        base_x = [0.1, -0.4, 0.7, -0.2, 0.5, -0.9, 0.3]
        base_y = [-0.2, 0.3, 0.1, 0.8, -0.5, 0.4, -0.7]
        r_centered = pearson(base_x, base_y)
        for offset, tol in ((1e6, 1e-8), (1e9, 1e-6), (1e12, 1e-3)):
            shifted_x = [v + offset for v in base_x]
            shifted_y = [v + offset for v in base_y]
            r_shifted = pearson(shifted_x, shifted_y)
            assert r_shifted == pytest.approx(r_centered, abs=tol), (
                f"offset {offset} changed pearson from {r_centered} to {r_shifted}"
            )

    def test_result_clamped_to_unit_interval(self):
        # Float overshoot on perfectly-correlated series must not leak past
        # ±1.0; downstream callers may feed the value into ``acos``.
        xs = [1e15 + i for i in range(10)]
        ys = [1e15 + 2 * i for i in range(10)]
        r = pearson(xs, ys)
        assert -1.0 <= r <= 1.0


class TestGini:
    def test_perfect_equality(self):
        assert gini([5, 5, 5, 5]) == pytest.approx(0.0)

    def test_maximum_inequality(self):
        # One person has everything
        values = [0, 0, 0, 0, 100]
        n = len(values)
        # Theoretical max Gini for n samples = (n-1)/n
        result = gini(values)
        assert result == pytest.approx((n - 1) / n, abs=1e-9)

    def test_empty_returns_zero(self):
        assert gini([]) == 0.0

    def test_all_zeros_returns_zero(self):
        assert gini([0, 0, 0]) == 0.0

    def test_single_nonzero_value(self):
        # Single sample: numerator = (2*0 - 1 + 1) * v = 0
        assert gini([42.0]) == pytest.approx(0.0)

    def test_known_two_values(self):
        # values=[1, 3], sorted, n=2, total=4
        # numerator = (2*0 - 2 + 1)*1 + (2*1 - 2 + 1)*3 = -1 + 3 = 2
        # gini = 2 / (2*4) = 0.25
        assert gini([1, 3]) == pytest.approx(0.25)

    def test_order_independent(self):
        a = [1, 2, 3, 4, 5]
        b = [5, 4, 3, 2, 1]
        c = [3, 1, 4, 5, 2]
        g = gini(a)
        assert gini(b) == pytest.approx(g)
        assert gini(c) == pytest.approx(g)

    def test_range_bounded(self):
        import random
        rng = random.Random(123)
        for _ in range(20):
            n = rng.randint(2, 30)
            values = [rng.uniform(0, 1000) for _ in range(n)]
            g = gini(values)
            assert 0.0 <= g <= 1.0

    def test_scale_invariance(self):
        v = [1.0, 2.0, 4.0, 8.0]
        g1 = gini(v)
        g2 = gini([100 * x for x in v])
        assert g1 == pytest.approx(g2, abs=1e-9)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 2, 3], [-1, -2, -3]) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_known_value(self):
        # [1,1] vs [1,0]: dot=1, mag=sqrt(2)*1 -> 1/sqrt(2)
        result = cosine_similarity([1, 1], [1, 0])
        assert result == pytest.approx(1 / math.sqrt(2))

    def test_scale_invariance(self):
        a = [1, 2, 3]
        b = [2, 4, 6]
        # Same direction, different magnitude
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_mismatched_lengths_returns_zero(self):
        assert cosine_similarity([1, 2, 3], [1, 2]) == 0.0

    def test_empty_returns_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
        assert cosine_similarity([1, 2, 3], [0, 0, 0]) == 0.0
        assert cosine_similarity([0, 0], [0, 0]) == 0.0

    def test_near_zero_magnitude_returns_zero(self):
        # Below 1e-9 threshold
        tiny = [1e-12, 1e-12, 1e-12]
        assert cosine_similarity(tiny, [1, 2, 3]) == 0.0

    def test_symmetry(self):
        a = [3.0, 1.0, 4.0, 1.0, 5.0]
        b = [2.0, 7.0, 1.0, 8.0, 2.0]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_range_bounded(self):
        import random
        rng = random.Random(7)
        for _ in range(20):
            n = rng.randint(1, 10)
            a = [rng.uniform(-10, 10) for _ in range(n)]
            b = [rng.uniform(-10, 10) for _ in range(n)]
            sim = cosine_similarity(a, b)
            assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9


class TestClamp:
    def test_within_bounds_returns_value(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_lo_returns_lo(self):
        assert clamp(-3.0, 0.0, 1.0) == 0.0

    def test_above_hi_returns_hi(self):
        assert clamp(7.0, 0.0, 1.0) == 1.0

    def test_inclusive_bounds(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0

    def test_arbitrary_range(self):
        assert clamp(150.0, -10.0, 100.0) == 100.0
        assert clamp(-50.0, -10.0, 100.0) == -10.0
        assert clamp(42.0, -10.0, 100.0) == 42.0

    def test_degenerate_range_collapses(self):
        # lo == hi forces the result regardless of value.
        assert clamp(99.0, 5.0, 5.0) == 5.0
        assert clamp(-99.0, 5.0, 5.0) == 5.0


class TestClamp01:
    def test_within_unit_interval(self):
        for v in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert clamp01(v) == v

    def test_below_zero_saturates(self):
        assert clamp01(-0.0001) == 0.0
        assert clamp01(-1e9) == 0.0

    def test_above_one_saturates(self):
        assert clamp01(1.0001) == 1.0
        assert clamp01(1e9) == 1.0

    def test_matches_inlined_form(self):
        # The whole point of the helper is to be behaviour-identical to
        # the ``max(0.0, min(1.0, x))`` it replaces across ~25 call sites.
        import random

        rng = random.Random(20260519)
        for _ in range(500):
            x = rng.uniform(-5.0, 5.0)
            assert clamp01(x) == max(0.0, min(1.0, x))
