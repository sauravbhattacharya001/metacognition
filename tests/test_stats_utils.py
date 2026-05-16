"""Tests for shared statistical utilities."""
from __future__ import annotations

import math

import pytest

from src.stats_utils import cosine_similarity, gini, pearson


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
