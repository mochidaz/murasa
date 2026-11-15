"""
Reference:
  BNPB (2023). Risiko Bencana Indonesia (RBI): Memahami Risiko Sistemik di Indonesia.
"""

import sys
import unittest
import numpy as np

sys.path.append(".")

from murasa.core.base import ParameterPlugin
from murasa.plugins.calculators import WeightedSumCalculator, BNPBCalculator


class MockParameter(ParameterPlugin):
    def __init__(self, name: str, weight: float = 1.0, group: str = "hazard"):
        super().__init__(name, weight, group)

    def validate_requirements(self, registry):
        return True

    def process(self, *args):
        return None


def make_param(name: str, weight: float = 1.0, group: str = "hazard") -> MockParameter:
    return MockParameter(name, weight, group)


def uniform(shape: tuple[int, int], value: float) -> np.ndarray:
    return np.full(shape, float(value), dtype=np.float32)


def expected_rbi_risk(h: float | np.ndarray, v: float | np.ndarray, c: float | np.ndarray) -> float | np.ndarray:
    """Reference mathematical formula from RBI BNPB 2023 (p. 55): R = (H * V * (1 - C))^(1/3)."""
    return np.cbrt(np.asarray(h, dtype=np.float64) * np.asarray(v, dtype=np.float64) * (1.0 - np.asarray(c, dtype=np.float64)))


def assert_allclose(test: unittest.TestCase, result: np.ndarray, expected: float | np.ndarray, rtol: float = 1e-5, atol: float = 1e-6, msg: str = ""):
    test.assertTrue(
        np.allclose(result, expected, rtol=rtol, atol=atol),
        msg=f"{msg} — expected ~{np.asarray(expected).flat[0]:.6f}, got {result.flat[0]:.6f}",
    )


class TestWeightedSumCalculator(unittest.TestCase):
    """
    Tests for WeightedSumCalculator (Weighted Linear Combination).
    Used both as a standalone calculator and as the intra-group aggregation
    method in BNPBCalculator.
    """

    S = (4, 4)

    def calc(self, params, results, shape=None):
        return WeightedSumCalculator().calculate(params, results, shape or self.S)

    def test_single_parameter_normalization(self):
        p = make_param("h1", weight=3.0)
        result = self.calc([p], {"h1": uniform(self.S, 0.7)})
        assert_allclose(self, result, 0.7, msg="single parameter with arbitrary positive weight")

    def test_two_parameters_equal_weights(self):
        p1 = make_param("h1", weight=1.0)
        p2 = make_param("h2", weight=1.0)
        results = {"h1": uniform(self.S, 1.0), "h2": uniform(self.S, 0.0)}
        result = self.calc([p1, p2], results)
        assert_allclose(self, result, 0.5, msg="equal weight average")

    def test_two_parameters_unequal_weights(self):
        p_heavy = make_param("h_heavy", weight=3.0)
        p_light = make_param("h_light", weight=1.0)
        results = {"h_heavy": uniform(self.S, 1.0), "h_light": uniform(self.S, 0.0)}
        result = self.calc([p_heavy, p_light], results)
        assert_allclose(self, result, 0.75, msg="3:1 weighted average")

    def test_unnormalized_weights_are_automatically_normalized(self):
        """Weights (40, 30, 30) must produce the exact same result as (0.4, 0.3, 0.3)."""
        p_pct = [make_param("s", 40.0), make_param("f", 30.0), make_param("e", 30.0)]
        p_dec = [make_param("s", 0.40), make_param("f", 0.30), make_param("e", 0.30)]
        results = {
            "s": uniform(self.S, 0.8),
            "f": uniform(self.S, 0.6),
            "e": uniform(self.S, 0.4),
        }
        r_pct = self.calc(p_pct, results)
        r_dec = self.calc(p_dec, results)
        assert_allclose(self, r_pct, 0.62, msg="percentage weights")
        assert_allclose(self, r_dec, 0.62, msg="decimal weights")
        self.assertTrue(np.allclose(r_pct, r_dec), "percentage and decimal weights must match")

    def test_all_parameters_zero(self):
        params = [make_param(f"p{i}", weight=1.0) for i in range(3)]
        results = {f"p{i}": uniform(self.S, 0.0) for i in range(3)}
        result = self.calc(params, results)
        assert_allclose(self, result, 0.0, msg="all zero inputs")

    def test_all_parameters_one(self):
        params = [make_param(f"p{i}", weight=float(i + 1)) for i in range(4)]
        results = {f"p{i}": uniform(self.S, 1.0) for i in range(4)}
        result = self.calc(params, results)
        assert_allclose(self, result, 1.0, msg="all max inputs")

    def test_spatial_heterogeneity_preserved(self):
        shape = (2, 2)
        p1 = make_param("a", weight=1.0)
        p2 = make_param("b", weight=1.0)
        arr_a = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
        arr_b = np.array([[1.0, 0.5], [0.0, 0.75]], dtype=np.float32)
        result = self.calc([p1, p2], {"a": arr_a, "b": arr_b}, shape=shape)
        expected = np.full(shape, 0.5, dtype=np.float32)
        self.assertTrue(np.allclose(result, expected), "cell-wise spatial average")

    def test_partial_nan_pixels_renormalize_weights(self):
        """If one parameter is NaN at a pixel, the remaining valid parameters take full weight."""
        shape = (1, 2)
        p1 = make_param("a", weight=0.6)
        p2 = make_param("b", weight=0.4)
        arr_a = np.array([[np.nan, 0.5]], dtype=np.float32)
        arr_b = np.array([[0.8, 0.5]], dtype=np.float32)
        result = self.calc([p1, p2], {"a": arr_a, "b": arr_b}, shape=shape)
        expected = np.array([[0.8, 0.5]], dtype=np.float32)
        self.assertTrue(np.allclose(result, expected), "NaN pixel should fall back to valid parameter")

    def test_zero_total_weight_raises_value_error(self):
        p1 = make_param("a", weight=0.0)
        with self.assertRaises(ValueError):
            self.calc([p1], {"a": uniform(self.S, 0.5)})

    def test_no_matching_results_raises_value_error(self):
        p1 = make_param("a", weight=1.0)
        with self.assertRaises(ValueError):
            self.calc([p1], {})


class TestBNPBCalculatorFormula(unittest.TestCase):
    """
    Tests for the core mathematical risk equation in RBI BNPB 2023 (p. 55):
        R = (H * V * (1 - C)) ** (1 / 3)
    """

    S = (4, 4)

    def calc(self, params, results, shape=None):
        return BNPBCalculator().calculate(params, results, shape or self.S)

    def test_basic_formula_calculation(self):
        """H=0.8, V=0.6, C=0.4 -> R = (0.8 * 0.6 * (1 - 0.4))^(1/3) = 0.288^(1/3)."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        results = {
            "h": uniform(self.S, 0.8),
            "v": uniform(self.S, 0.6),
            "c": uniform(self.S, 0.4),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, expected_rbi_risk(0.8, 0.6, 0.4), msg="basic RBI formula")

    def test_balanced_midpoint_inputs(self):
        """H=0.5, V=0.5, C=0.5 -> R = (0.5 * 0.5 * 0.5)^(1/3) = 0.5."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        results = {
            "h": uniform(self.S, 0.5),
            "v": uniform(self.S, 0.5),
            "c": uniform(self.S, 0.5),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, 0.5, msg="balanced midpoint risk")

    def test_zero_hazard_yields_zero_risk(self):
        """If H = 0, risk must be 0 regardless of vulnerability and capacity."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        results = {
            "h": uniform(self.S, 0.0),
            "v": uniform(self.S, 1.0),
            "c": uniform(self.S, 0.0),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, 0.0, msg="zero hazard")

    def test_zero_vulnerability_yields_zero_risk(self):
        """If V = 0, risk must be 0 even under maximum hazard (RBI 2023, pp. 34, 55)."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        results = {
            "h": uniform(self.S, 1.0),
            "v": uniform(self.S, 0.0),
            "c": uniform(self.S, 0.0),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, 0.0, msg="zero vulnerability")

    def test_zero_capacity_yields_unmitigated_risk(self):
        """If C = 0, (1 - C) = 1.0, so R = (H * V)^(1/3); for H=1, V=1 -> R = 1.0."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        results = {
            "h": uniform(self.S, 1.0),
            "v": uniform(self.S, 1.0),
            "c": uniform(self.S, 0.0),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, 1.0, msg="zero capacity with max H and V")

    def test_full_capacity_yields_zero_risk(self):
        """If C = 1.0, (1 - C) = 0.0, so R = (H * V * 0)^(1/3) = 0.0."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        results = {
            "h": uniform(self.S, 1.0),
            "v": uniform(self.S, 1.0),
            "c": uniform(self.S, 1.0),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, 0.0, msg="full capacity")

    def test_missing_vulnerability_and_capacity_use_neutral_defaults(self):
        """
        When vulnerability or capacity groups have no parameters, neutral factors
        in H * V * (1 - C) are V = 1.0 and C = 0.0 (so 1 - C = 1.0), yielding R = H^(1/3).
        """
        params = [make_param("h", weight=1.0, group="hazard")]
        for h_val in [0.25, 0.5, 0.75]:
            with self.subTest(h=h_val):
                result = self.calc(params, {"h": uniform(self.S, h_val)})
                assert_allclose(self, result, np.cbrt(h_val), msg=f"hazard-only H={h_val}")


class TestBNPBCalculatorGroupWeighting(unittest.TestCase):
    """
    Tests for intra-group parameter aggregation and official RBI BNPB 2023
    vulnerability weightings for Flood and Landslide (pp. 50-54).
    """

    S = (3, 3)

    def calc(self, params, results):
        return BNPBCalculator().calculate(params, results, self.S)

    def test_hazard_group_weighted_aggregation(self):
        """Two hazard parameters with 3:1 weights aggregate to H = 0.75 before cube-root."""
        p1 = make_param("h_heavy", weight=3.0, group="hazard")
        p2 = make_param("h_light", weight=1.0, group="hazard")
        p_v = make_param("v", weight=1.0, group="vulnerability")
        p_c = make_param("c", weight=1.0, group="capacity")
        results = {
            "h_heavy": uniform(self.S, 1.0),
            "h_light": uniform(self.S, 0.0),
            "v": uniform(self.S, 1.0),
            "c": uniform(self.S, 0.0),
        }
        result = self.calc([p1, p2, p_v, p_c], results)
        assert_allclose(self, result, expected_rbi_risk(0.75, 1.0, 0.0), msg="hazard 3:1 aggregation")

    def test_flood_vulnerability_weights(self):
        """
        RBI 2023 (p. 54) vulnerability weights for Flood (IKB):
            IKB = 0.40 * IKS + 0.25 * IKF + 0.25 * IKE + 0.10 * IKL
        With IKS=0.8, IKF=0.6, IKE=0.4, IKL=0.2 -> IKB = 0.32 + 0.15 + 0.10 + 0.02 = 0.59.
        """
        params = [
            make_param("hazard", weight=1.0, group="hazard"),
            make_param("v_social", weight=0.40, group="vulnerability"),
            make_param("v_physical", weight=0.25, group="vulnerability"),
            make_param("v_economic", weight=0.25, group="vulnerability"),
            make_param("v_environmental", weight=0.10, group="vulnerability"),
            make_param("capacity", weight=1.0, group="capacity"),
        ]
        results = {
            "hazard": uniform(self.S, 0.8),
            "v_social": uniform(self.S, 0.8),
            "v_physical": uniform(self.S, 0.6),
            "v_economic": uniform(self.S, 0.4),
            "v_environmental": uniform(self.S, 0.2),
            "capacity": uniform(self.S, 0.3),
        }
        result = self.calc(params, results)
        assert_allclose(
            self,
            result,
            expected_rbi_risk(0.8, 0.59, 0.3),
            msg="flood vulnerability weights (40/25/25/10)",
        )

    def test_landslide_vulnerability_weights(self):
        """
        RBI 2023 (p. 54) vulnerability weights for Landslide (IKTL):
            IKTL = 0.40 * IKS + 0.25 * IKF + 0.25 * IKE + 0.10 * IKL
        With IKS=0.8, IKF=0.6, IKE=0.4, IKL=0.2 -> IKTL = 0.32 + 0.15 + 0.10 + 0.02 = 0.59.
        """
        params = [
            make_param("hazard", weight=1.0, group="hazard"),
            make_param("v_social", weight=0.40, group="vulnerability"),
            make_param("v_physical", weight=0.25, group="vulnerability"),
            make_param("v_economic", weight=0.25, group="vulnerability"),
            make_param("v_environmental", weight=0.10, group="vulnerability"),
            make_param("capacity", weight=1.0, group="capacity"),
        ]
        results = {
            "hazard": uniform(self.S, 0.7),
            "v_social": uniform(self.S, 0.8),
            "v_physical": uniform(self.S, 0.6),
            "v_economic": uniform(self.S, 0.4),
            "v_environmental": uniform(self.S, 0.2),
            "capacity": uniform(self.S, 0.4),
        }
        result = self.calc(params, results)
        assert_allclose(
            self,
            result,
            expected_rbi_risk(0.7, 0.59, 0.4),
            msg="landslide vulnerability weights (40/25/25/10)",
        )

    def test_capacity_group_weighted_aggregation(self):
        """Multiple capacity components aggregate via weighted mean before (1 - C)."""
        p_h = make_param("h", weight=1.0, group="hazard")
        p_v = make_param("v", weight=1.0, group="vulnerability")
        p_c1 = make_param("c1", weight=0.6, group="capacity")
        p_c2 = make_param("c2", weight=0.4, group="capacity")

        results = {
            "h": uniform(self.S, 1.0),
            "v": uniform(self.S, 1.0),
            "c1": uniform(self.S, 0.5),
            "c2": uniform(self.S, 1.0),
        }
        result = self.calc([p_h, p_v, p_c1, p_c2], results)
        # C = 0.6 * 0.5 + 0.4 * 1.0 = 0.70 -> R = (1.0 * 1.0 * (1 - 0.70))^(1/3)
        assert_allclose(self, result, expected_rbi_risk(1.0, 1.0, 0.70), msg="capacity weighted aggregation")


class TestBNPBCalculatorScenarios(unittest.TestCase):
    """
    End-to-end disaster risk scenarios based on RBI BNPB 2023 risk classes:
      - Low Risk:    0.0 <= R <= 0.3
      - Medium Risk: 0.3 <  R <= 0.6
      - High Risk:   0.6 <  R <= 1.0
    """

    S = (5, 5)

    def calc(self, params, results, shape=None):
        return BNPBCalculator().calculate(params, results, shape or self.S)

    def _flood_parameters(self):
        return [
            make_param("flood_hazard", weight=1.0, group="hazard"),
            make_param("vuln_social", weight=0.40, group="vulnerability"),
            make_param("vuln_physical", weight=0.25, group="vulnerability"),
            make_param("vuln_economic", weight=0.25, group="vulnerability"),
            make_param("vuln_environmental", weight=0.10, group="vulnerability"),
            make_param("regional_capacity", weight=1.0, group="capacity"),
        ]

    def test_high_risk_scenario(self):
        """
        High hazard (0.9), high vulnerability (0.9), low capacity (0.1)
        -> R = (0.9 * 0.9 * (1 - 0.1))^(1/3) = 0.9 (High Risk class: > 0.6, <= 1.0).
        """
        params = self._flood_parameters()
        results = {
            "flood_hazard": uniform(self.S, 0.9),
            "vuln_social": uniform(self.S, 0.9),
            "vuln_physical": uniform(self.S, 0.9),
            "vuln_economic": uniform(self.S, 0.9),
            "vuln_environmental": uniform(self.S, 0.9),
            "regional_capacity": uniform(self.S, 0.1),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, 0.9, msg="high risk scenario value")
        self.assertTrue(np.all((result > 0.6) & (result <= 1.0)), "must fall in High Risk class (0.6, 1.0]")

    def test_medium_risk_scenario(self):
        """
        Moderate hazard (0.5), moderate vulnerability (0.5), moderate capacity (0.5)
        -> R = (0.5 * 0.5 * (1 - 0.5))^(1/3) = 0.5 (Medium Risk class: 0.3 < R <= 0.6).
        """
        params = self._flood_parameters()
        results = {
            "flood_hazard": uniform(self.S, 0.5),
            "vuln_social": uniform(self.S, 0.5),
            "vuln_physical": uniform(self.S, 0.5),
            "vuln_economic": uniform(self.S, 0.5),
            "vuln_environmental": uniform(self.S, 0.5),
            "regional_capacity": uniform(self.S, 0.5),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, 0.5, msg="medium risk scenario value")
        self.assertTrue(np.all((result > 0.3) & (result <= 0.6)), "must fall in Medium Risk class (0.3, 0.6]")

    def test_low_risk_scenario(self):
        """
        Low hazard (0.2), low vulnerability (0.2), high capacity (0.8)
        -> R = (0.2 * 0.2 * (1 - 0.8))^(1/3) = 0.2 (Low Risk class: <= 0.3).
        """
        params = self._flood_parameters()
        results = {
            "flood_hazard": uniform(self.S, 0.2),
            "vuln_social": uniform(self.S, 0.2),
            "vuln_physical": uniform(self.S, 0.2),
            "vuln_economic": uniform(self.S, 0.2),
            "vuln_environmental": uniform(self.S, 0.2),
            "regional_capacity": uniform(self.S, 0.8),
        }
        result = self.calc(params, results)
        assert_allclose(self, result, 0.2, msg="low risk scenario value")
        self.assertTrue(np.all((result >= 0.0) & (result <= 0.3)), "must fall in Low Risk class [0.0, 0.3]")

    def test_capacity_improvement_reduces_risk_class(self):
        """
        Increasing regional capacity from C=0.1 to C=0.9 under fixed H=0.7, V=0.6
        reduces risk by a factor of ((1 - 0.1) / (1 - 0.9))^(1/3) = 9^(1/3) ≈ 2.08,
        shifting the area from High Risk (~0.72 > 0.6) to Medium Risk (~0.35 <= 0.6).
        """
        params = self._flood_parameters()
        base_results = {
            "flood_hazard": uniform(self.S, 0.7),
            "vuln_social": uniform(self.S, 0.6),
            "vuln_physical": uniform(self.S, 0.6),
            "vuln_economic": uniform(self.S, 0.6),
            "vuln_environmental": uniform(self.S, 0.6),
        }
        risk_low_cap = self.calc(params, {**base_results, "regional_capacity": uniform(self.S, 0.1)})
        risk_high_cap = self.calc(params, {**base_results, "regional_capacity": uniform(self.S, 0.9)})

        ratio = risk_low_cap.flat[0] / risk_high_cap.flat[0]
        self.assertAlmostEqual(ratio, np.cbrt(9.0), places=4)
        self.assertGreater(ratio, 2.0, "capacity improvement from 0.1 to 0.9 must reduce risk > 2x")

    def test_spatial_risk_gradient_preserved(self):
        """Cell-by-cell spatial variations must be preserved without global averaging."""
        shape = (3, 3)
        hazard_map = np.array(
            [
                [0.2, 0.5, 0.8],
                [0.3, 0.6, 0.9],
                [0.1, 0.4, 0.7],
            ],
            dtype=np.float32,
        )
        vuln_map = np.array(
            [
                [0.3, 0.5, 0.7],
                [0.4, 0.6, 0.8],
                [0.2, 0.4, 0.6],
            ],
            dtype=np.float32,
        )
        cap_map = np.array(
            [
                [0.8, 0.5, 0.2],
                [0.7, 0.4, 0.1],
                [0.9, 0.6, 0.3],
            ],
            dtype=np.float32,
        )
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        results = {"h": hazard_map, "v": vuln_map, "c": cap_map}
        result = self.calc(params, results, shape=shape)
        expected = expected_rbi_risk(hazard_map, vuln_map, cap_map)
        self.assertTrue(np.allclose(result, expected, rtol=1e-5), "spatial risk gradient must match pixel-wise formula")

    def test_multi_hazard_aggregation(self):
        """
        Multi-hazard risk (RBI 2023, p. 134, Fig. 50):
            R = (H_m * V_m * (1 - C_m)) ** (1 / 3)
        Verifies weighted aggregation into H_m using sample weights (0.7 flood, 0.3 landslide).
        """
        p_flood = make_param("h_flood", weight=0.7, group="hazard")
        p_landslide = make_param("h_landslide", weight=0.3, group="hazard")
        p_vuln = make_param("v", weight=1.0, group="vulnerability")
        p_cap = make_param("c", weight=1.0, group="capacity")

        results = {
            "h_flood": uniform(self.S, 0.8),
            "h_landslide": uniform(self.S, 0.4),
            "v": uniform(self.S, 0.6),
            "c": uniform(self.S, 0.5),
        }
        result = self.calc([p_flood, p_landslide, p_vuln, p_cap], results)
        # H_m = 0.7 * 0.8 + 0.3 * 0.4 = 0.68
        assert_allclose(self, result, expected_rbi_risk(0.68, 0.6, 0.5), msg="multi-hazard risk")


class TestBNPBCalculatorEdgeCases(unittest.TestCase):
    """Edge cases, bounds enforcement, and numerical robustness."""

    S = (4, 4)

    def calc(self, params, results, shape=None):
        return BNPBCalculator().calculate(params, results, shape or self.S)

    def test_output_bounded_in_unit_interval(self):
        """For all H, V, C in [0.0, 1.0], risk index R must lie within [0.0, 1.0]."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        for h_val in [0.0, 0.3, 0.7, 1.0]:
            for v_val in [0.0, 0.5, 1.0]:
                for c_val in [0.0, 0.5, 1.0]:
                    with self.subTest(h=h_val, v=v_val, c=c_val):
                        results = {
                            "h": uniform(self.S, h_val),
                            "v": uniform(self.S, v_val),
                            "c": uniform(self.S, c_val),
                        }
                        result = self.calc(params, results)
                        self.assertTrue(
                            np.all((result >= 0.0) & (result <= 1.0)),
                            f"risk out of [0, 1] for H={h_val}, V={v_val}, C={c_val}: {result.flat[0]}",
                        )

    def test_no_nan_or_inf_for_boundary_capacities(self):
        """Output must not contain NaN or Inf at C = 0.0 or C = 1.0."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        for c_val in [0.0, 1e-12, 1.0 - 1e-12, 1.0]:
            with self.subTest(c=c_val):
                results = {
                    "h": uniform(self.S, 1.0),
                    "v": uniform(self.S, 1.0),
                    "c": uniform(self.S, c_val),
                }
                result = self.calc(params, results)
                self.assertFalse(np.any(np.isnan(result)), f"NaN in output for C={c_val}")
                self.assertFalse(np.any(np.isinf(result)), f"Inf in output for C={c_val}")

    def test_output_shape_and_dtype_preserved(self):
        """Output shape must match grid_shape and have dtype float32."""
        for shape in [(1, 1), (3, 3), (10, 10), (5, 7)]:
            with self.subTest(shape=shape):
                params = [
                    make_param("h", weight=1.0, group="hazard"),
                    make_param("v", weight=1.0, group="vulnerability"),
                    make_param("c", weight=1.0, group="capacity"),
                ]
                results = {
                    "h": uniform(shape, 0.6),
                    "v": uniform(shape, 0.8),
                    "c": uniform(shape, 0.4),
                }
                result = self.calc(params, results, shape=shape)
                self.assertEqual(result.shape, shape, f"shape mismatch for {shape}")
                self.assertEqual(result.dtype, np.float32, "output dtype must be float32")
                assert_allclose(self, result, expected_rbi_risk(0.6, 0.8, 0.4), msg=f"grid {shape}")

    def test_continuity_near_zero_capacity(self):
        """C = 0.0 and C = 1e-10 must yield virtually identical risk values (~1.0)."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        r_zero = self.calc(params, {"h": uniform(self.S, 1.0), "v": uniform(self.S, 1.0), "c": uniform(self.S, 0.0)})
        r_tiny = self.calc(params, {"h": uniform(self.S, 1.0), "v": uniform(self.S, 1.0), "c": uniform(self.S, 1e-10)})
        self.assertTrue(np.allclose(r_zero, r_tiny, rtol=1e-5), "continuity at C -> 0")

    def test_unknown_group_defaults_to_hazard(self):
        """Parameters with an unrecognized group name fall back to the 'hazard' group."""
        p_unknown = make_param("custom_param", weight=1.0, group="unknown_group")
        p_v = make_param("v", weight=1.0, group="vulnerability")
        p_c = make_param("c", weight=1.0, group="capacity")
        results = {
            "custom_param": uniform(self.S, 0.8),
            "v": uniform(self.S, 0.6),
            "c": uniform(self.S, 0.4),
        }
        result = self.calc([p_unknown, p_v, p_c], results)
        assert_allclose(self, result, expected_rbi_risk(0.8, 0.6, 0.4), msg="unknown group fallback to hazard")

    def test_deterministic_output(self):
        """Repeated calls with identical inputs must produce bit-identical arrays."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        results = {
            "h": np.random.RandomState(42).rand(*self.S).astype(np.float32),
            "v": np.random.RandomState(7).rand(*self.S).astype(np.float32),
            "c": (np.random.RandomState(13).rand(*self.S) * 0.9 + 0.05).astype(np.float32),
        }
        r1 = self.calc(params, results)
        r2 = self.calc(params, results)
        self.assertTrue(np.array_equal(r1, r2), "calculator output must be deterministic")


class TestCalculatorConsistency(unittest.TestCase):
    """Mathematical properties and invariants of the calculators."""

    S = (4, 4)

    def test_weighted_sum_monotonicity(self):
        """Increasing a parameter value while holding weights and other parameters fixed increases the result."""
        calc = WeightedSumCalculator()
        p1 = make_param("a", weight=1.0)
        p2 = make_param("b", weight=1.0)

        r_low = calc.calculate([p1, p2], {"a": uniform(self.S, 0.3), "b": uniform(self.S, 0.5)}, self.S)
        r_high = calc.calculate([p1, p2], {"a": uniform(self.S, 0.7), "b": uniform(self.S, 0.5)}, self.S)
        self.assertTrue(np.all(r_high > r_low), "WeightedSumCalculator must be strictly increasing")

    def test_bnpb_strictly_increasing_with_hazard(self):
        """For V > 0 and C < 1, risk strictly increases as H increases."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        common = {"v": uniform(self.S, 0.5), "c": uniform(self.S, 0.5)}
        calc = BNPBCalculator()

        prev = None
        for h_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            current = calc.calculate(params, {**common, "h": uniform(self.S, h_val)}, self.S)
            if prev is not None:
                self.assertTrue(np.all(current > prev), f"monotonicity failed at H={h_val}")
            prev = current

    def test_bnpb_strictly_increasing_with_vulnerability(self):
        """For H > 0 and C < 1, risk strictly increases as V increases."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        common = {"h": uniform(self.S, 0.6), "c": uniform(self.S, 0.4)}
        calc = BNPBCalculator()

        prev = None
        for v_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            current = calc.calculate(params, {**common, "v": uniform(self.S, v_val)}, self.S)
            if prev is not None:
                self.assertTrue(np.all(current > prev), f"monotonicity failed at V={v_val}")
            prev = current

    def test_bnpb_strictly_decreasing_with_capacity(self):
        """For H > 0 and V > 0, risk strictly decreases as C increases."""
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        common = {"h": uniform(self.S, 0.7), "v": uniform(self.S, 0.6)}
        calc = BNPBCalculator()

        prev = None
        for c_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            current = calc.calculate(params, {**common, "c": uniform(self.S, c_val)}, self.S)
            if prev is not None:
                self.assertTrue(np.all(current < prev), f"inverse monotonicity failed at C={c_val}")
            prev = current

    def test_bnpb_cube_root_scaling_with_hazard(self):
        """
        Under R = (H * V * (1 - C))^(1/3), doubling H while holding V and C constant
        scales R by 2^(1/3) (RBI 2023, p. 55).
        """
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        calc = BNPBCalculator()
        r1 = calc.calculate(params, {"h": uniform(self.S, 0.3), "v": uniform(self.S, 0.5), "c": uniform(self.S, 0.5)}, self.S)
        r2 = calc.calculate(params, {"h": uniform(self.S, 0.6), "v": uniform(self.S, 0.5), "c": uniform(self.S, 0.5)}, self.S)
        self.assertTrue(np.allclose(r2, r1 * np.cbrt(2.0), rtol=1e-5), "risk must scale with H^(1/3)")

    def test_bnpb_symmetry_across_factors(self):
        """
        Because R = (H * V * (1 - C))^(1/3) is the geometric mean of H, V, and (1 - C),
        permuting the values of (H, V, 1 - C) must yield the exact same risk index.
        """
        params = [
            make_param("h", weight=1.0, group="hazard"),
            make_param("v", weight=1.0, group="vulnerability"),
            make_param("c", weight=1.0, group="capacity"),
        ]
        calc = BNPBCalculator()
        # Factors (0.8, 0.5, 0.3) -> C = 1 - 0.3 = 0.7
        r_a = calc.calculate(params, {"h": uniform(self.S, 0.8), "v": uniform(self.S, 0.5), "c": uniform(self.S, 0.7)}, self.S)
        # Permuted factors (0.3, 0.8, 0.5) -> C = 1 - 0.5 = 0.5
        r_b = calc.calculate(params, {"h": uniform(self.S, 0.3), "v": uniform(self.S, 0.8), "c": uniform(self.S, 0.5)}, self.S)
        self.assertTrue(np.allclose(r_a, r_b, rtol=1e-5), "geometric mean symmetry violated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
