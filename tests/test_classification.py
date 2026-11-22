import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from murasa.core.classification import (
    CLASSIFIERS,
    classify,
    classify_equal_interval,
    classify_geometric,
    classify_jenks,
    classify_manual,
    classify_quantile,
    classify_std_dev,
)
from murasa.core.config import EngineConfig


class TestClassificationMethods(unittest.TestCase):

    def test_classifier_registry_contains_expected_methods(self):
        expected = {"jenks", "quantile", "equal_interval", "std_dev", "geometric", "manual"}
        self.assertTrue(expected.issubset(set(CLASSIFIERS.keys())))

    def test_classify_dispatcher_runs_all_methods(self):
        data = np.linspace(1.0, 100.0, 50, dtype=np.float32).reshape(5, 10)
        data[0, 0] = np.nan

        for method in ("jenks", "quantile", "equal_interval", "std_dev", "geometric", "manual"):
            kwargs = {"thresholds": [1.0, 25.0, 50.0, 75.0, 100.0]} if method == "manual" else {}
            n_classes = 4 if method == "manual" else 5

            classified, breaks = classify(data, method=method, num_classes=n_classes, **kwargs)
            self.assertEqual(classified.shape, data.shape, f"{method}: shape mismatch")
            self.assertEqual(classified.dtype, np.uint8, f"{method}: dtype should be uint8")
            self.assertEqual(classified[0, 0], 0, f"{method}: NaN pixel must remain 0")
            valid_classes = classified[np.isfinite(data)]
            self.assertTrue(
                np.all(valid_classes >= 1),
                f"{method}: finite pixels assigned to nodata 0: {valid_classes}",
            )
            self.assertGreaterEqual(len(breaks), 2, f"{method}: expected breaks list")
            self.assertEqual(breaks, sorted(breaks), f"{method}: breaks must be sorted")

    def test_classify_rejects_unknown_method_and_missing_manual_thresholds(self):
        data = np.array([0.1, 0.5, 0.9], dtype=np.float32)

        with self.assertRaises(ValueError):
            classify(data, method="unknown_method", num_classes=3)

        with self.assertRaises(ValueError):
            classify(data, method="manual", num_classes=3, thresholds=None)

        with self.assertRaises(ValueError):
            classify(data, method="manual", num_classes=3, thresholds=[])

    def test_all_nan_input_returns_zero_array_and_empty_breaks(self):
        data = np.full((3, 3), np.nan, dtype=np.float32)

        for method in ("jenks", "quantile", "equal_interval", "std_dev", "geometric", "manual"):
            kwargs = {"thresholds": [0.0, 0.5, 1.0]} if method == "manual" else {}
            classified, breaks = classify(data, method=method, num_classes=3, **kwargs)
            self.assertTrue(np.all(classified == 0), f"{method}: all-NaN should yield all 0")
            self.assertEqual(classified.dtype, np.uint8)
            self.assertEqual(breaks, [])

    def test_non_finite_values_remain_zero_while_valid_remain_nonzero(self):
        data = np.array(
            [
                [np.nan, np.inf, -np.inf],
                [1.0, 5.0, 10.0],
                [20.0, 50.0, 100.0],
            ],
            dtype=np.float32,
        )

        for method in ("jenks", "quantile", "equal_interval", "std_dev", "geometric", "manual"):
            kwargs = {"thresholds": [1.0, 10.0, 50.0, 100.0]} if method == "manual" else {}
            n_classes = 3 if method == "manual" else 4
            classified, _ = classify(data, method=method, num_classes=n_classes, **kwargs)
            self.assertTrue(np.all(classified[0, :] == 0), f"{method}: non-finite must stay 0")
            self.assertTrue(
                np.all(classified[1:, :] >= 1),
                f"{method}: finite pixels must be >= 1, got {classified[1:, :]}",
            )

    def test_classify_equal_interval_exact_bins_and_constant_data(self):
        data = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, np.nan], dtype=np.float32)
        classified, breaks = classify_equal_interval(data, n_classes=5)

        np.testing.assert_allclose(breaks, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], atol=1e-6)
        self.assertEqual(classified.tolist(), [1, 1, 2, 3, 4, 5, 5, 0])

        const = np.array([5.0, 5.0, np.nan], dtype=np.float32)
        const_cls, _ = classify_equal_interval(const, n_classes=5)
        self.assertTrue(np.all(const_cls[:2] >= 1))
        self.assertEqual(const_cls[2], 0)

    def test_classify_quantile_balances_counts(self):
        data = np.linspace(1.0, 100.0, 100, dtype=np.float32).reshape(10, 10)
        classified, breaks = classify_quantile(data, n_classes=5)

        self.assertEqual(len(breaks), 6)
        for cls_id in range(1, 6):
            count = int(np.sum(classified == cls_id))
            self.assertGreaterEqual(count, 18)
            self.assertLessEqual(count, 22)

    def test_classify_jenks_monotonic_and_quantile_fallback(self):
        data = np.array([[1.0, 1.2, 5.0, 5.2, 10.0, 10.2]], dtype=np.float32)
        classified, breaks = classify_jenks(data, n_classes=3)

        self.assertEqual(len(breaks), 4)
        self.assertTrue(np.all(np.diff(classified[0]) >= 0))
        self.assertEqual(int(classified[0, 0]), 1)
        self.assertEqual(int(classified[0, -1]), 3)

        real_import = builtins.__import__

        def _block_jenkspy(name, *args, **kwargs):
            if name == "jenkspy":
                raise ImportError("mocked missing jenkspy")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_jenkspy):
            fallback_cls, fallback_breaks = classify_jenks(data, n_classes=3)
        q_cls, q_breaks = classify_quantile(data, n_classes=3)
        np.testing.assert_array_equal(fallback_cls, q_cls)
        self.assertEqual(fallback_breaks, q_breaks)

    def test_classify_std_dev_symmetric_centered_and_clamps_outliers(self):
        rng = np.random.default_rng(42)
        core_vals = rng.normal(loc=50.0, scale=10.0, size=200).astype(np.float32)
        data = np.concatenate([core_vals, np.array([-500.0, 500.0], dtype=np.float32)])

        classified, breaks = classify_std_dev(data, n_classes=5)
        self.assertEqual(len(breaks), 6)
        mean_val = float(np.mean(data))
        midpoint = 0.5 * (breaks[0] + breaks[-1])
        self.assertAlmostEqual(midpoint, mean_val, places=3)

        self.assertEqual(int(classified[-2]), 1)
        self.assertEqual(int(classified[-1]), 5)

    def test_classify_geometric_positive_and_zero_inclusive_risk_raster(self):
        pos_data = np.array([1.0, 2.0, 10.0, 50.0, 100.0, 500.0, 1000.0], dtype=np.float32)
        classified, breaks = classify_geometric(pos_data, n_classes=3)
        self.assertEqual(len(breaks), 4)
        self.assertEqual(int(classified[0]), 1)
        self.assertEqual(int(classified[-1]), 3)

        risk_data = np.array([0.0, 0.05, 0.2, 0.6, 1.0, np.nan], dtype=np.float32)
        risk_cls, _ = classify_geometric(risk_data, n_classes=3)
        self.assertTrue(np.all(risk_cls[:5] >= 1))
        self.assertEqual(int(risk_cls[5]), 0)

    def test_classify_manual_unsorted_full_breaks_and_out_of_bounds_clamping(self):
        data = np.array([-0.1, 0.0, 0.2, 0.33, 0.5, 0.66, 0.9, 1.5, np.nan], dtype=np.float32)
        classified, breaks = classify_manual(data, thresholds=[0.66, 0.0, 1.0, 0.33])

        self.assertEqual(breaks, [0.0, 0.33, 0.66, 1.0])
        self.assertEqual(classified.tolist(), [1, 1, 1, 2, 2, 3, 3, 3, 0])

    def test_classify_thresholds_alias_with_internal_cutpoints_and_backcompat_exports(self):
        from murasa.core.normalize import classify_jenks as norm_jenks, classify_quantile as norm_quantile

        self.assertIsNotNone(norm_jenks)
        self.assertIsNotNone(norm_quantile)

        data = np.array([0.1, 0.33, 0.5, 0.66, 0.9, np.nan], dtype=np.float32)
        classified, breaks = classify(data, method="thresholds", num_classes=3, thresholds=[0.33, 0.66])
        self.assertEqual(breaks, [float(np.nanmin(data)), 0.33, 0.66, float(np.nanmax(data))])
        self.assertEqual(classified.tolist(), [1, 2, 2, 3, 3, 0])

    def test_classify_with_engine_config_from_yaml(self):
        data = np.array([[0.1, 0.4, 0.7, 0.9]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "cfg.yaml"
            yaml_path.write_text(
                "project:\n"
                "  name: cls_test\n"
                "classification:\n"
                "  method: manual\n"
                "  num_classes: 3\n"
                "  class_names: [Low, Medium, High]\n"
                "  thresholds: [0.0, 0.33, 0.66, 1.0]\n"
            )
            cfg = EngineConfig.from_yaml(yaml_path)
            classified, breaks = classify(
                data,
                method=cfg.classification.method,
                num_classes=cfg.classification.num_classes,
                thresholds=cfg.classification.thresholds,
            )
            self.assertEqual(breaks, [0.0, 0.33, 0.66, 1.0])
            self.assertEqual(classified[0].tolist(), [1, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
