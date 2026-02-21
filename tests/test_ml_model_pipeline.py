import unittest

import numpy as np
import pandas as pd

from ml_pipeline.model_pipeline import (
    build_sanity_checks,
    build_selector_signal,
    compute_regression_metrics,
    default_lgbm_params,
    evaluate_strategy,
    lgbm_candidate_params,
    min_threshold_from_cost,
    tune_threshold,
)


class MlModelPipelineTests(unittest.TestCase):
    def test_strategy_metrics_shape(self) -> None:
        y_true = np.array([0.01, -0.02, 0.015, -0.005, 0.007], dtype=float)
        pred = np.array([0.02, -0.01, 0.005, -0.02, 0.01], dtype=float)
        dates = pd.date_range("2021-01-01", periods=len(y_true), freq="B")

        result = evaluate_strategy(y_true=y_true, pred=pred, threshold=0.008, dates=dates)
        self.assertEqual(len(result.curve), len(y_true))
        self.assertIn("sharpe", result.metrics)
        self.assertIn("max_drawdown", result.metrics)

    def test_tune_threshold_returns_valid_threshold(self) -> None:
        y_val = np.array([0.01, -0.01, 0.02, -0.015, 0.005, -0.003], dtype=float)
        pred_val = np.array([0.008, -0.012, 0.018, -0.011, 0.004, -0.002], dtype=float)
        dates = pd.date_range("2021-02-01", periods=len(y_val), freq="B")

        threshold, table = tune_threshold(
            y_val=y_val,
            pred_val=pred_val,
            dates_val=pd.Series(dates),
            quantiles=(0.6, 0.7, 0.8, 0.9),
            min_threshold=0.005,
        )
        self.assertGreaterEqual(threshold, 0.005)
        self.assertTrue((table["threshold"] >= 0.005).all())
        self.assertFalse(table.empty)

    def test_regression_metrics_ic_respects_values_not_index_alignment(self) -> None:
        y_true = pd.Series([0.01, -0.01, 0.02, -0.02], index=[100, 101, 102, 103])
        pred = np.array([0.01, -0.01, 0.02, -0.02], dtype=float)
        metrics = compute_regression_metrics(y_true, pred)

        self.assertGreater(metrics["ic_pearson"], 0.99)
        self.assertGreater(metrics["ic_spearman"], 0.99)

    def test_min_threshold_from_cost_roundtrip(self) -> None:
        self.assertAlmostEqual(min_threshold_from_cost(cost_bps=10, threshold_cost_multiplier=1.0), 0.001)

    def test_selector_rule_cost_aware(self) -> None:
        q_low = np.array([0.002, 0.0005, -0.001], dtype=float)
        q_high = np.array([0.003, -0.002, -0.004], dtype=float)
        signal = build_selector_signal(pred_q_low=q_low, pred_q_high=q_high, thr_min=0.001, use_cost_rule=True)
        self.assertListEqual(signal.tolist(), [1, -1, -1])

    def test_sanity_warning_when_prediction_collapses(self) -> None:
        y = np.array([0.01, -0.01, 0.015, -0.02, 0.012], dtype=float)
        pred = np.zeros_like(y)
        sanity = build_sanity_checks(y_val=y, pred_val=pred, y_test=y, pred_test=pred)
        self.assertTrue(bool(sanity["pred_collapse_warning"]))

    def test_lgbm_candidates_include_non_overconstrained_options(self) -> None:
        params = default_lgbm_params(random_state=42, train_rows=1500)
        candidates = lgbm_candidate_params(random_state=42, train_rows=1500)

        self.assertLessEqual(params["min_child_samples"], 120)
        self.assertAlmostEqual(float(params["min_split_gain"]), 0.0)
        self.assertGreaterEqual(min(item["min_child_samples"] for item in candidates), 10)
        self.assertTrue(any(float(item["min_split_gain"]) == 0.0 for item in candidates))


if __name__ == "__main__":
    unittest.main()
