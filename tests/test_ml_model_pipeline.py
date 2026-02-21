import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from ml_pipeline.model_pipeline import (
    build_sanity_checks,
    build_selector_signal,
    choose_prediction_scale,
    compute_regression_metrics,
    default_lgbm_params,
    evaluate_strategy,
    lgbm_candidate_params,
    min_threshold_from_cost,
    predict_latest_signal,
    score_lgbm_candidate,
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

        self.assertEqual(params["n_estimators"], 2500)
        self.assertEqual(len(candidates), 12)
        self.assertAlmostEqual(float(params["min_split_gain"]), 0.0)
        self.assertSetEqual({item["learning_rate"] for item in candidates}, {0.03, 0.02})
        self.assertSetEqual({item["min_child_samples"] for item in candidates}, {15, 30, 60})

    def test_choose_prediction_scale_prefers_larger_scale_when_cost_floor_blocks_small_signal(self) -> None:
        y_val = np.array([0.004, -0.0035, 0.0038, -0.0042, 0.0039, -0.0036], dtype=float)
        pred_val = np.array([0.0002, -0.0002, 0.0002, -0.0002, 0.0002, -0.0002], dtype=float)
        dates = pd.Series(pd.date_range("2024-01-01", periods=len(y_val), freq="B"))

        result = choose_prediction_scale(
            y_val=y_val,
            pred_val=pred_val,
            dates_val=dates,
            threshold_quantiles=(0.6, 0.8, 0.9),
            min_threshold=0.001,
            cost_bps=10.0,
            overfit_gap=0.1,
            scale_grid=(1.0, 2.0, 4.0, 8.0, 12.0),
        )

        self.assertGreaterEqual(result["scale"], 8.0)
        self.assertGreater(result["val_strategy_exposure"], 0.0)

    def test_score_lgbm_candidate_penalizes_low_ratio_and_exposure(self) -> None:
        strong = score_lgbm_candidate(
            val_sharpe=1.0,
            val_ic_spearman=0.05,
            val_ic_pearson=0.04,
            overfit_gap=0.1,
            val_pred_to_y_std_ratio=0.02,
            val_exposure=0.30,
        )
        penalized = score_lgbm_candidate(
            val_sharpe=1.0,
            val_ic_spearman=0.05,
            val_ic_pearson=0.04,
            overfit_gap=0.1,
            val_pred_to_y_std_ratio=0.005,
            val_exposure=0.10,
        )
        self.assertLess(penalized, strong - 0.49)

    def test_predict_latest_signal_uses_prediction_scale(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            (artifacts / "models").mkdir(parents=True, exist_ok=True)

            dataset_path = root / "dataset.parquet"
            dataset = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-02-19"]),
                    "f1": [1.0],
                    "y": [0.0],
                    "y_dir": [1],
                }
            )
            dataset.to_parquet(dataset_path, index=False)

            (artifacts / "feature_columns.json").write_text('{"feature_columns": ["f1"]}', encoding="utf-8")
            (artifacts / "strategy_config.json").write_text(
                '{"threshold": 0.5, "prediction_scale_main": 2.0}',
                encoding="utf-8",
            )
            model = DummyRegressor(strategy="constant", constant=0.3).fit(np.array([[0.0], [1.0]]), np.array([0.3, 0.3]))
            joblib.dump(model, artifacts / "models" / "main_lgbm.joblib")

            result = predict_latest_signal(dataset_path=dataset_path, artifacts_dir=artifacts)
            self.assertAlmostEqual(float(result["prediction_raw"]), 0.3, places=6)
            self.assertAlmostEqual(float(result["prediction"]), 0.6, places=6)
            self.assertEqual(int(result["signal"]), 1)


if __name__ == "__main__":
    unittest.main()
