import unittest

import numpy as np
import pandas as pd

from ml_pipeline.model_pipeline import evaluate_strategy, tune_threshold


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
        )
        self.assertGreaterEqual(threshold, 0.0)
        self.assertFalse(table.empty)


if __name__ == "__main__":
    unittest.main()
