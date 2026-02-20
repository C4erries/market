import unittest

import numpy as np
import pandas as pd

from ml_pipeline.data_pipeline import finalize_dataset, make_features, make_target, time_split


class MlDataPipelineTests(unittest.TestCase):
    def _sample_frame(self, n: int = 80) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = np.linspace(100, 150, n) + np.sin(np.arange(n))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1000, 2000, n),
                "close_imoex": np.linspace(3000, 3500, n),
                "close_usdrub": np.linspace(60, 90, n),
                "dividend_flag_t": 0,
            }
        )
        return frame

    def test_make_target_drops_last_row_after_finalize(self) -> None:
        raw = self._sample_frame()
        featured = make_features(raw)
        targeted = make_target(featured, target_type="log")
        finalized = finalize_dataset(targeted.iloc[:-1].copy())

        self.assertEqual(len(finalized), len(raw) - 21)
        self.assertFalse(finalized[["y", "y_dir"]].isna().any().any())

    def test_time_split_is_ordered_and_non_overlapping(self) -> None:
        raw = self._sample_frame()
        featured = make_features(raw)
        targeted = make_target(featured, target_type="simple")
        finalized = finalize_dataset(targeted.iloc[:-1].copy())
        split = time_split(finalized, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        self.assertLessEqual(split.train["date"].max(), split.val["date"].min())
        self.assertLessEqual(split.val["date"].max(), split.test["date"].min())
        self.assertEqual(len(split.train) + len(split.val) + len(split.test), len(finalized))


if __name__ == "__main__":
    unittest.main()
