from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DATE_CANDIDATES = ("date", "ts", "datetime", "time")


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    return None


def _detect_date_col(columns: list[str]) -> Optional[str]:
    normalized = {col.lower(): col for col in columns}
    for candidate in DATE_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.floating, np.float32, np.float64)):
        converted = float(value)
        return converted if np.isfinite(converted) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _basic_table_stats(name: str, path: Path) -> dict[str, object]:
    frame = _read_table(path)
    if frame is None:
        return {"name": name, "path": str(path), "status": "missing_or_unsupported"}

    info: dict[str, object] = {
        "name": name,
        "path": str(path),
        "status": "ok",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
    }
    info["missing_ratio_top5"] = frame.isna().mean().sort_values(ascending=False).head(5).to_dict()

    date_col = _detect_date_col(list(frame.columns))
    info["date_col"] = date_col
    if date_col:
        parsed = pd.to_datetime(frame[date_col], utc=True, errors="coerce")
        valid = parsed.dropna().dt.tz_localize(None)
        info["date_non_null"] = int(valid.shape[0])
        info["date_min"] = str(valid.min()) if not valid.empty else None
        info["date_max"] = str(valid.max()) if not valid.empty else None
        info["duplicate_dates"] = int(valid.duplicated().sum())
        info["is_monotonic_by_date"] = bool(valid.is_monotonic_increasing)
    else:
        info["date_non_null"] = 0
        info["date_min"] = None
        info["date_max"] = None
        info["duplicate_dates"] = None
        info["is_monotonic_by_date"] = None

    normalized_columns = {col.lower(): col for col in frame.columns}
    volume_col = normalized_columns.get("volume")
    if volume_col:
        volume = pd.to_numeric(frame[volume_col], errors="coerce")
        info["volume_zero_count"] = int((volume == 0).sum())
        info["volume_negative_count"] = int((volume < 0).sum())

    open_col = normalized_columns.get("open")
    high_col = normalized_columns.get("high")
    low_col = normalized_columns.get("low")
    close_col = normalized_columns.get("close")
    if all([open_col, high_col, low_col, close_col]):
        open_price = pd.to_numeric(frame[open_col], errors="coerce")
        high = pd.to_numeric(frame[high_col], errors="coerce")
        low = pd.to_numeric(frame[low_col], errors="coerce")
        close = pd.to_numeric(frame[close_col], errors="coerce")
        info["ohlc_non_positive_rows"] = int(((open_price <= 0) | (high <= 0) | (low <= 0) | (close <= 0)).sum())
        info["high_below_low_rows"] = int((high < low).sum())
    return info


def _dataset_stats(path: Path) -> dict[str, object]:
    frame = _read_table(path)
    if frame is None:
        return {"path": str(path), "status": "missing_or_unsupported"}

    result: dict[str, object] = _basic_table_stats("model_ready", path)
    if "y" in frame.columns:
        y = pd.to_numeric(frame["y"], errors="coerce").dropna()
        if not y.empty:
            result["target"] = {
                "mean": float(y.mean()),
                "std": float(y.std(ddof=0)),
                "median": float(y.median()),
                "p01": float(y.quantile(0.01)),
                "p99": float(y.quantile(0.99)),
                "positive_rate": float((y > 0).mean()),
            }
    if "y_dir" in frame.columns:
        y_dir = pd.to_numeric(frame["y_dir"], errors="coerce").dropna()
        if not y_dir.empty:
            result["y_dir_positive_rate"] = float((y_dir > 0).mean())
    return result


def _common_window(stats: list[dict[str, object]]) -> dict[str, object]:
    valid = [item for item in stats if item.get("status") == "ok" and item.get("date_min") and item.get("date_max")]
    if not valid:
        return {"status": "unavailable"}

    start = max(pd.to_datetime(item["date_min"]) for item in valid)
    end = min(pd.to_datetime(item["date_max"]) for item in valid)
    if start > end:
        return {"status": "empty_intersection", "start": str(start), "end": str(end)}
    return {"status": "ok", "start": str(start), "end": str(end)}


def _artifacts_stats(artifacts: Path) -> dict[str, object]:
    result: dict[str, object] = {"artifacts_dir": str(artifacts)}

    quality_path = artifacts / "reports" / "model_quality.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        result["model_quality_rows"] = int(len(quality))
        if not quality.empty:
            key_cols = [col for col in ("model", "split", "task", "mae", "rmse", "ic_pearson", "accuracy", "sharpe") if col in quality.columns]
            result["model_quality_preview"] = quality[key_cols].head(10).to_dict(orient="records")
    else:
        result["model_quality_rows"] = 0

    predictions_path = artifacts / "data" / "predictions_test.parquet"
    if predictions_path.exists():
        pred = pd.read_parquet(predictions_path)
        result["predictions_rows"] = int(len(pred))
        if {"y", "pred_main_lgbm"}.issubset(pred.columns):
            y = pd.to_numeric(pred["y"], errors="coerce")
            p = pd.to_numeric(pred["pred_main_lgbm"], errors="coerce")
            pearson = float(pd.Series(p).corr(pd.Series(y), method="pearson"))
            spearman = float(pd.Series(p).corr(pd.Series(y), method="spearman"))
            result["prediction_corr"] = {
                "pearson": pearson if np.isfinite(pearson) else 0.0,
                "spearman": spearman if np.isfinite(spearman) else 0.0,
            }
    else:
        result["predictions_rows"] = 0

    curve_path = artifacts / "data" / "equity_curve_main_test.csv"
    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        result["equity_curve_rows"] = int(len(curve))
        if "signal" in curve.columns:
            counts = curve["signal"].value_counts().to_dict()
            result["signal_distribution"] = {str(k): int(v) for k, v in counts.items()}
        if "strategy_return" in curve.columns:
            sret = pd.to_numeric(curve["strategy_return"], errors="coerce").fillna(0.0)
            result["strategy_return_stats"] = {
                "mean": float(sret.mean()),
                "std": float(sret.std(ddof=0)),
                "non_zero_days": int((sret != 0).sum()),
            }
    else:
        result["equity_curve_rows"] = 0

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep diagnostics for raw data, model-ready dataset and ML artifacts")
    parser.add_argument("--raw-main", default=None)
    parser.add_argument("--raw-x5", default="./data/candles_mgnt_1d.parquet", help="Deprecated alias for --raw-main")
    parser.add_argument("--raw-imoex", default="./data/candles_imoex_1d.parquet")
    parser.add_argument("--raw-usdrub", default="./data/candles_usdrub_1d.parquet")
    parser.add_argument("--dataset", default="./data/model_ready/mgnt_next_day.parquet")
    parser.add_argument("--artifacts", default="./artifacts/ml")
    parser.add_argument("--output", default="./artifacts/ml/reports/diagnostics_deep.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_main = args.raw_main or args.raw_x5
    raw_stats = [
        _basic_table_stats("raw_main", Path(raw_main)),
        _basic_table_stats("raw_imoex", Path(args.raw_imoex)),
        _basic_table_stats("raw_usdrub", Path(args.raw_usdrub)),
    ]
    dataset_stats = _dataset_stats(Path(args.dataset))
    artifacts_stats = _artifacts_stats(Path(args.artifacts))
    common_window = _common_window(raw_stats + [dataset_stats])

    report = {
        "raw_tables": raw_stats,
        "dataset": dataset_stats,
        "common_window": common_window,
        "artifacts": artifacts_stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Deep diagnostics:")
    for item in raw_stats:
        print(
            f"  {item['name']}: status={item['status']} rows={item.get('rows')} "
            f"range={item.get('date_min')} -> {item.get('date_max')} dup_dates={item.get('duplicate_dates')}"
        )
    print(
        f"  dataset: status={dataset_stats.get('status')} rows={dataset_stats.get('rows')} "
        f"range={dataset_stats.get('date_min')} -> {dataset_stats.get('date_max')}"
    )
    if "target" in dataset_stats:
        target = dataset_stats["target"]
        print(
            "  target(y): "
            f"mean={target['mean']:.6f} std={target['std']:.6f} "
            f"p01={target['p01']:.6f} p99={target['p99']:.6f} pos_rate={target['positive_rate']:.3f}"
        )
    print(f"  common_window: {common_window}")
    print(
        f"  artifacts: quality_rows={artifacts_stats.get('model_quality_rows')} "
        f"pred_rows={artifacts_stats.get('predictions_rows')} equity_rows={artifacts_stats.get('equity_curve_rows')}"
    )
    if "prediction_corr" in artifacts_stats:
        corr = artifacts_stats["prediction_corr"]
        print(f"  prediction_corr: pearson={corr['pearson']:.6f} spearman={corr['spearman']:.6f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
