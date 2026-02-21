from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd


DATE_CANDIDATES = ("date", "ts", "datetime", "time")


def _load_matplotlib() -> object:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required for plotting. Run `make install-ml`.") from error
    return plt


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source)
    raise ValueError(f"Unsupported file format: {source}")


def _find_column_case_insensitive(columns: Iterable[str], wanted_names: Iterable[str]) -> Optional[str]:
    normalized = {col.lower(): col for col in columns}
    for wanted in wanted_names:
        if wanted.lower() in normalized:
            return normalized[wanted.lower()]
    return None


def normalize_date_column(frame: pd.DataFrame, *, required: bool = True) -> pd.Series:
    date_col = _find_column_case_insensitive(frame.columns, DATE_CANDIDATES)
    if date_col is None:
        if required:
            raise ValueError(f"Could not detect date column. Expected one of: {DATE_CANDIDATES}")
        return pd.Series(pd.NaT, index=frame.index)
    parsed = pd.to_datetime(frame[date_col], utc=True, errors="coerce")
    if parsed.isna().all() and required:
        raise ValueError(f"Failed to parse date column '{date_col}'")
    return parsed.dt.tz_localize(None)


def prepare_frame_with_date(frame: pd.DataFrame, *, required: bool = True) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = normalize_date_column(result, required=required)
    result = result.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return result.reset_index(drop=True)


def summarize_frame(frame: pd.DataFrame, *, name: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "name": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
    }
    if "date" in frame.columns and len(frame) > 0:
        summary["date_start"] = str(pd.to_datetime(frame["date"]).min())
        summary["date_end"] = str(pd.to_datetime(frame["date"]).max())
    else:
        summary["date_start"] = None
        summary["date_end"] = None
    return summary


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _save_plot(fig: object, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


def plot_price_and_volume(frame: pd.DataFrame, output_path: str | Path, *, title: str) -> Path:
    if "close" not in frame.columns or "volume" not in frame.columns:
        raise ValueError("Expected columns close and volume")
    plt = _load_matplotlib()
    close = _to_numeric(frame["close"])
    volume = _to_numeric(frame["volume"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frame["date"], close, color="#1f77b4", linewidth=1.2)
    axes[0].set_title(title)
    axes[0].set_ylabel("Close")
    axes[0].grid(alpha=0.25)
    axes[1].bar(frame["date"], volume, color="#2ca02c", width=1.0)
    axes[1].set_ylabel("Volume")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_context_normalized(
    main_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    imoex_frame: Optional[pd.DataFrame] = None,
    usdrub_frame: Optional[pd.DataFrame] = None,
) -> Path:
    if "close" not in main_frame.columns:
        raise ValueError("Main frame must contain close column")

    merged = main_frame[["date", "close"]].rename(columns={"close": "MAIN"}).copy()
    if imoex_frame is not None and "close" in imoex_frame.columns:
        merged = merged.merge(imoex_frame[["date", "close"]].rename(columns={"close": "IMOEX"}), on="date", how="left")
    if usdrub_frame is not None and "close" in usdrub_frame.columns:
        merged = merged.merge(usdrub_frame[["date", "close"]].rename(columns={"close": "USDRUB"}), on="date", how="left")

    for col in [item for item in merged.columns if item != "date"]:
        merged[col] = _to_numeric(merged[col])
        base = merged[col].dropna()
        if not base.empty and base.iloc[0] != 0:
            merged[col] = merged[col] / base.iloc[0] * 100.0

    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(12, 5))
    for col in [item for item in merged.columns if item != "date"]:
        axis.plot(merged["date"], merged[col], linewidth=1.2, label=col)
    axis.set_title("Normalized Context Series (base=100)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Index value")
    axis.grid(alpha=0.25)
    axis.legend()
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_target_distribution(frame: pd.DataFrame, output_path: str | Path, *, target_col: str = "y") -> Path:
    if target_col not in frame.columns:
        raise ValueError(f"Target column not found: {target_col}")
    target = _to_numeric(frame[target_col]).dropna()
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.hist(target, bins=60, color="#ff7f0e", alpha=0.9)
    axis.set_title(f"Target Distribution: {target_col}")
    axis.set_xlabel(target_col)
    axis.set_ylabel("Count")
    axis.grid(alpha=0.25)
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_missing_ratio(frame: pd.DataFrame, output_path: str | Path, *, top_n: int = 30) -> Path:
    missing_ratio = frame.isna().mean().sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0].head(top_n)
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(12, 5))
    if missing_ratio.empty:
        axis.text(0.5, 0.5, "No missing values", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.bar(missing_ratio.index, missing_ratio.values, color="#d62728", alpha=0.85)
        axis.tick_params(axis="x", rotation=60)
        for label in axis.get_xticklabels():
            label.set_ha("right")
        axis.set_ylabel("Missing ratio")
        axis.grid(alpha=0.25)
    axis.set_title("Missing Values Ratio")
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_top_feature_correlations(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    target_col: str = "y",
    top_n: int = 20,
) -> Path:
    if target_col not in frame.columns:
        raise ValueError(f"Target column not found: {target_col}")
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    rows = []
    target_values = pd.Series(_to_numeric(frame[target_col]))
    if target_values.nunique(dropna=True) <= 1:
        numeric_cols = []
    for col in numeric_cols:
        feature_values = pd.Series(_to_numeric(frame[col]))
        if feature_values.nunique(dropna=True) <= 1:
            continue
        corr = feature_values.corr(target_values, method="spearman")
        if pd.notna(corr):
            rows.append((col, float(corr)))
    ranked = sorted(rows, key=lambda item: abs(item[1]), reverse=True)[:top_n]

    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(12, 6))
    if not ranked:
        axis.text(0.5, 0.5, "No numeric feature correlations available", ha="center", va="center")
        axis.set_axis_off()
    else:
        names = [item[0] for item in ranked]
        values = [item[1] for item in ranked]
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
        axis.bar(names, values, color=colors, alpha=0.85)
        axis.tick_params(axis="x", rotation=60)
        for label in axis.get_xticklabels():
            label.set_ha("right")
        axis.set_ylabel("Spearman correlation with target")
        axis.grid(alpha=0.25)
    axis.set_title(f"Top Feature Correlations vs {target_col}")
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_predictions_vs_actual(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"date", "y", "pred_main_lgbm"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["date"], _to_numeric(predictions["y"]), label="Actual y", linewidth=1.2, color="#1f77b4")
    axis.plot(
        predictions["date"],
        _to_numeric(predictions["pred_main_lgbm"]),
        label="Predicted y",
        linewidth=1.2,
        color="#ff7f0e",
    )
    axis.set_title("Actual vs Predicted Return (Test)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Return")
    axis.grid(alpha=0.25)
    axis.legend()
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_prediction_scatter(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"y", "pred_main_lgbm"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    x = _to_numeric(predictions["pred_main_lgbm"])
    y = _to_numeric(predictions["y"])
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(x, y, s=18, alpha=0.55, color="#17becf")
    lim = max(np.nanmax(np.abs(x.values)), np.nanmax(np.abs(y.values)))
    if np.isfinite(lim) and lim > 0:
        axis.plot([-lim, lim], [-lim, lim], linestyle="--", color="gray", linewidth=1.0)
        axis.set_xlim(-lim, lim)
        axis.set_ylim(-lim, lim)
    axis.set_title("Prediction Scatter")
    axis.set_xlabel("Predicted y")
    axis.set_ylabel("Actual y")
    axis.grid(alpha=0.25)
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_residuals(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"y", "pred_main_lgbm"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    residuals = _to_numeric(predictions["y"]) - _to_numeric(predictions["pred_main_lgbm"])
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.hist(residuals.dropna(), bins=60, color="#9467bd", alpha=0.85)
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_title("Residual Distribution (y - pred)")
    axis.set_xlabel("Residual")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.25)
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_equity_and_drawdown(curve: pd.DataFrame, output_path: str | Path, *, title: str) -> Path:
    required = {"date", "equity", "drawdown"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(curve["date"], _to_numeric(curve["equity"]), linewidth=1.2, color="#1f77b4")
    axes[0].set_title(title)
    axes[0].set_ylabel("Equity")
    axes[0].grid(alpha=0.25)
    axes[1].fill_between(curve["date"], _to_numeric(curve["drawdown"]), 0.0, color="#d62728", alpha=0.7)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_signal_distribution(curve: pd.DataFrame, output_path: str | Path) -> Path:
    if "signal" not in curve.columns:
        raise ValueError("Curve does not contain signal column")
    counts = curve["signal"].value_counts().reindex([-1, 0, 1]).fillna(0)
    labels = ["short", "flat", "long"]
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, counts.values, color=["#d62728", "#7f7f7f", "#2ca02c"], alpha=0.9)
    axis.set_title("Signal Distribution")
    axis.set_ylabel("Days")
    axis.grid(alpha=0.25)
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_threshold_search(threshold_table: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"quantile", "sharpe", "cagr"}
    missing = required - set(threshold_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    ordered = threshold_table.sort_values("quantile").reset_index(drop=True)
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(ordered["quantile"], ordered["sharpe"], marker="o", label="Sharpe", color="#1f77b4")
    axis.plot(ordered["quantile"], ordered["cagr"], marker="o", label="CAGR", color="#ff7f0e")
    axis.set_title("Threshold Search on Validation")
    axis.set_xlabel("Quantile")
    axis.set_ylabel("Metric value")
    axis.grid(alpha=0.25)
    axis.legend()
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_walk_forward_metric(
    folds_table: pd.DataFrame,
    output_path: str | Path,
    *,
    metric_col: str,
    title: str,
    y_label: str,
) -> Path:
    required = {"fold", metric_col}
    missing = required - set(folds_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    table = folds_table.copy().sort_values("fold")
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(table["fold"], _to_numeric(table[metric_col]), marker="o", color="#1f77b4")
    axis.set_title(title)
    axis.set_xlabel("Fold")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved


def plot_feature_importance(
    model_path: str | Path,
    feature_columns_path: str | Path,
    output_path: str | Path,
    *,
    top_n: int = 25,
) -> Optional[Path]:
    model = joblib.load(model_path)
    feature_payload = json.loads(Path(feature_columns_path).read_text(encoding="utf-8"))
    feature_columns = feature_payload.get("feature_columns", [])
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    if len(feature_columns) != len(importances):
        return None
    ranked = sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)[:top_n]
    if not ranked:
        return None

    names = [item[0] for item in ranked]
    values = [item[1] for item in ranked]
    plt = _load_matplotlib()
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.barh(names[::-1], values[::-1], color="#1f77b4", alpha=0.9)
    axis.set_title("Main LGBM Feature Importance")
    axis.set_xlabel("Importance")
    axis.grid(alpha=0.25)
    saved = _save_plot(fig, output_path)
    plt.close(fig)
    return saved
