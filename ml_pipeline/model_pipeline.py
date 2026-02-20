from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline.data_pipeline import TimeSplitResult, load_dataset, time_split

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - runtime dependency
    lgb = None
    LGBMRegressor = None


@dataclass(frozen=True)
class StrategyResult:
    metrics: dict[str, float]
    curve: pd.DataFrame


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_serializable(data), file, indent=2, ensure_ascii=False)


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str = "y",
    direction_col: str = "y_dir",
    extra_exclude: Optional[Iterable[str]] = None,
) -> list[str]:
    excluded = {"date", target_col, direction_col}
    if extra_exclude:
        excluded.update(extra_exclude)
    features = [col for col in df.columns if col not in excluded and np.issubdtype(df[col].dtype, np.number)]
    if not features:
        raise ValueError("No numeric feature columns detected.")
    return sorted(features)


def compute_regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    ic_pearson = float(pd.Series(pred).corr(pd.Series(y_true), method="pearson"))
    ic_spearman = float(pd.Series(pred).corr(pd.Series(y_true), method="spearman"))
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "ic_pearson": ic_pearson if np.isfinite(ic_pearson) else 0.0,
        "ic_spearman": ic_spearman if np.isfinite(ic_spearman) else 0.0,
    }


def compute_direction_metrics(y_true_dir: pd.Series, pred_dir: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true_dir, pred_dir)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_dir, pred_dir)),
    }


def evaluate_strategy(
    *,
    y_true: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    cost_bps: float = 0.0,
    dates: Optional[pd.Series] = None,
) -> StrategyResult:
    signal = np.where(pred > threshold, 1, np.where(pred < -threshold, -1, 0))
    costs = (cost_bps / 10_000.0) * (np.abs(signal) > 0).astype(float)
    strategy_returns = signal * y_true - costs

    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).values if dates is not None else np.arange(len(y_true)),
            "y_true": y_true,
            "pred": pred,
            "signal": signal,
            "strategy_return": strategy_returns,
        }
    )
    curve["equity"] = np.exp(curve["strategy_return"].cumsum())
    curve["equity_peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["equity_peak"] - 1.0

    std = float(curve["strategy_return"].std(ddof=0))
    mean = float(curve["strategy_return"].mean())
    sharpe = 0.0 if std <= 0 else (mean / std) * np.sqrt(252)
    max_drawdown = float(curve["drawdown"].min())
    cagr = 0.0
    if len(curve) > 0 and curve["equity"].iloc[-1] > 0:
        cagr = float(curve["equity"].iloc[-1] ** (252 / len(curve)) - 1.0)

    active = curve["signal"] != 0
    hit_rate = float((np.sign(curve.loc[active, "y_true"]) == np.sign(curve.loc[active, "signal"])).mean()) if active.any() else 0.0
    metrics = {
        "threshold": float(threshold),
        "cost_bps": float(cost_bps),
        "total_return": float(curve["equity"].iloc[-1] - 1.0) if len(curve) > 0 else 0.0,
        "cagr": cagr,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "exposure": float(active.mean()),
        "hit_rate": hit_rate,
    }
    return StrategyResult(metrics=metrics, curve=curve)


def tune_threshold(
    *,
    y_val: np.ndarray,
    pred_val: np.ndarray,
    dates_val: pd.Series,
    quantiles: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    cost_bps: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    rows = []
    abs_pred = np.abs(pred_val)
    for quantile in quantiles:
        threshold = float(np.quantile(abs_pred, quantile))
        result = evaluate_strategy(
            y_true=y_val,
            pred=pred_val,
            threshold=threshold,
            cost_bps=cost_bps,
            dates=dates_val,
        )
        row = {"quantile": quantile, "threshold": threshold}
        row.update(result.metrics)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=[False, False]).reset_index(drop=True)
    top3 = summary.head(3).sort_values("threshold", ascending=True).reset_index(drop=True)
    chosen_threshold = float(top3.loc[0, "threshold"])
    return chosen_threshold, summary


def _ensure_lgbm_installed() -> None:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed. Install it before training.")


def default_lgbm_params(random_state: int) -> dict[str, Any]:
    return {
        "objective": "regression",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 100,
        "random_state": random_state,
    }


def _plot_equity_curve(curve: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(curve["date"], curve["equity"], label="Equity")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _fit_lgbm_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> Any:
    fit_kwargs: dict[str, Any] = {"eval_set": [(x_val, y_val)], "eval_metric": "l2"}
    if lgb is not None:
        fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def train_and_evaluate(
    *,
    dataset_path: str | Path,
    artifacts_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    threshold_quantiles: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    random_state: int = 42,
    cost_bps: float = 0.0,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    split: TimeSplitResult = time_split(dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    feature_cols = infer_feature_columns(split.train)

    x_train = split.train[feature_cols]
    y_train = split.train["y"]
    y_dir_train = split.train["y_dir"]

    x_val = split.val[feature_cols]
    y_val = split.val["y"]
    y_dir_val = split.val["y_dir"]

    x_test = split.test[feature_cols]
    y_test = split.test["y"]
    y_dir_test = split.test["y_dir"]

    if x_train.isna().any().any() or x_val.isna().any().any() or x_test.isna().any().any():
        raise ValueError("NaN detected in feature matrix. Check data preparation.")

    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    ridge.fit(x_train, y_train)

    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    logreg.fit(x_train, y_dir_train)

    _ensure_lgbm_installed()
    lgb_params = default_lgbm_params(random_state=random_state)
    main_model = LGBMRegressor(**lgb_params)
    main_model = _fit_lgbm_model(main_model, x_train, y_train, x_val, y_val)

    upper_model = LGBMRegressor(**{**lgb_params, "objective": "quantile", "alpha": 0.80})
    lower_model = LGBMRegressor(**{**lgb_params, "objective": "quantile", "alpha": 0.20})
    upper_model = _fit_lgbm_model(upper_model, x_train, y_train, x_val, y_val)
    lower_model = _fit_lgbm_model(lower_model, x_train, y_train, x_val, y_val)

    pred_val_ridge = ridge.predict(x_val)
    pred_test_ridge = ridge.predict(x_test)

    pred_val_logreg = logreg.predict(x_val)
    pred_test_logreg = logreg.predict(x_test)

    pred_val_main = main_model.predict(x_val)
    pred_test_main = main_model.predict(x_test)

    pred_test_upper = upper_model.predict(x_test)
    pred_test_lower = lower_model.predict(x_test)

    threshold, threshold_table = tune_threshold(
        y_val=y_val.to_numpy(),
        pred_val=pred_val_main,
        dates_val=split.val["date"],
        quantiles=threshold_quantiles,
        cost_bps=cost_bps,
    )

    main_strategy = evaluate_strategy(
        y_true=y_test.to_numpy(),
        pred=pred_test_main,
        threshold=threshold,
        cost_bps=cost_bps,
        dates=split.test["date"],
    )

    selector_signal = np.where(
        (pred_test_main > threshold) & (pred_test_lower > 0),
        1,
        np.where((pred_test_main < -threshold) & (pred_test_upper < 0), -1, 0),
    )
    selector_pred = selector_signal * np.abs(pred_test_main)
    selector_strategy = evaluate_strategy(
        y_true=y_test.to_numpy(),
        pred=selector_pred,
        threshold=0.0,
        cost_bps=cost_bps,
        dates=split.test["date"],
    )

    report = {
        "data": {
            "rows_total": int(len(dataset)),
            "train_rows": int(len(split.train)),
            "val_rows": int(len(split.val)),
            "test_rows": int(len(split.test)),
            "train_start": str(split.train["date"].min()),
            "train_end": str(split.train["date"].max()),
            "val_start": str(split.val["date"].min()),
            "val_end": str(split.val["date"].max()),
            "test_start": str(split.test["date"].min()),
            "test_end": str(split.test["date"].max()),
        },
        "metrics": {
            "ridge_val_regression": compute_regression_metrics(y_val, pred_val_ridge),
            "ridge_test_regression": compute_regression_metrics(y_test, pred_test_ridge),
            "logreg_val_direction": compute_direction_metrics(y_dir_val, pred_val_logreg),
            "logreg_test_direction": compute_direction_metrics(y_dir_test, pred_test_logreg),
            "lgbm_val_regression": compute_regression_metrics(y_val, pred_val_main),
            "lgbm_test_regression": compute_regression_metrics(y_test, pred_test_main),
            "lgbm_test_direction": compute_direction_metrics(y_dir_test, (pred_test_main > 0).astype(int)),
            "strategy_main_test": main_strategy.metrics,
            "strategy_selector_test": selector_strategy.metrics,
        },
        "strategy": {
            "threshold": float(threshold),
            "threshold_quantiles": list(threshold_quantiles),
            "cost_bps": float(cost_bps),
        },
        "feature_columns": feature_cols,
        "config": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_state": random_state,
        },
    }

    artifacts_root = Path(artifacts_dir)
    models_dir = artifacts_root / "models"
    reports_dir = artifacts_root / "reports"
    plots_dir = artifacts_root / "plots"
    data_dir = artifacts_root / "data"
    for folder in (models_dir, reports_dir, plots_dir, data_dir):
        folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(ridge, models_dir / "ridge.joblib")
    joblib.dump(logreg, models_dir / "logreg.joblib")
    joblib.dump(main_model, models_dir / "main_lgbm.joblib")
    joblib.dump(upper_model, models_dir / "quantile_upper_lgbm.joblib")
    joblib.dump(lower_model, models_dir / "quantile_lower_lgbm.joblib")

    _json_dump(artifacts_root / "feature_columns.json", {"feature_columns": feature_cols})
    _json_dump(artifacts_root / "strategy_config.json", report["strategy"])
    _json_dump(artifacts_root / "metrics.json", report["metrics"])
    _json_dump(artifacts_root / "run_report.json", report)

    threshold_table.to_csv(reports_dir / "threshold_search.csv", index=False)

    predictions = split.test[["date", "y", "y_dir"]].copy()
    predictions["pred_ridge"] = pred_test_ridge
    predictions["pred_main_lgbm"] = pred_test_main
    predictions["pred_quantile_upper"] = pred_test_upper
    predictions["pred_quantile_lower"] = pred_test_lower
    predictions.to_parquet(data_dir / "predictions_test.parquet", index=False)

    main_strategy.curve.to_csv(data_dir / "equity_curve_main_test.csv", index=False)
    selector_strategy.curve.to_csv(data_dir / "equity_curve_selector_test.csv", index=False)
    _plot_equity_curve(main_strategy.curve, plots_dir / "equity_curve_main_test.png", "Main Strategy Equity (Test)")
    _plot_equity_curve(selector_strategy.curve, plots_dir / "equity_curve_selector_test.png", "Selector Strategy Equity (Test)")

    return report


def predict_latest_signal(
    *,
    dataset_path: str | Path,
    artifacts_dir: str | Path,
    model_name: str = "main_lgbm",
) -> dict[str, Any]:
    artifacts_root = Path(artifacts_dir)
    dataset = load_dataset(dataset_path)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    feature_columns = json.loads((artifacts_root / "feature_columns.json").read_text(encoding="utf-8"))["feature_columns"]
    strategy_config = json.loads((artifacts_root / "strategy_config.json").read_text(encoding="utf-8"))
    threshold = float(strategy_config["threshold"])

    model_path = artifacts_root / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    latest_row = dataset.sort_values("date").iloc[-1]
    x_latest = pd.DataFrame([latest_row[feature_columns].to_dict()])
    pred = float(model.predict(x_latest)[0])

    signal = 0
    if pred > threshold:
        signal = 1
    elif pred < -threshold:
        signal = -1

    return {
        "date": str(pd.to_datetime(latest_row["date"]).date()),
        "prediction": pred,
        "threshold": threshold,
        "signal": signal,
        "signal_label": {1: "long", -1: "short", 0: "flat"}[signal],
        "model_name": model_name,
    }
