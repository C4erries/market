from __future__ import annotations

import json
import logging
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

from ml_pipeline.data_pipeline import TimeSplitResult, load_dataset, time_split, walk_forward_splits

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - runtime dependency
    lgb = None
    LGBMRegressor = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyResult:
    metrics: dict[str, float]
    curve: pd.DataFrame


@dataclass(frozen=True)
class SplitRunResult:
    feature_cols: list[str]
    metrics: dict[str, dict[str, float]]
    strategies: dict[str, StrategyResult]
    thresholds: dict[str, float]
    lgbm_params: dict[str, Any]
    lgbm_selection_table: pd.DataFrame
    threshold_tables: dict[str, pd.DataFrame]
    models: dict[str, Any]
    predictions_test: pd.DataFrame
    sanity_checks: dict[str, float | bool]


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
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
    y_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(pred, dtype=float)
    ic_pearson = float(pd.Series(pred_values).corr(pd.Series(y_values), method="pearson"))
    ic_spearman = float(pd.Series(pred_values).corr(pd.Series(y_values), method="spearman"))
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


def _safe_std(values: np.ndarray) -> float:
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return 0.0
    std = float(np.std(series, ddof=0))
    return std if np.isfinite(std) else 0.0


def _safe_mean(values: np.ndarray) -> float:
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return 0.0
    mean = float(np.mean(series))
    return mean if np.isfinite(mean) else 0.0


def build_sanity_checks(
    *,
    y_val: np.ndarray,
    pred_val: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    collapse_ratio: float = 0.1,
) -> dict[str, float | bool]:
    y_val_std = _safe_std(y_val)
    y_test_std = _safe_std(y_test)
    pred_val_std = _safe_std(pred_val)
    pred_test_std = _safe_std(pred_test)
    ratio_val = 0.0 if y_val_std <= 0 else pred_val_std / y_val_std
    ratio_test = 0.0 if y_test_std <= 0 else pred_test_std / y_test_std
    return {
        "pred_main_mean_val": _safe_mean(pred_val),
        "pred_main_std_val": pred_val_std,
        "y_mean_val": _safe_mean(y_val),
        "y_std_val": y_val_std,
        "pred_to_y_std_ratio_val": float(ratio_val),
        "pred_main_mean_test": _safe_mean(pred_test),
        "pred_main_std_test": pred_test_std,
        "y_mean_test": _safe_mean(y_test),
        "y_std_test": y_test_std,
        "pred_to_y_std_ratio_test": float(ratio_test),
        "pred_collapse_warning": bool(ratio_val < collapse_ratio or ratio_test < collapse_ratio),
    }


def build_selector_signal(
    *,
    pred_q_low: np.ndarray,
    pred_q_high: np.ndarray,
    thr_min: float,
    use_cost_rule: bool = True,
    pred_main: Optional[np.ndarray] = None,
    pred_main_threshold: float = 0.0,
) -> np.ndarray:
    q_low = np.asarray(pred_q_low, dtype=float)
    q_high = np.asarray(pred_q_high, dtype=float)
    if use_cost_rule:
        return np.where(q_low > thr_min, 1, np.where(q_high < -thr_min, -1, 0))

    if pred_main is None:
        raise ValueError("pred_main must be provided when use_cost_rule=False")
    main = np.asarray(pred_main, dtype=float)
    return np.where((main > pred_main_threshold) & (q_low > 0), 1, np.where((main < -pred_main_threshold) & (q_high < 0), -1, 0))


def evaluate_strategy(
    *,
    y_true: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    cost_bps: float = 0.0,
    dates: Optional[pd.Series] = None,
    signal_override: Optional[np.ndarray] = None,
) -> StrategyResult:
    if signal_override is None:
        signal = np.where(pred > threshold, 1, np.where(pred < -threshold, -1, 0))
    else:
        signal = np.asarray(signal_override, dtype=int)
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
    min_threshold: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    rows = []
    abs_pred = np.abs(pred_val)
    candidates: list[tuple[float, float]] = []
    for quantile in quantiles:
        threshold = max(float(np.quantile(abs_pred, quantile)), float(min_threshold))
        candidates.append((float(quantile), float(threshold)))
    if min_threshold > 0:
        for multiplier in (1.0, 1.25, 1.5, 2.0, 3.0):
            candidates.append((float("nan"), float(min_threshold) * multiplier))

    unique: dict[float, float] = {}
    for quantile, threshold in candidates:
        if not np.isfinite(threshold):
            continue
        threshold = max(float(min_threshold), float(threshold))
        threshold_key = round(float(threshold), 12)
        if threshold_key not in unique:
            unique[threshold_key] = quantile
    if not unique:
        unique[round(float(min_threshold), 12)] = float("nan")

    for threshold, quantile in sorted(unique.items()):
        result = evaluate_strategy(
            y_true=y_val,
            pred=pred_val,
            threshold=threshold,
            cost_bps=cost_bps,
            dates=dates_val,
        )
        row = {
            "quantile": quantile,
            "threshold": float(threshold),
            "thr_min": float(min_threshold),
            "min_threshold": float(min_threshold),
            "cost_bps": float(cost_bps),
        }
        row.update(result.metrics)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=[False, False]).reset_index(drop=True)
    top3 = summary.head(3).sort_values("threshold", ascending=True).reset_index(drop=True)
    chosen_threshold = max(float(top3.loc[0, "threshold"]), float(min_threshold))
    return chosen_threshold, summary


def min_threshold_from_cost(*, cost_bps: float, threshold_cost_multiplier: float) -> float:
    if cost_bps <= 0:
        return 0.0
    if threshold_cost_multiplier <= 0:
        raise ValueError("threshold_cost_multiplier must be > 0")
    return float(cost_bps / 10_000.0 * threshold_cost_multiplier)


def _ensure_lgbm_installed() -> None:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed. Install it before training.")


def default_lgbm_params(random_state: int, train_rows: int) -> dict[str, Any]:
    adaptive_min_child = max(100, min(300, max(100, train_rows // 6)))
    return {
        "objective": "regression",
        "n_estimators": 2200,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": adaptive_min_child,
        "min_split_gain": 0.02,
        "reg_lambda": 2.0,
        "reg_alpha": 0.5,
        "verbosity": -1,
        "random_state": random_state,
    }


def lgbm_candidate_params(random_state: int, train_rows: int) -> list[dict[str, Any]]:
    base = default_lgbm_params(random_state=random_state, train_rows=train_rows)
    child = base["min_child_samples"]
    return [
        {
            **base,
            "num_leaves": 31,
            "max_depth": 4,
            "min_child_samples": min(300, child + 20),
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "reg_alpha": 0.6,
            "min_split_gain": 0.03,
        },
        {
            **base,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": min(300, child + 40),
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_lambda": 2.5,
            "reg_alpha": 0.8,
            "min_split_gain": 0.05,
        },
        {
            **base,
            "num_leaves": 47,
            "max_depth": 5,
            "min_child_samples": min(300, child + 60),
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_lambda": 3.0,
            "reg_alpha": 1.0,
            "min_split_gain": 0.08,
        },
        {
            **base,
            "num_leaves": 63,
            "max_depth": 6,
            "min_child_samples": min(300, child + 80),
            "subsample": 0.70,
            "colsample_bytree": 0.70,
            "learning_rate": 0.015,
            "reg_lambda": 4.0,
            "reg_alpha": 1.2,
            "min_split_gain": 0.10,
        },
    ]


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
        fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=120, verbose=False)]
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def select_best_lgbm_model(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    candidates = lgbm_candidate_params(random_state=random_state, train_rows=len(x_train))
    scored_rows: list[dict[str, Any]] = []
    best_model: Any | None = None
    best_params: dict[str, Any] | None = None
    best_score = -np.inf

    for idx, params in enumerate(candidates):
        model = LGBMRegressor(**params)
        model = _fit_lgbm_model(model, x_train, y_train, x_val, y_val)
        pred_train = model.predict(x_train)
        pred_val = model.predict(x_val)
        train_metrics = compute_regression_metrics(y_train, pred_train)
        val_metrics = compute_regression_metrics(y_val, pred_val)

        overfit_gap = abs(train_metrics["ic_spearman"] - val_metrics["ic_spearman"])
        score = 0.65 * val_metrics["ic_spearman"] + 0.35 * val_metrics["ic_pearson"] - 0.60 * overfit_gap

        row = {
            "candidate_id": idx,
            "score": float(score),
            "train_ic_spearman": train_metrics["ic_spearman"],
            "val_ic_spearman": val_metrics["ic_spearman"],
            "val_ic_pearson": val_metrics["ic_pearson"],
            "val_rmse": val_metrics["rmse"],
            "overfit_gap": float(overfit_gap),
            "best_iteration": int(getattr(model, "best_iteration_", -1) or -1),
            "params": json.dumps(_to_serializable(params), ensure_ascii=False),
        }
        scored_rows.append(row)

        if score > best_score:
            best_score = float(score)
            best_model = model
            best_params = params

    if best_model is None or best_params is None:
        raise RuntimeError("Failed to select LGBM model from candidates.")

    scored = pd.DataFrame(scored_rows).sort_values("score", ascending=False).reset_index(drop=True)
    LOGGER.info(
        "Selected LGBM candidate id=%s score=%.6f val_ic_spearman=%.6f overfit_gap=%.6f",
        int(scored.loc[0, "candidate_id"]),
        float(scored.loc[0, "score"]),
        float(scored.loc[0, "val_ic_spearman"]),
        float(scored.loc[0, "overfit_gap"]),
    )
    return best_model, best_params, scored


def _logreg_signed_score(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba_up = np.asarray(model.predict_proba(features), dtype=float)[:, 1]
        return proba_up - 0.5
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(features), dtype=float)
        return np.tanh(raw / 4.0)
    raise RuntimeError("LogReg model does not expose predict_proba or decision_function.")


def _validate_feature_matrices(*, x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame) -> None:
    if x_train.isna().any().any() or x_val.isna().any().any() or x_test.isna().any().any():
        raise ValueError("NaN detected in feature matrix. Check data preparation.")


def _run_single_split(
    *,
    split: TimeSplitResult,
    threshold_quantiles: tuple[float, ...],
    random_state: int,
    cost_bps: float,
    threshold_cost_multiplier: float,
    selector_use_cost_rule: bool,
    selector_alpha_low: float,
    selector_alpha_high: float,
    feature_cols: Optional[list[str]] = None,
) -> SplitRunResult:
    if feature_cols is None:
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
    _validate_feature_matrices(x_train=x_train, x_val=x_val, x_test=x_test)

    ridge = Pipeline(steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge.fit(x_train, y_train)
    logreg = Pipeline(steps=[("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=random_state))])
    logreg.fit(x_train, y_dir_train)

    _ensure_lgbm_installed()
    main_model, lgb_params, lgbm_selection_table = select_best_lgbm_model(
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, random_state=random_state
    )
    upper_model = LGBMRegressor(**{**lgb_params, "objective": "quantile", "alpha": selector_alpha_high})
    lower_model = LGBMRegressor(**{**lgb_params, "objective": "quantile", "alpha": selector_alpha_low})
    upper_model = _fit_lgbm_model(upper_model, x_train, y_train, x_val, y_val)
    lower_model = _fit_lgbm_model(lower_model, x_train, y_train, x_val, y_val)

    pred_val_ridge = ridge.predict(x_val)
    pred_test_ridge = ridge.predict(x_test)
    pred_val_logreg = logreg.predict(x_val)
    pred_test_logreg = logreg.predict(x_test)
    pred_val_logreg_score = _logreg_signed_score(logreg, x_val)
    pred_test_logreg_score = _logreg_signed_score(logreg, x_test)
    pred_val_main = main_model.predict(x_val)
    pred_test_main = main_model.predict(x_test)
    pred_test_upper = upper_model.predict(x_test)
    pred_test_lower = lower_model.predict(x_test)

    min_threshold = min_threshold_from_cost(cost_bps=cost_bps, threshold_cost_multiplier=threshold_cost_multiplier)
    threshold_main, threshold_table_main = tune_threshold(
        y_val=y_val.to_numpy(),
        pred_val=pred_val_main,
        dates_val=split.val["date"],
        quantiles=threshold_quantiles,
        cost_bps=cost_bps,
        min_threshold=min_threshold,
    )
    threshold_logreg, threshold_table_logreg = tune_threshold(
        y_val=y_val.to_numpy(),
        pred_val=pred_val_logreg_score,
        dates_val=split.val["date"],
        quantiles=threshold_quantiles,
        cost_bps=cost_bps,
        min_threshold=min_threshold,
    )

    main_strategy = evaluate_strategy(y_true=y_test.to_numpy(), pred=pred_test_main, threshold=threshold_main, cost_bps=cost_bps, dates=split.test["date"])
    logreg_strategy = evaluate_strategy(y_true=y_test.to_numpy(), pred=pred_test_logreg_score, threshold=threshold_logreg, cost_bps=cost_bps, dates=split.test["date"])
    selector_signal = build_selector_signal(
        pred_q_low=pred_test_lower,
        pred_q_high=pred_test_upper,
        thr_min=min_threshold,
        use_cost_rule=selector_use_cost_rule,
        pred_main=pred_test_main,
        pred_main_threshold=threshold_main,
    )
    selector_threshold = float(min_threshold if selector_use_cost_rule else 0.0)
    selector_strategy = evaluate_strategy(
        y_true=y_test.to_numpy(),
        pred=pred_test_main,
        threshold=selector_threshold,
        cost_bps=cost_bps,
        dates=split.test["date"],
        signal_override=selector_signal,
    )

    metrics = {
        "ridge_val_regression": compute_regression_metrics(y_val, pred_val_ridge),
        "ridge_test_regression": compute_regression_metrics(y_test, pred_test_ridge),
        "logreg_val_direction": compute_direction_metrics(y_dir_val, pred_val_logreg),
        "logreg_test_direction": compute_direction_metrics(y_dir_test, pred_test_logreg),
        "lgbm_val_regression": compute_regression_metrics(y_val, pred_val_main),
        "lgbm_test_regression": compute_regression_metrics(y_test, pred_test_main),
        "lgbm_test_direction": compute_direction_metrics(y_dir_test, (pred_test_main > 0).astype(int)),
        "strategy_main_test": main_strategy.metrics,
        "strategy_logreg_test": logreg_strategy.metrics,
        "strategy_selector_test": selector_strategy.metrics,
    }
    sanity_checks = build_sanity_checks(y_val=y_val.to_numpy(), pred_val=pred_val_main, y_test=y_test.to_numpy(), pred_test=pred_test_main)
    predictions_test = split.test[["date", "y", "y_dir"]].copy()
    predictions_test["pred_ridge"] = pred_test_ridge
    predictions_test["pred_main_lgbm"] = pred_test_main
    predictions_test["pred_logreg_score"] = pred_test_logreg_score
    predictions_test["pred_logreg_dir"] = pred_test_logreg
    predictions_test["pred_quantile_upper"] = pred_test_upper
    predictions_test["pred_quantile_lower"] = pred_test_lower
    predictions_test["selector_signal"] = selector_signal

    return SplitRunResult(
        feature_cols=feature_cols,
        metrics=metrics,
        strategies={"main": main_strategy, "logreg": logreg_strategy, "selector": selector_strategy},
        thresholds={"threshold_main": float(threshold_main), "threshold_logreg": float(threshold_logreg), "threshold_min": float(min_threshold), "threshold_selector": float(selector_threshold)},
        lgbm_params=lgb_params,
        lgbm_selection_table=lgbm_selection_table,
        threshold_tables={"main": threshold_table_main, "logreg": threshold_table_logreg},
        models={"ridge": ridge, "logreg": logreg, "main_lgbm": main_model, "quantile_upper_lgbm": upper_model, "quantile_lower_lgbm": lower_model},
        predictions_test=predictions_test,
        sanity_checks=sanity_checks,
    )


def _walk_forward_layout(
    *,
    n_rows: int,
    wf_folds: int,
    val_ratio: float,
    test_ratio: float,
    wf_step_size: Optional[int],
    min_train_rows: int = 252,
) -> dict[str, int]:
    if wf_folds <= 0:
        raise ValueError("wf_folds must be positive")
    val_size = max(20, int(n_rows * val_ratio))
    test_size = max(20, int(n_rows * test_ratio))
    step_size = wf_step_size if wf_step_size is not None else max(5, test_size // max(1, wf_folds))
    if step_size <= 0:
        raise ValueError("wf_step_size must be positive")

    folds = wf_folds
    train_size = n_rows - val_size - test_size - (folds - 1) * step_size
    while folds > 1 and train_size < min_train_rows:
        folds -= 1
        train_size = n_rows - val_size - test_size - (folds - 1) * step_size
    if train_size < 60:
        raise ValueError(
            "Dataset is too small for requested walk-forward setup. "
            f"rows={n_rows} val={val_size} test={test_size} step={step_size} folds={folds}"
        )
    return {
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "step_size": int(step_size),
        "folds": int(folds),
    }


def _aggregate_walk_forward(folds_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if folds_frame.empty:
        summary = pd.DataFrame(columns=["metric", "mean", "median", "std", "min", "max"])
        return summary, {}

    numeric_cols = [col for col in folds_frame.columns if col != "fold" and np.issubdtype(folds_frame[col].dtype, np.number)]
    rows = []
    summary_dict: dict[str, dict[str, float]] = {}
    for col in sorted(numeric_cols):
        values = pd.to_numeric(folds_frame[col], errors="coerce").dropna()
        if values.empty:
            continue
        stats = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        rows.append({"metric": col, **stats})
        summary_dict[col] = stats
    summary = pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)
    return summary, summary_dict


def _run_walk_forward(
    *,
    dataset: pd.DataFrame,
    threshold_quantiles: tuple[float, ...],
    random_state: int,
    cost_bps: float,
    threshold_cost_multiplier: float,
    selector_use_cost_rule: bool,
    selector_alpha_low: float,
    selector_alpha_high: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    wf_folds: int,
    wf_expanding: bool,
    wf_step_size: Optional[int],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    layout = _walk_forward_layout(
        n_rows=len(dataset),
        wf_folds=wf_folds,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        wf_step_size=wf_step_size,
        min_train_rows=max(120, int(len(dataset) * max(0.2, train_ratio * 0.4))),
    )
    splits = walk_forward_splits(
        dataset,
        train_size=layout["train_size"],
        val_size=layout["val_size"],
        test_size=layout["test_size"],
        step_size=layout["step_size"],
        expanding=wf_expanding,
    )
    splits = splits[: layout["folds"]]

    rows: list[dict[str, Any]] = []
    for idx, split in enumerate(splits, start=1):
        result = _run_single_split(
            split=split,
            threshold_quantiles=threshold_quantiles,
            random_state=random_state,
            cost_bps=cost_bps,
            threshold_cost_multiplier=threshold_cost_multiplier,
            selector_use_cost_rule=selector_use_cost_rule,
            selector_alpha_low=selector_alpha_low,
            selector_alpha_high=selector_alpha_high,
            feature_cols=feature_cols,
        )
        row = {
            "fold": idx,
            "train_rows": int(len(split.train)),
            "val_rows": int(len(split.val)),
            "test_rows": int(len(split.test)),
            "train_start": str(split.train["date"].min()),
            "train_end": str(split.train["date"].max()),
            "val_start": str(split.val["date"].min()),
            "val_end": str(split.val["date"].max()),
            "test_start": str(split.test["date"].min()),
            "test_end": str(split.test["date"].max()),
            "lgbm_val_ic_pearson": result.metrics["lgbm_val_regression"]["ic_pearson"],
            "lgbm_val_ic_spearman": result.metrics["lgbm_val_regression"]["ic_spearman"],
            "lgbm_test_ic_pearson": result.metrics["lgbm_test_regression"]["ic_pearson"],
            "lgbm_test_ic_spearman": result.metrics["lgbm_test_regression"]["ic_spearman"],
            "strategy_main_test_sharpe": result.metrics["strategy_main_test"]["sharpe"],
            "strategy_main_test_cagr": result.metrics["strategy_main_test"]["cagr"],
            "strategy_main_test_max_drawdown": result.metrics["strategy_main_test"]["max_drawdown"],
            "strategy_main_test_exposure": result.metrics["strategy_main_test"]["exposure"],
            "strategy_selector_test_sharpe": result.metrics["strategy_selector_test"]["sharpe"],
            "strategy_selector_test_cagr": result.metrics["strategy_selector_test"]["cagr"],
            "strategy_selector_test_max_drawdown": result.metrics["strategy_selector_test"]["max_drawdown"],
            "strategy_selector_test_exposure": result.metrics["strategy_selector_test"]["exposure"],
            "threshold_main": result.thresholds["threshold_main"],
            "threshold_min": result.thresholds["threshold_min"],
            "sanity_pred_std_test": float(result.sanity_checks["pred_main_std_test"]),
            "sanity_y_std_test": float(result.sanity_checks["y_std_test"]),
            "sanity_pred_to_y_std_ratio_test": float(result.sanity_checks["pred_to_y_std_ratio_test"]),
        }
        rows.append(row)
        LOGGER.info(
            "WF fold=%s/%s test_ic=%.4f sharpe=%.4f thr=%.6f thr_min=%.6f",
            idx,
            len(splits),
            row["lgbm_test_ic_spearman"],
            row["strategy_main_test_sharpe"],
            row["threshold_main"],
            row["threshold_min"],
        )

    folds_frame = pd.DataFrame(rows)
    summary_frame, summary_dict = _aggregate_walk_forward(folds_frame)
    payload = {
        "enabled": True,
        "requested_folds": int(wf_folds),
        "actual_folds": int(len(folds_frame)),
        "expanding": bool(wf_expanding),
        "layout": layout,
        "summary": summary_dict,
    }
    return folds_frame, summary_frame, payload


def build_model_quality_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(model_name: str, split: str, task: str, payload: dict[str, float]) -> None:
        row: dict[str, Any] = {
            "model": model_name,
            "split": split,
            "task": task,
        }
        row.update(payload)
        rows.append(row)

    add_row("ridge", "val", "regression", metrics["ridge_val_regression"])
    add_row("ridge", "test", "regression", metrics["ridge_test_regression"])
    add_row("lgbm_main", "val", "regression", metrics["lgbm_val_regression"])
    add_row("lgbm_main", "test", "regression", metrics["lgbm_test_regression"])
    add_row("logreg", "val", "direction", metrics["logreg_val_direction"])
    add_row("logreg", "test", "direction", metrics["logreg_test_direction"])
    add_row("lgbm_main", "test", "direction", metrics["lgbm_test_direction"])
    add_row("strategy_main", "test", "strategy", metrics["strategy_main_test"])
    add_row("strategy_logreg", "test", "strategy", metrics["strategy_logreg_test"])
    add_row("strategy_selector", "test", "strategy", metrics["strategy_selector_test"])

    return pd.DataFrame(rows)


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
    threshold_cost_multiplier: float = 1.0,
    wf_enable: bool = True,
    wf_folds: int = 6,
    wf_expanding: bool = True,
    wf_step_size: Optional[int] = None,
    selector_use_cost_rule: bool = True,
    selector_alpha_low: float = 0.10,
    selector_alpha_high: float = 0.90,
) -> dict[str, Any]:
    if not (0 < selector_alpha_low < selector_alpha_high < 1):
        raise ValueError("selector_alpha_low and selector_alpha_high must satisfy 0 < low < high < 1")

    dataset = load_dataset(dataset_path)
    split: TimeSplitResult = time_split(dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    feature_cols = infer_feature_columns(split.train)
    split_result = _run_single_split(
        split=split,
        threshold_quantiles=threshold_quantiles,
        random_state=random_state,
        cost_bps=cost_bps,
        threshold_cost_multiplier=threshold_cost_multiplier,
        selector_use_cost_rule=selector_use_cost_rule,
        selector_alpha_low=selector_alpha_low,
        selector_alpha_high=selector_alpha_high,
        feature_cols=feature_cols,
    )
    LOGGER.info(
        "Thresholds selected: main=%.6f logreg=%.6f thr_min=%.6f",
        split_result.thresholds["threshold_main"],
        split_result.thresholds["threshold_logreg"],
        split_result.thresholds["threshold_min"],
    )
    LOGGER.info(
        "Sanity checks: pred_std_test=%.6f y_std_test=%.6f ratio=%.4f warning=%s",
        float(split_result.sanity_checks["pred_main_std_test"]),
        float(split_result.sanity_checks["y_std_test"]),
        float(split_result.sanity_checks["pred_to_y_std_ratio_test"]),
        bool(split_result.sanity_checks["pred_collapse_warning"]),
    )

    walk_forward_folds = pd.DataFrame()
    walk_forward_summary = pd.DataFrame(columns=["metric", "mean", "median", "std", "min", "max"])
    walk_forward_payload: dict[str, Any] = {"enabled": False, "requested_folds": int(wf_folds), "actual_folds": 0}
    if wf_enable:
        walk_forward_folds, walk_forward_summary, walk_forward_payload = _run_walk_forward(
            dataset=dataset,
            threshold_quantiles=threshold_quantiles,
            random_state=random_state,
            cost_bps=cost_bps,
            threshold_cost_multiplier=threshold_cost_multiplier,
            selector_use_cost_rule=selector_use_cost_rule,
            selector_alpha_low=selector_alpha_low,
            selector_alpha_high=selector_alpha_high,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            wf_folds=wf_folds,
            wf_expanding=wf_expanding,
            wf_step_size=wf_step_size,
            feature_cols=feature_cols,
        )

    selector_signal_rate = float((split_result.predictions_test["selector_signal"] != 0).mean())
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
        "metrics": dict(split_result.metrics),
        "strategy": {
            "threshold": float(split_result.thresholds["threshold_main"]),
            "threshold_logreg": float(split_result.thresholds["threshold_logreg"]),
            "threshold_selector": float(split_result.thresholds["threshold_selector"]),
            "threshold_min": float(split_result.thresholds["threshold_min"]),
            "threshold_cost_multiplier": float(threshold_cost_multiplier),
            "threshold_quantiles": list(threshold_quantiles),
            "cost_bps": float(cost_bps),
        },
        "selector_config": {
            "use_cost_rule": bool(selector_use_cost_rule),
            "alpha_low": float(selector_alpha_low),
            "alpha_high": float(selector_alpha_high),
            "thr_min": float(split_result.thresholds["threshold_min"]),
            "signal_rate_test": selector_signal_rate,
        },
        "sanity_checks": dict(split_result.sanity_checks),
        "walk_forward": walk_forward_payload,
        "feature_columns": feature_cols,
        "config": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_state": random_state,
            "lgbm_params": split_result.lgbm_params,
            "wf_enable": bool(wf_enable),
            "wf_folds": int(wf_folds),
            "wf_expanding": bool(wf_expanding),
            "wf_step_size": int(wf_step_size) if wf_step_size is not None else None,
        },
    }
    report["metrics"]["walk_forward_aggregate"] = walk_forward_payload.get("summary", {})

    artifacts_root = Path(artifacts_dir)
    models_dir = artifacts_root / "models"
    reports_dir = artifacts_root / "reports"
    plots_dir = artifacts_root / "plots"
    data_dir = artifacts_root / "data"
    for folder in (models_dir, reports_dir, plots_dir, data_dir):
        folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(split_result.models["ridge"], models_dir / "ridge.joblib")
    joblib.dump(split_result.models["logreg"], models_dir / "logreg.joblib")
    joblib.dump(split_result.models["main_lgbm"], models_dir / "main_lgbm.joblib")
    joblib.dump(split_result.models["quantile_upper_lgbm"], models_dir / "quantile_upper_lgbm.joblib")
    joblib.dump(split_result.models["quantile_lower_lgbm"], models_dir / "quantile_lower_lgbm.joblib")
    saved_models = sorted([item.name for item in models_dir.glob("*.joblib")])
    report["saved_models"] = saved_models

    _json_dump(artifacts_root / "feature_columns.json", {"feature_columns": feature_cols})
    _json_dump(artifacts_root / "strategy_config.json", report["strategy"])
    _json_dump(artifacts_root / "metrics.json", report["metrics"])
    _json_dump(artifacts_root / "run_report.json", report)

    split_result.threshold_tables["main"].to_csv(reports_dir / "threshold_search.csv", index=False)
    split_result.threshold_tables["logreg"].to_csv(reports_dir / "threshold_search_logreg.csv", index=False)
    split_result.lgbm_selection_table.to_csv(reports_dir / "lgbm_model_selection.csv", index=False)
    quality_table = build_model_quality_table(report["metrics"])
    quality_table.to_csv(reports_dir / "model_quality.csv", index=False)
    walk_forward_folds.to_csv(reports_dir / "walk_forward_folds.csv", index=False)
    walk_forward_summary.to_csv(reports_dir / "walk_forward_summary.csv", index=False)

    split_result.predictions_test.to_parquet(data_dir / "predictions_test.parquet", index=False)

    split_result.strategies["main"].curve.to_csv(data_dir / "equity_curve_main_test.csv", index=False)
    split_result.strategies["logreg"].curve.to_csv(data_dir / "equity_curve_logreg_test.csv", index=False)
    split_result.strategies["selector"].curve.to_csv(data_dir / "equity_curve_selector_test.csv", index=False)
    _plot_equity_curve(split_result.strategies["main"].curve, plots_dir / "equity_curve_main_test.png", "Main Strategy Equity (Test)")
    _plot_equity_curve(split_result.strategies["logreg"].curve, plots_dir / "equity_curve_logreg_test.png", "LogReg Strategy Equity (Test)")
    _plot_equity_curve(split_result.strategies["selector"].curve, plots_dir / "equity_curve_selector_test.png", "Selector Strategy Equity (Test)")

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
