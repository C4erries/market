from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml_pipeline.visualization import (
    plot_equity_and_drawdown,
    plot_feature_importance,
    plot_prediction_scatter,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_signal_distribution,
    plot_threshold_search,
    plot_walk_forward_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model diagnostics plots from saved artifacts")
    parser.add_argument("--artifacts", default="./artifacts/ml", help="Artifacts directory from training")
    parser.add_argument("--out", default=None, help="Output folder for plots (default: <artifacts>/plots/model_diagnostics)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = Path(args.artifacts)
    out_dir = Path(args.out) if args.out else artifacts / "plots" / "model_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = artifacts / "data" / "predictions_test.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions_test.parquet not found: {predictions_path}")
    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")

    main_curve_path = artifacts / "data" / "equity_curve_main_test.csv"
    if not main_curve_path.exists():
        raise FileNotFoundError(f"equity_curve_main_test.csv not found: {main_curve_path}")
    main_curve = pd.read_csv(main_curve_path)
    main_curve["date"] = pd.to_datetime(main_curve["date"], errors="coerce")

    plot_predictions_vs_actual(predictions, out_dir / "pred_vs_actual.png")
    plot_prediction_scatter(predictions, out_dir / "pred_scatter.png")
    plot_residuals(predictions, out_dir / "residuals_hist.png")
    plot_equity_and_drawdown(main_curve, out_dir / "equity_drawdown_main.png", title="Main Strategy: Equity and Drawdown")
    plot_signal_distribution(main_curve, out_dir / "signal_distribution_main.png")

    selector_curve_path = artifacts / "data" / "equity_curve_selector_test.csv"
    if selector_curve_path.exists():
        selector_curve = pd.read_csv(selector_curve_path)
        selector_curve["date"] = pd.to_datetime(selector_curve["date"], errors="coerce")
        plot_signal_distribution(selector_curve, out_dir / "selector_signal_distribution.png")

    threshold_path = artifacts / "reports" / "threshold_search.csv"
    if threshold_path.exists():
        threshold_table = pd.read_csv(threshold_path)
        plot_threshold_search(threshold_table, out_dir / "threshold_search.png")

    walk_forward_path = artifacts / "reports" / "walk_forward_folds.csv"
    if walk_forward_path.exists():
        walk_forward = pd.read_csv(walk_forward_path)
        if {"fold", "strategy_main_test_sharpe"}.issubset(walk_forward.columns):
            plot_walk_forward_metric(
                walk_forward,
                out_dir / "walk_forward_sharpe.png",
                metric_col="strategy_main_test_sharpe",
                title="Walk-forward Sharpe by Fold",
                y_label="Sharpe",
            )
        if {"fold", "lgbm_test_ic_spearman"}.issubset(walk_forward.columns):
            plot_walk_forward_metric(
                walk_forward,
                out_dir / "walk_forward_ic.png",
                metric_col="lgbm_test_ic_spearman",
                title="Walk-forward IC (Spearman) by Fold",
                y_label="IC Spearman",
            )

    feature_path = artifacts / "feature_columns.json"
    model_path = artifacts / "models" / "main_lgbm.joblib"
    if feature_path.exists() and model_path.exists():
        saved = plot_feature_importance(
            model_path=model_path,
            feature_columns_path=feature_path,
            output_path=out_dir / "feature_importance_main_lgbm.png",
            top_n=25,
        )
        if saved is None:
            print("Skipped feature importance plot: model does not expose feature_importances_.")

    print(f"Saved model plots to: {out_dir}")


if __name__ == "__main__":
    main()
