from __future__ import annotations

import argparse
import json
import logging

from ml_pipeline.model_pipeline import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate next-day ML models")
    parser.add_argument("--dataset", required=True, help="Path to model-ready dataset (.parquet/.csv)")
    parser.add_argument("--artifacts", default="./artifacts/ml", help="Output directory for models and reports")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--threshold-quantiles", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--selection-cost-bps", type=float, default=None, help="Optional cost (bps) used only for LGBM candidate selection")
    parser.add_argument("--threshold-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--wf-enable", type=int, default=1, help="1 to run walk-forward backtest, 0 to disable")
    parser.add_argument("--wf-folds", type=int, default=6, help="Requested number of walk-forward folds")
    parser.add_argument("--wf-expanding", type=int, default=1, help="1 for expanding train window, 0 for rolling")
    parser.add_argument("--wf-step-size", type=int, default=None, help="Optional walk-forward step size")
    parser.add_argument("--selector-use-cost-rule", type=int, default=1, help="1 to use cost-aware quantile risk filter; 0 to use zero-centered quantile filter")
    parser.add_argument("--selector-alpha-low", type=float, default=0.1)
    parser.add_argument("--selector-alpha-high", type=float, default=0.9)
    parser.add_argument("--selector-risk-multiple", type=float, default=1.0, help="Multiplier for quantile risk band in selector cost rule")
    parser.add_argument(
        "--selector-max-interval-width",
        type=float,
        default=None,
        help="Optional max width for [q_low, q_high] interval to filter uncertain selector signals",
    )
    return parser.parse_args()


def parse_quantiles(raw: str) -> tuple[float, ...]:
    quantiles = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not quantiles:
        raise ValueError("threshold quantiles list is empty")
    if any(q <= 0 or q >= 1 for q in quantiles):
        raise ValueError("Each threshold quantile must be in (0, 1)")
    return quantiles


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    report = train_and_evaluate(
        dataset_path=args.dataset,
        artifacts_dir=args.artifacts,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        threshold_quantiles=parse_quantiles(args.threshold_quantiles),
        random_state=args.random_state,
        cost_bps=args.cost_bps,
        selection_cost_bps=args.selection_cost_bps,
        threshold_cost_multiplier=args.threshold_cost_multiplier,
        wf_enable=bool(args.wf_enable),
        wf_folds=args.wf_folds,
        wf_expanding=bool(args.wf_expanding),
        wf_step_size=args.wf_step_size,
        selector_use_cost_rule=bool(args.selector_use_cost_rule),
        selector_alpha_low=args.selector_alpha_low,
        selector_alpha_high=args.selector_alpha_high,
        selector_risk_multiple=args.selector_risk_multiple,
        selector_max_interval_width=args.selector_max_interval_width,
    )

    logging.info("Training finished. Artifacts: %s", args.artifacts)
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
