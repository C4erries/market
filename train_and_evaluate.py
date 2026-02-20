from __future__ import annotations

import argparse
import json
import logging

from ml_pipeline.model_pipeline import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate X5 next-day ML models")
    parser.add_argument("--dataset", required=True, help="Path to model-ready dataset (.parquet/.csv)")
    parser.add_argument("--artifacts", default="./artifacts/ml", help="Output directory for models and reports")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--threshold-quantiles", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cost-bps", type=float, default=0.0)
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
    )

    logging.info("Training finished. Artifacts: %s", args.artifacts)
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
