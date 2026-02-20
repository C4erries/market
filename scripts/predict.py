from __future__ import annotations

import argparse
import json

from ml_pipeline.model_pipeline import predict_latest_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next-day signal from latest row")
    parser.add_argument("--dataset", required=True, help="Path to model-ready dataset (.parquet/.csv)")
    parser.add_argument("--artifacts", required=True, help="Artifacts directory from train_and_evaluate")
    parser.add_argument("--model-name", default="main_lgbm", help="Model file name without .joblib suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_latest_signal(
        dataset_path=args.dataset,
        artifacts_dir=args.artifacts,
        model_name=args.model_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
