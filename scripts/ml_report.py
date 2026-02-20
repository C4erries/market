from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show compact ML quality report from artifacts")
    parser.add_argument("--artifacts", default="./artifacts/ml", help="Artifacts directory from training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = Path(args.artifacts)
    run_report_path = artifacts / "run_report.json"
    quality_path = artifacts / "reports" / "model_quality.csv"

    if not run_report_path.exists():
        raise FileNotFoundError(f"run_report.json not found in {artifacts}")

    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    print("Data:")
    print(
        f"  rows={run_report['data']['rows_total']} "
        f"train={run_report['data']['train_rows']} "
        f"val={run_report['data']['val_rows']} "
        f"test={run_report['data']['test_rows']}"
    )
    print(
        f"  range: {run_report['data']['train_start']} -> {run_report['data']['test_end']}"
    )
    print("Saved models:")
    for model_name in run_report.get("saved_models", []):
        print(f"  - {model_name}")

    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        print("\nModel quality table (reports/model_quality.csv):")
        print(quality.to_string(index=False))
        return

    print("\nNo model_quality.csv found; showing raw metrics:")
    print(json.dumps(run_report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
