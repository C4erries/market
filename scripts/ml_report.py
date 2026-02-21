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

    sanity = run_report.get("sanity_checks")
    if isinstance(sanity, dict) and sanity:
        print("Sanity checks:")
        print(
            "  pred_std(test)={:.6f} y_std(test)={:.6f} ratio={:.4f} collapse_warning={}".format(
                float(sanity.get("pred_main_std_test", 0.0)),
                float(sanity.get("y_std_test", 0.0)),
                float(sanity.get("pred_to_y_std_ratio_test", 0.0)),
                bool(sanity.get("pred_collapse_warning", False)),
            )
        )

    walk_forward = run_report.get("walk_forward")
    if isinstance(walk_forward, dict) and walk_forward.get("enabled"):
        summary = walk_forward.get("summary", {})
        sharpe_stats = summary.get("strategy_main_test_sharpe", {})
        ic_stats = summary.get("lgbm_test_ic_spearman", {})
        print("Walk-forward:")
        print(
            "  folds={} median_sharpe={:.4f} sharpe_std={:.4f} median_ic={:.4f} ic_std={:.4f}".format(
                int(walk_forward.get("actual_folds", 0)),
                float(sharpe_stats.get("median", 0.0)),
                float(sharpe_stats.get("std", 0.0)),
                float(ic_stats.get("median", 0.0)),
                float(ic_stats.get("std", 0.0)),
            )
        )

    selector = run_report.get("selector_config")
    if isinstance(selector, dict) and selector:
        print("Selector config:")
        print(
            "  alpha_low={} alpha_high={} thr_min={:.6f} signal_rate={:.3f}".format(
                selector.get("alpha_low"),
                selector.get("alpha_high"),
                float(selector.get("thr_min", 0.0)),
                float(selector.get("signal_rate_test", 0.0)),
            )
        )

    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        print("\nModel quality table (reports/model_quality.csv):")
        print(quality.to_string(index=False))
        return

    print("\nNo model_quality.csv found; showing raw metrics:")
    print(json.dumps(run_report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
