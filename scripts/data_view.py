from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ml_pipeline.visualization import (
    plot_context_normalized,
    plot_missing_ratio,
    plot_price_and_volume,
    plot_target_distribution,
    plot_top_feature_correlations,
    prepare_frame_with_date,
    read_table,
    summarize_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data viewer: summaries and plots for raw/model-ready datasets")
    parser.add_argument("--main", default=None, help="Path to main symbol candles (1d)")
    parser.add_argument("--x5", default=None, help="Deprecated alias for --main")
    parser.add_argument("--imoex", default=None, help="Optional path to IMOEX candles (1d)")
    parser.add_argument("--usdrub", default=None, help="Optional path to USDRUB candles (1d)")
    parser.add_argument("--dataset", default=None, help="Optional path to model-ready dataset")
    parser.add_argument("--out", default="./artifacts/data_view", help="Output directory for summaries and plots")
    parser.add_argument("--head", type=int, default=5, help="How many rows to print from each table")
    return parser.parse_args()


def _read_optional(path: Optional[str]) -> Optional[object]:
    if not path:
        return None
    table_path = Path(path)
    if not table_path.exists():
        return None
    return read_table(table_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.main or args.x5
    if not main_path:
        raise ValueError("Missing required input: provide --main (or legacy --x5).")

    main_frame = prepare_frame_with_date(read_table(main_path))
    imoex_raw = _read_optional(args.imoex)
    usdrub_raw = _read_optional(args.usdrub)
    dataset_raw = _read_optional(args.dataset)

    imoex_frame = prepare_frame_with_date(imoex_raw) if imoex_raw is not None else None
    usdrub_frame = prepare_frame_with_date(usdrub_raw) if usdrub_raw is not None else None
    dataset_frame = prepare_frame_with_date(dataset_raw) if dataset_raw is not None else None

    summaries = [summarize_frame(main_frame, name="main")]
    if imoex_frame is not None:
        summaries.append(summarize_frame(imoex_frame, name="imoex"))
    if usdrub_frame is not None:
        summaries.append(summarize_frame(usdrub_frame, name="usdrub"))
    if dataset_frame is not None:
        summaries.append(summarize_frame(dataset_frame, name="model_ready"))

    summary_path = output_dir / "data_summary.json"
    summary_path.write_text(json.dumps({"tables": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Table summaries:")
    for item in summaries:
        print(
            f"  {item['name']}: rows={item['rows']} cols={item['columns']} "
            f"range={item['date_start']} -> {item['date_end']}"
        )

    print("\nHead previews:")
    print("main:")
    print(main_frame.head(args.head).to_string(index=False))
    if imoex_frame is not None:
        print("\nimoex:")
        print(imoex_frame.head(args.head).to_string(index=False))
    if usdrub_frame is not None:
        print("\nusdrub:")
        print(usdrub_frame.head(args.head).to_string(index=False))
    if dataset_frame is not None:
        print("\nmodel_ready:")
        print(dataset_frame.head(args.head).to_string(index=False))

    plot_price_and_volume(
        main_frame,
        output_dir / "main_price_volume.png",
        title="Main Symbol: Close and Volume",
    )
    plot_context_normalized(
        main_frame,
        output_dir / "context_normalized.png",
        imoex_frame=imoex_frame,
        usdrub_frame=usdrub_frame,
    )

    if dataset_frame is not None:
        plot_target_distribution(dataset_frame, output_dir / "target_distribution.png", target_col="y")
        plot_missing_ratio(dataset_frame, output_dir / "missing_ratio.png")
        plot_top_feature_correlations(dataset_frame, output_dir / "top_feature_correlations.png", target_col="y")

    print(f"\nSaved data view artifacts to: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
