from __future__ import annotations

import argparse
import logging

from ml_pipeline.data_pipeline import build_model_ready_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready dataset from local raw files")
    parser.add_argument("--x5", required=True, help="Path to X5 daily candles (Parquet/CSV)")
    parser.add_argument("--imoex", default=None, help="Optional path to IMOEX daily candles")
    parser.add_argument("--usdrub", default=None, help="Optional path to USDRUB daily candles")
    parser.add_argument("--calendar", default=None, help="Optional path to trading calendar")
    parser.add_argument("--dividends", default=None, help="Optional path to dividends table")
    parser.add_argument("--output", required=True, help="Output dataset path (.parquet or .csv)")
    parser.add_argument("--target-type", default="log", choices=["log", "simple"])
    parser.add_argument("--include-dividend-t-plus-1", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset = build_model_ready_dataset(
        x5_path=args.x5,
        output_path=args.output,
        imoex_path=args.imoex,
        usdrub_path=args.usdrub,
        calendar_path=args.calendar,
        dividends_path=args.dividends,
        target_type=args.target_type,
        include_dividend_t_plus_1=args.include_dividend_t_plus_1,
    )

    logging.info("Model-ready dataset saved to %s", args.output)
    logging.info("Rows: %d", len(dataset))
    logging.info("Date range: %s -> %s", dataset["date"].min(), dataset["date"].max())
    logging.info("Columns: %d", len(dataset.columns))


if __name__ == "__main__":
    main()
