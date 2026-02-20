from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


DATE_CANDIDATES = ("date", "ts", "datetime", "time")


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    return None


def _detect_date_col(columns: list[str]) -> Optional[str]:
    normalized = {col.lower(): col for col in columns}
    for candidate in DATE_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _summarize(path: Path) -> dict:
    frame = _read_table(path)
    if frame is None:
        return {"path": str(path), "status": "missing_or_unsupported"}

    date_col = _detect_date_col(list(frame.columns))
    if date_col is None:
        return {
            "path": str(path),
            "status": "ok_no_date_col",
            "rows": int(len(frame)),
            "date_col": None,
            "date_min": None,
            "date_max": None,
        }

    parsed = pd.to_datetime(frame[date_col], utc=True, errors="coerce").dropna()
    return {
        "path": str(path),
        "status": "ok",
        "rows": int(len(frame)),
        "date_col": date_col,
        "date_min": str(parsed.min()) if not parsed.empty else None,
        "date_max": str(parsed.max()) if not parsed.empty else None,
        "date_non_null": int(parsed.shape[0]),
    }


def _print_summary(name: str, summary: dict) -> None:
    print(f"{name}:")
    print(f"  path: {summary['path']}")
    print(f"  status: {summary['status']}")
    if summary["status"].startswith("ok"):
        print(f"  rows: {summary.get('rows')}")
        print(f"  date_col: {summary.get('date_col')}")
        print(f"  date_min: {summary.get('date_min')}")
        print(f"  date_max: {summary.get('date_max')}")
        if "date_non_null" in summary:
            print(f"  date_non_null: {summary.get('date_non_null')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ML raw/model-ready date ranges")
    parser.add_argument("--raw-x5", default="./data/candles_x5_1d.parquet")
    parser.add_argument("--raw-imoex", default="./data/candles_imoex_1d.parquet")
    parser.add_argument("--raw-usdrub", default="./data/candles_usdrub_1d.parquet")
    parser.add_argument("--dataset", default="./data/model_ready/x5_next_day.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sources = {
        "raw_x5": Path(args.raw_x5),
        "raw_imoex": Path(args.raw_imoex),
        "raw_usdrub": Path(args.raw_usdrub),
        "model_ready": Path(args.dataset),
    }

    summaries = {name: _summarize(path) for name, path in sources.items()}
    for name, summary in summaries.items():
        _print_summary(name, summary)

    print("\nDerived diagnostics:")
    ok_raw = [summaries[key] for key in ("raw_x5", "raw_imoex", "raw_usdrub") if summaries[key]["status"] == "ok"]
    if len(ok_raw) == 3:
        raw_start = max(item["date_min"] for item in ok_raw if item["date_min"] is not None)
        raw_end = min(item["date_max"] for item in ok_raw if item["date_max"] is not None)
        print(f"  common_raw_window_start: {raw_start}")
        print(f"  common_raw_window_end:   {raw_end}")
    else:
        print("  common_raw_window: unavailable (some raw files missing/invalid)")

    model = summaries["model_ready"]
    if model["status"] == "ok":
        print(f"  model_ready_start: {model['date_min']}")
        print(f"  model_ready_end:   {model['date_max']}")
    else:
        print("  model_ready_window: unavailable")


if __name__ == "__main__":
    main()
