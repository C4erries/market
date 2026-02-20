from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def deduplicate_and_sort(
    frame: pd.DataFrame,
    key_cols: Iterable[str],
    sort_cols: Iterable[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.drop_duplicates(subset=list(key_cols), keep="last")
    result = result.sort_values(list(sort_cols)).reset_index(drop=True)
    return result


def read_parquet_if_exists(
    path: Path,
    columns: Optional[list[str]] = None,
    filters: Optional[list[tuple[str, str, object]]] = None,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_parquet(path, columns=columns, filters=filters)
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame(columns=columns)


def read_existing_keys(
    path: Path,
    key_cols: list[str],
    filters: Optional[list[tuple[str, str, object]]] = None,
) -> set[tuple]:
    frame = read_parquet_if_exists(path, columns=key_cols, filters=filters)
    if frame.empty:
        return set()
    return set(frame.itertuples(index=False, name=None))


def filter_new_rows_by_keys(frame: pd.DataFrame, key_cols: list[str], existing_keys: set[tuple]) -> pd.DataFrame:
    if frame.empty or not existing_keys:
        return frame
    row_keys = frame[key_cols].itertuples(index=False, name=None)
    mask = [key not in existing_keys for key in row_keys]
    return frame.loc[mask].reset_index(drop=True)


def get_max_timestamp(
    path: Path,
    ts_col: str,
    filters: Optional[list[tuple[str, str, object]]] = None,
):
    frame = read_parquet_if_exists(path, columns=[ts_col], filters=filters)
    if frame.empty:
        return None
    max_ts = pd.to_datetime(frame[ts_col], utc=True).max()
    if pd.isna(max_ts):
        return None
    return max_ts


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    _ensure_parent(path)
    frame.to_parquet(path, index=False, engine="pyarrow")


def write_parquet_partitioned(frame: pd.DataFrame, path: Path, partition_cols: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        partition_cols=partition_cols,
    )
