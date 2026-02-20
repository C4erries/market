from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DATE_CANDIDATES = ("date", "ts", "datetime", "time")
OHLCV_FIELDS = ("open", "high", "low", "close", "volume")
DIVIDEND_DATE_CANDIDATES = ("record_date", "last_buy_date", "payment_date", "declared_date", "date", "ts")
DIVIDEND_AMOUNT_CANDIDATES = ("dividend_net", "amount", "value")


@dataclass(frozen=True)
class TimeSplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _read_local_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Input file not found: {table_path}")
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(table_path)
    raise ValueError(f"Unsupported file format: {table_path}")


def _find_column_case_insensitive(columns: Iterable[str], wanted_names: Iterable[str]) -> Optional[str]:
    normalized = {col.lower(): col for col in columns}
    for wanted in wanted_names:
        if wanted.lower() in normalized:
            return normalized[wanted.lower()]
    return None


def _normalize_date_column(frame: pd.DataFrame) -> pd.Series:
    date_col = _find_column_case_insensitive(frame.columns, DATE_CANDIDATES)
    if date_col is None:
        raise ValueError(f"Could not detect date column. Expected one of: {DATE_CANDIDATES}")
    parsed = pd.to_datetime(frame[date_col], utc=True, errors="coerce")
    if parsed.isna().all():
        raise ValueError(f"Failed to parse date column '{date_col}'")
    return parsed.dt.normalize().dt.tz_localize(None)


def _prepare_market_frame(frame: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    result = pd.DataFrame()
    result["date"] = _normalize_date_column(frame)

    for field in OHLCV_FIELDS:
        source_col = _find_column_case_insensitive(frame.columns, (field,))
        if source_col is None and suffix == "":
            raise ValueError(f"Missing required market column: {field}")
        if source_col is None:
            continue
        target_col = field if suffix == "" else f"{field}_{suffix}"
        result[target_col] = pd.to_numeric(frame[source_col], errors="coerce")

    result = result.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return result.sort_values("date").reset_index(drop=True)


def _prepare_calendar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame()
    calendar["date"] = _normalize_date_column(frame)
    flag_col = _find_column_case_insensitive(frame.columns, ("is_trading_day",))
    if flag_col is None:
        calendar["is_trading_day"] = True
    else:
        calendar["is_trading_day"] = frame[flag_col].astype(bool)
    calendar = calendar.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return calendar.sort_values("date").reset_index(drop=True)


def _prepare_dividends_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dividend_date_col = _find_column_case_insensitive(frame.columns, DIVIDEND_DATE_CANDIDATES)
    if dividend_date_col is None:
        raise ValueError(
            "Could not detect dividends date column. "
            f"Expected one of: {DIVIDEND_DATE_CANDIDATES}"
        )

    dividends = pd.DataFrame()
    dividends["date"] = pd.to_datetime(frame[dividend_date_col], utc=True, errors="coerce").dt.normalize().dt.tz_localize(None)
    amount_col = _find_column_case_insensitive(frame.columns, DIVIDEND_AMOUNT_CANDIDATES)
    if amount_col is None:
        dividends["dividend_amount_t"] = 1.0
    else:
        dividends["dividend_amount_t"] = pd.to_numeric(frame[amount_col], errors="coerce").fillna(0.0)
        dividends.loc[dividends["dividend_amount_t"] == 0.0, "dividend_amount_t"] = 1.0

    dividends = dividends.dropna(subset=["date"]).groupby("date", as_index=False)["dividend_amount_t"].sum()
    dividends["dividend_flag_t"] = (dividends["dividend_amount_t"] > 0).astype(int)
    return dividends


def load_data(
    x5_path: str | Path,
    *,
    imoex_path: str | Path | None = None,
    usdrub_path: str | Path | None = None,
    calendar_path: str | Path | None = None,
    dividends_path: str | Path | None = None,
) -> pd.DataFrame:
    x5_frame = _prepare_market_frame(_read_local_table(x5_path))
    result = x5_frame.copy()

    if imoex_path:
        imoex = _prepare_market_frame(_read_local_table(imoex_path), suffix="imoex")
        result = result.merge(imoex, on="date", how="left")

    if usdrub_path:
        usdrub = _prepare_market_frame(_read_local_table(usdrub_path), suffix="usdrub")
        result = result.merge(usdrub, on="date", how="left")

    if calendar_path:
        calendar = _prepare_calendar_frame(_read_local_table(calendar_path))
        result = result.merge(calendar, on="date", how="left")
        result["is_trading_day"] = result["is_trading_day"].fillna(False)
        result = result[result["is_trading_day"]].drop(columns=["is_trading_day"])

    if dividends_path:
        dividends = _prepare_dividends_frame(_read_local_table(dividends_path))
        result = result.merge(dividends, on="date", how="left")
    else:
        result["dividend_amount_t"] = np.nan
        result["dividend_flag_t"] = 0

    result["dividend_amount_t"] = pd.to_numeric(result["dividend_amount_t"], errors="coerce").fillna(0.0)
    result["dividend_flag_t"] = result["dividend_flag_t"].fillna(0).astype(int)
    result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return result


def make_features(df: pd.DataFrame, *, include_dividend_t_plus_1: bool = False) -> pd.DataFrame:
    features = df.copy().sort_values("date").reset_index(drop=True)
    close = features["close"].replace(0, np.nan)
    open_price = features["open"].replace(0, np.nan)

    features["ret_1"] = close.pct_change(1)
    features["ret_2"] = close.pct_change(2)
    features["ret_5"] = close.pct_change(5)
    features["ret_10"] = close.pct_change(10)

    for window in (5, 10, 20):
        features[f"volatility_{window}"] = features["ret_1"].rolling(window).std()
        features[f"momentum_{window}"] = features["ret_1"].rolling(window).mean()

    features["hl_range"] = (features["high"] - features["low"]) / close
    features["co_move"] = (features["close"] - features["open"]) / open_price

    features["log_volume"] = np.log(features["volume"].clip(lower=1.0))
    features["log_volume_mean_5"] = features["log_volume"].rolling(5).mean()
    features["log_volume_mean_20"] = features["log_volume"].rolling(20).mean()
    volume_std_20 = features["log_volume"].rolling(20).std()
    features["volume_zscore_20"] = (features["log_volume"] - features["log_volume_mean_20"]) / volume_std_20

    if "close_imoex" in features.columns:
        features["ret_imoex_1"] = features["close_imoex"].replace(0, np.nan).pct_change(1)
        features["ret_imoex_mom_5"] = features["ret_imoex_1"].rolling(5).mean()
        features["ret_imoex_vol_20"] = features["ret_imoex_1"].rolling(20).std()

    if "close_usdrub" in features.columns:
        features["ret_usdrub_1"] = features["close_usdrub"].replace(0, np.nan).pct_change(1)
        features["ret_usdrub_mom_5"] = features["ret_usdrub_1"].rolling(5).mean()
        features["ret_usdrub_vol_20"] = features["ret_usdrub_1"].rolling(20).std()

    features["day_of_week"] = features["date"].dt.dayofweek.astype(int)
    features["month"] = features["date"].dt.month.astype(int)
    features["day_of_month"] = features["date"].dt.day.astype(int)
    features["is_month_end"] = features["date"].dt.is_month_end.astype(int)

    features["dividend_flag_t"] = features.get("dividend_flag_t", 0).fillna(0).astype(int)
    if include_dividend_t_plus_1:
        # Use only if dividend calendar for t+1 is known ex-ante.
        features["dividend_flag_t_plus_1"] = features["dividend_flag_t"].shift(-1).fillna(0).astype(int)

    return features


def make_target(df: pd.DataFrame, *, target_type: str = "log") -> pd.DataFrame:
    result = df.copy().sort_values("date").reset_index(drop=True)
    close = result["close"].replace(0, np.nan)
    if target_type == "log":
        result["y"] = np.log(close.shift(-1) / close)
    elif target_type == "simple":
        result["y"] = close.shift(-1) / close - 1.0
    else:
        raise ValueError("target_type must be one of: log, simple")
    result["y_dir"] = (result["y"] > 0).astype(int)
    return result


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna().sort_values("date").reset_index(drop=True)
    if prepared.empty:
        raise ValueError("Dataset is empty after dropping NaN/inf values.")
    if not prepared["date"].is_monotonic_increasing:
        raise ValueError("Date column must be sorted in ascending order.")
    return prepared


def build_model_ready_dataset(
    x5_path: str | Path,
    *,
    output_path: str | Path,
    imoex_path: str | Path | None = None,
    usdrub_path: str | Path | None = None,
    calendar_path: str | Path | None = None,
    dividends_path: str | Path | None = None,
    target_type: str = "log",
    include_dividend_t_plus_1: bool = False,
) -> pd.DataFrame:
    raw = load_data(
        x5_path=x5_path,
        imoex_path=imoex_path,
        usdrub_path=usdrub_path,
        calendar_path=calendar_path,
        dividends_path=dividends_path,
    )
    featured = make_features(raw, include_dividend_t_plus_1=include_dividend_t_plus_1)
    targeted = make_target(featured, target_type=target_type)
    targeted = targeted.iloc[:-1].copy()  # Last row has no t+1 target by construction.
    dataset = finalize_dataset(targeted)
    save_dataset(dataset, output_path)
    return dataset


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return
    if suffix in {".csv", ".txt"}:
        df.to_csv(output_path, index=False)
        return
    raise ValueError(f"Unsupported output format: {output_path}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    data = _read_local_table(path)
    if "date" not in data.columns:
        data["date"] = _normalize_date_column(data)
    else:
        data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce").dt.normalize().dt.tz_localize(None)
    return data.sort_values("date").reset_index(drop=True)


def time_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> TimeSplitResult:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive")

    ordered = df.sort_values("date").reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows < 30:
        raise ValueError("Dataset is too small for train/val/test split")

    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("Invalid split boundaries.")

    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    return TimeSplitResult(train=train, val=val, test=test)


def walk_forward_splits(
    df: pd.DataFrame,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: Optional[int] = None,
    expanding: bool = True,
) -> list[TimeSplitResult]:
    ordered = df.sort_values("date").reset_index(drop=True)
    n_rows = len(ordered)
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("train_size, val_size and test_size must be positive")

    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    splits: list[TimeSplitResult] = []
    train_start = 0
    train_end = train_size
    while True:
        val_end = train_end + val_size
        test_end = val_end + test_size
        if test_end > n_rows:
            break

        if expanding:
            current_train = ordered.iloc[:train_end].copy()
        else:
            current_train = ordered.iloc[train_start:train_end].copy()
        current_val = ordered.iloc[train_end:val_end].copy()
        current_test = ordered.iloc[val_end:test_end].copy()
        splits.append(TimeSplitResult(train=current_train, val=current_val, test=current_test))

        train_end += step
        if not expanding:
            train_start += step

    return splits
