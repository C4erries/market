from __future__ import annotations

import argparse
import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from t_tech.invest.utils import now

from etl.safety_guard import run_startup_safety_checks
from etl.storage import (
    deduplicate_and_sort,
    filter_new_rows_by_keys,
    get_max_timestamp,
    read_existing_keys,
    read_parquet_if_exists,
    write_parquet,
    write_parquet_partitioned,
)
from etl.tinvest_client import InstrumentMeta, TInvestClient


LOGGER = logging.getLogger("download_data")

INTERVAL_ALIASES = {
    "1d": "1d",
    "d": "1d",
    "day": "1d",
    "5m": "5m",
    "5min": "5m",
}

SYMBOL_CLASS_HINTS = {
    "X5": ["TQBR"],
    "IMOEX": ["INDX", "TQBR"],
    "USDRUB": ["CETS"],
}

SYMBOL_QUERY_HINTS = {
    "USDRUB": ["USD000UTSTOM", "USD000000TOD", "USDRUB"],
}

SYMBOL_DISALLOW_CLASS_CODES = {
    "USDRUB": ["SPBFUT"],
}

MIN_REASONABLE_HISTORY_START = datetime(1990, 1, 1, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download T-Invest historical market data")
    parser.add_argument("--symbols", default="X5,IMOEX,USDRUB", help="Comma-separated symbols")
    parser.add_argument("--intervals", default="1d,5m", help="Comma-separated intervals")
    parser.add_argument("--start", default="2018-01-01", help="ISO date or 'max'")
    parser.add_argument("--end", default="now", help="ISO date or 'now'")
    parser.add_argument("--out", default="./data", help="Output directory")
    parser.add_argument("--mode", default="incremental", choices=["full", "incremental"])
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_datetime(value: str) -> datetime:
    normalized = value.strip().lower()
    if normalized == "now":
        return to_utc(now())
    parsed = datetime.fromisoformat(value)
    return to_utc(parsed)


def parse_csv(raw: str) -> list[str]:
    items = []
    seen = set()
    for token in raw.split(","):
        value = token.strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        items.append(value)
    return items


def normalize_intervals(raw: str) -> list[str]:
    result = []
    seen = set()
    for token in raw.split(","):
        key = token.strip().lower()
        if key not in INTERVAL_ALIASES:
            raise ValueError(f"Unsupported interval: {token}")
        interval = INTERVAL_ALIASES[key]
        if interval in seen:
            continue
        seen.add(interval)
        result.append(interval)
    return result


def interval_step(interval: str) -> timedelta:
    if interval == "1d":
        return timedelta(days=1)
    if interval == "5m":
        return timedelta(minutes=5)
    raise ValueError(f"Unsupported interval: {interval}")


def resolve_start_for_instrument(
    start_arg: str,
    interval: str,
    instrument: InstrumentMeta,
    end_dt: datetime,
) -> datetime:
    def valid_history_date(value: datetime | None) -> bool:
        if value is None:
            return False
        if value < MIN_REASONABLE_HISTORY_START:
            return False
        if value >= end_dt:
            return False
        return True

    if start_arg.strip().lower() != "max":
        return parse_datetime(start_arg)

    if interval == "1d":
        if valid_history_date(instrument.first_1day_candle_date):
            return instrument.first_1day_candle_date
        if instrument.first_1day_candle_date is not None:
            LOGGER.warning(
                "Ignore suspicious first_1day_candle_date for %s: %s; using fallback 10 years.",
                instrument.requested_symbol,
                instrument.first_1day_candle_date,
            )
        return end_dt - timedelta(days=3650)

    if interval == "5m":
        two_years_ago = end_dt - timedelta(days=730)
        if valid_history_date(instrument.first_1min_candle_date):
            return max(two_years_ago, instrument.first_1min_candle_date)
        if instrument.first_1min_candle_date is not None:
            LOGGER.warning(
                "Ignore suspicious first_1min_candle_date for %s: %s; using fallback 2 years.",
                instrument.requested_symbol,
                instrument.first_1min_candle_date,
            )
        return two_years_ago

    raise ValueError(f"Unsupported interval: {interval}")


def upsert_table(path: Path, frame: pd.DataFrame, key_cols: list[str], sort_cols: list[str]) -> int:
    if frame.empty:
        return 0
    existing = read_parquet_if_exists(path)
    if existing.empty:
        combined = frame
    else:
        combined = pd.concat([existing, frame], ignore_index=True)
    combined = deduplicate_and_sort(combined, key_cols=key_cols, sort_cols=sort_cols)
    write_parquet(path, combined)
    return len(frame)


def prepare_full_mode(out_dir: Path, symbols: list[str]) -> None:
    if not symbols:
        return

    candles_root = out_dir / "candles"
    corporate_actions_root = out_dir / "corporate_actions"

    for symbol in symbols:
        symbol_dir = candles_root / f"symbol={symbol}"
        if symbol_dir.exists():
            shutil.rmtree(symbol_dir, ignore_errors=True)

    if "X5" in symbols:
        dividends_path = corporate_actions_root / "dividends_symbol=X5.parquet"
        if dividends_path.exists():
            dividends_path.unlink()


def main() -> None:
    args = parse_args()
    run_startup_safety_checks()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_dir = Path(args.out)
    candles_path = out_dir / "candles"
    calendars_dir = out_dir / "calendars"
    corporate_actions_dir = out_dir / "corporate_actions"
    out_dir.mkdir(parents=True, exist_ok=True)
    calendars_dir.mkdir(parents=True, exist_ok=True)
    corporate_actions_dir.mkdir(parents=True, exist_ok=True)

    symbols = parse_csv(args.symbols)
    intervals = normalize_intervals(args.intervals)
    end_dt = parse_datetime(args.end)
    if args.mode == "full":
        prepare_full_mode(out_dir, symbols=symbols)

    LOGGER.info("Starting download for symbols=%s intervals=%s mode=%s", symbols, intervals, args.mode)
    LOGGER.info("Output directory: %s", out_dir.resolve())

    with TInvestClient(max_retry_attempt=args.max_retries) as api:
        resolved = {}
        instrument_rows = []

        for symbol in symbols:
            instrument = api.resolve_instrument(
                symbol=symbol,
                prefer_class_codes=SYMBOL_CLASS_HINTS.get(symbol),
                query_candidates=SYMBOL_QUERY_HINTS.get(symbol),
                disallow_class_codes=SYMBOL_DISALLOW_CLASS_CODES.get(symbol),
            )
            resolved[symbol] = instrument
            instrument_rows.append(
                {
                    "symbol": instrument.requested_symbol,
                    "ticker": instrument.ticker,
                    "instrument_id": instrument.instrument_id,
                    "figi": instrument.figi,
                    "class_code": instrument.class_code,
                    "exchange": instrument.exchange,
                    "currency": instrument.currency,
                    "lot": instrument.lot,
                    "min_price_increment": instrument.min_price_increment,
                    "name": instrument.name,
                    "first_1day_candle_date": instrument.first_1day_candle_date,
                    "first_1min_candle_date": instrument.first_1min_candle_date,
                    "loaded_at": to_utc(now()),
                }
            )
            LOGGER.info(
                "Resolved symbol=%s ticker=%s instrument_id=%s class_code=%s exchange=%s",
                instrument.requested_symbol,
                instrument.ticker,
                instrument.instrument_id,
                instrument.class_code,
                instrument.exchange,
            )

        instruments_df = pd.DataFrame(instrument_rows)
        upsert_table(
            path=out_dir / "instruments.parquet",
            frame=instruments_df,
            key_cols=["symbol"],
            sort_cols=["symbol"],
        )

        total_candles = 0
        for symbol in symbols:
            instrument = resolved[symbol]
            for interval in intervals:
                start_dt = resolve_start_for_instrument(args.start, interval, instrument, end_dt)
                if args.mode == "incremental":
                    existing_max = get_max_timestamp(
                        candles_path,
                        ts_col="ts",
                        filters=[("symbol", "==", instrument.requested_symbol), ("interval", "==", interval)],
                    )
                    if existing_max is not None:
                        start_dt = max(start_dt, existing_max.to_pydatetime() + interval_step(interval))

                if start_dt >= end_dt:
                    LOGGER.info(
                        "Skip candles symbol=%s interval=%s (start=%s >= end=%s)",
                        instrument.requested_symbol,
                        interval,
                        start_dt,
                        end_dt,
                    )
                    continue

                LOGGER.info(
                    "Loading candles symbol=%s interval=%s start=%s end=%s",
                    instrument.requested_symbol,
                    interval,
                    start_dt,
                    end_dt,
                )
                candles_started_at = time.perf_counter()
                frame = api.load_candles(
                    instrument=instrument,
                    interval=interval,
                    start=start_dt,
                    end=end_dt,
                )
                LOGGER.info(
                    "Loaded candles symbol=%s interval=%s raw_rows=%d elapsed=%.2fs",
                    instrument.requested_symbol,
                    interval,
                    len(frame),
                    time.perf_counter() - candles_started_at,
                )
                if frame.empty:
                    LOGGER.info("No candles returned symbol=%s interval=%s", instrument.requested_symbol, interval)
                    continue

                frame["year"] = frame["ts"].dt.year.astype("int32")
                frame = deduplicate_and_sort(
                    frame,
                    key_cols=["symbol", "interval", "ts"],
                    sort_cols=["symbol", "interval", "ts"],
                )
                existing_keys = read_existing_keys(
                    candles_path,
                    key_cols=["symbol", "interval", "ts"],
                    filters=[("symbol", "==", instrument.requested_symbol), ("interval", "==", interval)],
                )
                frame = filter_new_rows_by_keys(
                    frame,
                    key_cols=["symbol", "interval", "ts"],
                    existing_keys=existing_keys,
                )

                if frame.empty:
                    LOGGER.info("All candles already exist symbol=%s interval=%s", instrument.requested_symbol, interval)
                    continue

                write_parquet_partitioned(
                    frame=frame,
                    path=candles_path,
                    partition_cols=["symbol", "interval", "year"],
                )
                total_candles += len(frame)
                LOGGER.info(
                    "Saved candles symbol=%s interval=%s rows=%d from=%s to=%s",
                    instrument.requested_symbol,
                    interval,
                    len(frame),
                    frame["ts"].min(),
                    frame["ts"].max(),
                )

        LOGGER.info("Saved candle rows total=%d", total_candles)

        if "X5" in resolved:
            x5 = resolved["X5"]
            dividends_path = corporate_actions_dir / "dividends_symbol=X5.parquet"
            dividends_start = resolve_start_for_instrument(args.start, "1d", x5, end_dt)
            if args.mode == "incremental":
                max_dividend_date = get_max_timestamp(dividends_path, ts_col="record_date")
                if max_dividend_date is not None:
                    dividends_start = max(dividends_start, max_dividend_date.to_pydatetime() + timedelta(days=1))

            LOGGER.info("Loading dividends symbol=%s start=%s end=%s", x5.requested_symbol, dividends_start, end_dt)
            dividends_started_at = time.perf_counter()
            dividends = api.load_dividends(x5, start=dividends_start, end=end_dt)
            LOGGER.info(
                "Loaded dividends symbol=%s raw_rows=%d elapsed=%.2fs",
                x5.requested_symbol,
                len(dividends),
                time.perf_counter() - dividends_started_at,
            )
            written = upsert_table(
                path=dividends_path,
                frame=dividends,
                key_cols=["symbol", "record_date", "payment_date", "last_buy_date", "dividend_net"],
                sort_cols=["symbol", "record_date", "payment_date"],
            )
            LOGGER.info("Saved dividends rows=%d path=%s", written, dividends_path)

        exchanges = sorted({item.exchange for item in resolved.values() if item.exchange})
        schedules_path = calendars_dir / "trading_schedules.parquet"
        schedule_frames = []
        if resolved:
            earliest_start = min(
                resolve_start_for_instrument(args.start, "1d", instrument, end_dt)
                for instrument in resolved.values()
            )
        else:
            earliest_start = end_dt - timedelta(days=30)

        for exchange in exchanges:
            schedule_start = earliest_start
            if args.mode == "incremental":
                schedule_max = get_max_timestamp(
                    schedules_path,
                    ts_col="date",
                    filters=[("exchange", "==", exchange)],
                )
                if schedule_max is not None:
                    schedule_start = max(schedule_start, schedule_max.to_pydatetime() + timedelta(days=1))

            if schedule_start >= end_dt:
                LOGGER.info(
                    "Skip trading schedules exchange=%s (start=%s >= end=%s)",
                    exchange,
                    schedule_start,
                    end_dt,
                )
                continue
            LOGGER.info(
                "Loading trading schedules exchange=%s start=%s end=%s",
                exchange,
                schedule_start,
                end_dt,
            )
            schedule_started_at = time.perf_counter()
            schedule_frame = api.load_trading_schedules(exchange, start=schedule_start, end=end_dt)
            LOGGER.info(
                "Loaded trading schedules exchange=%s rows=%d elapsed=%.2fs",
                exchange,
                len(schedule_frame),
                time.perf_counter() - schedule_started_at,
            )
            if not schedule_frame.empty:
                schedule_frames.append(schedule_frame)

        if schedule_frames:
            schedules = pd.concat(schedule_frames, ignore_index=True)
            written = upsert_table(
                path=schedules_path,
                frame=schedules,
                key_cols=["exchange", "date"],
                sort_cols=["exchange", "date"],
            )
            LOGGER.info("Saved trading schedule rows=%d", written)
        else:
            LOGGER.info("No trading schedules to save.")

        LOGGER.info("Loading trading statuses for %d instruments", len(resolved))
        statuses_started_at = time.perf_counter()
        statuses = api.load_trading_statuses(list(resolved.values()))
        LOGGER.info(
            "Loaded trading statuses rows=%d elapsed=%.2fs",
            len(statuses),
            time.perf_counter() - statuses_started_at,
        )
        if not statuses.empty:
            written = upsert_table(
                path=calendars_dir / "trading_statuses.parquet",
                frame=statuses,
                key_cols=["symbol", "instrument_id", "loaded_at"],
                sort_cols=["symbol", "loaded_at"],
            )
            LOGGER.info("Saved trading statuses rows=%d", written)
        else:
            LOGGER.info("No trading statuses to save.")

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
