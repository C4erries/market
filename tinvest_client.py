from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd
from t_tech.invest import CandleInterval
from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from t_tech.invest.retrying.settings import RetryClientSettings
from t_tech.invest.retrying.sync.client import RetryingClient
from t_tech.invest.schemas import InstrumentIdType
from t_tech.invest.utils import now


INTERVAL_MAP = {
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
}

TINVEST_ENV_VAR = "TINVEST_ENV"
TINVEST_SANDBOX_TOKEN_VAR = "TINVEST_SANDBOX_TOKEN"
LEGACY_TOKEN_VARS = ("INVEST_TOKEN", "TINVEST_TOKEN", "TOKEN")


def enforce_sandbox_environment() -> None:
    tinvest_env = os.getenv(TINVEST_ENV_VAR, "sandbox").strip().lower()
    if tinvest_env != "sandbox":
        raise RuntimeError(
            f"{TINVEST_ENV_VAR} must be 'sandbox', got '{tinvest_env}'. "
            "This project does not support production mode."
        )


def read_sandbox_token_from_env() -> str:
    sandbox_token = os.getenv(TINVEST_SANDBOX_TOKEN_VAR)
    if sandbox_token and sandbox_token.strip():
        return sandbox_token.strip()

    legacy_vars_found = [name for name in LEGACY_TOKEN_VARS if os.getenv(name)]
    if legacy_vars_found:
        raise RuntimeError(
            f"Use {TINVEST_SANDBOX_TOKEN_VAR} only. Found legacy token env vars: "
            f"{', '.join(legacy_vars_found)}."
        )

    raise RuntimeError(
        f"Missing required env var {TINVEST_SANDBOX_TOKEN_VAR}. "
        "Production tokens are not supported in this repository."
    )


def require_sandbox_target(target: str) -> str:
    if target == INVEST_GRPC_API:
        raise RuntimeError("Production target is forbidden for this data collector.")
    if target != INVEST_GRPC_API_SANDBOX:
        raise RuntimeError(
            f"Invalid target '{target}'. Only INVEST_GRPC_API_SANDBOX is allowed."
        )
    assert target == INVEST_GRPC_API_SANDBOX
    return target


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _quotation_to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    units = getattr(value, "units", None)
    nano = getattr(value, "nano", None)
    if units is None or nano is None:
        return None
    return float(units) + float(nano) / 1_000_000_000


def _enum_name(value: object) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "name"):
        return value.name
    return str(value)


@dataclass(frozen=True)
class InstrumentMeta:
    requested_symbol: str
    ticker: str
    instrument_id: str
    figi: str
    class_code: str
    exchange: str
    currency: str
    lot: int
    min_price_increment: Optional[float]
    name: str
    first_1day_candle_date: Optional[datetime]
    first_1min_candle_date: Optional[datetime]


class TInvestClient:
    def __init__(self, max_retry_attempt: int = 5):
        enforce_sandbox_environment()
        self._token = read_sandbox_token_from_env()
        self._target = require_sandbox_target(INVEST_GRPC_API_SANDBOX)
        settings = RetryClientSettings(use_retry=True, max_retry_attempt=max_retry_attempt)
        self._client = RetryingClient(self._token, settings=settings, target=self._target)
        if getattr(self._client, "_target", self._target) != INVEST_GRPC_API_SANDBOX:
            raise RuntimeError("Client target guard failed: non-sandbox target detected.")
        self._services = None

    def __enter__(self) -> "TInvestClient":
        self._services = self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self._client.__exit__(exc_type, exc_val, exc_tb)

    @property
    def services(self):
        if self._services is None:
            raise RuntimeError("Client is not initialized, use context manager")
        return self._services

    def resolve_instrument(
        self,
        symbol: str,
        prefer_class_codes: Optional[Iterable[str]] = None,
    ) -> InstrumentMeta:
        symbol_upper = symbol.strip().upper()
        preferred = {code.upper() for code in (prefer_class_codes or [])}
        found = self.services.instruments.find_instrument(query=symbol_upper).instruments
        if not found:
            raise ValueError(f"Instrument not found for symbol/query: {symbol_upper}")

        def score(candidate) -> int:
            candidate_ticker = (candidate.ticker or "").upper()
            candidate_name = (candidate.name or "").upper()
            candidate_class_code = (candidate.class_code or "").upper()
            result = 0
            if candidate_ticker == symbol_upper:
                result += 100
            elif candidate_ticker.startswith(symbol_upper):
                result += 30
            if symbol_upper in candidate_name:
                result += 20
            if candidate.api_trade_available_flag:
                result += 10
            if preferred and candidate_class_code in preferred:
                result += 25
            return result

        best = max(found, key=score)
        instrument = self.services.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_UID,
            id=best.uid,
        ).instrument

        return InstrumentMeta(
            requested_symbol=symbol_upper,
            ticker=instrument.ticker,
            instrument_id=instrument.uid,
            figi=instrument.figi,
            class_code=instrument.class_code,
            exchange=instrument.exchange,
            currency=instrument.currency,
            lot=instrument.lot,
            min_price_increment=_quotation_to_float(instrument.min_price_increment),
            name=instrument.name,
            first_1day_candle_date=_as_utc(instrument.first_1day_candle_date),
            first_1min_candle_date=_as_utc(instrument.first_1min_candle_date),
        )

    def load_candles(
        self,
        instrument: InstrumentMeta,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        interval_key = interval.strip().lower()
        if interval_key not in INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")

        rows = []
        for candle in self.services.get_all_candles(
            instrument_id=instrument.instrument_id,
            interval=INTERVAL_MAP[interval_key],
            from_=_as_utc(start),
            to=_as_utc(end),
        ):
            rows.append(
                {
                    "symbol": instrument.requested_symbol,
                    "ticker": instrument.ticker,
                    "figi": instrument.figi,
                    "instrument_id": instrument.instrument_id,
                    "interval": interval_key,
                    "ts": _as_utc(candle.time),
                    "open": _quotation_to_float(candle.open),
                    "high": _quotation_to_float(candle.high),
                    "low": _quotation_to_float(candle.low),
                    "close": _quotation_to_float(candle.close),
                    "volume": candle.volume,
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "ticker",
                    "figi",
                    "instrument_id",
                    "interval",
                    "ts",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

        frame = pd.DataFrame(rows)
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        return frame

    def load_dividends(
        self,
        instrument: InstrumentMeta,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        response = self.services.instruments.get_dividends(
            instrument_id=instrument.instrument_id,
            from_=_as_utc(start),
            to=_as_utc(end),
        )
        rows = []
        for item in response.dividends:
            rows.append(
                {
                    "symbol": instrument.requested_symbol,
                    "ticker": instrument.ticker,
                    "figi": instrument.figi,
                    "instrument_id": instrument.instrument_id,
                    "currency": instrument.currency,
                    "last_buy_date": _as_utc(item.last_buy_date),
                    "record_date": _as_utc(item.record_date),
                    "payment_date": _as_utc(item.payment_date),
                    "declared_date": _as_utc(item.declared_date),
                    "created_at": _as_utc(item.created_at),
                    "dividend_net": _quotation_to_float(item.dividend_net),
                    "close_price": _quotation_to_float(item.close_price),
                    "yield_value": _quotation_to_float(item.yield_value),
                    "regularity": _enum_name(item.regularity),
                    "dividend_type": _enum_name(item.dividend_type),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "ticker",
                    "figi",
                    "instrument_id",
                    "currency",
                    "last_buy_date",
                    "record_date",
                    "payment_date",
                    "declared_date",
                    "created_at",
                    "dividend_net",
                    "close_price",
                    "yield_value",
                    "regularity",
                    "dividend_type",
                ]
            )

        frame = pd.DataFrame(rows)
        for col in ("last_buy_date", "record_date", "payment_date", "declared_date", "created_at"):
            frame[col] = pd.to_datetime(frame[col], utc=True)
        return frame

    def load_trading_schedules(
        self,
        exchange: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        response = self.services.instruments.trading_schedules(
            exchange=exchange,
            from_=_as_utc(start),
            to=_as_utc(end),
        )
        rows = []
        for schedule in response.exchanges:
            for day in schedule.days:
                rows.append(
                    {
                        "exchange": schedule.exchange,
                        "date": _as_utc(day.date),
                        "is_trading_day": day.is_trading_day,
                        "start_time": _as_utc(day.start_time),
                        "end_time": _as_utc(day.end_time),
                        "opening_auction_start_time": _as_utc(day.opening_auction_start_time),
                        "opening_auction_end_time": _as_utc(day.opening_auction_end_time),
                        "closing_auction_start_time": _as_utc(day.closing_auction_start_time),
                        "closing_auction_end_time": _as_utc(day.closing_auction_end_time),
                        "evening_start_time": _as_utc(day.evening_start_time),
                        "evening_end_time": _as_utc(day.evening_end_time),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "exchange",
                    "date",
                    "is_trading_day",
                    "start_time",
                    "end_time",
                    "opening_auction_start_time",
                    "opening_auction_end_time",
                    "closing_auction_start_time",
                    "closing_auction_end_time",
                    "evening_start_time",
                    "evening_end_time",
                ]
            )

        frame = pd.DataFrame(rows)
        for col in (
            "date",
            "start_time",
            "end_time",
            "opening_auction_start_time",
            "opening_auction_end_time",
            "closing_auction_start_time",
            "closing_auction_end_time",
            "evening_start_time",
            "evening_end_time",
        ):
            frame[col] = pd.to_datetime(frame[col], utc=True)
        return frame

    def load_trading_statuses(self, instruments: list[InstrumentMeta]) -> pd.DataFrame:
        if not instruments:
            return pd.DataFrame(columns=["symbol", "instrument_id", "figi", "trading_status", "loaded_at"])

        id_to_symbol = {item.instrument_id: item.requested_symbol for item in instruments}
        figi_to_symbol = {item.figi: item.requested_symbol for item in instruments}
        response = self.services.market_data.get_trading_statuses(
            instrument_ids=[item.instrument_id for item in instruments]
        )
        loaded_at = now()
        rows = []
        for item in response.trading_statuses:
            symbol = id_to_symbol.get(item.instrument_uid) or figi_to_symbol.get(item.figi)
            rows.append(
                {
                    "symbol": symbol,
                    "instrument_id": item.instrument_uid,
                    "figi": item.figi,
                    "trading_status": _enum_name(item.trading_status),
                    "limit_order_available_flag": item.limit_order_available_flag,
                    "market_order_available_flag": item.market_order_available_flag,
                    "api_trade_available_flag": item.api_trade_available_flag,
                    "bestprice_order_available_flag": item.bestprice_order_available_flag,
                    "only_best_price": item.only_best_price,
                    "loaded_at": _as_utc(loaded_at),
                }
            )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["loaded_at"] = pd.to_datetime(frame["loaded_at"], utc=True)
        return frame
