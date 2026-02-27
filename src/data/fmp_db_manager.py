"""
fmp_db_manager.py
=================
FMP JSON 데이터 → SQLite DB 저장/조회 매니저.

기존 yfinance용 db_manager.py와 완전 분리.
FmpBase → fmp_stocks.db 전용.

사용법:
    with FmpDatabaseManager() as fmp_db:
        fmp_db.save_tickers_from_profiles()
        fmp_db.save_prices(symbols=["AAPL"])
        prices = fmp_db.load_prices(["AAPL"])
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from src.data.fmp_db_models import (
    FmpBase,
    FmpTicker,
    FmpDailyPrice,
    FmpMarketCap,
    FmpFinancial,
    FmpTreasuryRate,
    FmpUniverseCalendar,
    FINANCIAL_COLUMN_MAP,
    convert_fmp_date,
    parse_symbol_from_filename,
)

logger = logging.getLogger(__name__)

# ── JSON 키 → DB 컬럼 매핑 ──────────────────────────────────────────────────
PRICE_KEY_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "change": "change",
    "changePercent": "change_percent",
    "vwap": "vwap",
}

TREASURY_KEY_MAP = {
    "month1": "month_1",
    "month2": "month_2",
    "month3": "month_3",
    "month6": "month_6",
    "year1": "year_1",
    "year2": "year_2",
    "year3": "year_3",
    "year5": "year_5",
    "year7": "year_7",
    "year10": "year_10",
    "year20": "year_20",
    "year30": "year_30",
}

PROFILE_KEY_MAP = {
    "symbol": "symbol",
    "companyName": "company_name",
    "exchange": "exchange",
    "industry": "industry",
    "sector": "sector",
    "country": "country",
    "currency": "currency",
    "ipoDate": "ipo_date",
    "isActivelyTrading": "is_actively_trading",
    "isEtf": "is_etf",
    "isAdr": "is_adr",
    "isFund": "is_fund",
    "cik": "cik",
    "isin": "isin",
    "cusip": "cusip",
}

# 청크 크기 (대량 INSERT 시 메모리 관리)
CHUNK_SIZE = 10_000


class FmpDatabaseManager:
    """FMP 데이터 전용 데이터베이스 매니저"""

    def __init__(self, db_path: str = "data/database/fmp_stocks.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            pool_pre_ping=True,
        )
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # 테이블 생성
        FmpBase.metadata.create_all(self.engine)
        logger.info(f"FMP Database initialized: {db_path}")

    # ──────────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_json(filepath: str) -> list:
        """JSON 파일을 읽어 list[dict]로 반환."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_or_create_ticker(self, symbol: str) -> int:
        """FmpTicker 조회 또는 생성 후 ticker_id 반환."""
        ticker = self.session.query(FmpTicker).filter_by(symbol=symbol).first()
        if ticker:
            return ticker.ticker_id

        new_ticker = FmpTicker(symbol=symbol)
        self.session.add(new_ticker)
        self.session.commit()
        logger.debug(f"Created FMP ticker: {symbol} (ID: {new_ticker.ticker_id})")
        return new_ticker.ticker_id

    def _bulk_get_or_create_tickers(self, symbols: List[str]) -> Dict[str, int]:
        """여러 심볼의 ticker_id를 일괄 조회/생성."""
        result = {}
        # 기존 조회
        existing = (
            self.session.query(FmpTicker.symbol, FmpTicker.ticker_id)
            .filter(FmpTicker.symbol.in_(symbols))
            .all()
        )
        result = {sym: tid for sym, tid in existing}

        # 없는 심볼 생성
        missing = [s for s in symbols if s not in result]
        if missing:
            new_tickers = [FmpTicker(symbol=s) for s in missing]
            self.session.add_all(new_tickers)
            self.session.commit()
            for t in new_tickers:
                result[t.symbol] = t.ticker_id
            logger.debug(f"Created {len(missing)} new FMP tickers")

        return result

    # ──────────────────────────────────────────────────────────────────────
    # SAVE: Tickers (company_profile / stock-list)
    # ──────────────────────────────────────────────────────────────────────
    def save_tickers_from_profiles(
        self, folder: str = "data/fmp/company_profile"
    ) -> int:
        """company_profile/*.json → FmpTicker UPSERT."""
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Profile folder not found: {folder}")
            return 0

        json_files = sorted(folder_path.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files in {folder}")
            return 0

        saved = 0
        with self.engine.begin() as conn:
            for fpath in json_files:
                try:
                    data = self._load_json(str(fpath))
                    if not data:
                        continue
                    profile = data[0] if isinstance(data, list) else data

                    record = {}
                    for json_key, db_col in PROFILE_KEY_MAP.items():
                        val = profile.get(json_key)
                        if db_col == "ipo_date":
                            val = convert_fmp_date(val)
                        record[db_col] = val

                    if not record.get("symbol"):
                        continue

                    stmt = sqlite_insert(FmpTicker.__table__).values(record)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["symbol"],
                        set_={
                            k: stmt.excluded[k]
                            for k in record
                            if k != "symbol"
                        },
                    )
                    conn.execute(stmt)
                    saved += 1
                except Exception as e:
                    logger.error(f"Failed to save profile {fpath.name}: {e}")

        logger.info(f"Saved {saved} ticker profiles from {folder}")
        return saved

    def save_tickers_from_stock_list(self, filepath: str) -> int:
        """stock-list/{EXCHANGE}.json → FmpTicker UPSERT (간략 정보)."""
        fpath = Path(filepath)
        if not fpath.exists():
            logger.warning(f"Stock list file not found: {filepath}")
            return 0

        data = self._load_json(str(fpath))
        if not data:
            return 0

        saved = 0
        for chunk_start in range(0, len(data), CHUNK_SIZE):
            chunk = data[chunk_start : chunk_start + CHUNK_SIZE]
            records = []
            for item in chunk:
                symbol = item.get("symbol")
                if not symbol:
                    continue
                records.append({
                    "symbol": symbol,
                    "company_name": item.get("companyName"),
                    "exchange": item.get("exchangeShortName") or item.get("exchange"),
                    "sector": item.get("sector"),
                    "industry": item.get("industry"),
                    "country": item.get("country"),
                    "is_etf": item.get("isEtf"),
                    "is_fund": item.get("isFund"),
                    "is_actively_trading": item.get("isActivelyTrading"),
                })

            if not records:
                continue

            with self.engine.begin() as conn:
                stmt = sqlite_insert(FmpTicker.__table__).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol"],
                    set_={
                        "company_name": stmt.excluded.company_name,
                        "exchange": stmt.excluded.exchange,
                        "sector": stmt.excluded.sector,
                        "industry": stmt.excluded.industry,
                        "country": stmt.excluded.country,
                        "is_etf": stmt.excluded.is_etf,
                        "is_fund": stmt.excluded.is_fund,
                        "is_actively_trading": stmt.excluded.is_actively_trading,
                    },
                )
                conn.execute(stmt)
                saved += len(records)

        logger.info(f"Saved {saved} tickers from stock list: {filepath}")
        return saved

    # ──────────────────────────────────────────────────────────────────────
    # SAVE: Prices
    # ──────────────────────────────────────────────────────────────────────
    def save_prices(
        self,
        folder: str = "data/fmp/price/ticker",
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """price/ticker/{SYMBOL}_*.json → FmpDailyPrice UPSERT."""
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Price folder not found: {folder}")
            return {}

        json_files = sorted(folder_path.glob("*.json"))
        if symbols:
            symbols_set = set(symbols)
            json_files = [
                f for f in json_files
                if parse_symbol_from_filename(f.name) in symbols_set
            ]

        results = {}
        # 심볼 목록 수집 후 일괄 ticker_id 생성
        file_symbols = {}
        for fpath in json_files:
            sym = parse_symbol_from_filename(fpath.name)
            file_symbols[fpath] = sym

        ticker_ids = self._bulk_get_or_create_tickers(list(set(file_symbols.values())))

        for fpath, symbol in file_symbols.items():
            try:
                data = self._load_json(str(fpath))
                if not data:
                    results[symbol] = 0
                    continue

                ticker_id = ticker_ids[symbol]
                records = []
                for row in data:
                    dt = convert_fmp_date(row.get("date"))
                    if dt is None:
                        continue
                    rec = {"ticker_id": ticker_id, "date": dt}
                    for json_key, db_col in PRICE_KEY_MAP.items():
                        rec[db_col] = row.get(json_key)
                    records.append(rec)

                if not records:
                    results[symbol] = 0
                    continue

                # 청크 단위 UPSERT
                total = 0
                for i in range(0, len(records), CHUNK_SIZE):
                    chunk = records[i : i + CHUNK_SIZE]
                    with self.engine.begin() as conn:
                        stmt = sqlite_insert(FmpDailyPrice.__table__).values(chunk)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["ticker_id", "date"],
                            set_={
                                db_col: stmt.excluded[db_col]
                                for db_col in PRICE_KEY_MAP.values()
                            },
                        )
                        conn.execute(stmt)
                    total += len(chunk)

                results[symbol] = total
                logger.debug(f"Saved {total} price records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to save prices for {symbol}: {e}")
                results[symbol] = 0

        logger.info(f"Price save completed for {len(results)} tickers")
        return results

    # ──────────────────────────────────────────────────────────────────────
    # SAVE: Market Caps
    # ──────────────────────────────────────────────────────────────────────
    def save_market_caps(
        self,
        folder: str = "data/fmp/market_cap",
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """market_cap/{SYMBOL}_*.json → FmpMarketCap UPSERT."""
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Market cap folder not found: {folder}")
            return {}

        json_files = sorted(folder_path.glob("*.json"))
        if symbols:
            symbols_set = set(symbols)
            json_files = [
                f for f in json_files
                if parse_symbol_from_filename(f.name) in symbols_set
            ]

        file_symbols = {}
        for fpath in json_files:
            sym = parse_symbol_from_filename(fpath.name)
            file_symbols[fpath] = sym

        ticker_ids = self._bulk_get_or_create_tickers(list(set(file_symbols.values())))
        results = {}

        for fpath, symbol in file_symbols.items():
            try:
                data = self._load_json(str(fpath))
                if not data:
                    results[symbol] = 0
                    continue

                ticker_id = ticker_ids[symbol]
                records = []
                for row in data:
                    dt = convert_fmp_date(row.get("date"))
                    if dt is None:
                        continue
                    records.append({
                        "ticker_id": ticker_id,
                        "date": dt,
                        "market_cap": row.get("marketCap"),
                    })

                if not records:
                    results[symbol] = 0
                    continue

                total = 0
                for i in range(0, len(records), CHUNK_SIZE):
                    chunk = records[i : i + CHUNK_SIZE]
                    with self.engine.begin() as conn:
                        stmt = sqlite_insert(FmpMarketCap.__table__).values(chunk)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["ticker_id", "date"],
                            set_={"market_cap": stmt.excluded.market_cap},
                        )
                        conn.execute(stmt)
                    total += len(chunk)

                results[symbol] = total
                logger.debug(f"Saved {total} market cap records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to save market caps for {symbol}: {e}")
                results[symbol] = 0

        logger.info(f"Market cap save completed for {len(results)} tickers")
        return results

    # ──────────────────────────────────────────────────────────────────────
    # SAVE: Financials
    # ──────────────────────────────────────────────────────────────────────
    def save_financials(
        self,
        folder: str = "data/fmp/financial/ticker",
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """financial/ticker/{SYMBOL}/all_financial_*.json → FmpFinancial UPSERT."""
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Financial folder not found: {folder}")
            return {}

        # 각 심볼은 하위 폴더에 존재
        sym_dirs = sorted([d for d in folder_path.iterdir() if d.is_dir()])
        if symbols:
            symbols_set = set(symbols)
            sym_dirs = [d for d in sym_dirs if d.name in symbols_set]

        results = {}
        # 전체 심볼 목록으로 ticker_id 일괄 생성
        all_symbols = [d.name for d in sym_dirs]
        ticker_ids = self._bulk_get_or_create_tickers(all_symbols)

        for sym_dir in sym_dirs:
            symbol = sym_dir.name
            try:
                # all_financial_*.json 파일 찾기
                fin_files = sorted(sym_dir.glob("all_financial_*.json"))
                if not fin_files:
                    results[symbol] = 0
                    continue

                # 가장 최신 파일 사용
                fpath = fin_files[-1]
                data = self._load_json(str(fpath))
                if not data:
                    results[symbol] = 0
                    continue

                ticker_id = ticker_ids[symbol]
                records = []
                for row in data:
                    dt = convert_fmp_date(row.get("date"))
                    if dt is None:
                        continue

                    rec = {
                        "ticker_id": ticker_id,
                        "date": dt,
                        "period": row.get("period"),
                        "fiscal_year": row.get("fiscalYear"),
                        "reported_currency": row.get("reportedCurrency"),
                        "filing_date": convert_fmp_date(row.get("filingDate")),
                        "accepted_date": row.get("acceptedDate"),
                    }

                    # 119개 수치 컬럼 매핑 (camelCase → snake_case)
                    for camel_key, snake_col in FINANCIAL_COLUMN_MAP.items():
                        rec[snake_col] = row.get(camel_key)

                    records.append(rec)

                if not records:
                    results[symbol] = 0
                    continue

                # 수치 컬럼 목록 (UPSERT set_ 용)
                update_cols = (
                    ["period", "fiscal_year", "reported_currency",
                     "filing_date", "accepted_date"]
                    + list(FINANCIAL_COLUMN_MAP.values())
                )

                total = 0
                for i in range(0, len(records), CHUNK_SIZE):
                    chunk = records[i : i + CHUNK_SIZE]
                    with self.engine.begin() as conn:
                        stmt = sqlite_insert(FmpFinancial.__table__).values(chunk)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["ticker_id", "date", "period"],
                            set_={col: stmt.excluded[col] for col in update_cols},
                        )
                        conn.execute(stmt)
                    total += len(chunk)

                results[symbol] = total
                logger.debug(f"Saved {total} financial records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to save financials for {symbol}: {e}")
                results[symbol] = 0

        logger.info(f"Financial save completed for {len(results)} tickers")
        return results

    # ──────────────────────────────────────────────────────────────────────
    # SAVE: Treasury Rates
    # ──────────────────────────────────────────────────────────────────────
    def save_treasury_rates(self, folder: str = "data/fmp/treasury") -> int:
        """treasury/treasury_*.json → FmpTreasuryRate UPSERT."""
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Treasury folder not found: {folder}")
            return 0

        json_files = sorted(folder_path.glob("treasury_*.json"))
        if not json_files:
            logger.warning(f"No treasury JSON files in {folder}")
            return 0

        total = 0
        for fpath in json_files:
            try:
                data = self._load_json(str(fpath))
                if not data:
                    continue

                records = []
                for row in data:
                    dt = convert_fmp_date(row.get("date"))
                    if dt is None:
                        continue
                    rec = {"date": dt}
                    for json_key, db_col in TREASURY_KEY_MAP.items():
                        rec[db_col] = row.get(json_key)
                    records.append(rec)

                if not records:
                    continue

                for i in range(0, len(records), CHUNK_SIZE):
                    chunk = records[i : i + CHUNK_SIZE]
                    with self.engine.begin() as conn:
                        stmt = sqlite_insert(FmpTreasuryRate.__table__).values(chunk)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["date"],
                            set_={
                                db_col: stmt.excluded[db_col]
                                for db_col in TREASURY_KEY_MAP.values()
                            },
                        )
                        conn.execute(stmt)
                    total += len(chunk)

            except Exception as e:
                logger.error(f"Failed to save treasury data from {fpath.name}: {e}")

        logger.info(f"Saved {total} treasury rate records")
        return total

    # ──────────────────────────────────────────────────────────────────────
    # SAVE: Universe Calendar
    # ──────────────────────────────────────────────────────────────────────
    def save_universe_calendar(
        self,
        folder: str = "data/fmp/universe_calendar",
        exchanges: Optional[List[str]] = None,
    ) -> int:
        """universe_calendar/{EXCHANGE}/{EXCHANGE}_{year}.json → FmpUniverseCalendar UPSERT."""
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Universe calendar folder not found: {folder}")
            return 0

        exchange_dirs = sorted([d for d in folder_path.iterdir() if d.is_dir()])
        if exchanges:
            exchanges_set = set(e.upper() for e in exchanges)
            exchange_dirs = [d for d in exchange_dirs if d.name.upper() in exchanges_set]

        total = 0
        for ex_dir in exchange_dirs:
            json_files = sorted(ex_dir.glob("*.json"))
            for fpath in json_files:
                try:
                    data = self._load_json(str(fpath))
                    if not data or not isinstance(data, dict):
                        continue

                    records = []
                    for date_str, symbols_list in data.items():
                        if not symbols_list:
                            continue
                        dt = convert_fmp_date(date_str)
                        if dt is None:
                            continue
                        for sym in symbols_list:
                            records.append({"date": dt, "symbol": sym})

                    if not records:
                        continue

                    for i in range(0, len(records), CHUNK_SIZE):
                        chunk = records[i : i + CHUNK_SIZE]
                        with self.engine.begin() as conn:
                            stmt = sqlite_insert(
                                FmpUniverseCalendar.__table__
                            ).values(chunk)
                            stmt = stmt.on_conflict_do_nothing(
                                index_elements=["date", "symbol"]
                            )
                            conn.execute(stmt)
                        total += len(chunk)

                    logger.debug(
                        f"Saved calendar: {fpath.name} ({len(records)} records)"
                    )

                except Exception as e:
                    logger.error(f"Failed to save calendar {fpath.name}: {e}")

        logger.info(f"Saved {total} universe calendar records")
        return total

    # ──────────────────────────────────────────────────────────────────────
    # LOAD: Prices
    # ──────────────────────────────────────────────────────────────────────
    def load_prices(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """FmpDailyPrice 데이터를 심볼별 DataFrame으로 반환."""
        df_dict = {}
        for symbol in symbols:
            try:
                stmt = (
                    select(
                        FmpDailyPrice.date,
                        FmpDailyPrice.open,
                        FmpDailyPrice.high,
                        FmpDailyPrice.low,
                        FmpDailyPrice.close,
                        FmpDailyPrice.volume,
                        FmpDailyPrice.change,
                        FmpDailyPrice.change_percent,
                        FmpDailyPrice.vwap,
                    )
                    .join(FmpTicker, FmpTicker.ticker_id == FmpDailyPrice.ticker_id)
                    .where(FmpTicker.symbol == symbol)
                )
                if start_date:
                    stmt = stmt.where(FmpDailyPrice.date >= start_date)
                if end_date:
                    stmt = stmt.where(FmpDailyPrice.date <= end_date)
                stmt = stmt.order_by(FmpDailyPrice.date)

                df = pd.read_sql(stmt, self.engine, parse_dates=["date"])
                df_dict[symbol] = df
                logger.debug(f"Loaded {len(df)} price records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load prices for {symbol}: {e}")

        return df_dict

    # ──────────────────────────────────────────────────────────────────────
    # LOAD: Market Caps
    # ──────────────────────────────────────────────────────────────────────
    def load_market_caps(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """FmpMarketCap 데이터를 심볼별 DataFrame으로 반환."""
        df_dict = {}
        for symbol in symbols:
            try:
                stmt = (
                    select(FmpMarketCap.date, FmpMarketCap.market_cap)
                    .join(FmpTicker, FmpTicker.ticker_id == FmpMarketCap.ticker_id)
                    .where(FmpTicker.symbol == symbol)
                )
                if start_date:
                    stmt = stmt.where(FmpMarketCap.date >= start_date)
                if end_date:
                    stmt = stmt.where(FmpMarketCap.date <= end_date)
                stmt = stmt.order_by(FmpMarketCap.date)

                df = pd.read_sql(stmt, self.engine, parse_dates=["date"])
                df_dict[symbol] = df
                logger.debug(f"Loaded {len(df)} market cap records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load market caps for {symbol}: {e}")

        return df_dict

    # ──────────────────────────────────────────────────────────────────────
    # LOAD: Financials
    # ──────────────────────────────────────────────────────────────────────
    def load_financials(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """FmpFinancial 데이터를 심볼별 DataFrame으로 반환."""
        # 조회할 수치 컬럼
        fin_columns = [
            getattr(FmpFinancial, col)
            for col in FINANCIAL_COLUMN_MAP.values()
            if hasattr(FmpFinancial, col)
        ]

        df_dict = {}
        for symbol in symbols:
            try:
                stmt = (
                    select(
                        FmpFinancial.date,
                        FmpFinancial.period,
                        FmpFinancial.fiscal_year,
                        FmpFinancial.reported_currency,
                        FmpFinancial.filing_date,
                        FmpFinancial.accepted_date,
                        *fin_columns,
                    )
                    .join(FmpTicker, FmpTicker.ticker_id == FmpFinancial.ticker_id)
                    .where(FmpTicker.symbol == symbol)
                )
                if start_date:
                    stmt = stmt.where(FmpFinancial.date >= start_date)
                if end_date:
                    stmt = stmt.where(FmpFinancial.date <= end_date)
                stmt = stmt.order_by(FmpFinancial.date)

                df = pd.read_sql(stmt, self.engine, parse_dates=["date"])
                df_dict[symbol] = df
                logger.debug(f"Loaded {len(df)} financial records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load financials for {symbol}: {e}")

        return df_dict

    # ──────────────────────────────────────────────────────────────────────
    # LOAD: Treasury Rates
    # ──────────────────────────────────────────────────────────────────────
    def load_treasury_rates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """FmpTreasuryRate 데이터를 DataFrame으로 반환."""
        stmt = select(
            FmpTreasuryRate.date,
            *[
                getattr(FmpTreasuryRate, col)
                for col in TREASURY_KEY_MAP.values()
            ],
        )
        if start_date:
            stmt = stmt.where(FmpTreasuryRate.date >= start_date)
        if end_date:
            stmt = stmt.where(FmpTreasuryRate.date <= end_date)
        stmt = stmt.order_by(FmpTreasuryRate.date)

        return pd.read_sql(stmt, self.engine, parse_dates=["date"])

    # ──────────────────────────────────────────────────────────────────────
    # LOAD: Universe Calendar
    # ──────────────────────────────────────────────────────────────────────
    def load_universe_calendar(
        self,
        date: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """FmpUniverseCalendar 데이터를 DataFrame으로 반환."""
        stmt = select(FmpUniverseCalendar.date, FmpUniverseCalendar.symbol)
        if date:
            stmt = stmt.where(FmpUniverseCalendar.date == date)
        if symbol:
            stmt = stmt.where(FmpUniverseCalendar.symbol == symbol)
        stmt = stmt.order_by(FmpUniverseCalendar.date, FmpUniverseCalendar.symbol)

        return pd.read_sql(stmt, self.engine, parse_dates=["date"])

    # ──────────────────────────────────────────────────────────────────────
    # LOAD: Ticker Metadata
    # ──────────────────────────────────────────────────────────────────────
    def load_ticker_metadata(
        self,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """FmpTicker 메타데이터를 DataFrame으로 반환."""
        stmt = select(
            FmpTicker.symbol,
            FmpTicker.company_name,
            FmpTicker.exchange,
            FmpTicker.industry,
            FmpTicker.sector,
            FmpTicker.country,
            FmpTicker.currency,
            FmpTicker.ipo_date,
            FmpTicker.is_actively_trading,
            FmpTicker.is_etf,
            FmpTicker.is_adr,
            FmpTicker.is_fund,
        )
        if symbols:
            stmt = stmt.where(FmpTicker.symbol.in_(symbols))

        df = pd.read_sql(stmt, self.engine)
        if not df.empty:
            df.set_index("symbol", inplace=True)
        return df

    # ──────────────────────────────────────────────────────────────────────
    # 리소스 정리
    # ──────────────────────────────────────────────────────────────────────
    def close(self):
        self.session.close()
        self.engine.dispose()
        logger.info("FMP Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
