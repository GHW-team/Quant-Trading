# DatabaseManager
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import and_, create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from src.data.models import Base, DailyPrice, TechnicalIndicator, Ticker

logger = logging.getLogger(__name__)


class DatabaseManager:
    """데이터베이스 접근/저장 래퍼"""

    def __init__(self, db_path: str = "data/database/stocks.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            pool_pre_ping=True,
        )
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        Base.metadata.create_all(self.engine)
        logger.info("Database initialized: %s", db_path)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _infer_market_from_ticker(self, ticker_code: str) -> str:
        ticker_upper = ticker_code.upper()
        if ticker_upper.endswith(".KS"):
            return "KOSPI"
        if ticker_upper.endswith(".KQ"):
            return "KOSDAQ"
        if "-USD" in ticker_upper:
            return "CRYPTO"
        if ticker_upper.endswith(".T"):
            return "TOKYO"
        if ticker_upper.endswith(".HK"):
            return "HONGKONG"
        if "." not in ticker_upper:
            return "US"
        return "UNKNOWN"

    def _get_ticker_id(self, ticker_code: str, market: Optional[str] = None) -> int:
        ticker = self.session.query(Ticker).filter_by(ticker_code=ticker_code).first()
        if ticker:
            return ticker.ticker_id

        if market is None:
            market = self._infer_market_from_ticker(ticker_code)

        new_ticker = Ticker(ticker_code=ticker_code, market=market)
        self.session.add(new_ticker)
        self.session.commit()
        logger.info("Created ticker: %s (ID: %s)", ticker_code, new_ticker.ticker_id)
        return new_ticker.ticker_id

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    def save_ticker_metadata(self, ticker_code: str, metadata: dict) -> None:
        ticker = self.session.query(Ticker).filter_by(ticker_code=ticker_code).first()
        if not ticker:
            logger.warning("Ticker %s not found. Create it first.", ticker_code)
            return

        updated = False
        if "name" in metadata and ticker.name != metadata["name"]:
            ticker.name = metadata["name"]
            updated = True
        if "sector" in metadata and ticker.sector != metadata["sector"]:
            ticker.sector = metadata["sector"]
            updated = True

        if updated:
            self.session.commit()
            logger.info("Updated metadata for %s", ticker_code)

    def save_price_data(
        self,
        df_dict: Dict[str, pd.DataFrame],
        update_if_exists: bool,
    ) -> Dict[str, int]:
        results: Dict[str, int] = {}

        ticker_ids: Dict[str, int] = {}
        for ticker_code in df_dict.keys():
            try:
                ticker_ids[ticker_code] = self._get_ticker_id(ticker_code)
            except Exception as exc:
                logger.error("Failed to get ticker_id for %s: %s", ticker_code, exc)
                results[ticker_code] = 0

        with self.engine.begin() as conn:
            for ticker_code, stock_df in df_dict.items():
                try:
                    if ticker_code not in ticker_ids:
                        continue

                    df_save = stock_df[
                        ["date", "open", "high", "low", "close", "volume", "adj_close"]
                    ].copy()
                    df_save["ticker_id"] = ticker_ids[ticker_code]

                    records = df_save.to_dict(orient="records")

                    stmt = sqlite_insert(DailyPrice.__table__).values(records)
                    if update_if_exists:
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["ticker_id", "date"],
                            set_=dict(
                                open=stmt.excluded.open,
                                high=stmt.excluded.high,
                                low=stmt.excluded.low,
                                close=stmt.excluded.close,
                                volume=stmt.excluded.volume,
                                adj_close=stmt.excluded.adj_close,
                            ),
                        )
                    else:
                        stmt = stmt.on_conflict_do_nothing(index_elements=["ticker_id", "date"])

                    conn.execute(stmt)
                    results[ticker_code] = len(records)
                    logger.debug("Bulk saved %d price records for %s", len(records), ticker_code)
                except Exception as exc:
                    logger.error("Failed to save price data for %s: %s", ticker_code, exc)
                    results[ticker_code] = 0

        logger.info("Bulk price save completed for %d tickers", len(results))
        return results

    def save_indicators(
        self,
        indicator_data_dict: Dict[str, pd.DataFrame],
        version: str = "v1.0",
    ) -> Dict[str, int]:
        results: Dict[str, int] = {}
        indicators = [
            "ma_5",
            "ma_10",
            "ma_20",
            "ma_50",
            "ma_60",
            "ma_120",
            "ma_200",
            "macd",
            "macd_signal",
            "macd_hist"
        ]

        ticker_ids: Dict[str, int] = {}
        for ticker_code in indicator_data_dict.keys():
            try:
                ticker_ids[ticker_code] = self._get_ticker_id(ticker_code)
            except Exception as exc:
                logger.error("Failed to get ticker_id for %s: %s", ticker_code, exc)
                results[ticker_code] = 0

        with self.engine.begin() as conn:
            for ticker_code, df in indicator_data_dict.items():
                try:
                    if ticker_code not in ticker_ids:
                        continue
                    ticker_id = ticker_ids[ticker_code]

                    available_indicators = [ind for ind in indicators if ind in df.columns]

                    df_save = df[available_indicators + ["date"]].copy()
                    df_save["ticker_id"] = ticker_id
                    df_save["calculated_version"] = version
                    df_save = df_save.replace({np.nan: None})

                    records = df_save.to_dict(orient="records")

                    stmt = sqlite_insert(TechnicalIndicator.__table__).values(records)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["ticker_id", "date"],
                        set_={ind: stmt.excluded[ind] for ind in available_indicators},
                    )

                    conn.execute(stmt)
                    results[ticker_code] = len(records)
                    logger.debug("Bulk saved %d indicator records for %s", len(records), ticker_code)
                except Exception as exc:
                    logger.error("Failed to save indicators for %s: %s", ticker_code, exc)
                    results[ticker_code] = 0

        logger.info("Bulk indicator save completed for %d tickers", len(results))
        return results

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load_ticker_metadata(self, ticker_codes: Optional[List[str]] = None) -> pd.DataFrame:
        stmt = select(
            Ticker.ticker_code,
            Ticker.name,
            Ticker.market,
            Ticker.sector,
        )
        if ticker_codes:
            stmt = stmt.where(Ticker.ticker_code.in_(ticker_codes))

        df = pd.read_sql(stmt, self.engine)
        if not df.empty:
            df.set_index("ticker_code", inplace=True)
        return df

    def load_price_data(
        self,
        ticker_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        df_dict: Dict[str, pd.DataFrame] = {}
        for ticker_code in ticker_codes:
            try:
                stmt = (
                    select(
                        DailyPrice.date,
                        DailyPrice.open,
                        DailyPrice.high,
                        DailyPrice.low,
                        DailyPrice.close,
                        DailyPrice.volume,
                        DailyPrice.adj_close,
                    )
                    .join(Ticker, Ticker.ticker_id == DailyPrice.ticker_id)
                    .where(Ticker.ticker_code == ticker_code)
                )
                if start_date:
                    stmt = stmt.where(DailyPrice.date >= start_date)
                if end_date:
                    stmt = stmt.where(DailyPrice.date <= end_date)
                stmt = stmt.order_by(DailyPrice.date)

                df = pd.read_sql(stmt, self.engine, parse_dates=["date"])
                if not df.empty:
                    df.columns = df.columns.str.strip().str.lower()
                df_dict[ticker_code] = df
                logger.debug("Loaded %d price records for %s", len(df), ticker_code)
            except Exception as exc:
                logger.error("Failed to load price data for %s: %s", ticker_code, exc)
                raise

        logger.info("Bulk price load completed for %d tickers", len(ticker_codes))
        return df_dict

    def load_indicators(
        self,
        ticker_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        df_dict: Dict[str, pd.DataFrame] = {}
        for ticker_code in ticker_codes:
            try:
                stmt = (
                    select(
                        TechnicalIndicator.date,
                        TechnicalIndicator.ma_5,
                        TechnicalIndicator.ma_10,
                        TechnicalIndicator.ma_20,
                        TechnicalIndicator.ma_50,
                        TechnicalIndicator.ma_60,
                        TechnicalIndicator.ma_120,
                        TechnicalIndicator.ma_200,
                        TechnicalIndicator.macd,
                        TechnicalIndicator.macd_signal,
                        TechnicalIndicator.macd_hist
                    )
                    .join(Ticker, Ticker.ticker_id == TechnicalIndicator.ticker_id)
                    .where(Ticker.ticker_code == ticker_code)
                )
                if start_date:
                    stmt = stmt.where(TechnicalIndicator.date >= start_date)
                if end_date:
                    stmt = stmt.where(TechnicalIndicator.date <= end_date)
                stmt = stmt.order_by(TechnicalIndicator.date)

                df = pd.read_sql(stmt, self.engine, parse_dates=["date"])
                if not df.empty:
                    df.columns = df.columns.str.strip().str.lower()
                df_dict[ticker_code] = df
                logger.debug("Loaded %d indicator records for %s", len(df), ticker_code)
            except Exception as exc:
                logger.error("Failed to load indicators for %s: %s", ticker_code, exc)
                raise

        logger.info("Bulk indicator load completed for %d tickers", len(ticker_codes))
        return df_dict

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.session.close()
        self.engine.dispose()
        logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
