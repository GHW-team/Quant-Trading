# DatabaseManager
# SQLite 기반으로 가격/지표 데이터를 저장/조회하는 헬퍼 (주요 흐름만 주석 추가)

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from src.data.db_models import Base, Ticker, DailyPrice, TechnicalIndicator

logger = logging.getLogger(__name__)


class DatabaseManager:
    """데이터베이스 관리 클래스"""

    def __init__(self, db_path: str = "data/database/stocks.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # SQLAlchemy 엔진/세션 초기화
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            echo=False,
            pool_pre_ping=True,
        )
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # 테이블 생성
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {db_path}")
    
    # ------------------------------------------------------------------ #
    # 헬퍼: 티커 시장 추론 및 ID 확보
    # ------------------------------------------------------------------ #
    def _infer_market_from_ticker(self, ticker_code: str) -> str:
        """티커 문자열을 분석해 시장 정보를 추론"""
        ticker_upper = ticker_code.upper()
        if ticker_upper.endswith('.KS'):
            return 'KOSPI'
        if ticker_upper.endswith('.KQ'):
            return 'KOSDAQ'
        if '-USD' in ticker_upper:
            return 'CRYPTO'
        if ticker_upper.endswith('.T'):
            return 'TOKYO'
        if ticker_upper.endswith('.HK'):
            return 'HONGKONG'
        if '.' not in ticker_upper:
            return 'US'
        return 'UNKNOWN'

    def _get_ticker_id(
        self,
        ticker_code: str,
        market: Optional[str] = None,
    ):
        """
        종목 조회 또는 생성
        
        Returns:
            ticker_id
        """
        ticker = self.session.query(Ticker).filter_by(ticker_code=ticker_code).first()
        if ticker:
            return ticker.ticker_id

        if market is None:
            market = self._infer_market_from_ticker(ticker_code)

        new_ticker = Ticker(
            ticker_code=ticker_code,
            market=market,
        )

        self.session.add(new_ticker)
        self.session.commit()

        logger.info(f"✓ Created ticker: {ticker_code} (ID: {new_ticker.ticker_id})")
        return new_ticker.ticker_id

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    def save_ticker_metadata(
        self, 
        ticker_code: str, 
        metadata: dict,
    ):
        """티커 메타데이터(name/sector)를 업데이트"""
        ticker = self.session.query(Ticker).filter_by(ticker_code=ticker_code).first()
        if not ticker:
            logger.warning(f"Ticker {ticker_code} not found. Create it first.")
            return

        updated = False
        if 'name' in metadata and ticker.name != metadata['name']:
            ticker.name = metadata['name']
            updated = True
        if 'sector' in metadata and ticker.sector != metadata['sector']:
            ticker.sector = metadata['sector']
            updated = True

        if updated:
            self.session.commit()
            logger.info(f"Updated metadata for {ticker_code}")

    def save_price_data(
        self,
        df_dict: Dict[str, pd.DataFrame],
        update_if_exists: bool,
    ) -> Dict[str,int]:
        """
        여러 티커의 가격 데이터를 하나의 트랜잭션으로 db에 저장
        - update_if_exists=True: UPSERT (중복 시 갱신)
        - update_if_exists=False: INSERT OR IGNORE
        """
        results = {}

        # 먼저 모든 ticker_id를 미리 조회/생성 (트랜잭션 밖에서)
        ticker_ids = {}
        for ticker_code in df_dict.keys():
            try:
                ticker_ids[ticker_code] = self._get_ticker_id(ticker_code)
            except Exception as e:
                logger.error(f"✗ Failed to get ticker_id for {ticker_code}: {e}")
                results[ticker_code] = 0

        with self.engine.begin() as conn:
            for ticker_code, stock_df in df_dict.items():
                try:
                    if ticker_code not in ticker_ids:
                        continue

                    # 필요한 열만 선택 후 ticker_id 추가
                    df_save = stock_df[['date','open','high','low','close','volume','adj_close']].copy()
                    df_save['ticker_id'] = ticker_ids[ticker_code]

                    records = df_save.to_dict(orient='records')

                    stmt = sqlite_insert(DailyPrice.__table__).values(records)
                    if update_if_exists:
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['ticker_id', 'date'],
                            set_=dict(
                                open=stmt.excluded.open,
                                high=stmt.excluded.high,
                                low=stmt.excluded.low,
                                close=stmt.excluded.close,
                                volume=stmt.excluded.volume,
                                adj_close=stmt.excluded.adj_close,
                            )
                        )
                    else:
                        stmt = stmt.on_conflict_do_nothing(index_elements=['ticker_id', 'date'])

                    conn.execute(stmt)
                    results[ticker_code] = len(records)
                    logger.debug(f"✓ Bulk saved {len(records)} price records for {ticker_code}")

                except Exception as e:
                    logger.error(f"✗ Failed to save price data for {ticker_code}: {e}")
                    results[ticker_code] = 0
                    continue

        logger.info(f"Bulk price save completed for {len(results)} tickers in single transaction")
        return results
    
    def save_indicators(
        self,
        indicator_data_dict: Dict[str, pd.DataFrame],
        version: str = "v1.0",
    ) -> Dict[str,int]:
        """
        여러 티커의 지표 데이터를 하나의 트랜잭션으로 db에 저장
        - 지표 컬럼만 선택, NaN을 None으로 치환 후 UPSERT
        """
        results = {}
        indicators = ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60', 'ma_100', 'ma_120', 'ma_200', 'macd', 'macd_hist', 'macd_signal',
                      'rsi', 'bb_upper', 'bb_mid', 'bb_lower', 'bb_pct', 'atr', 'hv', 'stoch_k', 'stoch_d', 'obv']

        ticker_ids = {}
        for ticker_code in indicator_data_dict.keys():
            try:
                ticker_ids[ticker_code] = self._get_ticker_id(ticker_code)
            except Exception as e:
                logger.error(f"✗ Failed to get ticker_id for {ticker_code}: {e}")
                results[ticker_code] = 0

        with self.engine.begin() as conn:
            for ticker_code, df in indicator_data_dict.items():
                try:
                    if ticker_code not in ticker_ids:
                        continue
                    ticker_id = ticker_ids[ticker_code]

                    # 사용 가능한 지표만 선택
                    available_indicators = [ind for ind in indicators if ind in df.columns]
                    df_save = df[available_indicators + ['date']].copy()
                    df_save['ticker_id'] = ticker_id
                    df_save['calculated_version'] = version
                    df_save = df_save.replace({np.nan: None})

                    records = df_save.to_dict(orient='records')

                    stmt = sqlite_insert(TechnicalIndicator.__table__).values(records)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['ticker_id', 'date'],
                        set_={ind: stmt.excluded[ind] for ind in available_indicators},
                    )

                    conn.execute(stmt)
                    results[ticker_code] = len(records)
                    logger.debug(f"✓ Bulk saved {len(records)} indicator records for {ticker_code}")

                except Exception as e:
                    logger.error(f"✗ Failed to save indicators for {ticker_code}: {e}")
                    results[ticker_code] = 0

        logger.info(f"Bulk indicator save completed for {len(results)} tickers in single transaction")
        return results

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load_ticker_metadata(
        self,
        ticker_codes: Optional[List[str]] = None,
    ):
        """티커 메타데이터 로드 (없으면 빈 DataFrame)"""
        stmt = select(
            Ticker.ticker_code,
            Ticker.name,
            Ticker.market,
            Ticker.sector,
        )
        if ticker_codes:
            stmt = stmt.where(Ticker.ticker_code.in_(ticker_codes))

        df = pd.read_sql(stmt, self.engine)
        df.set_index('ticker_code', inplace=True) # 티커를 인덱스로

        return df

    def load_price_data(
        self,
        ticker_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 티커의 가격 데이터를 일괄로 불러옵니다.
        - 날짜 필터 적용, 컬럼명은 소문자/snake_case로 통일
        """
        df_dict = {}

        for ticker_code in ticker_codes:
            try:
                stmt = select(
                    DailyPrice.date,
                    DailyPrice.open,
                    DailyPrice.high,
                    DailyPrice.low,
                    DailyPrice.close,
                    DailyPrice.volume,
                    DailyPrice.adj_close,
                ).join(
                    Ticker, Ticker.ticker_id == DailyPrice.ticker_id
                ).where(
                    Ticker.ticker_code == ticker_code
                )

                if start_date:
                    stmt = stmt.where(DailyPrice.date >= start_date)
                if end_date:
                    stmt = stmt.where(DailyPrice.date <= end_date)

                stmt = stmt.order_by(DailyPrice.date)

                df = pd.read_sql(stmt, self.engine, parse_dates=['date'])
                if not df.empty:
                    df.columns = df.columns.str.strip().str.lower()

                df_dict[ticker_code] = df
                logger.debug(f"✓ Loaded {len(df)} price records for {ticker_code}")

            except Exception as e:
                logger.error(f"✗ Failed to load price data for {ticker_code}: {e}")
                continue

        logger.info(f"Bulk price load completed for {len(ticker_codes)} tickers")
        return df_dict

    def load_indicators(
        self,
        ticker_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 티커의 기술적 지표를 일괄로 불러옵니다.
        - 날짜 필터 적용, 컬럼명은 소문자/snake_case로 통일
        """
        df_dict = {}

        for ticker_code in ticker_codes:
            try:
                stmt = select(
                    TechnicalIndicator.date,
                    TechnicalIndicator.ma_5,
                    TechnicalIndicator.ma_20,
                    TechnicalIndicator.ma_200,
                    TechnicalIndicator.macd,
                ).join(
                    Ticker, Ticker.ticker_id == TechnicalIndicator.ticker_id
                ).where(
                    Ticker.ticker_code == ticker_code
                )

                if start_date:
                    stmt = stmt.where(TechnicalIndicator.date >= start_date)
                if end_date:
                    stmt = stmt.where(TechnicalIndicator.date <= end_date)

                stmt = stmt.order_by(TechnicalIndicator.date)

                df = pd.read_sql(stmt, self.engine, parse_dates=['date'])
                if not df.empty:
                    df.columns = df.columns.str.strip().str.lower()

                df_dict[ticker_code] = df
                logger.debug(f"✓ Loaded {len(df)} indicator records for {ticker_code}")

            except Exception as e:
                logger.error(f"✗ Failed to load indicators for {ticker_code}: {e}")
                continue

        logger.info(f"Bulk indicator load completed for {len(ticker_codes)} tickers")
        return df_dict

    # ------------------------------------------------------------------ #
    # 리소스 정리
    # ------------------------------------------------------------------ #
    def close(self):
        self.session.close()
        self.engine.dispose()
        logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

