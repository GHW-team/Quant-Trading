# DatabaseManager
# SQLite 기반으로 가격/지표 데이터를 저장/조회하는 헬퍼 (주요 흐름만 주석 추가)

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterable, Iterator, TypeVar

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from src.data.db_models import Base, Ticker, DailyPrice, TechnicalIndicator

logger = logging.getLogger(__name__)


class DatabaseManager:
    """데이터베이스 관리 클래스"""
    INDICATORS = [
        'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60', 'ma_100', 'ma_120', 'ma_200',
        'macd', 'macd_hist', 'macd_signal',
        'rsi', 'bb_upper', 'bb_mid', 'bb_lower', 'bb_pct',
        'atr', 'hv', 'stoch_k', 'stoch_d', 'obv'
    ]

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

        logger.debug(f"✓ Created ticker: {ticker_code} (ID: {new_ticker.ticker_id})")
        return new_ticker.ticker_id

    T = TypeVar("T")
    def _chunked(self, seq: List[T], chunk_size: int) -> Iterator[List[T]]:
        """리스트를 chunk_size 단위로 잘라서 yield"""
        if chunk_size <= 0:
            chunk_size = 100
        for i in range(0, len(seq), chunk_size):
            yield seq[i:i + chunk_size]

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    def save_ticker_metadata(
        self,
        ticker_code_or_dict: Union[str, Dict[str, dict]],
        metadata: Optional[dict] = None,
    ):
        """티커 메타데이터(name/sector)를 업데이트 (단일/배치 지원)

        지원 입력 형태
        - 단일: save_ticker_metadata("005930.KS", {"name": "Samsung", "sector": "IT"})
        - 배치: save_ticker_metadata({"005930.KS": {...}, "000660.KS": {...}})
        - 배치(dict, value가 1-row DF): save_ticker_metadata({"005930.KS": df_1row, ...})

        Returns:
            Dict[str, bool]: {ticker_code: 업데이트 수행 여부}
        """

        # ---- 입력을 표준 형태로 통일 ----
        # 단일: (ticker_code, metadata_dict)
        # 배치(dict): {ticker_code: metadata_dict} 또는 {ticker_code: 1-row DataFrame}
        items: List[Tuple[str, dict]]
        if isinstance(ticker_code_or_dict, str):
            items = [(ticker_code_or_dict, metadata or {})]
        else:
            items = []
            for k, v in ticker_code_or_dict.items(): # k는 ticker, v는 메타데이터
                if isinstance(v, pd.DataFrame):
                    if v.empty:
                        md = {}
                    else:
                        # 1행을 메타데이터로 간주 (name/sector 같은 컬럼을 사용)
                        md = {kk: vv for kk, vv in v.iloc[0].to_dict().items() if vv is not None}
                    items.append((k, md))
                else:
                    items.append((k, (v or {})))

        results: Dict[str, bool] = {k: False for k, _ in items}
        if not items:
            return results

        # ---- 배치 조회 (IN 절) ----
        codes = [t for t, _ in items]
        tickers = self.session.query(Ticker).filter(Ticker.ticker_code.in_(codes)).all()
        ticker_map = {t.ticker_code: t for t in tickers}

        updated_any = False
        for ticker_code, md in items:
            ticker = ticker_map.get(ticker_code)
            if not ticker:
                logger.warning(f"Ticker {ticker_code} not found. Create it first.")
                continue

            updated = False
            if 'name' in md and ticker.name != md['name']:
                ticker.name = md['name']
                updated = True
            if 'sector' in md and ticker.sector != md['sector']:
                ticker.sector = md['sector']
                updated = True

            if updated:
                updated_any = True
                results[ticker_code] = True

        # 여러 개 수정해도 commit은 1번만
        if updated_any:
            self.session.commit()
            logger.info(f"Updated ticker metadata: {sum(results.values())}/{len(results)} updated")

        return results

    def save_price_data(
        self,
        df_dict: Dict[str, pd.DataFrame],
        update_if_exists: bool,
        chunk_size: int = 100,
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

        items = list(df_dict.items())
        for batch in self._chunked(items, chunk_size):
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
            logger.info(f"Price save batch committed: {len(batch)} tickers")
                
        logger.info(f"Price save completed for {len(results)} tickers")
        return results
    
    def save_indicators(
        self,
        indicator_data_dict: Dict[str, pd.DataFrame],
        version: str = "v1.0",
        chunk_size: int = 100,
    ) -> Dict[str,int]:
        """
        여러 티커의 지표 데이터를 하나의 트랜잭션으로 db에 저장
        - 지표 컬럼만 선택, NaN을 None으로 치환 후 UPSERT
        """
        results = {}
        indicators = self.INDICATORS

        ticker_ids = {}
        for ticker_code in indicator_data_dict.keys():
            try:
                ticker_ids[ticker_code] = self._get_ticker_id(ticker_code)
            except Exception as e:
                logger.error(f"✗ Failed to get ticker_id for {ticker_code}: {e}")
                results[ticker_code] = 0
        
        items = list(indicator_data_dict.items())
        for batch in self._chunked(items, chunk_size):
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
            logger.info(f"Indicator save batch committed: {len(batch)} tickers")

        logger.info(f"Bulk indicator save completed for {len(results)} tickers in single transaction")
        return results

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load_ticker_metadata(
        self,
        ticker_codes: Optional[Union[str, List[str], Dict[str, object]]] = None,
    ):
        """티커 메타데이터 로드 (단일/배치 지원, 없으면 빈 DataFrame)

        Args:
            ticker_codes: None(전체 로드) | str(단일 티커) | List[str](여러 티커) | Dict[str, ...](key를 티커로 사용)

        Returns:
            pd.DataFrame: index=ticker_code, columns=[name, market, sector]
        """
        # Dict 형태(예: {ticker: df})가 들어오면 key만 사용
        if isinstance(ticker_codes, dict):
            ticker_codes = list(ticker_codes.keys())

        # 단일 str이면 리스트로 변환
        if isinstance(ticker_codes, str):
            ticker_codes = [ticker_codes]

        stmt = select(
            Ticker.ticker_code,
            Ticker.name,
            Ticker.market,
            Ticker.sector,
        )
        if ticker_codes:
            stmt = stmt.where(Ticker.ticker_code.in_(ticker_codes))

        df = pd.read_sql(stmt, self.engine)
        df.set_index('ticker_code', inplace=True)  # 티커를 인덱스로
        return df

    def load_price_data(
        self,
        ticker_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        chunk_size: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 티커의 가격 데이터를 일괄로 불러옵니다.
        - 날짜 필터 적용, 컬럼명은 소문자/snake_case로 통일
        - IN 절 + chunk 조회로 티커별 반복 조회 제거
        """
        if not ticker_codes:
            return {}
        
        # 중복 제거(입력 순서 유지) + 기본 빈 DF 템플릿
        uniq_codes = list(dict.fromkeys(ticker_codes))
        base_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        empty_tpl = pd.DataFrame(columns=base_cols)
        df_dict: Dict[str, pd.DataFrame] = {c: empty_tpl.copy() for c in uniq_codes}
        
        for codes_chunk in self._chunked(uniq_codes, chunk_size):
            try:
                stmt = select(
                    Ticker.ticker_code.label('ticker_code'),
                    DailyPrice.date,
                    DailyPrice.open,
                    DailyPrice.high,
                    DailyPrice.low,
                    DailyPrice.close,
                    DailyPrice.volume,
                    DailyPrice.adj_close,
                ).select_from(
                    DailyPrice
                ).join(
                    Ticker, Ticker.ticker_id == DailyPrice.ticker_id
                ).where(
                    Ticker.ticker_code.in_(codes_chunk)
                )

                if start_date:
                    stmt = stmt.where(DailyPrice.date >= start_date)
                if end_date:
                    stmt = stmt.where(DailyPrice.date <= end_date)

                # ticker_code로 그룹핑할 거라 ticker_code -> date 순 정렬
                stmt = stmt.order_by(Ticker.ticker_code, DailyPrice.date)

                df_all = pd.read_sql(stmt, self.engine, parse_dates=['date'])
                if df_all.empty:
                    continue

                df_all.columns = df_all.columns.str.strip().str.lower()

                for code, g in df_all.groupby('ticker_code', sort=False):
                    df_dict[code] = g.drop(columns=['ticker_code']).reset_index(drop=True)
                    logger.debug(f"✓ Loaded {len(df_dict[code])} price records for {code}")

            except Exception as e:
                logger.error(f"✗ Failed to load price data for chunk({len(codes_chunk)} tickers): {e}")
                continue

        logger.info(f"Bulk price load completed for {len(ticker_codes)} tickers")
        return df_dict

    def load_indicators(
        self,
        ticker_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        indicators: Optional[List[str]] = None,
        chunk_size: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 티커의 기술적 지표를 일괄로 불러옵니다.
        - 날짜 필터 적용, 컬럼명은 소문자/snake_case로 통일
        - IN 절 + chunk 조회로 티커별 반복 조회 제거
        - indicators 파라미터로 필요한 지표만 SELECT 하여 메모리 최적화
        """
        if not ticker_codes:
            return {}
        
        # ---- 1) 요청 지표 확정 (유효성 검사 + 기본값) ----
        if indicators is None:
            selected_inds = list(self.INDICATORS)  # 기존 동작 유지(전체)
        else:
            # 중복 제거 + 입력 순서 유지
            indicators = list(dict.fromkeys(indicators))

            # DB 모델에 존재하는 지표만 허용
            valid_set = set(self.INDICATORS)
            selected_inds = [ind for ind in indicators if ind in valid_set]

            invalid = [ind for ind in indicators if ind not in valid_set]
            if invalid:
                logger.warning(f"Invalid indicators ignored: {invalid}")

            # 전부 invalid면 안전하게 "전체"로 fallback (원하면 빈 컬럼만 로드로 바꿔도 됨)
            if not selected_inds:
                logger.warning("No valid indicators requested. Fallback to all indicators.")
                selected_inds = list(self.INDICATORS)
                
        uniq_codes = list(dict.fromkeys(ticker_codes))
        
        # ---- 2) 빈 DF 템플릿 (선택 지표만) ----
        ind_cols = ['date'] + selected_inds
        empty_tpl = pd.DataFrame(columns=ind_cols)
        df_dict: Dict[str, pd.DataFrame] = {c: empty_tpl.copy() for c in uniq_codes}

        # ---- 3) chunk 단위 IN 조회 ----
        for codes_chunk in self._chunked(uniq_codes, chunk_size):
            try:
                stmt = select(
                    Ticker.ticker_code.label('ticker_code'),
                    TechnicalIndicator.date,
                    *[getattr(TechnicalIndicator, ind) for ind in selected_inds]
                ).select_from(
                    TechnicalIndicator
                ).join(
                    Ticker, Ticker.ticker_id == TechnicalIndicator.ticker_id
                ).where(
                    Ticker.ticker_code.in_(codes_chunk)
                )

                if start_date:
                    stmt = stmt.where(TechnicalIndicator.date >= start_date)
                if end_date:
                    stmt = stmt.where(TechnicalIndicator.date <= end_date)

                stmt = stmt.order_by(Ticker.ticker_code, TechnicalIndicator.date)

                df_all = pd.read_sql(stmt, self.engine, parse_dates=['date'])
                if df_all.empty:
                    continue

                df_all.columns = df_all.columns.str.strip().str.lower()

                for code, g in df_all.groupby('ticker_code', sort=False):
                    df_dict[code] = g.drop(columns=['ticker_code']).reset_index(drop=True)
                    logger.debug(f"✓ Loaded {len(df_dict[code])} indicator records for {code}")

            except Exception as e:
                logger.error(f"✗ Failed to load price data for chunk({len(codes_chunk)} tickers): {e}")
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

