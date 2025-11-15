# src/data/db_manager.py - 데이터 저장 관리

import sqlite3
import pandas as pd
from typing import Optional, Dict, List, Sequence
from pathlib import Path
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
import logging

try:
    from .models import Base, Ticker, DailyPrice, TechnicalIndicator
except ImportError:
    from models import Base, Ticker, DailyPrice, TechnicalIndicator

logger = logging.getLogger(__name__)


class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "data/database/stocks.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # SQLAlchemy 엔진
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            echo=False,
            pool_pre_ping=True  # 연결 유효성 검사
        )
        
        # 세션 팩토리
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # 테이블 생성
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {db_path}")
    
    def fetch_ticker_codes(
        self,
        market: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Load ticker codes stored in the database with optional filtering.
        """
        query = self.session.query(Ticker.ticker_code)
        if market:
            query = query.filter(Ticker.market == market)
        query = query.order_by(Ticker.ticker_code)
        if limit is not None:
            query = query.limit(limit)
        return [row.ticker_code for row in query.all()]

    def fetch_ticker_codes_by_indexes(
        self,
        indexes: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Load ticker codes grouped by their listing indexes (market column).
        """
        query = self.session.query(Ticker.market, Ticker.ticker_code)
        if indexes:
            query = query.filter(Ticker.market.in_(indexes))
        query = query.order_by(Ticker.market, Ticker.ticker_code)

        grouped: Dict[str, List[str]] = {}
        for market, ticker_code in query.all():
            key = market or "UNKNOWN"
            grouped.setdefault(key, []).append(ticker_code)
        return grouped
    
    def get_or_create_ticker(
        self, 
        ticker_code: str, 
        name: Optional[str] = None,
        market: str = "KOSPI",
    ) -> int:
        """
        종목 조회 또는 생성
        
        Returns:
            ticker_id
        """
        # 기존 종목 조회
        ticker = self.session.query(Ticker).filter_by(ticker_code=ticker_code).first()
        
        if ticker:
            return ticker.ticker_id
        
        # 새 종목 생성
        new_ticker = Ticker(
            ticker_code=ticker_code,
            name=name,
            market=market,
        )
        self.session.add(new_ticker)
        self.session.commit()
        
        logger.info(f"✓ Created ticker: {ticker_code} (ID: {new_ticker.ticker_id})")
        return new_ticker.ticker_id
    
    def save_price_data_bulk(
        self, 
        ticker_code: str, 
        df: pd.DataFrame,
        update_if_exists: bool = False
    ) -> int:
        """
        가격 데이터 대량 저장 (SQLAlchemy Core 사용)
        
        Args:
            ticker_code: 종목 코드
            df: yfinance에서 가져온 DataFrame (index=Date)
            update_if_exists: True면 UPSERT, False면 INSERT OR IGNORE
            
        Returns:
            저장된 행 수
        """
        ticker_id = self.get_or_create_ticker(ticker_code)
        #이 함수는 ticker_code하나 가지고 그 종목에 대한 정보들만 저장하는건지
        #아니면 ticker_code리스트를 받아서 처리할 수 있는건지.
        
        # DataFrame 전처리
        df_reset = df.reset_index()
        df_reset.rename(columns={'Date': 'date'}, inplace=True)#?? inplace가 뭐지. 그리고 없어도되는 줄 아닌가.
        
        # 컬럼명 소문자 변환
        df_reset.columns = df_reset.columns.str.lower()
        
        # adjusted_close 처리
        if 'adj close' in df_reset.columns:
            df_reset.rename(columns={'adj close': 'adjusted_close'}, inplace=True)#?? 왜 adj_close로 안함
        elif 'adjusted_close' not in df_reset.columns:
            df_reset['adjusted_close'] = df_reset['close']
        
        # 필요한 컬럼만 선택
        df_save = df_reset[['date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']].copy()
        df_save['ticker_id'] = ticker_id 
        
        # 딕셔너리 리스트로 변환
        records = df_save.to_dict(orient='records') #딕셔너리 리스트가 뭐지.
        
        # SQLAlchemy Core로 bulk insert
        if update_if_exists:
            # UPSERT (SQLite 3.24+)
            stmt = sqlite_insert(DailyPrice.__table__).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['ticker_id', 'date'],
                set_=dict(#??: 이건 무슨구조지.
                    open=stmt.excluded.open,#??: excluded는 뭐야
                    high=stmt.excluded.high,
                    low=stmt.excluded.low,
                    close=stmt.excluded.close,
                    volume=stmt.excluded.volume,
                    adjusted_close=stmt.excluded.adjusted_close
                )
            )
        else:
            # INSERT OR IGNORE
            stmt = sqlite_insert(DailyPrice.__table__).values(records)
            stmt = stmt.on_conflict_do_nothing(index_elements=['ticker_id', 'date'])
        
        result = self.engine.execute(stmt)
        logger.info(f"✓ Saved {len(records)} price records for {ticker_code}")
        
        return len(records)
    
    def save_indicators_bulk(
        self,
        ticker_code: str,
        df: pd.DataFrame,
        version: str = "v1.0"
    ) -> int:
        """
        지표 데이터 대량 저장
        
        Args:
            ticker_code: 종목 코드
            df: 지표가 포함된 DataFrame (index=date)
            version: 계산 버전
            
        Returns:
            저장된 행 수
        """
        ticker_id = self.get_or_create_ticker(ticker_code)
        
        # DataFrame 전처리
        df_reset = df.reset_index()
        if 'Date' in df_reset.columns:
            df_reset.rename(columns={'Date': 'date'}, inplace=True)
        
        # 필요한 컬럼 선택 (존재하는 것만)
        indicator_cols = [
            'date', 'ma_5', 'ma_20', 'ma_60', 'ma_200',
            'macd', 'macd_signal', 'macd_hist',
            'rsi_14', 'bb_upper', 'bb_middle', 'bb_lower'
        ]
        
        available_cols = [col for col in indicator_cols if col in df_reset.columns]
        df_save = df_reset[available_cols].copy()
        df_save['ticker_id'] = ticker_id
        df_save['calculation_version'] = version
        
        # NaN을 None으로 변환 (SQL NULL)
        df_save = df_save.where(pd.notnull(df_save), None)# ??: where의 자세한 문법 파악
        
        # 딕셔너리 리스트로 변환
        records = df_save.to_dict(orient='records')
        
        # UPSERT
        stmt = sqlite_insert(TechnicalIndicator.__table__).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=['ticker_id', 'date'],
            set_={col: stmt.excluded[col] for col in available_cols if col != 'date'}
        )
        
        result = self.engine.execute(stmt)
        logger.info(f"✓ Saved {len(records)} indicator records for {ticker_code}")
        
        return len(records)
    
    def load_price_data(
        self,
        ticker_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        가격 데이터 조회
        
        Returns:
            DataFrame (index=date)
        """
        query = """
        SELECT 
            dp.date,
            dp.open,
            dp.high,
            dp.low,
            dp.close,
            dp.volume,
            dp.adjusted_close
        FROM daily_prices dp
        JOIN tickers t ON dp.ticker_id = t.ticker_id 
        WHERE t.ticker_code = :ticker_code
        """
        #??:왜 join을 썼는지
        #??: :ticker_code는 무슨 문법인지.
        
        params = {'ticker_code': ticker_code}
        
        if start_date:
            query += " AND dp.date >= :start_date"
            #??:이게 뭐지
            params['start_date'] = start_date
        
        if end_date:
            query += " AND dp.date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY dp.date"
        
        df = pd.read_sql(query, self.engine, params=params, parse_dates=['date'])
        #??: engine들어가는거면 pandas가 SQLAlchemy와 연동된건가? 근데 기본함수가 어떻게?
        df.set_index('date', inplace=True)
        
        return df
    
    def load_data_with_indicators(
        self,
        ticker_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        가격 + 지표 데이터 조회
        
        Returns:
            DataFrame (index=date)
        """
        query = """
        SELECT 
            dp.date,
            dp.close,
            dp.volume,
            ti.ma_5,
            ti.ma_20,
            ti.ma_60,
            ti.ma_200,
            ti.macd,
            ti.macd_signal,
            ti.macd_hist,
            ti.rsi_14
        FROM daily_prices dp
        JOIN tickers t ON dp.ticker_id = t.ticker_id
        LEFT JOIN technical_indicators ti 
            ON dp.ticker_id = ti.ticker_id 
            AND dp.date = ti.date
        WHERE t.ticker_code = :ticker_code
        """
        
        params = {'ticker_code': ticker_code}
        
        if start_date:
            query += " AND dp.date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND dp.date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY dp.date"
        
        df = pd.read_sql(query, self.engine, params=params, parse_dates=['date'])
        df.set_index('date', inplace=True)
        
        return df
    
    def close(self):
        """연결 종료"""
        self.session.close()
        self.engine.dispose()#단어뜻
        logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
