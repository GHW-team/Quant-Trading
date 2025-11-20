#DatabaseManager

import pandas as pd
import numpy as np
from typing import Optional,List
from sqlalchemy import create_engine,select,and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from src.data.models import Base,Ticker,DailyPrice,TechnicalIndicator
from pathlib import Path
import logging

#test
from src.data.data_fetcher import StockDataFetcher

logger = logging.getLogger(__name__)

class DatabaseManager():
    """데이터베이스 관리 클래스"""

    def __init__(self, db_path: str = "data/database/stocks.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        #SQLAlchemy 엔진
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            echo=False,
            pool_pre_ping=True,
        )

        #세션 팩토리
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        #테이블 생성
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {db_path}")
    
    # 헬퍼 함수 추가 (내부적으로 사용)
    def _infer_market_from_ticker(self, ticker_code: str) -> str:
        """티커 문자열을 분석해 시장 정보를 추론"""
        ticker_upper = ticker_code.upper()

        # 한국 시장
        if ticker_upper.endswith('.KS'):
            return 'KOSPI'
        if ticker_upper.endswith('.KQ'):
            return 'KOSDAQ'
        
        # 암호화폐 (BTC-USD 등)
        if '-USD' in ticker_upper:
            return 'CRYPTO'
            
        # 일본
        if ticker_upper.endswith('.T'):
            return 'TOKYO'
            
        # 홍콩
        if ticker_upper.endswith('.HK'):
            return 'HONGKONG'

        # 점(.)이 없으면 대부분 미국 주식 (NYSE, NASDAQ, AMEX 등)
        # 주의: 티커 문자열만으로는 NYSE인지 NASDAQ인지 구분이 불가능함 -> 통칭 'US'로 저장
        if '.' not in ticker_upper:
            return 'US'
            
        return 'UNKNOWN'

    def get_ticker_id(
        self,
        ticker_code: str,
        market: Optional[str] = None,
    ):
        """
        종목 조회 또는 생성
        
        Returns:
            ticker_id
        """
        
        #티커 아이디 가져오기
        ticker = self.session.query(Ticker).filter_by(ticker_code=ticker_code).first()
        
        if ticker:
            return ticker.ticker_id

        # 만약 market 인자가 안 들어왔다면, 티커 코드로 추론한다.
        if market is None:
            market = self._infer_market_from_ticker(ticker_code)

        #db에 없으면 추가
        new_ticker = Ticker(
            ticker_code=ticker_code,
            market=market,
        )

        self.session.add(new_ticker)
        self.session.commit()

        logger.info(f"✓ Created ticker: {ticker_code} (ID: {new_ticker.ticker_id})")
        return new_ticker.ticker_id

    def save_ticker_metadata(
        self, 
        ticker_code: str, 
        metadata: dict,
    ):
        """
        종목의 상세 메타데이터(이름, 섹터, 산업, 시가총액 등)를 업데이트
        """
        ticker = self.session.query(Ticker).filter_by(ticker_code=ticker_code).first()
        
        if not ticker:
            logger.warning(f"Ticker {ticker_code} not found. Create it first.")
            return

        # 메타데이터 딕셔너리를 이용해 업데이트
        updated = False
        if 'name' in metadata and ticker.name != metadata['name']:
            ticker.name = metadata['name']
            updated = True
            
        if 'sector' in metadata and ticker.sector != metadata['sector']:
            ticker.sector = metadata['sector']
            updated = True
            
        # 나중에 항목이 늘어나면 여기에 if문만 추가하면 됩니다.
        # if 'industry' in metadata...
        # if 'ceo' in metadata...

        if updated:
            self.session.commit()
            logger.info(f"Updated metadata for {ticker_code}")

    def save_price_data(
        self,
        ticker_code,
        stock_df: pd.DataFrame,
        update_if_exists: bool,
    )-> int:
        """
        가격 데이터 대량 저장 (SQLAlchemy Core 사용)
        
        Args:
            ticker_code: 종목 코드
            df: yfinance에서 가져온 DataFrame (index=Date)
            update_if_exists: True면 UPSERT, False면 INSERT OR IGNORE
            
        Returns:
            저장된 행 수
        """
        ticker_id = self.get_ticker_id(ticker_code)

        #stock_preprocess
        df_reset = stock_df.reset_index()

        df_reset.columns = df_reset.columns.str.strip()
        df_reset.columns = df_reset.columns.str.lower()
        df_reset.columns = df_reset.columns.str.replace(' ','_')

        #select columns
        df_save = df_reset[['date','open','high','low','close','volume','adj_close']].copy()
        df_save['ticker_id'] = ticker_id

        #convert to records
        records = df_save.to_dict(orient='records')

        #make statement
        if update_if_exists:
            stmt = sqlite_insert(DailyPrice.__table__).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements= ['ticker_id', 'date'],
                set_=dict(
                    open = stmt.excluded.open,
                    high = stmt.excluded.high,
                    low = stmt.excluded.low,
                    close = stmt.excluded.close,
                    volume = stmt.excluded.volume,
                    adj_close = stmt.excluded.adj_close,
                )
            )
        else:
            stmt = sqlite_insert(DailyPrice.__table__).values(records)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=['ticker_id','date'],
            )
        
        #excute statement
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
        logger.info(f"Saved {len(records)} price records for {ticker_code}")

        return len(records)
    
    def save_indicators(
        self,
        ticker_code: str,
        df: pd.DataFrame,
        version: str = "v1.0",
    ):
        """
        지표 데이터 대량 저장
        
        Args:
            ticker_code: 종목 코드
            df: 지표가 포함된 DataFrame (index=date)
            version: 계산 버전
            
        Returns:
            저장된 행 수
        """
        #loading ticker_id 
        ticker_id = self.get_ticker_id(ticker_code)

        #preprocess dataframe
        df_reset = df.reset_index()
        
        df_reset.columns = df_reset.columns.str.strip()
        df_reset.columns = df_reset.columns.str.lower()
        df_reset.columns = df_reset.columns.str.replace(' ','_')
        
        #select indicators
        indicators = [
            'ma_5','ma_20','ma_200','macd',
        ]
        available_indicators = [ind for ind in indicators if ind in df_reset.columns]

        df_save = df_reset[available_indicators + ['date']].copy()
        df_save['ticker_id'] = ticker_id
        df_save['calculated_version'] = version 

        #fill NaN -> None
        df_save = df_save.replace({np.nan : None})

        #convert to dictionary list
        records = df_save.to_dict(orient='records')

        #make statement
        stmt = sqlite_insert(TechnicalIndicator.__table__).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=['ticker_id','date'],
            set_={
                ind : stmt.excluded[ind] for ind in available_indicators
            }
        )
        
        #execute stmt
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
        logger.info(f"Saved {len(records)} indicator records for {ticker_code}")

        return len(records)
    
    def load_ticker_metadata(
        self,
        ticker_codes: Optional[List[str]] = None,
    ):
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
            df.set_index('ticker_code', inplace=True) # 티커를 인덱스로

        return df

    def load_price_data(
            self,
            ticker_code: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
    ):
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
        
        df = pd.read_sql(stmt,self.engine,parse_dates=['date'])
        if not df.empty:
            df.set_index('date',inplace=True)

        return df

    def load_indicators(
            self,
            ticker_code: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
    ):
        #Make Query
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
            df.set_index('date',inplace=True)

        return df
    
    def close(self):
        self.session.close()
        self.engine.dispose()
        logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    fetcher = StockDataFetcher()

    korean_stocks = [
        "005930.KS",  # 삼성전자
        "000660.KS",  # SK하이닉스
        "035720.KS",  # 카카오
        "035420.KS",  # NAVER
    ]    

    data_dict = fetcher.fetch_multiple_stocks(korean_stocks)
    with DatabaseManager() as db_manager:
        for ticker, df in data_dict.items():
            db_manager.save_price_data(ticker_code=ticker, stock_df=df, update_if_exists=True)
