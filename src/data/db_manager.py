#DatabaseManager

import pandas as pd
import numpy as np
from typing import Optional,List,Dict
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
        df_dict: Dict[str, pd.DataFrame],
        update_if_exists: bool,
    ) -> Dict[str,int]:
        """
        여러 티커의 가격 데이터를 하나의 트랜잭션으로 db에 저장

        Args:
            df_dict: {ticker_code: DataFrame} 딕셔너리
            update_if_exists: True면 UPSERT, False면 INSERT OR IGNORE

        Returns:
            {ticker_code: saved_count} 딕셔너리
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

        # 하나의 트랜잭션으로 모든 티커의 가격 데이터 저장
        with self.engine.begin() as conn:
            for ticker_code, stock_df in df_dict.items():
                try:
                    # 이미 조회한 ticker_id 사용
                    if ticker_code not in ticker_ids:
                        continue

                    # 필요한 열만 선택
                    logger.debug(f"Columns in stock_df for {ticker_code}: {list(stock_df.columns)}")
                    logger.debug(f"stock_df shape: {stock_df.shape}")
                    logger.debug(f"stock_df index name: {stock_df.index.name}")

                    df_save = stock_df[['date','open','high','low','close','volume','adj_close']].copy()
                    df_save['ticker_id'] = ticker_ids[ticker_code]

                    # 레코드로 변환
                    records = df_save.to_dict(orient='records')

                    # SQL 문 생성
                    if update_if_exists:
                        stmt = sqlite_insert(DailyPrice.__table__).values(records)
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
                        stmt = sqlite_insert(DailyPrice.__table__).values(records)
                        stmt = stmt.on_conflict_do_nothing(
                            index_elements=['ticker_id', 'date']
                        )

                    # 같은 트랜잭션 안에서 실행
                    conn.execute(stmt)
                    results[ticker_code] = len(records)
                    logger.debug(f"✓ Bulk saved {len(records)} price records for {ticker_code}")

                except Exception as e:
                    logger.error(f"✗ Failed to save price data for {ticker_code}: {e}")
                    results[ticker_code] = 0

        logger.info(f"Bulk price save completed for {len(results)} tickers in single transaction")
        return results
    
    def save_indicators(
        self,
        indicator_data_dict: Dict[str, pd.DataFrame],
        version: str = "v1.0",
    ) -> Dict[str,int]:
        """
        여러 티커의 지표 데이터를 하나의 트랜잭션으로 db에 저장

        Args:
            indicator_data_dict: {ticker_code: DataFrame} 딕셔너리
            version: 계산 버전

        Returns:
            {ticker_code: saved_count} 딕셔너리
        """
        results = {}

        # 지표 목록 정의
        indicators = ['ma_5', 'ma_20', 'ma_200', 'macd']

        # 먼저 모든 ticker_id를 미리 조회/생성 (트랜잭션 밖에서)
        ticker_ids = {}
        for ticker_code in indicator_data_dict.keys():
            try:
                ticker_ids[ticker_code] = self._get_ticker_id(ticker_code)
            except Exception as e:
                logger.error(f"✗ Failed to get ticker_id for {ticker_code}: {e}")
                results[ticker_code] = 0

        # 하나의 트랜잭션으로 모든 티커의 지표 데이터 저장
        with self.engine.begin() as conn:
            for ticker_code, df in indicator_data_dict.items():
                try:
                    # 이미 조회한 ticker_id 사용
                    if ticker_code not in ticker_ids:
                        continue

                    ticker_id = ticker_ids[ticker_code]

                    # 사용 가능한 지표 선택
                    available_indicators = [ind for ind in indicators if ind in df.columns]

                    df_save = df[available_indicators + ['date']].copy()
                    df_save['ticker_id'] = ticker_id
                    df_save['calculated_version'] = version

                    # NaN을 None으로 변환
                    df_save = df_save.replace({np.nan: None})

                    # 레코드로 변환
                    records = df_save.to_dict(orient='records')

                    # SQL 문 생성
                    stmt = sqlite_insert(TechnicalIndicator.__table__).values(records)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['ticker_id', 'date'],
                        set_={
                            ind: stmt.excluded[ind] for ind in available_indicators
                        }
                    )

                    # 같은 트랜잭션 안에서 실행
                    conn.execute(stmt)
                    results[ticker_code] = len(records)
                    logger.debug(f"✓ Bulk saved {len(records)} indicator records for {ticker_code}")

                except Exception as e:
                    logger.error(f"✗ Failed to save indicators for {ticker_code}: {e}")
                    results[ticker_code] = 0

        logger.info(f"Bulk indicator save completed for {len(results)} tickers in single transaction")
        return results

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
        ticker_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 티커의 가격 데이터를 일괄로 불러옵니다.

        Args:
            ticker_codes: 티커 코드 리스트
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            {ticker_code: DataFrame} 딕셔너리
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
                # date를 column으로 유지 (index로 설정하지 않음)
                # fetch된 데이터 형식과 통일: date column, lowercase 컬럼명
                if not df.empty:
                    df.columns = df.columns.str.strip().str.lower()

                df_dict[ticker_code] = df
                logger.debug(f"✓ Loaded {len(df)} price records for {ticker_code}")

            except Exception as e:
                logger.error(f"✗ Failed to load price data for {ticker_code}: {e}")
                raise

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

        Args:
            ticker_codes: 티커 코드 리스트
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            {ticker_code: DataFrame} 딕셔너리
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
                # date를 column으로 유지 (index로 설정하지 않음)
                # 통일된 데이터 형식: date column, lowercase 컬럼명
                if not df.empty:
                    df.columns = df.columns.str.strip().str.lower()

                df_dict[ticker_code] = df
                logger.debug(f"✓ Loaded {len(df)} indicator records for {ticker_code}")

            except Exception as e:
                logger.error(f"✗ Failed to load indicators for {ticker_code}: {e}")
                raise

        logger.info(f"Bulk indicator load completed for {len(ticker_codes)} tickers")
        return df_dict

    def close(self):
        self.session.close()
        self.engine.dispose()
        logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DatabaseManager 함수 테스트")
    print("="*70)

    # 1. 테스트 데이터 준비
    print("\n[1] 테스트 데이터 다운로드...")
    fetcher = StockDataFetcher()
    korean_stocks = [
        "005930.KS",  # 삼성전자
        "000660.KS",  # SK하이닉스
    ]

    data_dict = fetcher.fetch_multiple_by_period(
        ticker_list=korean_stocks,
        period="1y"
    )
    print(f"✓ {len(data_dict)} 개 티커 데이터 준비 완료")

    with DatabaseManager() as db_manager:
        # 2. _infer_market_from_ticker() 테스트
        print("\n[2] _infer_market_from_ticker() 테스트...")
        try:
            for ticker in korean_stocks:
                market = db_manager._infer_market_from_ticker(ticker)
                print(f"✓ {ticker} → market: {market}")
        except Exception as e:
            print(f"✗ 시장 추론 실패: {e}")

        # 3. save_price_data() 테스트
        print("\n[3] save_price_data() 테스트 (여러 티커 저장)...")
        try:
            save_results = db_manager.save_price_data(
                df_dict=data_dict,
                update_if_exists=True
            )
            print(f"✓ 가격 데이터 저장 완료: {save_results}")
        except Exception as e:
            print(f"✗ 가격 데이터 저장 실패: {e}")

        # 4. _get_ticker_id() 테스트
        print("\n[4] _get_ticker_id() 테스트...")
        try:
            for ticker in korean_stocks:
                ticker_id = db_manager._get_ticker_id(ticker)
                print(f"✓ {ticker} → ticker_id: {ticker_id}")
        except Exception as e:
            print(f"✗ 티커 ID 조회 실패: {e}")

        # 5. load_ticker_metadata() 테스트
        print("\n[5] load_ticker_metadata() 테스트...")
        try:
            metadata = db_manager.load_ticker_metadata(ticker_codes=korean_stocks)
            print(f"✓ 티커 메타데이터 로드:")
            print(metadata)
        except Exception as e:
            print(f"✗ 메타데이터 로드 실패: {e}")

        # 6. save_ticker_metadata() 테스트
        print("\n[6] save_ticker_metadata() 테스트...")
        try:
            metadata_info = {
                'name': '삼성전자',
                'sector': 'IT'
            }
            db_manager.save_ticker_metadata(
                ticker_code=korean_stocks[0],
                metadata=metadata_info
            )
            print(f"✓ {korean_stocks[0]} 메타데이터 저장 완료")
        except Exception as e:
            print(f"✗ 메타데이터 저장 실패: {e}")

        # 7. load_price_data() 테스트 (여러 티커 로드)
        print("\n[7] load_price_data() 테스트 (여러 티커 로드)...")
        try:
            df_price_dict = db_manager.load_price_data(
                ticker_codes=korean_stocks,
                start_date="2024-01-01"
            )
            for ticker, df in df_price_dict.items():
                print(f"✓ {ticker}: {len(df)} 행")
                print(f"  컬럼: {list(df.columns)}")
        except Exception as e:
            print(f"✗ 가격 데이터 로드 실패: {e}")

        # 8. save_indicators() 테스트
        print("\n[8] save_indicators() 테스트 (여러 티커 지표 저장)...")
        try:
            from src.data.indicator_calculator import IndicatorCalculator

            indicator_data = {}
            calculator = IndicatorCalculator()

            for ticker, df_price in df_price_dict.items():
                df_with_indicators = calculator.calculate_indicators(
                    df=df_price,
                    indicator_list=['ma_5', 'ma_20', 'ma_200', 'macd']
                )
                indicator_data[ticker] = df_with_indicators

            indicator_results = db_manager.save_indicators(
                indicator_data_dict=indicator_data,
                version="v1.0"
            )
            print(f"✓ 지표 데이터 저장 완료: {indicator_results}")
        except Exception as e:
            print(f"✗ 지표 데이터 저장 실패: {e}")

        # 9. load_indicators() 테스트 (여러 티커 로드)
        print("\n[9] load_indicators() 테스트 (여러 티커 로드)...")
        try:
            df_indicators_dict = db_manager.load_indicators(
                ticker_codes=korean_stocks,
                start_date="2024-01-01"
            )
            for ticker, df in df_indicators_dict.items():
                print(f"✓ {ticker}: {len(df)} 행")
                print(f"  컬럼: {list(df.columns)}")
        except Exception as e:
            print(f"✗ 지표 데이터 로드 실패: {e}")

    print("\n" + "="*70)
    print("모든 테스트 완료!")
    print("="*70)