#StockDataFetcher
import pandas as pd
import yfinance as yf
import logging
from typing import Optional,List,Dict
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor,as_completed
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#추가기능 
    #API rate limiting
    #데이터 검증
    #티커 배치 조회 groupby.
    #요청 간 최소 대기 시간
    #이미 존재하는 데이터 다운 여부.

class StockDataFetcher:
    """yfinance를 사용한 주식 데이터 수집기"""
    def __init__(self,max_workers: int=5,max_retries: int=3):
        """
        Args:
            max_workers: 동시 실행 스레드 수 (Yahoo API 부담 고려)
            max_retries: 재시도 횟수
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        
    def _fetch_single_by_period(
            self, 
            ticker: str, 
            period: str = "1y", 
            interval: str = "1d",
            actions : bool = False
        ) -> Optional[pd.DataFrame]:
        """
        단일 종목 데이터 수집 (재시도 포함)
        
        Args:
            ticker: 종목 코드 (예: "005930.KS")
            period: 기간 ("1y", "2y", "max" 등)
            interval: 간격 ("1d", "1h" 등)
            
        Returns:
            DataFrame 또는 None (실패 시)
        """
        for attempt in range(1,self.max_retries+1):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(
                    period = period,
                    interval = interval,
                    auto_adjust = False,
                    actions = actions
                )
                #data validation
                if df.empty:
                    logger.warning(f"{ticker}: Empty data returns")
                    return None

                if len(df) < 10:
                    logger.warning(f"{ticker}: Insufficient data ({len(df)} rows)")

                logger.info(f"{ticker}: {len(df)} records fetched")

                #형식 통일 (대문자 -> 소문자) (Date 인덱스 -> 컬럼) (띄어쓰기 -> '_')
                df = df.reset_index()
                df.columns = (df.columns.str.strip()
                                        .str.lower()
                                        .str.replace(' ','_'))

                return df

            except RequestException as e:
                logger.warning(
                    f"{ticker}: Network error (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    sleep_time = 2**attempt
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"{ticker}: Unexpected error: {e}")
                return None

        logger.error(f"{ticker}: Failed after {self.max_retries} attempts")
        return None

    def fetch_multiple_by_period(
            self,
            ticker_list: List[str],
            period: str,
            interval: str = "1d",
            actions: bool = False,
        )-> Dict[str,pd.DataFrame]:
        """
        여러 종목 병렬 수집
        
        Args:
            ticker_list: 종목 코드 리스트
            period: 기간
            interval: 간격
            
        Returns:
            {ticker: DataFrame} 딕셔너리
        """
        results = {}

        if not ticker_list:
            logger.warning("Ticker list is empty")
            return results

        logger.info(f"🚀 Fetching {len(ticker_list)} stocks with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    self._fetch_single_by_period, 
                    ticker, 
                    period, 
                    interval,
                    actions,
                ): ticker 
                for ticker in ticker_list
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[ticker] = df
                except Exception as e:
                    logger.error(f"{ticker}: Exception in thread: {e}")

        success_rate = len(results) / len(ticker_list) * 100 if ticker_list else 0
        logger.info(
            f"Collection complete: {len(results)}/{len(ticker_list)}"
            f"({success_rate}% success)"
        )

        return results

    def _fetch_single_by_date(
            self,
            ticker: str,
            start_date: str,
            end_date: str,
            interval: str = "1d",
            actions : bool = False,)-> Optional[pd.DataFrame]:
        """날짜 범위 지정 헬퍼 메서드"""

        for attempt in range(1,self.max_retries+1):
            try:
                stock = yf.Ticker(ticker=ticker)
                df = stock.history(
                    start = start_date,
                    end = end_date,
                    interval = interval,
                    auto_adjust = False,
                    actions = actions
                )


                if df.empty:
                    logger.warning(f"{ticker}: Empty data returned")
                    return None

                if len(df) < 10:
                    logger.warning(f"{ticker}: Insufficient data {len(df)} rows")
                    return None

                #형식 통일 (대문자 -> 소문자) (Date 인덱스 -> 컬럼) (띄어쓰기 -> '_')
                df = df.reset_index()
                df.columns = (df.columns.str.strip()
                                        .str.lower()
                                        .str.replace(' ','_'))
                
                logger.info(f"{ticker}: {len(df)} records fetched")
                return df

            except RequestException as e:
                logger.warning(
                    f"{ticker}: Network error (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    sleep_time = 2**attempt
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"{ticker}: Unexpected error: {e}")
                return None
        
        logger.info(f"{ticker}: Failed after {self.max_retries} attempts")
        return None
    
    def fetch_multiple_by_date(
        self, 
        ticker_list: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = "1d",
        actions: bool = False,
        ) -> Dict[str,pd.DataFrame]:
        """
        특정 날짜 범위로 수집 (백테스팅용)
        
        Args:
            ticker_list: 종목 코드 리스트
            start_date: 시작일 "YYYY-MM-DD"
            end_date: 종료일 "YYYY-MM-DD"
        """

        results = {}
        if not ticker_list:
            logger.warning("Ticker list is empty")
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    self._fetch_single_by_date, 
                    ticker, 
                    start_date, 
                    end_date,
                    interval,
                    actions,
                ): ticker
                for ticker in ticker_list
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                df = future.result()
                if df is not None:
                    results[ticker] = df
            
        success_rate = len(results)/len(ticker_list) * 100 if ticker_list else 0
        logger.info(f"Fetch Complete: {len(results)}/{len(ticker_list)}")
        logger.info(f"({success_rate}%) Success")
        return results


if __name__ == '__main__':
    print("\n" + "="*70)
    print("StockDataFetcher 테스트")
    print("="*70)

    # [1] StockDataFetcher 초기화
    print("\n[1] StockDataFetcher 초기화...")
    try:
        fetcher = StockDataFetcher(max_workers=5, max_retries=3)
        print(f"✓ Fetcher 생성 완료")
        print(f"  - max_workers: {fetcher.max_workers}")
        print(f"  - max_retries: {fetcher.max_retries}")
    except Exception as e:
        print(f"✗ Fetcher 생성 실패: {e}")
        exit(1)

    korean_tickers = ["005930.KS", "000660.KS"]  # 삼성전자, SK하이닉스

    # [2] fetch_multiple_by_period() 테스트
    print("\n[2] fetch_multiple_by_period() 테스트 (기간 기반)...")
    try:
        df_dict_period = fetcher.fetch_multiple_by_period(
            ticker_list=korean_tickers,
            period="1y",
            interval="1d"
        )
        print(f"✓ 데이터 수집 완료: {len(df_dict_period)}/{len(korean_tickers)} 종목")

        for ticker, df in df_dict_period.items():
            print(f"\n  {ticker}:")
            print(f"    - 행 수: {len(df)}")
            print(f"    - 컬럼: {list(df.columns)}")
            print(f"    - 날짜 범위: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
            # 필수 컬럼 확인
            required_cols = {'open', 'high', 'low', 'close', 'volume'}
            actual_cols = set(df.columns)
            if required_cols.issubset(actual_cols):
                print(f"    - 필수 컬럼: ✓")
            else:
                print(f"    - 필수 컬럼: ✗ (누락: {required_cols - actual_cols})")
    except Exception as e:
        print(f"✗ fetch_multiple_by_period 실패: {e}")

    # [3] fetch_multiple_by_date() 테스트 (날짜 범위 기반)
    print("\n[3] fetch_multiple_by_date() 테스트 (날짜 범위 기반)...")
    try:
        df_dict_date = fetcher.fetch_multiple_by_date(
            ticker_list=korean_tickers,
            start_date="2024-01-01",
            end_date="2024-12-31",
            interval="1d"
        )
        print(f"✓ 데이터 수집 완료: {len(df_dict_date)}/{len(korean_tickers)} 종목")

        for ticker, df in df_dict_date.items():
            print(df.head(5))
            print(f"\n  {ticker}:")
            print(f"    - 행 수: {len(df)}")
            print(f"    - 날짜 범위: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    except Exception as e:
        print(f"✗ fetch_multiple_by_date 실패: {e}")

    # [4] 에러 처리 테스트 - 빈 리스트
    print("\n[4] 에러 처리 테스트 (빈 리스트)...")
    try:
        result = fetcher.fetch_multiple_by_period(ticker_list=[], period="1y")
        if len(result) == 0:
            print(f"✓ 빈 리스트 처리 정상: {result}")
        else:
            print(f"✗ 빈 리스트 처리 오류: {result}")
    except Exception as e:
        print(f"✗ 에러 처리 실패: {e}")

    # [5] 에러 처리 테스트 - 잘못된 티커
    print("\n[5] 에러 처리 테스트 (잘못된 티커)...")
    try:
        result = fetcher.fetch_multiple_by_period(
            ticker_list=["INVALID.KS"],
            period="1y"
        )
        if len(result) == 0:
            print(f"✓ 잘못된 티커 처리 정상: 데이터 없음")
        else:
            print(f"⚠ 잘못된 티커도 데이터 반환: {len(result)}개")
    except Exception as e:
        print(f"✗ 에러 처리 실패: {e}")

    # [6] 병렬 처리 성능 테스트
    print("\n[6] 병렬 처리 검증 (여러 종목 동시 조회)...")
    try:
        many_tickers = ["005930.KS", "000660.KS", "035720.KS", "035420.KS"]
        result = fetcher.fetch_multiple_by_period(
            ticker_list=many_tickers,
            period="1d"
        )
        print(f"✓ {len(many_tickers)}개 종목 조회: {len(result)}/{len(many_tickers)} 성공")
    except Exception as e:
        print(f"✗ 병렬 처리 실패: {e}")

    print("\n" + "="*70)
    print("모든 테스트 완료!")
    print("="*70)