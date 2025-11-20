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
    def __init__(self, max_workers: int = 3, max_retries: int = 3, per_request_delay_sec: float = 1.5): #기본 시간 지연 추가
        """
        Args:
            max_workers: 동시 실행 스레드 수 (Yahoo API 부담 고려)
            max_retries: 재시도 횟수
            per_request_delay_sec: 개별 다운로드 사이 강제 지연(초)  # 변경 사항
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.per_request_delay_sec = per_request_delay_sec  # 변경 사항

    def fetch_single_stock(
            self, 
            ticker: str, 
            period: str = "1y", 
            interval: str = "1d",
            auto_adjust : bool = False,
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
                    auto_adjust = auto_adjust,
                    actions = actions
                )
                #data validation
                if df.empty:
                    logger.warning(f"{ticker}: Empty data returns")
                    return None

                if len(df) < 10:
                    logger.warning(f"{ticker}: Insufficient data ({len(df)} rows)")

                logger.info(f"{ticker}: {len(df)} records fetched")
                if self.per_request_delay_sec > 0:  # 변경 사항: 요청 간 지연
                    time.sleep(self.per_request_delay_sec)
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

    def fetch_multiple_stocks(
            self,
            ticker_list: List[str],
            period: str = "1y",
            interval: str = "1d",
            auto_adjust: bool = False,
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
                    self.fetch_single_stock, 
                    ticker, 
                    period, 
                    interval,
                    auto_adjust,
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

    def _fetch_with_dates(
            self,
            ticker: str,
            start_date: str,
            end_date: str,
            interval: str = "1d",
            auto_adjust : bool = False,
            actions : bool = False,)-> Optional[pd.DataFrame]:
        """날짜 범위 지정 헬퍼 메서드"""

        for attempt in range(1,self.max_retries+1):
            try:
                stock = yf.Ticker(ticker=ticker)
                df = stock.history(
                    start = start_date,
                    end = end_date,
                    interval = interval,
                    auto_adjust = auto_adjust,
                    actions = actions
                )

                #data validation
                if df.empty:
                    logger.warning(f"{ticker}: Empty data returned")
                    return None

                if len(df) < 10:
                    logger.warning(f"{ticker}: Insufficient data {len(df)} rows")
                    return None
                
                logger.info(f"{ticker}: {len(df)} records fetched")
                if self.per_request_delay_sec > 0:  # 변경 사항: 요청 간 지연
                    time.sleep(self.per_request_delay_sec)
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
    
    def fetch_with_date_range(
        self, 
        ticker_list: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = "1d",
        auto_adjust: bool = False,
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
                    self._fetch_with_dates, 
                    ticker, 
                    start_date, 
                    end_date,
                    interval,
                    auto_adjust, 
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
    fetcher = StockDataFetcher()

    exp = '035420.KS'

    #df = fetcher.get_single_stock(exp)
    df = fetcher._fetch_with_dates(ticker=exp, start_date = "2025-09-01", end_date = "2025-10-01")
    print(df.head())

    tickers = ["005930.KS", "000660.KS", "035720.KS", "035420.KS"]
    #df_dict = fetcher.fetch_multiple_stocks(tickers=tickers)
    df_dict = fetcher.fetch_with_date_range(ticker_list=tickers, start_date = "2025-09-01", end_date = "2025-10-01")
    for ticker in tickers:
        print(f"\nticker : {ticker}")
        print(df_dict[ticker].head(5))
