# src/data/data_fetcher.py

import time
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
from requests.exceptions import RequestException

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
    
    def __init__(self, max_workers: int = 5, max_retries: int = 3):
        """
        Args:
            max_workers: 동시 실행 스레드 수 (Yahoo API 부담 고려)
            max_retries: 재시도 횟수
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
    
    def fetch_single_stock(
        self, 
        ticker: str, 
        period: str = "1y",
        interval: str = "1d"
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
        for attempt in range(1, self.max_retries + 1):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(
                    period=period, 
                    interval=interval,
                    auto_adjust=False, # 분할/배당 자동 조정
                    actions = False
                )
                
                # 데이터 검증
                if df.empty:
                    logger.warning(f"⚠ {ticker}: Empty data returned")
                    return None
                
                if len(df) < 10:
                    logger.warning(f"⚠ {ticker}: Insufficient data ({len(df)} rows)")
                    return None
                
                logger.info(f"✓ {ticker}: {len(df)} records fetched")
                return df
                
            except RequestException as e:
                logger.warning(
                    f"⚠ {ticker}: Network error (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    sleep_time = 2 ** attempt  # Exponential backoff: 2, 4, 8초
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"✗ {ticker}: Unexpected error: {e}")
                return None
        
        logger.error(f"✗ {ticker}: Failed after {self.max_retries} attempts")
        return None
    
    def fetch_multiple_stocks(
        self,
        ticker_list: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
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
        
        logger.info(f"🚀 Fetching {len(ticker_list)} stocks with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 종목에 대한 Future 생성
            future_to_ticker = {
                executor.submit(
                    self.fetch_single_stock, ticker, period, interval
                ): ticker
                for ticker in ticker_list
            }
            
            # 완료된 순서대로 처리
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[ticker] = df
                except Exception as e:
                    logger.error(f"✗ {ticker}: Exception in thread: {e}")
        
        success_rate = len(results) / len(ticker_list) * 100
        logger.info(
            f"📊 Collection complete: {len(results)}/{len(ticker_list)} "
            f"({success_rate:.1f}% success)"
        )
        
        return results
    
    def fetch_with_date_range(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        특정 날짜 범위로 수집 (백테스팅용)
        
        Args:
            ticker_list: 종목 코드 리스트
            start_date: 시작일 "YYYY-MM-DD"
            end_date: 종료일 "YYYY-MM-DD"
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    self._fetch_with_dates, ticker, start_date, end_date
                ): ticker
                for ticker in ticker_list
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                df = future.result()
                if df is not None:
                    results[ticker] = df
        
        return results
    
    def _fetch_with_dates(self, ticker, start, end):
        """날짜 범위 지정 헬퍼 메서드"""
        for attempt in range(1, self.max_retries + 1):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start, 
                                    end=end, 
                                    auto_adjust=True,
                                    actions = False
                                    )
                
                if not df.empty:
                    logger.info(f"✓ {ticker}: {len(df)} records")
                    return df
                    
            except Exception as e:
                logger.warning(f"⚠ {ticker}: Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        
        return None


# 사용 예시
if __name__ == "__main__":
    fetcher = StockDataFetcher(max_workers=5, max_retries=3)
    
    korean_stocks = [
        "005930.KS",  # 삼성전자
        "000660.KS",  # SK하이닉스
        "035720.KS",  # 카카오
        "035420.KS",  # NAVER
    ]
    
    data_dict = fetcher.fetch_multiple_stocks(korean_stocks, period="2y")
    
    for ticker, df in data_dict.items():
        print(f"\n{ticker}:")
        print(df.head())