# StockDataFetcher
# yfinance로 OHLCV를 수집하고 병렬 처리/재시도/컬럼 정규화를 수행하는 헬퍼
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from requests.exceptions import RequestException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """yfinance 기반 주식 데이터를 기간/날짜 범위로 수집하는 유틸리티"""

    def __init__(
        self,
        max_workers: int = 3,
        max_retries: int = 3,
    ):
        """
        Args:
            max_workers: 동시 실행 스레드 수
            max_retries: 실패 시 재시도 횟수 (지수 백오프)
        """
        self.max_workers = max_workers
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    # 단일 수집
    # ------------------------------------------------------------------ #
    def _fetch_single_by_period(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        actions: bool = False,
    ) -> Optional[pd.DataFrame]:
        # 기간 단위로 단일 종목 조회, 실패 시 지수 백오프 재시도
        for attempt in range(1, self.max_retries + 1):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    actions=actions,
                )
                if df.empty:
                    logger.warning("%s: Empty data returned", ticker)
                    return None

                # Date 인덱스를 컬럼으로 풀고 snake_case로 정규화
                df = df.reset_index()
                df.columns = (
                    df.columns.str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                )

                logger.info("%s: %d records fetched", ticker, len(df))
                return df

            except RequestException as exc:
                logger.warning(
                    "%s: Network error (attempt %d/%d): %s",
                    ticker,
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2**attempt)

            except Exception as exc:
                logger.error("%s: Unexpected error: %s", ticker, exc)
                return None

        logger.error("%s: Failed after %d attempts", ticker, self.max_retries)
        return None

    def _fetch_single_by_date(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        actions: bool = False,
    ) -> Optional[pd.DataFrame]:
        # 날짜 범위로 단일 종목 조회
        for attempt in range(1, self.max_retries + 1):
            try:
                stock = yf.Ticker(ticker=ticker)
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,
                    actions=actions,
                )

                if df.empty:
                    logger.warning("%s: Empty data returned", ticker)
                    return None

                # Date 인덱스를 컬럼으로 풀고 snake_case로 정규화
                df = df.reset_index()
                df.columns = (
                    df.columns.str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                )

                logger.info("%s: %d records fetched", ticker, len(df))
                return df

            except RequestException as exc:
                logger.warning(
                    "%s: Network error (attempt %d/%d): %s",
                    ticker,
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2**attempt)

            except Exception as exc:
                logger.error("%s: Unexpected error: %s", ticker, exc)
                return None

        logger.info("%s: Failed after %d attempts", ticker, self.max_retries)
        return None

    # ------------------------------------------------------------------ #
    # 병렬 수집
    # ------------------------------------------------------------------ #
    def fetch_multiple_by_period(
        self,
        ticker_list: List[str],
        period: str,
        interval: str = "1d",
        actions: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        # 여러 종목을 기간 기준 병렬 수집
        results: Dict[str, pd.DataFrame] = {}
        if not ticker_list:
            logger.warning("Ticker list is empty")
            return results

        logger.info("Fetching %d tickers (period=%s, interval=%s)", len(ticker_list), period, interval)

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
                except Exception as exc:
                    logger.error("%s: Exception in thread: %s", ticker, exc)

        success_rate = len(results) / len(ticker_list) * 100 if ticker_list else 0
        logger.info("Collection complete: %d/%d (%.1f%%)", len(results), len(ticker_list), success_rate)
        return results

    def fetch_multiple_by_date(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        actions: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        # 여러 종목을 날짜 범위 기준 병렬 수집
        results: Dict[str, pd.DataFrame] = {}
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
                try:
                    df = future.result()
                    if df is not None:
                        results[ticker] = df
                except Exception as exc:
                    logger.error("%s: Exception in thread: %s", ticker, exc)

        success_rate = len(results) / len(ticker_list) * 100 if ticker_list else 0
        logger.info("Fetch complete: %d/%d (%.1f%%)", len(results), len(ticker_list), success_rate)
        return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("StockDataFetcher quick self-test")
    print("=" * 70)

    fetcher = StockDataFetcher(max_workers=2, max_retries=2)
    sample = ["005930.KS"]
    df_dict = fetcher.fetch_multiple_by_period(ticker_list=sample, period="1mo")
    for t, df in df_dict.items():
        print(f"{t}: {len(df)} rows, columns={list(df.columns)}")
