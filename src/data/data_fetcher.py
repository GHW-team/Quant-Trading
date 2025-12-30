# StockDataFetcher
# yfinance로 OHLCV를 수집하고 병렬 처리/재시도/컬럼 정규화를 수행하는 헬퍼
import logging
import time
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from ratelimit import limits, sleep_and_retry

import re
import pandas as pd
import yfinance as yf
import exchange_calendars as xcals
from requests.exceptions import RequestException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """yfinance 기반 주식 데이터를 기간/날짜 범위로 수집하는 유틸리티"""

    def __init__(
        self,
        max_workers: int = 3,
        max_retries: int = 3,
        custom_holidays_path: str = "config/market_holidays.yaml"
    ):
        """
        Args:
            max_workers: 동시 실행 스레드 수
            max_retries: 실패 시 재시도 횟수 (지수 백오프)
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.max_workers = max_workers
        self.max_retries = max_retries

        # 커스텀 휴장일 로드
        self.custom_holidays = self._load_custom_holidays(custom_holidays_path)

        # calendar 캐시 추가
        self.calendar_cache = {}

    
    # ------------------------------------------------------------------ #
    # 헬퍼 함수
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convert_period_to_dates(period: str) -> Tuple[str,str]:
        """
        Period 문자열을 start_date, end_date로 변환합니다
        
        Args:
            period: 기간 문자열 (예: "1d", "3y", "6m")

        Returns:
            (start_date, end_date) 튜플 (YYYY-MM-DD) 형식
        """
        end_date = datetime.now()

        if period == "ytd":
            start_date = datetime(end_date.year,1,1)
        elif period == "max":
            start_date = datetime(1970,1,1)
        else:
            match = re.match(r'^(\d+)(y|mo|d)$',period)

            if not match:
                raise ValueError(f"Invalid period format: {period}. Expected format: <number><unit> (e.g., '1y','6mo','5d')")
            
            value = int(match.group(1))
            unit = match.group(2)

            if unit == "d":
                start_date = end_date - timedelta(days=value)
            elif unit == "mo":
                start_date = end_date - relativedelta(months=value)
            elif unit == "y":
                start_date = end_date - relativedelta(years=value)
            else:
                raise ValueError(f"Unsupported unit: {unit}")
        
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    @staticmethod    
    def _get_exchange_code(ticker_code: str) -> str:
        """티커 문자열을 분석해 시장 정보를 추론"""
        ticker_upper = ticker_code.upper()
        if ticker_upper.endswith('.KS'):
            return 'XKRX'
        if ticker_upper.endswith('.KQ'):
            return 'XKRX'
        if ticker_upper.endswith('.T'):
            return 'XTKS'
        if ticker_upper.endswith('.HK'):
            return 'XHKG'
        if '.' not in ticker_upper:
            return 'XNYS'
        return None
    
    @staticmethod
    def _load_custom_holidays(config_path: str) -> dict:
        """
        수동으로 추가한 휴장일을 YAML에서 로드
        
        Returns:
            {
                'XKRX': [datetime(2022, 1, 3), datetime(2022, 5, 9), ...],
                'XNYS': [...],
            }
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Custom holidays file not found: {config_path}")
                return {}
            
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 문자열 날짜를 datetime으로 변환
            holidays_dict = {}
            for exchange, date_list in config.items():
                if date_list:
                    holidays_dict[exchange] = [
                        pd.to_datetime(date) for date in date_list
                    ]
                else:
                    holidays_dict[exchange] = []
            
            logger.info(f"Loaded custom holidays: {holidays_dict}")
            return holidays_dict
        
        except Exception as e:
            logger.error(f"Failed to load custom holidays: {e}")
            return {}
    
    def _validate_and_fill_gaps(
            self, 
            start_date: str,
            end_date: str,
            df_dict: Dict[str,pd.DataFrame],
        )-> Dict[str,pd.DataFrame]:
        """
        1.시장 거래일의 데이터와 다운받은 데이터를 대조하여 누락된 날짜를 NaN으로 채움
        2.목표 fetch 기간과 실제 fetch 된 데이터 거래 기간 대조 및 로깅
        3.수동으로 추가한 휴장일 제외 (exchange_calendars 라이브러라가 놓친 날짜)
        """
        validated_df_dict = {}
        logged_holidays = set()

        for ticker, df in df_dict.items():
            try:
                # 시장 정보 추론
                exchange = self._get_exchange_code(ticker)
                if exchange is None:
                    logger.warning(f"Cannot infer exchange from ticker: {ticker}")
                    continue
                
                # date 열 index로 전환
                df = df.set_index('date')

                original_tz = df.index.tz
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # 날짜 설정
                df_start_date = df.index.min()
                df_end_date = df.index.max()

                # 거래일 달력 불러오기
                if exchange in self.calendar_cache:
                    # 이미 존재하면 활용
                    cal = self.calendar_cache[exchange]
                else:
                    # 없는 경우 다운로드
                    cal = None
                    for attempt in range(1, self.max_retries + 1):
                        try:
                            cal = xcals.get_calendar(exchange)   
                            self.calendar_cache[exchange] = cal
                            logger.debug(f"Loaded calendar for {exchange}")
                            break
                        except Exception as e:
                            # 실패 시 재시도 (최대 max_retires번)
                            if attempt < self.max_retries:
                                time.sleep(2**attempt)
                    # 다운로드 실패 시 해당 종목 스킵
                    if cal is None:    
                        logger.error(
                            f"{ticker}: Failed to get {exchange} calendar after {self.max_retries} attempts. Skipping validation."
                        )
                        continue

                # 지정 범위 거래일 불러오기
                date_list = cal.sessions_in_range(df_start_date, df_end_date)
                if date_list.tz is not None:
                    date_list = date_list.tz_localize(None)
                
                # 수동 휴장일 제외
                if exchange in self.custom_holidays:
                    custom_holidays_pd = pd.DatetimeIndex(self.custom_holidays[exchange])
                    
                    # 타임존 제거
                    if custom_holidays_pd.tz is not None:
                        custom_holidays_pd = custom_holidays_pd.tz_localize(None)
                    
                    # 수동 휴장일 제외
                    excluded_dates = date_list.intersection(custom_holidays_pd)
                    if not excluded_dates.empty and exchange not in logged_holidays:
                        logger.debug(
                            f"{exchange}: Excluding {len(excluded_dates)} custom holidays from trading days: "
                            f"{[d.strftime('%Y-%m-%d') for d in excluded_dates]}"
                        )
                        logged_holidays.add(exchange)

                    date_list = date_list.difference(custom_holidays_pd)

                # 목표 fetch기간보다 fetch된 기간이 짧은 경우 로깅
                target_date = cal.sessions_in_range(start_date,end_date)
                target_start_date = target_date.min()
                target_end_date = target_date.max()

                if target_start_date != df_start_date or target_end_date != df_end_date:
                    logger.warning(f"{ticker}: Loaded less than the target period")
                    logger.warning(f"Fetched Date: {df_start_date} ~ {df_end_date}, Length: {len(df)}")

                # 누락된 날짜 탐지
                missing_dates = date_list.difference(df.index)
                if not missing_dates.empty:
                    logger.warning(f"{ticker}: Found {len(missing_dates)} missing trading days. Filling with NaN")
                    logger.info(f"Missing date list: {missing_dates.date}")

                    # 누락된 날짜 채우기
                    df = df.reindex(date_list)
                    df['volume'] = df['volume'].fillna(0)
                    df = df.ffill()
                    df.index.name = 'date'

                # tz 복원
                if original_tz is not None:
                    df.index = df.index.tz_localize(original_tz)

                # 인덱스 복원
                df = df.reset_index()

                validated_df_dict[ticker] = df

            except Exception as e:
                logger.warning(f"{ticker}: Failed to validate market calendar: {e}")
                continue

        return validated_df_dict
        
    # ------------------------------------------------------------------ #
    # 단일 수집
    # ------------------------------------------------------------------ #
    @sleep_and_retry
    @limits(calls=20, period=1)
    def _fetch_single_by_date(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        actions: bool = False,
    ) -> Optional[pd.DataFrame]:
        # 날짜 파라미터 검증
        date_format = "%Y-%m-%d"
        try:
            start_dt = datetime.strptime(start_date, date_format)
            end_dt = datetime.strptime(end_date, date_format)
        except ValueError as e:
            raise ValueError(
                f"Invalid date format. Expected YYYY-MM-DD format. "
                f"start_date='{start_date}', end_date='{end_date}'. Error: {e}"
            )

        # 논리적 유효성 검사
        if start_dt >= end_dt:
            raise ValueError(
                f"start_date ({start_date}) cannot be later than end_date ({end_date})"
            )

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

                logger.debug("%s: %d records fetched", ticker, len(df))

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

                    # 정상 fetch된 경우
                    if df is not None:
                        # NaN 값 탐지
                        nan_count = df.isna().sum().sum()
                        if nan_count:
                            logger.info(f"NaN values found in fetched data: {ticker}, NaN count: {nan_count}")

                        #결과 추가
                        results[ticker] = df
                except Exception as exc:
                    logger.error("%s: Exception in thread: %s", ticker, exc)


        # 시장 거래일의 데이터가 모두 존재하는지 검증 및 채우기
        results = self._validate_and_fill_gaps(start_date, end_date, results)

        failed_tickers = set(ticker_list) - set(results.keys())
        success_rate = len(results) / len(ticker_list) * 100 if ticker_list else 0

        # Fetch 실패한 티커 로깅
        if failed_tickers:
            logger.info(f"Failed Tickers: {failed_tickers}")

        # 성공률 로깅
        logger.info("Fetch complete: %d/%d (%.1f%%)", len(results), len(ticker_list), success_rate)
        return results