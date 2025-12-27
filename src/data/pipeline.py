"""데이터 수집·저장·지표계산 파이프라인 (로직 변경 없이 주석만 추가)."""
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

from src.data.data_fetcher import StockDataFetcher
from src.data.db_manager import DatabaseManager
from src.data.indicator_calculator import IndicatorCalculator

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(
        self,
        db_path: str = 'data/database/stocks.db',
        max_workers: int = 5,
        max_retries: int = 3,
        batch_size_default: int = 100,
    ):
        self.db_path = db_path 
        self.batch_size_default = batch_size_default
        max_workers = max_workers
        max_retries = max_retries

        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.fetcher = StockDataFetcher(
            max_retries=max_retries,
            max_workers=max_workers,
        )
        self.calculator = IndicatorCalculator()

        logger.info("DataPipeline initialized")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _calculate_extended_start_date(self, start_date: str, indicator_list: List[str]) -> Optional[str]:
        """
        지표 계산을 위해 '안전 마진'이 포함된 과거 시작 날짜를 계산합니다.

        영업일/휴일을 고려하여 1.6배수를 적용하고 추가 10일의 여유를 더합니다.

        Args:
            start_date: 원래 요청한 시작 날짜 (YYYY-MM-DD 형식)
            indicator_list: 지표 리스트

        Returns:
            확장된 시작 날짜 (YYYY-MM-DD 형식), start_date가 None이면 None
        """

        # 필요한 룩백 일수 조회 (예: ma_200 -> 200일)
        lookback_days = IndicatorCalculator.get_lookback_days(indicator_list)

        # 영업일/휴일 고려하여 1.6배수 적용 + 10일 여유 (넉넉하게 잡음)
        # 이유: 일주일에 5개 영업일만 있으므로 약 1.4~1.6배가 필요함
        safe_lookback = int(lookback_days * 1.6) + 10

        dt_start = pd.to_datetime(start_date)
        dt_extended_start = dt_start - timedelta(days=safe_lookback)

        extended_date_str = dt_extended_start.strftime('%Y-%m-%d')
        logger.info(f"Date extended for lookback: {start_date} → {extended_date_str} (-{safe_lookback} days, "
                   f"base lookback: {lookback_days} days)")
        return extended_date_str

    def _validate_period_and_date(self, start_date: str, end_date: str, period: str):
        #start/end or period validation
        if (start_date or end_date) and period:
            raise ValueError(
                """
                Ambiguous configuration!!
                날짜 지정 방식(start/end)과 기간 지정 방식(period)둘 중 하나만 선택하세요!!
                """
            )

        if (start_date is not None and end_date is None) or \
            (start_date is None and end_date is not None):
                raise ValueError(
                    """
                    Incomplete date range!!
                    start_date 와 end_date는 모두 입력되거나 모두 입력되지 않아야 합니다.
                    """
                )

        if start_date and end_date:
            # 날짜 형식 검증
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

    def _to_datetime_safe(self, date_like):
        if date_like is None:
            return None
        return pd.to_datetime(date_like)

    def _coverage_ok(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
        tol_days: int = 7,
        *,
        ticker_code: Optional[str] = None,
        check_gaps: bool = True,
        # 아래 3개가 "중간 공백 허용치" (기본=0이면 1개라도 비면 False)
        max_missing_trading_days: int = 0,
        max_missing_ratio: float = 0.0,
        # 캘린더를 못 쓰는 경우(예외/라이브러리 문제) fallback: 날짜 간격이 이 값보다 크면 gap으로 판단
        max_gap_days_fallback: int = 10,
        # 상장폐지/가격정지의 경우(누락 데이터 시작부터 끝 커버리지까지의 경우)
        allow_early_end: bool = True,
    ) -> bool:
        """DB에서 불러온 데이터가 요청 구간을 커버하는지 검사한다.

        기존: min/max 날짜만 확인
        확장: (가능하면) 거래일 캘린더 기반으로 중간 누락(날짜 row 누락)까지 검사

        - tol_days: start/end는 주말/휴일로 인해 +-tol_days 오차 허용
        - check_gaps=True면 중간 거래일 누락도 검사
        - 기본 설정(누락 허용 0): 중간에 거래일이 1개라도 빠지면 False
        """

        # ---------- 기본 유효성 ----------
        if df is None or df.empty or "date" not in df.columns:
            return False

        # date 정리
        d = pd.to_datetime(df["date"], errors="coerce")
        d = d.dropna()
        if d.empty:
            return False

        s = self._to_datetime_safe(start_date)
        e = self._to_datetime_safe(end_date)
        tol = pd.Timedelta(days=tol_days)

        dmin = d.min()
        dmax = d.max()

        # ---------- 1) 시작 커버리지 ----------
        if s is not None and dmin > s + tol:
            return False
        if s is not None and dmax < s - tol:
            return False
        
        # 끝 커버리지: early-end 허용이면 '실패'시키지 말고 검사 구간을 dmax로 축소
        effective_end = e
        if e is not None and dmax < e - tol:
            if not allow_early_end:
                return False
            effective_end = dmax

        # gap 검사 안 하면 여기서 끝
        if not check_gaps:
            return True

        # 검사 구간(요청 구간이 없으면 df 구간 사용)
        rs = s if s is not None else dmin
        re_ = effective_end if effective_end is not None else dmax

        # ---------- 2) "가격 데이터"면 NaN row도 누락으로 취급 (지표는 초반 NaN이 정상이라 제외) ----------
        price_cols = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in df.columns]
        nan_like_missing_dates = pd.DatetimeIndex([])
        if price_cols:
            mask_all_nan = df[price_cols].isna().all(axis=1)
            if mask_all_nan.any():
                nan_dates = pd.to_datetime(df.loc[mask_all_nan, "date"], errors="coerce").dropna()
                if not nan_dates.empty:
                    nan_like_missing_dates = pd.DatetimeIndex(nan_dates)

        # ---------- 3) 거래일 캘린더 기반 gap 체크 (가능하면) ----------
        try:
            import exchange_calendars as xcals

            exchange_code = None
            if ticker_code:
                # data_fetcher.py에 있는 추론 로직 재사용 (.KS/.KQ -> XKRX 등)
                exchange_code = self.fetcher._get_exchange_code(ticker_code)

            cal = None
            if exchange_code:
                cal = self.fetcher.calendar_cache.get(exchange_code)
                if cal is None:
                    cal = xcals.get_calendar(exchange_code)
                    self.fetcher.calendar_cache[exchange_code] = cal

            if cal is not None:
                expected = cal.sessions_in_range(rs, re_)
                if expected.tz is not None:
                    expected = expected.tz_localize(None)

                # 수동 휴장일 제외(프로젝트에서 쓰는 것과 동일)
                if hasattr(self.fetcher, "custom_holidays") and exchange_code in self.fetcher.custom_holidays:
                    custom = pd.DatetimeIndex(self.fetcher.custom_holidays[exchange_code])
                    if custom.tz is not None:
                        custom = custom.tz_localize(None)
                    expected = expected.difference(custom)

                # 실제 날짜 집합
                actual = pd.DatetimeIndex(pd.to_datetime(df["date"], errors="coerce").dropna().unique())
                if actual.tz is not None:
                    actual = actual.tz_localize(None)

                missing = expected.difference(actual)

                # 가격데이터면 "row는 있는데 전부 NaN"도 사실상 누락으로 합침
                if not nan_like_missing_dates.empty:
                    if nan_like_missing_dates.tz is not None:
                        nan_like_missing_dates = nan_like_missing_dates.tz_localize(None)
                    missing = missing.union(nan_like_missing_dates.intersection(expected))

                total = len(expected)
                miss_n = len(missing)
                miss_ratio = (miss_n / total) if total > 0 else 0.0

                # 누락 허용치 판단 (기본=0이면 1개라도 누락이면 False)
                if miss_n > max_missing_trading_days:
                    return False
                if miss_ratio > max_missing_ratio:
                    return False

                return True

        except Exception:
            # 캘린더 로딩 실패/라이브러리 문제면 fallback로 내려감
            pass

        # ---------- 4) fallback: 연속 날짜 간격이 너무 크면 gap으로 판단 ----------
        dd = pd.Series(pd.to_datetime(df["date"], errors="coerce").dropna().sort_values().unique())
        if len(dd) < 2:
            return True

        max_gap = (dd.diff().dt.days.max())
        if pd.notna(max_gap) and max_gap > max_gap_days_fallback:
            return False

        # fallback에서는 NaN row까지는 강하게 판단하지 않고(오탐 방지), 필요하면 여기서 추가로 확장 가능
        return True

    def run_full_pipeline(
        self,
        ticker_list: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        actions: bool = False,
        update_if_exists: bool = True,
        indicator_list: List[str] = None,
        version: Optional[str] = 'v1.0',
        batch_size: int = 100,
        prefer_db: bool = True,
    )-> Dict[str,pd.DataFrame]:
        """
        다음 단계 실행 (배치 처리)
        1.주식 종목 OHLCV데이터 불러오기
        2.OHLCV 데이터 DB 저장
        3.가격 데이터로 지표 계산
        4.지표 데이터 DB 저장

        Args:
            ticker_list : 종목 리스트
            start_date : 시작 날짜
            end_date : 종료 날짜
            period : 기간 (현재 날짜 기준)
            ==> (start/end) 와 period 둘중 하나만 입력
            interval : 데이터 간격
            actions : 분할 등 부가정보 포함 여부
            update_if_exists: db에 존재하는 데이터 새로 업데이트하기
            indicator_list : 지표 리스트
            version : 지표 계산 로직 버전
            batch_size: 배치 크기

        Returns:
            {ticker : DataFrame} 딕셔너리
            DataFrame: OHLCV데이터 + 지표데이터
        """

        # 기본 배치 크기 설정 (None/0 방지)
        batch_size = batch_size or self.batch_size_default

        results = {
            'step1_fetch': {},
            'step2_save_price' : {},
            'step3_calculate': {},
            'step4_save_indicator' : {},
            'summary': {},
        }

        try:
            # ============ start_date, end_date 와 period 충돌 검증, 형식 검증 ============
            self._validate_period_and_date(start_date, end_date, period)

            # ============ Pre-processing: Period를 start_date/end_date로 변환 ============
            if period:
                start_date, end_date = self.fetcher._convert_period_to_dates(period)
                logger.info(f"Period '{period}' converted to start_date={start_date}, end_date={end_date}")

            # ============ Lookback 계산 및 적용 (안전 마진 포함) ============
            # 룩백 적용하여 확장된 시작날짜
            extended_start_date = self._calculate_extended_start_date(start_date, indicator_list)

            # 가격 파이프라인
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"Price data pipeline Start (included lookback period)")
            logger.info(f"{'='*60}")
            all_price_dict = self.run_price_pipeline(
                ticker_list=ticker_list,
                start_date=extended_start_date,
                end_date=end_date,
                period=period,
                interval=interval,
                actions=actions,
                update_if_exists=update_if_exists,
                batch_size=batch_size,
                prefer_db=prefer_db
            )

            # 가격 필터링
            filtered_price_dict = {}
            logger.info(f"____Filtering price data by original start date...____")
            for ticker, df in all_price_dict.items():
                filtered_dict = df[df['date'] >= start_date]
                if filtered_dict.empty:
                    logger.info(f"{ticker}: excluded from price data. No price data in original date period")
                else:
                    filtered_price_dict[ticker] = filtered_dict

            success_rate = (len(filtered_price_dict.keys()) / len(all_price_dict.keys())) * 100 if all_price_dict.keys() else 0 
            logger.info(f"Filtering Complete. Success rate: {success_rate:.2f} %")

            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"Indicator data pipeline Start")
            logger.info(f"{'='*60}")
            logger.info("")

            # 지표 파이프라인
            all_calculated_dict = self.run_indicator_pipeline(
                ticker_list=ticker_list,
                indicator_list=indicator_list,
                start_date=start_date,
                end_date=end_date,
                version=version,
                batch_size=batch_size,
                period=period,
                interval=interval,
                actions=actions,
                update_if_exists=update_if_exists,
                prefer_db=prefer_db
            )

            # 가격 데이터와 지표 데이터 병합
            merged_dict = {}
            for ticker in ticker_list:
                price_df = all_price_dict.get(ticker)
                indicator_df = all_calculated_dict.get(ticker)

                if price_df is not None and indicator_df is not None:
                    merged_df = pd.merge(price_df, indicator_df, how="inner", on="date")
                    merged_dict[ticker] = merged_df
                else:
                    logger.info(f"{ticker}: excluded from final result. Missing data exist")
            logger.info(f"Merged {len(merged_dict)} tickers successfully.")


            final_success_rate = (len(merged_dict.keys()) / len(ticker_list)) * 100
            logger.info(f"{'='*60}")
            logger.info(f"All data pipeline Completed.")
            logger.info(f"Final Success Count: {len(merged_dict.keys())}")
            logger.info(f"Final Success Rate: {final_success_rate:.2f} %")
            logger.info(f"{'='*60}")

            return merged_dict

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def run_price_pipeline(
            self,
            ticker_list: List[str],
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            period: Optional[str] = None,
            interval: str = "1d",
            actions: bool = False,
            update_if_exists: bool = True,
            batch_size: int = 100,
            prefer_db: bool = True,
    )-> Dict[str,pd.DataFrame]:
        """
        주식 종목 OHLCV데이터 불러오기 및 DB 저장 (배치 처리)

        Args:
            ticker_list : 종목 리스트
            start_date : 시작 날짜
            end_date : 종료 날짜
            period : 기간 (현재 날짜 기준)
            ==> (start/end) 와 period 둘중 하나만 입력
            interval : 데이터 간격
            actions : 분할 등 부가정보 포함 여부
            update_if_exists: db에 존재하는 데이터 새로 업데이트하기
            batch_size: 배치 크기

        Returns:
            {ticker : DataFrame} 딕셔너리
            DataFrame: OHLCV데이터
        """
        # 기본 배치 크기 설정 (None/0 방지)
        batch_size = batch_size or self.batch_size_default

        results = {
            'step1_fetch': {},
            'step2_save_price' : {},
        }
        try:
            # ============ start_date, end_date 와 period 충돌 검증, 형식 검증 ============
            self._validate_period_and_date(start_date, end_date, period)

            # ============ Pre-processing: Period를 start_date/end_date로 변환 ============
            if period:
                start_date, end_date = self.fetcher._convert_period_to_dates(period)
                logger.info(f"Period '{period}' converted to start_date={start_date}, end_date={end_date}")

            # ============ DB에 데이터 존재하는지 검증하는 로직. 존재하면 가져와서 반환 ===============
            requested_tickers = list(ticker_list)

            if prefer_db and not update_if_exists:
                db_price_dict = self.db_manager.load_price_data(
                    ticker_codes=ticker_list,
                    start_date=start_date,
                    end_date=end_date,
                )

                missing_tickers = []
                ok_count = 0
                for ticker in ticker_list:
                    df = db_price_dict.get(ticker)
                    if self._coverage_ok(df, start_date, end_date, ticker_code=ticker):
                        ok_count += 1
                    else:
                        missing_tickers.append(ticker)

                if not missing_tickers:
                    logger.info(f"✓ Price data loaded from DB for all tickers (n={ok_count})")
                    return db_price_dict

                logger.info(
                    f"DB missing/incomplete price data for {len(missing_tickers)}/{len(ticker_list)} tickers. "
                    f"Fetching only missing tickers..."
                )
                ticker_list = missing_tickers
            # ============ DB에 목표 데이터가 온전히 존재하지 않는다면, 밑의 fetch/save 작업 진행 ============

            # 전체 결과를 저장할 dict
            all_price_dict = {}

            # 배치로 처리
            total_tickers = len(ticker_list)
            num_batches = (total_tickers + batch_size - 1) // batch_size
            logger.info(f"Processing {total_tickers} tickers in {num_batches} batches of size {batch_size}")

            for i in range(0, total_tickers, batch_size):
                batch_tickers = ticker_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(f"{'-'*60}")
                logger.info(f"Batch {batch_num}/{num_batches}: Fetching {len(batch_tickers)} tickers")
                logger.info(f"{'-'*60}")

                # Step 1: Fetch batch data
                try:
                    logger.info("")
                    logger.info(f"____Step 1: Fetching data for batch {batch_num}...____")
                    batch_price_dict = self.fetcher.fetch_multiple_by_date(
                        ticker_list=batch_tickers,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        actions=actions,
                    )

                    results['step1_fetch'].update({
                        ticker : len(df) for ticker, df in batch_price_dict.items()
                    })

                except Exception as e:
                    logger.error(f"Failed to fetch batch {batch_num}: {e}")
                    for ticker in batch_tickers:
                        results['step1_fetch'][ticker] = 0
                    continue

                # Step 2: Save batch data using bulk save function
                try:
                    logger.info("")
                    logger.info(f"____Step 2: Saving price data for batch {batch_num}...____")

                    logger.info(f"Saving batch {batch_num} to database...")
                    batch_save_results = self.db_manager.save_price_data(
                        df_dict=batch_price_dict,
                        update_if_exists=update_if_exists,
                    )
                    results['step2_save_price'].update(batch_save_results)
                    logger.info(f"Batch {batch_num} saved successfully")

                except Exception as e:
                    logger.error(f"Failed to save batch {batch_num}: {e}")
                    for ticker in batch_price_dict.keys():
                        results['step2_save_price'][ticker] = 0
                    continue

                # 메모리 정리
                all_price_dict.update(batch_price_dict)
                del batch_price_dict

            #============Summary=============
            logger.info("")
            logger.info("="*60)
            logger.info("Price Pipeline Summary")
            logger.info("="*60)

            total_fetched = sum(results['step1_fetch'].values())
            total_saved_prices = sum(results['step2_save_price'].values())

            logger.info(f"Total fetched records: {total_fetched}")
            logger.info(f"Total saved price records: {total_saved_prices}")
                
            #결과 출력
            failed_tickers = set(ticker_list) - set(all_price_dict.keys())
            success_rate = (len(all_price_dict) / len(ticker_list)) * 100 if ticker_list else 0 

            logger.info(f"Total processed: {len(all_price_dict)}/{len(ticker_list)}")
            logger.info(f"Total success rate: {success_rate:.2f} %")
            if failed_tickers:
                logger.info(f"Total failed tickers: {failed_tickers}")

            logger.info(f"{'='*60}")
            logger.info(f"✓ Price pipeline completed successfully")
            logger.info(f"{'='*60}")
            logger.info("")
            
            if prefer_db and not update_if_exists:
                return self.db_manager.load_price_data(
                    ticker_codes=requested_tickers,
                    start_date=start_date,
                    end_date=end_date,
                )
            return all_price_dict

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
    def run_indicator_pipeline(
        self,
        ticker_list: List[str],
        indicator_list: List[str],
        start_date: str,
        end_date: str,
        version: Optional[str] = "v1.0",
        batch_size: int = 100,
        period: Optional[str] = None,
        interval: str = "1d",
        actions: bool = False,
        update_if_exists: bool = True,
        prefer_db: bool = True,
    )-> Dict[str,pd.DataFrame]:
        """
        db에서 가격 정보 불러와서 지표 계산 및 db에 지표 저장 (배치 처리)\n
        **주의**
            -지표계산에 필요한 과거데이터까지 추가로 db에서 불러오기 때문에 db에 데이터가 존재해야함.

        Args:
            ticker_list : 종목 리스트
            indicator_list : 지표 리스트
            start_date : 시작 날짜
            end_date : 종료 날짜
            version : 지표 계산 로직 버전
            batch_size: 배치 크기

        Returns:
            {ticker : DataFrame} 딕셔너리
            DataFrame: OHLCV데이터 + 지표데이터
        """
        # 기본 배치 크기 설정 (None/0 방지)
        batch_size = batch_size or self.batch_size_default

        results = {
            'step1_load_data' : {},
            'step2_calculate' : {},
            'step3_save_indicator' : {},
        }

        try:
            # ============ 날짜 유효성 검증 =============
            self._validate_period_and_date(start_date=start_date, end_date=end_date, period=None)

            # ============ DB에 데이터 존재하는지 검증하는 로직. 존재하면 가져와서 반환 ===============
            requested_tickers = list(ticker_list)
            
            if prefer_db and not update_if_exists:
                db_ind_dict = self.db_manager.load_indicators(
                    ticker_codes=ticker_list,
                    start_date=start_date,
                    end_date=end_date,
                )

                missing_tickers = []
                ok_count = 0
                required_cols = set(["date"] + (indicator_list or []))

                for ticker in ticker_list:
                    df = db_ind_dict.get(ticker)
                    if not self._coverage_ok(df, start_date, end_date, ticker_code=ticker):
                        missing_tickers.append(ticker)
                        continue
                    if df is None or df.empty:
                        missing_tickers.append(ticker)
                        continue
                    if not required_cols.issubset(set(df.columns)): #required_cols가 set(df.columns)의 하위집합인지
                        missing_tickers.append(ticker)
                        continue
                    ok_count += 1

                if not missing_tickers:
                    logger.info(f"✓ Indicator data loaded from DB for all tickers (n={ok_count})")
                    return db_ind_dict

                logger.info(
                    f"DB missing/incomplete indicator data for {len(missing_tickers)}/{len(ticker_list)} tickers. "
                    f"Calculating only missing tickers..."
                )
                ticker_list = missing_tickers

            # ============ DB에 목표 데이터가 온전히 존재하지 않는다면, 밑의 load/calculate/save 작업 진행 ============

            # ============ Pre-processing: Lookback 계산 및 적용 (안전 마진 포함) ============
            original_start_date = start_date  # 원래 요청한 start_date 저장
            start_date = self._calculate_extended_start_date(start_date, indicator_list)

            # ============ Price 데이터 불러오기 =============
            logger.info(f"____Loading price data...____")
            price_dict =  self.run_price_pipeline(
                ticker_list=ticker_list,
                start_date=start_date,
                end_date=end_date,
                period=period,
                interval=interval,
                actions=actions,
                update_if_exists=update_if_exists,
                batch_size=batch_size,
                prefer_db=prefer_db,
            )
            logger.info(f"Loaded {len(price_dict)} tickers")
            results['step1_load_data'] = {
                ticker: len(df) for ticker, df in price_dict.items()
            }

            # ============ 지표 계산 및 저장 시작 ==============
            # 전체 결과를 저장할 dict
            all_calculated_dict = {}

            # 배치로 처리
            total_tickers = len(ticker_list)
            num_batches = (total_tickers + batch_size - 1) // batch_size
            logger.info(f"Processing {total_tickers} tickers in {num_batches} batches of size {batch_size}")

            for i in range(0, total_tickers, batch_size):
                batch_tickers = ticker_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(f"{'-'*60}")
                logger.info(f"Batch {batch_num}/{num_batches}: Processing {len(batch_tickers)} tickers")
                logger.info(f"{'-'*60}")

                # Step 0: Set price data for batch
                batch_price_dict = {ticker : price_dict[ticker].copy() for ticker in batch_tickers if ticker in price_dict}

                # Step 1: Calculate indicators for batch
                batch_calculated_dict = {}
                try:
                    logger.info("")
                    logger.info(f"____Step1: Calculating indicators for batch {batch_num}...____")
                    for ticker, df in batch_price_dict.items():
                        try:
                            calculated_df = self.calculator.calculate_indicators(
                                df=df,
                                indicator_list=indicator_list
                            )

                            batch_calculated_dict[ticker] = calculated_df
                            results['step2_calculate'][ticker] = len(calculated_df)
                            logger.debug(f"Calculated {len(calculated_df)} records for {ticker}")

                        except Exception as e:
                            logger.error(f"Failed to calculate indicators for {ticker}: {e}")
                            results['step2_calculate'][ticker] = 0

                    logger.info(f"Calculated indicators for {len(batch_calculated_dict)} tickers in batch {batch_num}")

                except Exception as e:
                    logger.error(f"Failed to calculate indicators for batch {batch_num}: {e}")
                    continue

                # Step 2: Save indicators (original_start_date 기준으로 필터링)
                try:
                    logger.info("")
                    logger.info(f"____Step2: Saving indicators for batch {batch_num} to database...____")

                    # original_start_date 이후의 지표 데이터만 필터링
                    filtered_indicator_dict = {}

                    logger.info(f"___Filtering indicator data by original start date...___")
                    for ticker, df in batch_calculated_dict.items():
                        # 필터링
                        filtered_df = df[df['date'] >= original_start_date].copy()

                        if filtered_df.empty:
                            logger.info(f"{ticker}: excluded from indicator data. No indicator data in original date period")
                        else:
                            # NaN 검증: original_start_date 범위에 NaN이 있으면 경고
                            nan_cols = filtered_df.columns[filtered_df.isna().any()].tolist()
                            if nan_cols:
                                ticker_nan_count = 0
                                for col in nan_cols:
                                    #Ticker의 각 컬럼별 통계
                                    nan_count = filtered_df[col].isna().sum()
                                    ticker_nan_count += nan_count
                                    logger.debug(f"{ticker}[{col}]:: NaN count: {nan_count}/{len(filtered_df)}")
                            
                                #Ticker별 통계
                                ticker_total_cells = filtered_df.size
                                ticker_nan_ratio = (ticker_nan_count / ticker_total_cells) * 100
                                logger.warning(f"{ticker}: NaN ratio: {ticker_nan_ratio:.2f} %, NaN count: {ticker_nan_count}/{ticker_total_cells}")
                                
                            # 결과에 추가
                            filtered_indicator_dict[ticker] = filtered_df
                    
                    filtered_success_rate = (len(batch_calculated_dict) / len(filtered_indicator_dict)) * 100
                    logger.info(f"Filtering Complete. Success rate: {filtered_success_rate:.2f} %")

                    if filtered_indicator_dict:
                        batch_save_results = self.db_manager.save_indicators(
                            indicator_data_dict=filtered_indicator_dict,
                            version=version,
                        )
                        results['step3_save_indicator'].update(batch_save_results)
                        logger.info(f"Batch {batch_num} indicators saved successfully (filtered to original_start_date={original_start_date})")

                except Exception as e:
                    logger.error(f"Failed to save indicators for batch {batch_num}: {e}")
                    for ticker in batch_calculated_dict.keys():
                        results['step3_save_indicator'][ticker] = 0
                    continue

                # 메모리 정리
                all_calculated_dict.update(batch_calculated_dict)
                del batch_price_dict
                del batch_calculated_dict

            #============Summary=============
            logger.info("")
            logger.info("="*60)
            logger.info("Indicator Pipeline Summary")
            logger.info("="*60)

            total_loaded_price = sum(results['step1_load_data'].values())
            total_calculated = sum(results['step2_calculate'].values())
            total_saved_indicators = sum(results['step3_save_indicator'].values())

            logger.info(f"Total loaded price data: {total_loaded_price}")
            logger.info(f"Total calculated indicator records: {total_calculated}")
            logger.info(f"Total saved indicator records: {total_saved_indicators}")

            failed_tickers = set(price_dict.keys()) - set(all_calculated_dict)
            success_rate = (len(all_calculated_dict) / len(price_dict)) * 100 if price_dict else 0

            logger.info(f"Total processed: {len(all_calculated_dict)}/{len(price_dict.keys())}")
            logger.info(f"Total success rate: {success_rate:.2f} %")
            if failed_tickers:
                logger.info(f"Total failed tickers: {failed_tickers}")

            #결과 출력
            logger.info(f"{'='*60}")
            logger.info(f"✓ Indicator pipeline completed successfully")
            logger.info(f"{'='*60}")
            logger.info("")
            
            if prefer_db and not update_if_exists:
                return self.db_manager.load_indicators(
                    ticker_codes=requested_tickers,
                    start_date=original_start_date,
                    end_date=end_date,
                )
            return all_calculated_dict

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
            
    def close(self):
        try:
            self.db_manager.close()
            logger.info("DataPipeline closed successfully")
        except Exception as e:
            logger.error(f"Error closing pipeline: {e}")
