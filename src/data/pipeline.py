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

    @staticmethod
    def _convert_period_to_dates(period: str) -> Tuple[str, str]:
        """
        Period 문자열을 start_date, end_date로 변환합니다.

        Args:
            period: 기간 문자열 (예: "1y", "6m", "3m", "1m")

        Returns:
            (start_date, end_date) 튜플 (YYYY-MM-DD 형식)
        """
        today = datetime.now()

        # Period에서 숫자와 단위 추출
        if not period or len(period) < 2:
            raise ValueError(f"Invalid period format: {period}")

        amount = int(period[:-1])
        unit = period[-1].lower()

        if unit == 'y':
            start_dt = today.replace(year=today.year - amount)
        elif unit == 'm':
            # 월 계산
            month = today.month - amount
            year = today.year
            while month <= 0:
                month += 12
                year -= 1
            start_dt = today.replace(year=year, month=month)
        elif unit == 'd':
            start_dt = today - timedelta(days=amount)
        elif unit == 'w':
            start_dt = today - timedelta(weeks=amount)
        else:
            raise ValueError(f"Unknown period unit: {unit}. Use 'y', 'm', 'w', or 'd'")

        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')

        logger.debug(f"Converted period '{period}' to start_date={start_date}, end_date={end_date}")
        return start_date, end_date

    def _calculate_extended_start_date(self, start_date: Optional[str], indicator_list: List[str]) -> Optional[str]:
        """
        지표 계산을 위해 '안전 마진'이 포함된 과거 시작 날짜를 계산합니다.

        영업일/휴일을 고려하여 1.6배수를 적용하고 추가 10일의 여유를 더합니다.

        Args:
            start_date: 원래 요청한 시작 날짜 (YYYY-MM-DD 형식)
            indicator_list: 지표 리스트

        Returns:
            확장된 시작 날짜 (YYYY-MM-DD 형식), start_date가 None이면 None
        """
        if not start_date:
            return None

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

            # ============ Pre-processing: Period를 start_date/end_date로 변환 ============
            if period:
                start_date, end_date = self._convert_period_to_dates(period)
                logger.info(f"Period '{period}' converted to start_date={start_date}, end_date={end_date}")

            # ============ Lookback 계산 및 적용 (안전 마진 포함) ============
            original_start_date = start_date  # 원래 요청한 start_date 저장
            start_date = self._calculate_extended_start_date(start_date, indicator_list)

            # 전체 결과를 저장할 dict
            all_calculated_dict = {}

            # 배치로 처리
            total_tickers = len(ticker_list)
            num_batches = (total_tickers + batch_size - 1) // batch_size
            logger.info(f"\nProcessing {total_tickers} tickers in {num_batches} batches of size {batch_size}")

            for i in range(0, total_tickers, batch_size):
                batch_tickers = ticker_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(f"\n{'='*60}")
                logger.info(f"Batch {batch_num}/{num_batches}: Processing {len(batch_tickers)} tickers")
                logger.info(f"{'='*60}")

                # Step 1: Fetch batch data (조정된 date range로 fetch)
                try:
                    logger.info(f"Step 1: Fetching data for batch {batch_num}...")
                    batch_df_dict = self.fetcher.fetch_multiple_by_date(
                        ticker_list=batch_tickers,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        actions=actions,
                    )

                    logger.info(f"Fetched {len(batch_df_dict)} tickers")
                    results['step1_fetch'].update({
                        ticker : len(df) for ticker, df in batch_df_dict.items()
                    })

                except Exception as e:
                    logger.error(f"Failed to fetch batch {batch_num}: {e}")
                    for ticker in batch_tickers:
                        results['step1_fetch'][ticker] = 0
                    continue

                # Step 2: Save batch price data (original_start_date 기준으로 필터링)
                try:
                    logger.info(f"Step 2: Saving price data for batch {batch_num}...")

                    # original_start_date 이후의 데이터만 필터링
                    filtered_price_dict = {}
                    for ticker, df in batch_df_dict.items():
                        if 'date' in df.columns:
                            filtered_df = df[df['date'] >= original_start_date].copy()
                        else:
                            # date column이 없으면 오류!!
                            raise ValueError(f"There is no 'date' column in {ticker} DataFrame")

                        if not filtered_df.empty:
                            filtered_price_dict[ticker] = filtered_df

                    if filtered_price_dict:
                        batch_save_price_results = self.db_manager.save_price_data(
                            df_dict=filtered_price_dict,
                            update_if_exists=update_if_exists,
                        )
                        results['step2_save_price'].update(batch_save_price_results)
                        logger.info(f"Saved price data for batch {batch_num} (filtered to original_start_date={original_start_date})")

                except Exception as e:
                    logger.error(f"Failed to save price data for batch {batch_num}: {e}")
                    for ticker in batch_df_dict.keys():
                        results['step2_save_price'][ticker] = 0
                    continue

                # Step 3: Calculate indicators for batch
                batch_calculated_dict = {}
                try:
                    logger.info(f"Step 3: Calculating indicators for batch {batch_num}...")
                    for ticker, df in batch_df_dict.items():
                        try:
                            calculated_df = self.calculator.calculate_indicators(
                                df=df,
                                indicator_list=indicator_list
                            )

                            batch_calculated_dict[ticker] = calculated_df
                            results['step3_calculate'][ticker] = len(calculated_df)
                            logger.debug(f"Calculated {len(calculated_df)} records for {ticker}")

                        except Exception as e:
                            logger.error(f"Failed to calculate indicators for {ticker}: {e}")
                            results['step3_calculate'][ticker] = 0

                    logger.info(f"Calculated indicators for {len(batch_calculated_dict)} tickers in batch {batch_num}")

                except Exception as e:
                    logger.error(f"Failed to calculate indicators for batch {batch_num}: {e}")
                    continue

                # Step 4: Save indicators (original_start_date 기준으로 필터링)
                try:
                    logger.info(f"Step 4: Saving indicators for batch {batch_num}...")

                    # original_start_date 이후의 지표 데이터만 필터링
                    filtered_indicator_dict = {}
                    for ticker, df in batch_calculated_dict.items():
                        # index.name이 대문자 'Date'일 수 있으므로 대소문자 무시
                        has_date_index = df.index.name and df.index.name.lower() == 'date'
                        has_date_column = 'date' in df.columns

                        if has_date_index:
                            # Index가 Date인 경우 - Date를 컬럼으로 변환 후 필터링
                            df_with_date_col = df.reset_index()
                            df_with_date_col.columns = df_with_date_col.columns.str.strip().str.lower()
                            filtered_df = df_with_date_col[
                                pd.to_datetime(df_with_date_col['date']).dt.date.astype(str) >= original_start_date
                            ].copy()
                        elif has_date_column:
                            # date가 column인 경우
                            filtered_df = df[df['date'] >= original_start_date].copy()
                        else:
                            # date column이 없으면 전체 데이터 사용
                            filtered_df = df.copy()

                        if not filtered_df.empty:
                            # NaN 검증: original_start_date 범위에 NaN이 있으면 경고
                            nan_cols = filtered_df.columns[filtered_df.isna().any()].tolist()
                            if nan_cols:
                                logger.warning(
                                    f"⚠ {ticker}: NaN values found in original date range for columns: {nan_cols}. "
                                    f"Lookback may be insufficient or DB data incomplete."
                                )

                            filtered_indicator_dict[ticker] = filtered_df

                    if filtered_indicator_dict:
                        batch_save_indicator_results = self.db_manager.save_indicators(
                            indicator_data_dict=filtered_indicator_dict,
                            version=version,
                        )
                        results['step4_save_indicator'].update(batch_save_indicator_results)
                        logger.info(f"Saved indicators for batch {batch_num} (filtered to original_start_date={original_start_date})")

                except Exception as e:
                    logger.error(f"Failed to save indicators for batch {batch_num}: {e}")
                    for ticker in batch_calculated_dict.keys():
                        results['step4_save_indicator'][ticker] = 0
                    continue

                # 메모리 정리
                all_calculated_dict.update(batch_calculated_dict)
                del batch_df_dict
                del batch_calculated_dict

            #============Summary=============
            logger.info("="*60)
            logger.info("Pipeline Summary")
            logger.info("="*60)

            total_fetched = sum(results['step1_fetch'].values())
            total_saved_prices = sum(results['step2_save_price'].values())
            total_calculated = sum(results['step3_calculate'].values())
            total_saved_indicators = sum(results['step4_save_indicator'].values())

            results['summary'] = {
                'total_fetched_records': total_fetched,
                'total_saved_prices': total_saved_prices,
                'total_calculated_indicators': total_calculated,
                'total_saved_indicators' : total_saved_indicators,
                'tickers_processed': len(ticker_list),
                'batches_processed': num_batches,
                'status': 'success'
            }

            logger.info(f"Total fetched records: {total_fetched}")
            logger.info(f"Total saved price records: {total_saved_prices}")
            logger.info(f"Total calculated indicator records: {total_calculated}")
            logger.info(f"Total saved indicator records: {total_saved_indicators}")
            logger.info(f"✓ Pipeline completed successfully")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ToTal Results: {results}")
            logger.info(f"{'='*60}")

            return all_calculated_dict

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['summary'] = {
                'status': 'failed',
                'error': str(e)
            }
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
            'summary' : {},
        }
        try:
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

            # ============ Pre-processing: Period를 start_date/end_date로 변환 ============
            if period:
                start_date, end_date = self._convert_period_to_dates(period)
                logger.info(f"Period '{period}' converted to start_date={start_date}, end_date={end_date}")

            # 전체 결과를 저장할 dict
            all_df_dict = {}

            # 배치로 처리
            total_tickers = len(ticker_list)
            num_batches = (total_tickers + batch_size - 1) // batch_size
            logger.info(f"Processing {total_tickers} tickers in {num_batches} batches of size {batch_size}")

            for i in range(0, total_tickers, batch_size):
                batch_tickers = ticker_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(f"\n{'='*60}")
                logger.info(f"Batch {batch_num}/{num_batches}: Fetching {len(batch_tickers)} tickers")
                logger.info(f"{'='*60}")

                # Step 1: Fetch batch data
                try:
                    batch_df_dict = self.fetcher.fetch_multiple_by_date(
                        ticker_list=batch_tickers,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        actions=actions,
                    )

                    logger.info(f"Fetched {len(batch_df_dict)} tickers")
                    results['step1_fetch'].update({
                        ticker : len(df) for ticker, df in batch_df_dict.items()
                    })

                except Exception as e:
                    logger.error(f"Failed to fetch batch {batch_num}: {e}")
                    for ticker in batch_tickers:
                        results['step1_fetch'][ticker] = 0
                    continue

                # Step 2: Save batch data using bulk save function
                try:
                    logger.info(f"Saving batch {batch_num} to database...")
                    batch_save_results = self.db_manager.save_price_data(
                        df_dict=batch_df_dict,
                        update_if_exists=update_if_exists,
                    )
                    results['step2_save_price'].update(batch_save_results)
                    logger.info(f"Batch {batch_num} saved successfully")

                except Exception as e:
                    logger.error(f"Failed to save batch {batch_num}: {e}")
                    for ticker in batch_df_dict.keys():
                        results['step2_save_price'][ticker] = 0
                    continue

                # 메모리 정리
                all_df_dict.update(batch_df_dict)
                del batch_df_dict

            #============Summary=============
            logger.info("="*60)
            logger.info("Pipeline Summary")
            logger.info("="*60)

            total_fetched = sum(results['step1_fetch'].values())
            total_saved_prices = sum(results['step2_save_price'].values())

            results['summary'] = {
                'total_fetched_records': total_fetched,
                'total_saved_prices': total_saved_prices,
                'tickers_processed': len(ticker_list),
                'batches_processed': num_batches,
                'status': 'success'
            }

            logger.info(f"Total fetched records: {total_fetched}")
            logger.info(f"Total saved price records: {total_saved_prices}")
            logger.info(f"✓ Pipeline completed successfully")
                
            #결과 출력
            logger.info(f"\n{'='*60}")
            logger.info(f"ToTal Results: {results}")
            logger.info(f"{'='*60}")

            return all_df_dict

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['summary'] = {
                'status' : 'failed',
                'error': str(e)
            }
            raise
        
    def run_indicator_pipeline(
        self,
        ticker_list: List[str],
        indicator_list: List[str],
        start_date: str,
        end_date: str,
        version: Optional[str] = "v1.0",
        batch_size: int = 100,
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
            'summary': {},
        }

        try:
            # ============ Pre-processing: Lookback 계산 및 적용 (안전 마진 포함) ============
            original_start_date = start_date  # 원래 요청한 start_date 저장
            start_date = self._calculate_extended_start_date(start_date, indicator_list)

            # 전체 결과를 저장할 dict
            all_calculated_dict = {}

            # 배치로 처리
            total_tickers = len(ticker_list)
            num_batches = (total_tickers + batch_size - 1) // batch_size
            logger.info(f"Processing {total_tickers} tickers in {num_batches} batches of size {batch_size}")

            for i in range(0, total_tickers, batch_size):
                batch_tickers = ticker_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(f"\n{'='*60}")
                logger.info(f"Batch {batch_num}/{num_batches}: Processing {len(batch_tickers)} tickers")
                logger.info(f"{'='*60}")

                # Step 1: Load price data for batch
                batch_df_dict = {}
                try:
                    logger.info(f"Loading price data for batch {batch_num}...")
                    # 배치 전체를 한번에 로드
                    batch_price_dict = self.db_manager.load_price_data(
                        ticker_codes=batch_tickers,
                        start_date=start_date,
                        end_date=end_date,
                    )

                    for ticker in batch_tickers:
                        if ticker in batch_price_dict:
                            df = batch_price_dict[ticker]
                            if df.empty:
                                logger.warning(f"No price data for {ticker}")
                                results['step1_load_data'][ticker] = 0
                            else:
                                batch_df_dict[ticker] = df
                                results['step1_load_data'][ticker] = len(df)
                                logger.debug(f"Loaded {len(df)} records for {ticker}")
                        else:
                            logger.warning(f"No price data for {ticker}")
                            results['step1_load_data'][ticker] = 0

                    logger.info(f"Loaded {len(batch_df_dict)} tickers for batch {batch_num}")

                except Exception as e:
                    logger.error(f"Failed to load data for batch {batch_num}: {e}")
                    for ticker in batch_tickers:
                        results['step1_load_data'][ticker] = 0
                    continue

                # Step 2: Calculate indicators for batch
                batch_calculated_dict = {}
                try:
                    logger.info(f"Calculating indicators for batch {batch_num}...")
                    for ticker, df in batch_df_dict.items():
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

                # Step 3: Save indicators (original_start_date 기준으로 필터링)
                try:
                    logger.info(f"Saving indicators for batch {batch_num} to database...")

                    # original_start_date 이후의 지표 데이터만 필터링
                    filtered_indicator_dict = {}
                    for ticker, df in batch_calculated_dict.items():
                        if 'date' in df.columns or df.index.name == 'date':
                            # index가 date인 경우
                            if df.index.name == 'date':
                                filtered_df = df[df.index >= original_start_date].copy()
                            # date가 column인 경우
                            else:
                                filtered_df = df[df['date'] >= original_start_date].copy()
                        else:
                            # date column이 없으면 전체 데이터 사용
                            filtered_df = df.copy()

                        if not filtered_df.empty:
                            # NaN 검증: original_start_date 범위에 NaN이 있으면 경고
                            nan_cols = filtered_df.columns[filtered_df.isna().any()].tolist()
                            if nan_cols:
                                logger.warning(
                                    f"⚠ {ticker}: NaN values found in original date range for columns: {nan_cols}. "
                                    f"Lookback may be insufficient or DB data incomplete."
                                )

                            filtered_indicator_dict[ticker] = filtered_df

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
                del batch_df_dict
                del batch_calculated_dict

            #============Summary=============
            logger.info("="*60)
            logger.info("Pipeline Summary")
            logger.info("="*60)

            total_loaded_price = sum(results['step1_load_data'].values())
            total_calculated = sum(results['step2_calculate'].values())
            total_saved_indicators = sum(results['step3_save_indicator'].values())

            results['summary'] = {
                'total_loaded_price_data' : total_loaded_price,
                'total_calculated_indicators': total_calculated,
                'total_saved_indicators' : total_saved_indicators,
                'tickers_processed': len(ticker_list),
                'batches_processed': num_batches,
                'status': 'success'
            }

            logger.info(f"Total loaded price data: {total_loaded_price}")
            logger.info(f"Total calculated indicator records: {total_calculated}")
            logger.info(f"Total saved indicator records: {total_saved_indicators}")
            logger.info(f"✓ Pipeline completed successfully")

            #결과 출력
            logger.info(f"\n{'='*60}")
            logger.info(f"ToTal Results: {results}")
            logger.info(f"{'='*60}")

            return all_calculated_dict

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['summary'] = {
                'status' : 'failed',
                'error': str(e)
            }
            raise
            
    def close(self):
        try:
            self.db_manager.close()
            logger.info("DataPipeline closed successfully")
        except Exception as e:
            logger.error(f"Error closing pipeline: {e}")
