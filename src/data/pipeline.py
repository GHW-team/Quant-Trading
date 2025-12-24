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
