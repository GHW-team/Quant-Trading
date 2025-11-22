#IndicatorCalculator
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import logging 
from src.data.data_fetcher import StockDataFetcher
from typing import List, Optional

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    @staticmethod
    def _calc_sma(df: pd.DataFrame, length: int)-> pd.Series:
        return ta.sma(df['adj_close'],length=length)

    @staticmethod
    def _calc_ema(df: pd.DataFrame, length: int)-> pd.Series:
        return ta.ema(df['adj_close'],length=length)

    @staticmethod
    def _calc_macd(df: pd.DataFrame, 
                   fast:int,
                   slow:int,
                   signal:int)-> pd.Series:
        return ta.macd(df['adj_close'],fast=fast, slow=slow, signal=signal)

    INDICATORS_FUNCTIONS = {
        'ma_5' : lambda df : IndicatorCalculator._calc_ema(df,5),
        'ma_10' : lambda df : IndicatorCalculator._calc_ema(df,10),
        'ma_20' : lambda df : IndicatorCalculator._calc_ema(df,20),
        'ma_50' : lambda df : IndicatorCalculator._calc_ema(df,50),
        'ma_60' : lambda df : IndicatorCalculator._calc_ema(df,60),
        'ma_120' : lambda df : IndicatorCalculator._calc_ema(df,120),
        'ma_200' : lambda df : IndicatorCalculator._calc_ema(df,200),
        'macd' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,0],
        'macd_signal' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,1],
        'macd_hist' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,2],
    }

    def __init__(self):
        self.calculated_indicators = []

    @staticmethod
    def get_lookback_days(indicator_list: Optional[List[str]] = None) -> int:
        """
        지표별 필요한 과거 기간(lookback)을 계산합니다.

        Args:
            indicator_list: 지표 리스트 (예: ['ma_5', 'ma_20', 'ma_60', 'ma_200'])

        Returns:
            필요한 최대 과거 기간 (일수)
        """
        if indicator_list is None:
            indicator_list = list(IndicatorCalculator.INDICATORS_FUNCTIONS.keys())

        # 지표별 필요한 과거 기간
        lookback_map = {
            'ma_5': 5,
            'ma_10': 10,
            'ma_20': 20,
            'ma_50': 50,
            'ma_60': 60,
            'ma_120': 120,
            'ma_200': 200,
            'macd': 26,              # slow period
            'macd_signal': 26,
            'macd_hist': 26,
            'rsi': 14,               # RSI 기본값 (향후 추가 가능)
        }

        # 요청한 지표 중 최대 lookback 일수 구하기
        max_lookback = 0
        for indicator in indicator_list:
            days = lookback_map.get(indicator, 0)
            max_lookback = max(max_lookback, days)

        # 안전마진 추가 (계산 오차 방지)
        return max_lookback + 5

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        indicator_list: Optional[List[str]] = None
        )-> Optional[pd.DataFrame]:

        # DataFrame validation
        if df.empty:
            logger.error("Empty dataframe provided")
            raise ValueError("Cannot calculate indicators: empty dataframe")

        # Check for required column 'adj_close'
        if 'adj_close' not in df.columns:
            available_cols = list(df.columns)
            logger.error(f"Required column 'adj_close' not found. Available columns: {available_cols}")
            raise ValueError(f"Cannot calculate indicators: required column 'adj_close' not found. Available: {available_cols}")

        # Default setting: All indicators
        if indicator_list == None:
            indicator_list = self.get_available_indicators()
            logger.warning(f"No input indicators specified. Calculating all indicators: {indicator_list}")

        # Log input data info
        logger.info(f"Input DataFrame: {len(df)} rows, columns={list(df.columns)}")

        # Validate indicators before calculation
        self.validate_indicators(indicator_list)

        # Calculation
        result_df = df.copy()
        errors = {}

        logger.debug(f"Start calculating {len(indicator_list)} indicators for {len(df)} rows")
        for indicator in indicator_list:
            try:
                logger.debug(f"Calculating {indicator}...")

                # Get calculation function
                calc_func = self.INDICATORS_FUNCTIONS[indicator]

                # Check minimum data requirement for this indicator
                lookback = self.get_lookback_days([indicator])
                if len(df) < lookback:
                    logger.warning(
                        f"Insufficient data for {indicator}: need {lookback} rows (warmup), "
                        f"but have {len(df)} rows. Continuing with available data."
                    )

                # Calculate indicator
                calculated_indicator = calc_func(df)

                # Validate result
                if calculated_indicator is None:
                    error_detail = (
                        f"Calculation returned None. This suggests the indicator function "
                        f"({calc_func.__name__ if hasattr(calc_func, '__name__') else 'lambda'}) "
                        f"did not produce output. Check if data length ({len(df)}) is sufficient."
                    )
                    logger.error(f"{indicator}: {error_detail}")
                    errors[indicator] = error_detail
                    continue

                # Check if result is Series and has valid length
                if not isinstance(calculated_indicator, pd.Series):
                    error_detail = f"Expected Series but got {type(calculated_indicator).__name__}"
                    logger.error(f"{indicator}: {error_detail}")
                    errors[indicator] = error_detail
                    continue

                if len(calculated_indicator) != len(df):
                    error_detail = f"Result length ({len(calculated_indicator)}) != input length ({len(df)})"
                    logger.error(f"{indicator}: {error_detail}")
                    errors[indicator] = error_detail
                    continue

                # Assign to result
                result_df[indicator] = calculated_indicator

                # Log NaN info
                nan_count = calculated_indicator.isna().sum()
                nan_ratio = (nan_count / len(calculated_indicator) * 100) if len(calculated_indicator) > 0 else 0
                logger.debug(
                    f"✓ {indicator}: {len(calculated_indicator)} values, "
                    f"{nan_count} NaN ({nan_ratio:.1f}%)"
                )

            except Exception as e:
                error_detail = f"Exception during calculation: {str(e)} (Type: {type(e).__name__})"
                logger.error(f"{indicator}: {error_detail}")
                errors[indicator] = error_detail

        # Check results
        if errors:
            error_msg = f"Failed to calculate {len(errors)}/{len(indicator_list)} indicators: {errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.calculated_indicators = indicator_list
        logger.info(f"✓ All {len(indicator_list)} indicators calculated successfully")
        logger.debug(f"Result DataFrame: {len(result_df)} rows, {len(result_df.columns)} columns")

        return result_df    

    @staticmethod
    def get_available_indicators()-> List[str]:
        return list(IndicatorCalculator.INDICATORS_FUNCTIONS.keys())

    @staticmethod
    def validate_indicators(indicator_list)->None:
        invalid = []
        available = IndicatorCalculator.INDICATORS_FUNCTIONS.keys()

        invalid = [ind for ind in indicator_list if ind not in available]
        if invalid:
            logger.warning(f"Invalid indicator exists: {invalid}")
            raise ValueError(
                f"Invalid indicators: {invalid}\n"
                f"Available: {available}"
            )
    
    @staticmethod
    def add_indicators(
        indicator: str,
        calculation_func
    ):
        if indicator in IndicatorCalculator.INDICATORS_FUNCTIONS:
            logger.warning(f"Indicator {indicator} already exists. Overwritting..")
        
        IndicatorCalculator.INDICATORS_FUNCTIONS[indicator] = calculation_func
        logger.info(f"Added new indicator: {indicator}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("IndicatorCalculator 테스트")
    print("="*70)

    # [1] IndicatorCalculator 초기화
    print("\n[1] IndicatorCalculator 초기화...")
    try:
        calculator = IndicatorCalculator()
        print(f"✓ IndicatorCalculator 생성 완료")
    except Exception as e:
        print(f"✗ 초기화 실패: {e}")
        exit(1)

    # [2] get_available_indicators() 테스트
    print("\n[2] get_available_indicators() 테스트...")
    try:
        available = IndicatorCalculator.get_available_indicators()
        print(f"✓ 사용 가능한 지표: {available}")
        print(f"  - 총 {len(available)}개")
    except Exception as e:
        print(f"✗ 지표 조회 실패: {e}")

    # [3] validate_indicators() 테스트 - 정상 케이스
    print("\n[3] validate_indicators() 테스트 (정상)...")
    try:
        valid_indicators = ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60', 'ma_120', 'ma_200', 'macd']
        IndicatorCalculator.validate_indicators(valid_indicators)
        print(f"✓ 유효한 지표 검증 통과: {valid_indicators}")
    except Exception as e:
        print(f"✗ 유효한 지표 검증 실패: {e}")

    # [4] validate_indicators() 테스트 - 에러 케이스
    print("\n[4] validate_indicators() 테스트 (에러)...")
    try:
        invalid_indicators = ['ma_5', 'invalid_ma', 'macd']
        IndicatorCalculator.validate_indicators(invalid_indicators)
        print(f"✗ 무효한 지표를 통과시킴 (오류)")
    except ValueError as e:
        print(f"✓ 무효한 지표 감지 정상: {str(e)[:50]}...")

    # [5] get_lookback_days() 테스트
    print("\n[5] get_lookback_days() 테스트...")
    try:
        test_cases = [
            ['ma_5'],
            ['ma_200'],
            ['ma_5', 'ma_60'],
            ['ma_5', 'ma_20', 'ma_50', 'ma_120', 'macd'],
            None,  # 모든 지표
        ]
        for indicators in test_cases:
            lookback = IndicatorCalculator.get_lookback_days(indicators)
            print(f"✓ {indicators}: {lookback} 일")
    except Exception as e:
        print(f"✗ lookback 계산 실패: {e}")

    # [6] 테스트 데이터 준비 (fetch)
    print("\n[6] 테스트 데이터 준비...")
    try:
        fetcher = StockDataFetcher()
        ticker = '005930.KS'
        df_dict = fetcher.fetch_multiple_by_period(
            ticker_list=[ticker],
            period="1y"
        )

        if not df_dict:
            print(f"✗ 데이터 수집 실패")
            exit(1)

        stock_df = df_dict[ticker]
        # 컬럼명 정규화
        stock_df.columns = (stock_df.columns
                            .str.strip()
                            .str.lower()
                            .str.replace(' ', '_'))

        print(f"✓ 데이터 준비 완료: {ticker}")
        print(f"  - 행 수: {len(stock_df)}")
        print(f"  - 컬럼: {list(stock_df.columns)}")
    except Exception as e:
        print(f"✗ 데이터 준비 실패: {e}")
        exit(1)

    # [7] calculate_indicators() 테스트 - 정상 케이스
    print("\n[7] calculate_indicators() 테스트 (정상)...")
    try:
        indicator_list = ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60', 'ma_120', 'ma_200', 'macd']
        calculated_df = calculator.calculate_indicators(
            stock_df.copy(),
            indicator_list=indicator_list
        )
        print(f"✓ 지표 계산 완료: {indicator_list}")
        print(f"  - 원본 행 수: {len(stock_df)}")
        print(f"  - 계산 후 행 수: {len(calculated_df)}")
        print(f"  - 계산된 지표 컬럼: {[col for col in calculated_df.columns if col in indicator_list]}")
        print(f"  - NaN 값 비율:")
        for ind in indicator_list:
            nan_ratio = calculated_df[ind].isna().sum() / len(calculated_df) * 100
            print(f"    - {ind}: {nan_ratio:.1f}%")
    except Exception as e:
        print(f"✗ 지표 계산 실패: {e}")

    # [8] calculate_indicators() 테스트 - 부분 지표
    print("\n[8] calculate_indicators() 테스트 (부분 지표)...")
    try:
        partial_indicators = ['ma_5', 'ma_60', 'macd']
        calculated_df2 = calculator.calculate_indicators(
            stock_df.copy(),
            indicator_list=partial_indicators
        )
        print(f"✓ 부분 지표 계산 완료: {partial_indicators}")
        print(f"  - 계산된 지표: {[col for col in calculated_df2.columns if col in partial_indicators]}")
    except Exception as e:
        print(f"✗ 부분 지표 계산 실패: {e}")

    # [9] calculate_indicators() 테스트 - 기본값 (모든 지표)
    print("\n[9] calculate_indicators() 테스트 (기본값 - 모든 지표)...")
    try:
        calculated_df3 = calculator.calculate_indicators(stock_df.copy(), indicator_list=None)
        calculated_cols = [col for col in calculated_df3.columns if col in available]
        print(f"✓ 모든 지표 계산 완료")
        print(f"  - 계산된 지표: {calculated_cols}")
    except Exception as e:
        print(f"✗ 모든 지표 계산 실패: {e}")

    # [10] add_indicators() 테스트
    print("\n[10] add_indicators() 테스트 (커스텀 지표)...")
    try:
        # 커스텀 지표 함수 정의
        def custom_ma_10(df):
            return ta.sma(df['adj_close'], length=10)

        IndicatorCalculator.add_indicators('custom_ma_10', custom_ma_10)
        print(f"✓ 커스텀 지표 추가 완료: custom_ma_10")

        # 추가된 지표로 계산
        calculated_df4 = calculator.calculate_indicators(
            stock_df.copy(),
            indicator_list=['custom_ma_10', 'ma_5']
        )
        print(f"✓ 커스텀 지표 계산 완료")
        print(f"  - 포함된 지표: {[col for col in calculated_df4.columns if col in ['custom_ma_10', 'ma_5']]}")
    except Exception as e:
        print(f"✗ 커스텀 지표 추가 실패: {e}")

    # [11] 에러 처리 - 빈 DataFrame
    print("\n[11] 에러 처리 테스트 (빈 DataFrame)...")
    try:
        empty_df = pd.DataFrame()
        calculated_df5 = calculator.calculate_indicators(
            empty_df,
            indicator_list=['ma_5']
        )
        print(f"✗ 빈 DataFrame을 통과시킴 (오류)")
    except ValueError as e:
        print(f"✓ 빈 DataFrame 감지 정상: {str(e)}")

    # [12] 에러 처리 - 필수 컬럼 누락
    print("\n[12] 에러 처리 테스트 (필수 컬럼 누락)...")
    try:
        bad_df = pd.DataFrame({'open': [1, 2, 3]})  # adj_close 없음
        calculated_df6 = calculator.calculate_indicators(
            bad_df,
            indicator_list=['ma_5']
        )
        print(f"⚠ 필수 컬럼 없이도 계산됨 (또는 에러)")
    except Exception as e:
        print(f"✓ 필수 컬럼 누락 감지: {str(e)[:50]}...")

    print("\n" + "="*70)
    print("모든 테스트 완료!")
    print("="*70)
