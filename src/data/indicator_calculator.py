# IndicatorCalculator
# OHLCV DataFrame에 이동평균·MACD 등을 계산해 컬럼으로 추가하는 유틸
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import logging 
from src.data.data_fetcher import StockDataFetcher
from typing import List, Optional

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    # 계산에 사용되는 지표와 대응 함수 맵
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
    
    @staticmethod
    def _calc_rsi(df: pd.DataFrame, length: int)->pd.Series:
         # 필요하다면 컬럼 검사/NaN 처리 추가
        return ta.rsi(df['adj_close'], length = length)
    
    @staticmethod
    def _calc_bbands(df: pd.DataFrame, length: int, std: int)->pd.DataFrame:
        #upper,mid,lower,%b 컬럼을 반환함
        return ta.bbands(df['adj_close'], length = length, std = std)
    
    @staticmethod
    def _calc_atr(df:pd.DataFrame, length: int)->pd.Series:
        return ta.atr(df['high'], df['low'], df['close'], length = length)
    
    @staticmethod
    def _calc_hv(df: pd.DataFrame, length: int, annual_factor: int=252)-> pd.Series:
        log_ret = np.log(df['adj_close']).diff()
        # 연환산 변동성 (스케일은 0.24 = 연 24%)
        return log_ret.rolling(length).std() * np.sqrt(annual_factor)
    
    @staticmethod
    def _calc_stoch(df: pd.DataFrame, k: int, d: int)-> pd.DataFrame:
        #%K, %D컬럼 반환
        stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=k, d=d)
        #부족분은 NaN으로 유지
        return stoch_df.reindex(df.index)
    
    @staticmethod
    def _calc_obv(df: pd.DataFrame)->pd.Series:
        return ta.obv(df['close'], df['volume'])
    

    #rsi,볼린저밴드,atr,hv,스토캐스틱 등은 현재 기본값으로 설정해뒀으니 이 값을 바꿀 경우 이름을 바꿔주기바람
    #ex) 볼린저밴드의 이동평균선을 50일, 표준편차 2를 사용할 경우: bb_upper_50_2
    INDICATORS_FUNCTIONS = {
        'ma_5' : lambda df : IndicatorCalculator._calc_ema(df,5),
        'ma_10' : lambda df : IndicatorCalculator._calc_ema(df,10),
        'ma_20' : lambda df : IndicatorCalculator._calc_ema(df,20),
        'ma_50' : lambda df : IndicatorCalculator._calc_ema(df,50),
        'ma_60' : lambda df : IndicatorCalculator._calc_ema(df,60),
        'ma_100': lambda df : IndicatorCalculator._calc_ema(df,100),
        'ma_120' : lambda df : IndicatorCalculator._calc_ema(df,120),
        'ma_200' : lambda df : IndicatorCalculator._calc_ema(df,200),
        'macd' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,0],
        'macd_signal' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,1],
        'macd_hist' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,2],
        'rsi' : lambda df : IndicatorCalculator._calc_rsi(df,14),
        'bb_upper': lambda df: IndicatorCalculator._calc_bbands(df, length=20, std=2).iloc[:, 0],
        'bb_mid':   lambda df: IndicatorCalculator._calc_bbands(df, length=20, std=2).iloc[:, 1],
        'bb_lower': lambda df: IndicatorCalculator._calc_bbands(df, length=20, std=2).iloc[:, 2],
        'bb_pct':   lambda df: IndicatorCalculator._calc_bbands(df, length=20, std=2).iloc[:, 3],
        'atr':  lambda df: IndicatorCalculator._calc_atr(df, 14),
        "hv":   lambda df: IndicatorCalculator._calc_hv(df, 20, 252),
        'stoch_k': lambda df: IndicatorCalculator._calc_stoch(df, 14, 3).iloc[:, 0],
        'stoch_d': lambda df: IndicatorCalculator._calc_stoch(df, 14, 3).iloc[:, 1],
        'obv':     lambda df: IndicatorCalculator._calc_obv(df)
    }

    def __init__(self):
        self.calculated_indicators = []

    @staticmethod
    def get_lookback_days(indicator_list: Optional[List[str]] = None) -> int:
        """
        지표별 필요한 과거 기간(lookback)을 계산합니다.

        Args:
            indicator_list: 지표 리스트 (예: ['ma_5', 'ma_20', 'ma_200'])

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
            'ma_100': 100,
            'ma_120': 120,
            'ma_200': 200,
            'macd': 26,              # slow period
            'macd_signal': 26,
            'macd_hist': 26,
            'rsi': 14,               # RSI 기본값 (향후 추가 가능)
            'bb_upper': 20,          #볼린저밴드 기본값
            'bb_mid': 20,
            'bb_lower': 20,
            'bb_pct': 20,
            'atr' : 14,              #ATR 기본값
            'hv' : 20,               #HV 기본값 
            'stoch_k' : 14,          #스토캐스틱 기본값: k=14,d=3
            'stoch_d' : 14,          #D는 k의 3일 이동평균이므로 룩백기간은 k와 동일해야 함.
            'obv' : 1
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

        # -------- 입력 검증 단계 --------
        # DataFrame validation
        if df.empty:
            logger.error("Empty dataframe provided")
            raise ValueError("Cannot calculate indicators: empty dataframe")

        # Check for required column 'adj_close'
        if 'adj_close' not in df.columns:
            available_cols = list(df.columns)
            logger.error(f"Required column 'adj_close' not found. Available columns: {available_cols}")
            raise ValueError(f"Cannot calculate indicators: required column 'adj_close' not found. Available: {available_cols}")

        # Check for 'date' column and validate ordering
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                logger.error(f"'date' column exists but is not datetime type: {df['date'].dtype}")
                raise ValueError(
                    f"Cannot calculate indicators: 'date' column must be datetime type, "
                    f"got {df['date'].dtype}"
                )

            if not df['date'].is_monotonic_increasing:
                logger.error("'date' column is not sorted in chronological order")
                raise ValueError(
                    "Cannot calculate indicators: 'date' column must be sorted in "
                    "chronological (ascending) order for accurate time-series calculations. "
                    "Please sort DataFrame by date before calling calculate_indicators()."
                )

            logger.debug(f"Date column validated: {len(df)} rows from {df['date'].min()} to {df['date'].max()}")

        # Default setting: All indicators
        if indicator_list == None:
            indicator_list = self.get_available_indicators()
            logger.warning(f"No input indicators specified. Calculating all indicators: {indicator_list}")

        # Log input data info
        logger.info(f"Input DataFrame: {len(df)} rows, columns={list(df.columns)}")

        # Validate indicators before calculation
        self.validate_indicators(indicator_list)

        # -------- 지표 계산 단계 --------
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