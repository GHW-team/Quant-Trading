# src/data/indicator_calculator.py - 기술적 지표 계산

import pandas as pd
import pandas_ta_classic as ta
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """기술적 지표 계산 클래스"""

    # 지표별 계산 함수 매핑 (확장 가능한 구조)
    INDICATOR_FUNCTIONS = {
        'ma_5': lambda df: ta.sma(df['close'], length=5),
        'ma_20': lambda df: ta.sma(df['close'], length=20),
        'ma_200': lambda df: ta.sma(df['close'], length=200),
        'macd': lambda df: ta.macd(df['close'], fast=12, slow=26, signal=9),
    }

    def __init__(self):
        """초기화"""
        self.calculated_indicators = {}

    @staticmethod
    def validate_indicators(indicators: List[str]) -> bool:
        """
        요청한 지표가 유효한지 검증

        Args:
            indicators: 계산할 지표 목록

        Returns:
            모두 유효하면 True
        """
        invalid = [ind for ind in indicators if ind not in IndicatorCalculator.INDICATOR_FUNCTIONS]
        if invalid:
            logger.warning(f"Invalid indicators: {invalid}")
            return False
        return True

    @staticmethod
    def _calculate_ma(series: pd.Series, length: int) -> pd.Series:
        """
        단순 이동평균 계산

        Args:
            series: 가격 시리즈 (보통 close)
            length: 기간

        Returns:
            이동평균 시리즈
        """
        return ta.sma(series, length=length)

    @staticmethod
    def _calculate_macd(series: pd.Series) -> pd.DataFrame:
        """
        MACD 계산 (MACD, Signal, Histogram)

        Args:
            series: 가격 시리즈 (보통 close)

        Returns:
            MACD 데이터프레임 (macd, macd_signal, macd_hist)
        """
        macd_result = ta.macd(series, fast=12, slow=26, signal=9)

        # pandas-ta는 여러 컬럼을 반환하므로, 'macd' 컬럼만 추출
        if macd_result is not None and len(macd_result.columns) > 0:
            # MACD 결과 구조: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            macd_df = pd.DataFrame({
                'macd': macd_result.iloc[:, 0] if macd_result.shape[1] > 0 else None
            })
            return macd_df

        return pd.DataFrame({'macd': [None] * len(series)})

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        지정된 지표 계산

        Args:
            df: OHLCV 데이터프레임 (index=date, columns: open, high, low, close, volume)
            indicators: 계산할 지표 목록 (None이면 모든 지표 계산)
                       예: ['ma_5', 'ma_20', 'ma_200', 'macd']

        Returns:
            계산된 지표가 추가된 데이터프레임
        """
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df

        # 기본값: 모든 지표 계산
        if indicators is None:
            indicators = list(self.INDICATOR_FUNCTIONS.keys())

        # 유효성 검증
        if not self.validate_indicators(indicators):
            logger.error(f"Some indicators are invalid: {indicators}")
            raise ValueError(f"Invalid indicators provided: {indicators}")

        logger.info(f"Calculating indicators: {indicators}")

        result_df = df.copy()

        for indicator in indicators:
            try:
                if indicator == 'ma_5':
                    result_df['ma_5'] = self._calculate_ma(df['close'], 5)
                    logger.debug(f"✓ Calculated ma_5")

                elif indicator == 'ma_20':
                    result_df['ma_20'] = self._calculate_ma(df['close'], 20)
                    logger.debug(f"✓ Calculated ma_20")

                elif indicator == 'ma_200':
                    result_df['ma_200'] = self._calculate_ma(df['close'], 200)
                    logger.debug(f"✓ Calculated ma_200")

                elif indicator == 'macd':
                    macd_df = self._calculate_macd(df['close'])
                    result_df['macd'] = macd_df['macd']
                    logger.debug(f"✓ Calculated macd")

            except Exception as e:
                logger.error(f"Error calculating {indicator}: {str(e)}")
                raise

        self.calculated_indicators = indicators
        logger.info(f"✓ All requested indicators calculated successfully")

        return result_df

    @staticmethod
    def get_available_indicators() -> List[str]:
        """
        사용 가능한 지표 목록 반환

        Returns:
            지표 목록
        """
        return list(IndicatorCalculator.INDICATOR_FUNCTIONS.keys())

    @staticmethod
    def add_indicator(
        name: str,
        calculation_func
    ) -> None:
        """
        새로운 지표 추가 (확장용)

        Args:
            name: 지표 이름
            calculation_func: 계산 함수 (DataFrame -> Series 또는 DataFrame)
        """
        if name in IndicatorCalculator.INDICATOR_FUNCTIONS:
            logger.warning(f"Indicator '{name}' already exists. Overwriting...")

        IndicatorCalculator.INDICATOR_FUNCTIONS[name] = calculation_func
        logger.info(f"✓ Added new indicator: {name}")
