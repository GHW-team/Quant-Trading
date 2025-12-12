"""
IndicatorCalculator 클래스 단위 테스트
기술 지표 계산 및 lookback 로직 검증
"""

import pytest
import pandas as pd
import numpy as np
from src.data.indicator_calculator import IndicatorCalculator

# get_lookback_days
class TestIndicatorCalculatorGetLookbackDays:
    """lookback 일수 계산 테스트"""

    @pytest.mark.parametrize("indicators,expected_min", [
        (['ma_5'], 5),
        (['ma_10'], 10),
        (['ma_20'], 20),
        (['ma_50'], 50),
        (['ma_60'], 60),
        (['ma_100'], 100),
        (['ma_120'], 120),
        (['ma_200'], 200),
        (['macd'], 26),
        (['macd_signal'], 26),
        (['macd_hist'], 26),
        (['rsi'], 14),
        (['bb_upper'], 20),
        (['bb_mid'], 20),
        (['bb_lower'], 20),
        (['bb_pct'], 20),
        (['atr'], 14),
        (['hv'], 20),
        (['stoch_k'], 14),
        (['stoch_d'], 14),
        (['obv'], 1),
        (['unkown'], 0),
    ])
    def test_lookback_for_single_indicator(self, indicators, expected_min):
        """단일 지표 lookback 계산"""
        lookback = IndicatorCalculator.get_lookback_days(indicators)
        assert lookback == expected_min + 5

    @pytest.mark.parametrize("indicators,expected_min", [
        (['ma_5','ma_10','ma_20','ma_50','ma_60','ma_100'], 100),
        (['ma_120','ma_200','macd','macd_signal','macd_hist'], 200),
        (['rsi','bb_upper','bb_mid','bb_lower','bb_pct'], 20),
        (['atr','hv','stoch_k','stoch_d','obv','unknown'], 20),
    ])
    def test_lookback_for_multiple_indicator(self, indicators, expected_min):
        """다중 지표 lookback 계산"""
        lookback = IndicatorCalculator.get_lookback_days(indicators)
        assert lookback == expected_min + 5

# 
class TestIndicatorCalculatorCalculateIndicators:
    """지표 계산 메인 메서드 테스트"""

    @pytest.mark.parametrize("indicator", ['ma_5', 'ma_20', 'macd'])
    def test_calculate_single_indicator(self, sample_df_basic, indicator):
        """단일 지표 계산"""
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(sample_df_basic, [indicator])

        assert indicator in result.columns
        assert len(result) == len(sample_df_basic)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_ma_200(self, sample_df_basic):
        """ma_200 계산 - 데이터 부족 시 실패"""
        calc = IndicatorCalculator()

        # sample_df_basic은 100개 행인데, ma_200은 205개 필요
        # 따라서 ValueError 발생
        with pytest.raises(ValueError, match="Calculation returned None"):
            calc.calculate_indicators(sample_df_basic, ['ma_200'])

    def test_calculate_multiple_indicators(self, sample_df_basic):
        """여러 지표 동시 계산"""
        calc = IndicatorCalculator()
        # ma_200은 데이터 부족으로 제외 (100개 행에서는 불가능)
        indicators = ['ma_5', 'ma_20', 'macd']
        result = calc.calculate_indicators(sample_df_basic, indicators)

        for indicator in indicators:
            assert indicator in result.columns

    def test_calculate_preserves_original_columns(self, sample_df_basic):
        """원본 컬럼 유지"""
        original_cols = set(sample_df_basic.columns)
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(sample_df_basic, ['ma_5'])

        for col in original_cols:
            assert col in result.columns

    def test_calculate_with_none_indicators(self, sample_df_basic):
        """indicator_list=None (기본값) - 모든 지표 계산 시도"""
        calc = IndicatorCalculator()

        # indicator_list=None일 때 모든 지표를 계산하려고 함
        # 하지만 ma_200이 포함되어 있어 데이터 부족으로 실패
        with pytest.raises(ValueError, match="Calculation returned None"):
            calc.calculate_indicators(sample_df_basic, None)

    def test_calculate_default_indicators(self, sample_df_basic):
        """기본 지표 목록"""
        calc = IndicatorCalculator()
        available = calc.get_available_indicators()

        assert isinstance(available, list)
        assert len(available) > 0


class TestIndicatorCalculatorErrorHandling:
    """에러 처리 테스트"""

    def test_empty_dataframe_raises_error(self):
        """빈 DataFrame에서 에러"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({'adj_close': []})

        with pytest.raises(ValueError, match="empty dataframe"):
            calc.calculate_indicators(df, ['ma_5'])

    def test_missing_adj_close_column_raises_error(self):
        """adj_close 컬럼 누락 시 에러"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(ValueError, match="adj_close"):
            calc.calculate_indicators(df, ['ma_5'])

    def test_invalid_dataframe_type_raises_error(self):
        """DataFrame이 아닌 입력"""
        calc = IndicatorCalculator()
        data = np.array([100, 101, 102])

        # numpy array 입력 시 AttributeError 발생 (df.empty 속성 없음)
        with pytest.raises((ValueError, TypeError, AttributeError)):
            calc.calculate_indicators(data, ['ma_5'])

    def test_unsorted_date_column_raises_error(self):
        """date 컬럼이 시간순으로 정렬되지 않은 경우 에러"""
        calc = IndicatorCalculator()

        # date가 섞여 있는 DataFrame 생성
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02']),
            'adj_close': [100, 101, 102]
        })

        # ValueError 발생 확인
        with pytest.raises(ValueError, match="chronological.*order"):
            calc.calculate_indicators(df, ['ma_5'])

    def test_date_column_wrong_type_raises_error(self):
        """date 컬럼이 datetime이 아닌 경우 에러"""
        calc = IndicatorCalculator()

        # date가 문자열인 DataFrame
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],  # str
            'adj_close': [100, 101, 102]
        })

        # ValueError 발생 확인
        with pytest.raises(ValueError, match="datetime type"):
            calc.calculate_indicators(df, ['ma_5'])


class TestIndicatorCalculatorEdgeCases:
    """엣지 케이스 테스트"""

    def test_small_dataframe_for_long_window(self):
        """작은 DataFrame에 긴 window 지표 - 데이터 부족"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({
            'adj_close': [100, 101, 102, 103, 104]  # 5개
        })

        # ma_200은 200 + 5(안전마진) = 205개 필요
        # 5개만 제공하면 ta.ema()가 None 반환 → ValueError 발생
        with pytest.raises(ValueError, match="Calculation returned None"):
            calc.calculate_indicators(df, ['ma_200'])

    def test_all_nan_values(self):
        """모든 NaN 값"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({'adj_close': [np.nan] * 10})

        result = calc.calculate_indicators(df, ['ma_5'])
        assert 'ma_5' in result.columns
        assert result['ma_5'].isna().all()

    def test_large_dataframe(self):
        """큰 DataFrame"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({
            'adj_close': np.arange(10000)
        })

        result = calc.calculate_indicators(df, ['ma_5', 'ma_20', 'macd'])
        assert len(result) == 10000


class TestIndicatorCalculatorAvailableIndicators:
    """사용 가능한 지표 테스트"""

    def test_get_available_indicators(self):
        """사용 가능한 지표 목록"""
        calc = IndicatorCalculator()
        available = calc.get_available_indicators()

        assert isinstance(available, list)
        assert len(available) > 0


class TestIndicatorCalculatorDataTypePreservation:
    """데이터 타입 보존 테스트"""

    def test_float_preservation(self, sample_df_basic):
        """float 데이터 타입 보존"""
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(sample_df_basic, ['ma_5'])

        assert pd.api.types.is_float_dtype(result['ma_5'])

    def test_datetime_column_preserved(self, sample_df_basic):
        """datetime 컬럼 보존"""
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(sample_df_basic, ['ma_5'])

        assert 'date' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
