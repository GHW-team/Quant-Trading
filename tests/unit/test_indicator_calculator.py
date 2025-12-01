"""
IndicatorCalculator 클래스 단위 테스트
기술 지표 계산 및 lookback 로직 검증
"""

import pytest
import pandas as pd
import numpy as np
from src.data.indicator_calculator import IndicatorCalculator


class TestIndicatorCalculatorGetLookbackDays:
    """lookback 일수 계산 테스트"""

    def test_lookback_for_ma_5(self):
        """ma_5 lookback"""
        lookback = IndicatorCalculator.get_lookback_days(['ma_5'])
        assert lookback >= 5

    def test_lookback_for_ma_20(self):
        """ma_20 lookback"""
        lookback = IndicatorCalculator.get_lookback_days(['ma_20'])
        assert lookback >= 20

    def test_lookback_for_ma_200(self):
        """ma_200 lookback"""
        lookback = IndicatorCalculator.get_lookback_days(['ma_200'])
        assert lookback >= 200

    def test_lookback_for_macd(self):
        """macd lookback"""
        lookback = IndicatorCalculator.get_lookback_days(['macd'])
        assert lookback >= 26  # MACD slow period

    def test_lookback_multiple_indicators(self):
        """여러 지표의 최대 lookback"""
        lookback = IndicatorCalculator.get_lookback_days(['ma_5', 'ma_20', 'macd'])
        # ma_200의 lookback이 최대
        assert lookback >= 26

    def test_lookback_with_ma_200(self):
        """ma_200 포함 시 최대값"""
        lookback_short = IndicatorCalculator.get_lookback_days(['ma_5', 'ma_20'])
        lookback_long = IndicatorCalculator.get_lookback_days(['ma_5', 'ma_20', 'ma_200'])
        assert lookback_long > lookback_short

    def test_lookback_default_all_indicators(self):
        """기본값 (모든 지표)"""
        lookback = IndicatorCalculator.get_lookback_days(None)
        assert lookback >= 200  # ma_200이 최대

    def test_lookback_unknown_indicator(self):
        """알 수 없는 지표는 0으로 처리"""
        lookback = IndicatorCalculator.get_lookback_days(['unknown_indicator'])
        assert lookback > 0  # 최소 안전마진


class TestIndicatorCalculatorInitialization:
    """IndicatorCalculator 초기화 테스트"""

    def test_initialization(self):
        """초기화"""
        calc = IndicatorCalculator()
        assert calc is not None
        assert hasattr(calc, 'calculated_indicators')


class TestIndicatorCalculatorCalculateIndicators:
    """지표 계산 메인 메서드 테스트"""

    def test_calculate_ma_5(self, sample_df_basic):
        """ma_5 계산"""
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(sample_df_basic, ['ma_5'])

        assert 'ma_5' in result.columns
        assert len(result) == len(sample_df_basic)

    def test_calculate_ma_20(self, sample_df_basic):
        """ma_20 계산"""
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(sample_df_basic, ['ma_20'])

        assert 'ma_20' in result.columns

    def test_calculate_ma_200(self, sample_df_basic):
        """ma_200 계산 - 데이터 부족 시 실패"""
        calc = IndicatorCalculator()

        # sample_df_basic은 100개 행인데, ma_200은 205개 필요
        # 따라서 ValueError 발생
        with pytest.raises(ValueError, match="Calculation returned None"):
            calc.calculate_indicators(sample_df_basic, ['ma_200'])

    def test_calculate_macd(self, sample_df_basic):
        """macd 계산"""
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(sample_df_basic, ['macd'])

        assert 'macd' in result.columns

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

    def test_calculate_returns_dataframe(self, sample_df_basic):
        """반환값이 DataFrame"""
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(sample_df_basic, ['ma_5'])

        assert isinstance(result, pd.DataFrame)

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


class TestIndicatorCalculatorEdgeCases:
    """엣지 케이스 테스트"""

    def test_single_row_dataframe(self):
        """1개 행 DataFrame - 데이터 부족으로 계산 실패"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({'adj_close': [100]})

        # 1개 행은 ma_5 계산에 필요한 5개 행(+5 안전마진 = 10)보다 부족
        # ta.ema()가 None을 반환 → ValueError 발생
        with pytest.raises(ValueError, match="Calculation returned None"):
            calc.calculate_indicators(df, ['ma_5'])

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

    def test_constant_price(self):
        """일정한 가격"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({
            'adj_close': [100] * 100
        })

        result = calc.calculate_indicators(df, ['ma_5', 'ma_20', 'macd'])

        # ma들은 모두 100
        assert (result['ma_5'].dropna() == 100).all()
        assert (result['ma_20'].dropna() == 100).all()
        # MACD는 0 근처
        assert result['macd'].dropna().abs().max() < 1e-10

    def test_increasing_price(self):
        """증가하는 가격"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({
            'adj_close': np.arange(100, 200)
        })

        result = calc.calculate_indicators(df, ['ma_5', 'macd'])

        # MACD는 양수 (가격이 계속 증가)
        assert result['macd'].dropna().iloc[-1] > 0

    def test_decreasing_price(self):
        """감소하는 가격"""
        calc = IndicatorCalculator()
        df = pd.DataFrame({
            'adj_close': np.arange(200, 100, -1)
        })

        result = calc.calculate_indicators(df, ['macd'])

        # MACD는 음수 (가격이 계속 감소)
        assert result['macd'].dropna().iloc[-1] < 0


class TestIndicatorCalculatorIntegration:
    """통합 테스트"""

    def test_full_workflow(self, sample_df_basic):
        """완전한 지표 계산 워크플로우"""
        calc = IndicatorCalculator()

        # ma_200은 100개 데이터로는 계산 불가능 (205개 필요)
        # ma_5, ma_20, macd만 테스트
        indicator_list = ['ma_5', 'ma_20', 'macd']

        # 1. lookback 계산
        lookback = calc.get_lookback_days(indicator_list)
        assert lookback > 0

        # 2. 지표 계산
        result = calc.calculate_indicators(sample_df_basic, indicator_list)

        # 3. 검증
        assert all(ind in result.columns for ind in indicator_list)
        assert len(result) == len(sample_df_basic)

    def test_with_realistic_data(self, sample_df_with_indicators):
        """현실적인 데이터로 테스트"""
        calc = IndicatorCalculator()
        result = calc.calculate_indicators(
            sample_df_with_indicators,
            ['ma_5', 'macd']
        )

        assert 'ma_5' in result.columns
        assert 'macd' in result.columns


class TestIndicatorCalculatorAvailableIndicators:
    """사용 가능한 지표 테스트"""

    def test_get_available_indicators(self):
        """사용 가능한 지표 목록"""
        calc = IndicatorCalculator()
        available = calc.get_available_indicators()

        assert isinstance(available, list)
        assert len(available) > 0
        assert 'ma_5' in available or 'ma_20' in available  # 적어도 하나는 있어야 함

    def test_indicators_functions_dict(self):
        """지표 함수 맵"""
        assert 'ma_5' in IndicatorCalculator.INDICATORS_FUNCTIONS
        assert 'ma_20' in IndicatorCalculator.INDICATORS_FUNCTIONS
        assert 'ma_200' in IndicatorCalculator.INDICATORS_FUNCTIONS
        assert 'macd' in IndicatorCalculator.INDICATORS_FUNCTIONS


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
