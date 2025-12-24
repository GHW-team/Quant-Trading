"""
IndicatorCalculator 클래스 단위 테스트
기술 지표 계산 및 lookback 로직 검증
"""

import pytest
import pandas as pd
import numpy as np
from src.data.indicator_calculator import IndicatorCalculator

INDICATORS = [
    'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60', 'ma_100', 'ma_120', 'ma_200',
    'macd', 'macd_signal', 'macd_hist',
    'rsi',
    'bb_upper', 'bb_mid', 'bb_lower', 'bb_pct',
    'atr',
    'hv',
    'stoch_k', 'stoch_d',
    'obv' 
]

#==========================================
# 지표계산 함수
#==========================================
# _calc_atr
class TestCalcAtr:
    def test_denominator_zero(self, sample_df_basic):
        #Assert 
        calc = IndicatorCalculator()
        df = sample_df_basic.copy()
        df['close'] = 0

        #Act
        result = calc._calc_atr(df, 14)

        #Assert
        assert result.isna().sum() == len(df)

# calculate_indicators
class TestIndicatorCalculatorCalculateIndicators:
    """지표 계산 메인 메서드 테스트"""
    #==========================================
    # 함수 정상 동작 테스트
    #==========================================
    def test_calculate_multiple_indicators(self, sample_df_basic):
        """여러 지표 동시 계산"""
        calc = IndicatorCalculator()
        indicators = INDICATORS
        result = calc.calculate_indicators(sample_df_basic, indicators)

        for indicator in indicators:
            # 룩백 기간 제외하고, 모두 유효한 값을 반환했는지 검증
            lookback = calc.get_lookback_days(indicator_list=[indicator])
            expected_count = len(sample_df_basic) - lookback + 1

            # 예상기간의 90퍼 이상이만 통과
            assert result[indicator].count() == expected_count
    
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
        indicators = INDICATORS
        #모든 데이터에 대해 계산

        result = calc.calculate_indicators(sample_df_basic, None)

        for ind in indicators:
            assert ind in result.columns

    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_calculate_ma_200(self, sample_df_small):
        """ma_200 계산 - 데이터 부족 시 None으로 fill"""
        #Arrange
        calc = IndicatorCalculator()

        #Act
        # sample_df_basic은 100개 행인데, ma_200은 205개 필요
        df = calc.calculate_indicators(sample_df_small, ['ma_200'])

        #Assert
        #전부 NaN으로 채워짐
        assert len(df['ma_200']) == len(sample_df_small)
        assert df['ma_200'].isna().sum() == len(sample_df_small)
        

    def test_calculate_default_indicators(self, sample_df_basic):
        """기본 지표 목록"""
        calc = IndicatorCalculator()
        available = calc.get_available_indicators()

        assert isinstance(available, list)
        assert len(available) > 0
    
    #==========================================
    # 함수 에러 케이스 테스트
    #==========================================
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
            'open': [100,201,302],
            'high': [101, 201, 301],
            'low': [99, 199, 299],
            'close': [100,200,300],
            'adj_close': [100, 101, 102],
            'volume': [100000,200000,1100000],
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
            'open': [100,201,302],
            'high': [101, 201, 301],
            'low': [99, 199, 299],
            'close': [100,200,300],
            'adj_close': [100, 101, 102],
            'volume': [100000,200000,1100000],
        })

        # ValueError 발생 확인
        with pytest.raises(ValueError, match="datetime type"):
            calc.calculate_indicators(df, ['ma_5'])

    def test_all_nan_values(self):
        """모든 NaN 값"""
        #Arrange
        calc = IndicatorCalculator()
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01',periods=10), 
            'open': [np.nan] * 10,
            'high': [np.nan] * 10,
            'low': [np.nan] * 10,
            'close': [np.nan] * 10,
            'adj_close': [np.nan] * 10,
            'volume': [np.nan] * 10,
        })

        #Act
        result = calc.calculate_indicators(df, ['ma_5'])

        #Assert
        assert 'ma_5' in result.columns
        assert result['ma_5'].isna().all()

#==========================================
# 헬퍼 함수
#==========================================

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
        (['macd_signal'], 34),
        (['macd_hist'], 34),
        (['rsi'], 15),
        (['bb_upper'], 20),
        (['bb_mid'], 20),
        (['bb_lower'], 20),
        (['bb_pct'], 20),
        (['atr'], 15),
        (['hv'], 21),
        (['stoch_k'], 16),
        (['stoch_d'], 18),
        (['obv'], 1),
        (['unkown'], 0),
    ])
    def test_lookback_for_single_indicator(self, indicators, expected_min):
        """단일 지표 lookback 계산"""
        lookback = IndicatorCalculator.get_lookback_days(indicators)
        assert lookback == expected_min

# get_available_indicators
class TestGetAvailableIndicators:
    """지표 리스트 get 함수 테스트"""
    def test_calculate_default_indicators(self, sample_df_basic):
        """기본 지표 목록"""
        calc = IndicatorCalculator()
        available = calc.get_available_indicators()

        assert isinstance(available, list)
        assert len(available) > 0 

# _validate_indicators
class TestValidateIndicators:
    """_validate_indicators 함수 테스트"""
    
    def test_valid_indicators_pass(self):
        """유효한 지표 리스트는 통과"""
        IndicatorCalculator._validate_indicators(['ma_5', 'macd', 'rsi'])
        # 에러 없이 통과해야 함
    
    def test_invalid_indicators_raise_error(self):
        """무효한 지표는 ValueError 발생"""
        with pytest.raises(ValueError, match="Invalid indicators"):
            IndicatorCalculator._validate_indicators(['invalid_indicator'])
    
    def test_mixed_valid_invalid_indicators(self):
        """유효/무효 혼합 시 에러"""
        with pytest.raises(ValueError, match="Invalid indicators"):
            IndicatorCalculator._validate_indicators(['ma_5', 'fake_indicator'])
    
    def test_empty_list(self):
        """빈 리스트는 통과"""
        IndicatorCalculator._validate_indicators([])