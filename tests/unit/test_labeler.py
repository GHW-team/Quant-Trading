"""
Labeler 클래스 단위 테스트
고정 기간 수익률 기반 라벨링 검증
"""

import pytest
import pandas as pd
import numpy as np
from src.ml.labeler import Labeler


class TestLabeler:
    """라벨 생성 기능 테스트"""

    # ==================================
    # 함수 정상 작동 테스트
    # ==================================
    def test_label_column_created(self, sample_df_basic):
        """label 컬럼이 생성되는가?"""
        labeler = Labeler()
        result = labeler.label_data(sample_df_basic)

        # label 컬럼 생성 여부
        assert 'label' in result.columns
        assert len(result['label']) > 0

        # DataFrame 행 개수 유지        
        assert len(result) == len(sample_df_basic)

        #return_n 컬럼은 제거되어야 함
        assert 'return_n' not in result.columns

    def test_label_only_contains_0_1_nan(self, sample_df_basic):
        """label 값이 0, 1, NaN만 포함"""
        labeler = Labeler()
        result = labeler.label_data(sample_df_basic)

        assert result['label'].isin([1,0,np.nan]).all()

    def test_last_horizon_rows_are_nan(self, sample_df_basic):
        """마지막 horizon개 행은 NaN"""
        horizon = 5
        labeler = Labeler(horizon=horizon)
        result = labeler.label_data(sample_df_basic)

        # 마지막 horizon개 행은 미래 데이터가 없으므로 NaN
        assert result['label'].iloc[-horizon:].isna().all()

        # 마지막 horizon개 이전 행들은 NaN이 아님 (1/0)
        before_last = result['label'].iloc[:-horizon]
        assert not before_last.isna().any()
    
    @pytest.mark.parametrize("price_start,price_end,threshold,expected_label", [
        # 상승 케이스 (label=1)
        (100, 110, 0.02, 1),      # 10% > 2%
        (100, 110, 0.05, 1),      # 10% > 5%
        (100, 110, 0.10, 1),      # 10% = 10%
        (100, 102.5, 0.02, 1),    # 2.5% > 2%
        (100, 100.01, 0.0, 1),    # 0.01%, threshold=0
        # 하락/중립 케이스 (label=0)
        (100, 101.5, 0.02, 0),    # 1.5% < 2%
        (100, 98, 0.02, 0),       # -2% (음수)
    ])
    def test_threshold_logic(self, price_start, price_end, threshold, expected_label):
        """다양한 threshold와 가격 변동 테스트"""
        labeler = Labeler(horizon=1, threshold=threshold)
        df = pd.DataFrame({
            'adj_close': [price_start, price_end],
            'date': pd.date_range('2020-01-01',periods=2)
        })

        result = labeler.label_data(df)
        assert result['label'].iloc[0] == expected_label
    
    def test_sorts_by_date_column(self):
        """date 컬럼이 있으면 정렬"""
        labeler = Labeler(horizon=1)
        df = pd.DataFrame({
            'date': ['2020-01-03', '2020-01-01', '2020-01-02'],
            'adj_close': [103, 101, 102]
        })

        result = labeler.label_data(df)

        # 날짜순으로 정렬되었는가?
        dates = pd.to_datetime(result['date'])
        assert (dates.diff()[1:] >= pd.Timedelta(days=0)).all()

    # ==================================
    # 함수 엣지 케이스 테스트
    # ==================================

    def test_empty_dataframe(self):
        """빈 DataFrame에서 작동"""
        labeler = Labeler()
        df = pd.DataFrame({
            'adj_close': [],
            'date': [],
        })

        result = labeler.label_data(df)
        assert len(result) == 0

    def test_single_row_dataframe(self):
        """1개 행 DataFrame 처리"""
        labeler = Labeler(horizon=5)
        df = pd.DataFrame({
            'adj_close': [100],
            'date': ['2020-01-01'],
        })

        result = labeler.label_data(df)
        assert len(result) == 1
        # 1개 행만 있으면 모두 NaN (미래 데이터 없음)
        assert pd.isna(result['label'].iloc[0])

    def test_handles_nan_values(self, sample_df_with_nan):
        """NaN 값이 있는 데이터 처리"""
        labeler = Labeler(horizon=1)
        result = labeler.label_data(sample_df_with_nan)

        assert 'label' in result.columns
        assert len(result) == len(sample_df_with_nan)

    def test_nan_propagates_in_calculation(self, sample_df_with_nan):
        """NaN은 계산에서 처리됨 (NaN >= threshold → False → 0)"""
        labeler = Labeler(horizon=1)
        result = labeler.label_data(sample_df_with_nan)

        # NaN이 가격에 있으면: (NaN - NaN) / NaN = NaN
        # NaN >= threshold는 False이므로 label=0 (정수 변환)
        # 마지막 horizon 행(index 9)은 미래 데이터 없음 → NaN
        nan_indices = sample_df_with_nan[sample_df_with_nan['adj_close'].isna()].index
        # NaN 가격 행들은 label=0이어야 함 (NaN >= threshold → False → 0)
        assert (result.loc[nan_indices, 'label'] == 0.0).all(), \
            "NaN 가격은 label=0을 생성해야 함"

    # ==================================
    # 함수 에러 케이스 테스트
    # ==================================

    def test_raises_error_missing_price_column(self):
        """필수 컬럼(adj_close)이 없으면 ValueError"""
        labeler = Labeler()
        df = pd.DataFrame({'wrong_col': [100, 101, 102]})
        df = pd.DataFrame({
            'wrong_col': [100, 101, 102],
            'date': ['2020-01-01','2020-01-02','2020-01-03'],
        })

        with pytest.raises(ValueError, match="Column 'adj_close' not found"):
            labeler.label_data(df)

    def test_raises_error_missing_date_column(self):
        """필수 컬럼(date)이 없으면 ValueError"""
        labeler = Labeler()
        df = pd.DataFrame({
            'adj_close': [100,101,102],
            'worng_col': [50, 51, 52], 
        })
        with pytest.raises(ValueError, match="Column 'date' not found in DataFrame"):
            labeler.label_data(df)