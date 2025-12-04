"""
Labeler 클래스 단위 테스트
고정 기간 수익률 기반 라벨링 검증
"""

import pytest
import pandas as pd
import numpy as np
from src.ml.labeler import Labeler


class TestLabelerInitialization:
    """Labeler 초기화 테스트"""

    def test_default_initialization(self):
        """기본값으로 초기화"""
        labeler = Labeler()
        assert labeler.horizon == 5
        assert labeler.threshold == 0.02


class TestLabelerBasicFunctionality:
    """라벨 생성 기본 기능 테스트"""

    def test_label_column_created(self, sample_df_basic):
        """label 컬럼이 생성되는가?"""
        labeler = Labeler()
        result = labeler.label_data(sample_df_basic)

        assert 'label' in result.columns
        assert len(result['label']) > 0

    def test_dataframe_shape_preserved(self, sample_df_basic):
        """데이터프레임 행 개수 유지"""
        labeler = Labeler()
        result = labeler.label_data(sample_df_basic)

        assert len(result) == len(sample_df_basic)
        assert result.shape[0] == sample_df_basic.shape[0]

    def test_return_n_column_removed(self, sample_df_basic):
        """return_n 컬럼은 제거되어야 함"""
        labeler = Labeler()
        result = labeler.label_data(sample_df_basic)

        assert 'return_n' not in result.columns

    def test_label_only_contains_0_1_nan(self, sample_df_basic):
        """label 값이 0, 1, NaN만 포함"""
        labeler = Labeler()
        result = labeler.label_data(sample_df_basic)

        valid_labels = {0.0, 1.0, np.nan}
        actual_labels = set(result['label'].dropna().unique())
        assert actual_labels.issubset({0.0, 1.0})


class TestLabelerLastHorizonNaN:
    """마지막 horizon 행의 NaN 처리 테스트"""

    def test_last_horizon_rows_are_nan(self, sample_df_basic):
        """마지막 horizon개 행은 NaN"""
        horizon = 5
        labeler = Labeler(horizon=horizon)
        result = labeler.label_data(sample_df_basic)

        # 마지막 5개 행은 미래 데이터가 없으므로 NaN
        assert result['label'].iloc[-5:].isna().all()

    def test_nan_count_equals_horizon(self, sample_df_basic):
        """NaN 개수가 정확히 horizon개"""
        horizon = 7
        labeler = Labeler(horizon=horizon)
        result = labeler.label_data(sample_df_basic)

        nan_count = result['label'].isna().sum()
        assert nan_count == horizon

    def test_before_last_horizon_not_all_nan(self, sample_df_basic):
        """마지막 horizon 행 이전은 NaN이 아님"""
        horizon = 5
        labeler = Labeler(horizon=horizon)
        result = labeler.label_data(sample_df_basic)

        # 마지막 5개 이전 행들은 NaN이 아님 (1/0)
        before_last = result['label'].iloc[:-horizon]
        assert not before_last.isna().any()


class TestLabelerErrorHandling:
    """에러 처리 테스트"""

    def test_raises_error_missing_price_column(self):
        """필수 컬럼(adj_close)이 없으면 ValueError"""
        labeler = Labeler()
        df = pd.DataFrame({'wrong_col': [100, 101, 102]})

        with pytest.raises(ValueError, match="Column 'adj_close' not found"):
            labeler.label_data(df)

    def test_custom_price_column_not_found(self, sample_df_basic):
        """커스텀 price_col이 없으면 ValueError"""
        labeler = Labeler()
        df = sample_df_basic.copy()

        with pytest.raises(ValueError, match="Column 'custom_price' not found"):
            labeler.label_data(df, price_col='custom_price')

    def test_empty_dataframe(self):
        """빈 DataFrame에서 작동"""
        labeler = Labeler()
        df = pd.DataFrame({'adj_close': []})

        result = labeler.label_data(df)
        assert len(result) == 0

    def test_single_row_dataframe(self):
        """1개 행 DataFrame 처리"""
        labeler = Labeler(horizon=5)
        df = pd.DataFrame({'adj_close': [100]})

        result = labeler.label_data(df)
        assert len(result) == 1
        # 1개 행만 있으면 모두 NaN (미래 데이터 없음)
        assert pd.isna(result['label'].iloc[0])


class TestLabelerThresholdLogic:
    """threshold 로직 테스트"""

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
        df = pd.DataFrame({'adj_close': [price_start, price_end]})

        result = labeler.label_data(df)
        assert result['label'].iloc[0] == expected_label


class TestLabelerDateSorting:
    """날짜 정렬 테스트"""

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

    def test_without_date_column(self, sample_df_basic):
        """date 컬럼이 없어도 작동"""
        labeler = Labeler(horizon=1)
        df = pd.DataFrame({
            'adj_close': [100, 101, 102, 103]
        })

        result = labeler.label_data(df)
        assert 'label' in result.columns

    def test_date_column_preserved(self, sample_df_basic):
        """date 컬럼이 원본에서 유지됨"""
        labeler = Labeler()
        result = labeler.label_data(sample_df_basic)

        assert 'date' in result.columns
        pd.testing.assert_series_equal(
            result['date'].reset_index(drop=True),
            sample_df_basic['date'].reset_index(drop=True)
        )


class TestLabelerWithNaNValues:
    """NaN 값이 포함된 데이터프레임 테스트"""

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


class TestLabelerCopyBehavior:
    """원본 DataFrame 수정 여부 테스트"""

    def test_does_not_modify_original(self, sample_df_basic):
        """원본 DataFrame을 수정하지 않음"""
        labeler = Labeler()
        df_copy = sample_df_basic.copy()

        labeler.label_data(sample_df_basic)

        # 원본이 변경되지 않았는지 확인
        pd.testing.assert_frame_equal(sample_df_basic, df_copy)

    def test_returns_new_dataframe(self, sample_df_basic):
        """새로운 DataFrame을 반환"""
        labeler = Labeler()
        result = labeler.label_data(sample_df_basic)

        # 다른 객체여야 함
        assert result is not sample_df_basic