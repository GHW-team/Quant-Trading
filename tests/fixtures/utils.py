"""
유틸리티 Fixture
DataFrame/Series 비교 함수 등
"""

import pytest
import pandas as pd


@pytest.fixture
def assert_df_equal():
    """DataFrame 비교 함수"""
    def _assert_equal(df1, df2, check_dtype=True):
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    return _assert_equal


@pytest.fixture
def assert_series_equal():
    """Series 비교 함수"""
    def _assert_equal(s1, s2, check_dtype=True):
        pd.testing.assert_series_equal(s1, s2, check_dtype=check_dtype)
    return _assert_equal
