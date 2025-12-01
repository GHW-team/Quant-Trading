"""
샘플 테스트 데이터 Fixture
시계열 데이터, 지표, 라벨, 학습 데이터 등
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df_basic():
    """기본 시계열 데이터프레임 (가격 데이터)"""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'open': np.arange(100, 200),
        'high': np.arange(101, 201),
        'low': np.arange(99, 199),
        'close': np.arange(100, 200),
        'adj_close': np.arange(100, 200),
        'volume': np.arange(1000000, 1000000 + 100)
    })


@pytest.fixture
def sample_df_with_indicators(sample_df_basic):
    """지표가 포함된 데이터프레임"""
    df = sample_df_basic.copy()

    # 간단한 이동평균 추가
    df['ma_5'] = df['adj_close'].rolling(window=5).mean()
    df['ma_20'] = df['adj_close'].rolling(window=20).mean()
    df['ma_200'] = df['adj_close'].rolling(window=200).mean()

    # MACD 간단 버전
    df['macd'] = df['adj_close'].ewm(span=12).mean() - df['adj_close'].ewm(span=26).mean()

    return df


@pytest.fixture
def sample_df_labeled(sample_df_with_indicators):
    """라벨이 있는 데이터프레임"""
    df = sample_df_with_indicators.copy()

    # 간단한 라벨 추가
    df['label'] = (df['adj_close'].pct_change(5) > 0.02).astype(int)

    # 마지막 5개 행은 NaN
    df.loc[df.index[-5:], 'label'] = np.nan

    return df


@pytest.fixture
def sample_training_data(sample_df_labeled):
    """모델 학습용 X, y 데이터"""
    feature_cols = ['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']

    # NaN 제거
    clean_data = sample_df_labeled.dropna(subset=feature_cols + ['label'])

    X = clean_data[feature_cols].astype(float)
    y = clean_data['label'].astype(int)

    return X, y


@pytest.fixture
def sample_df_empty():
    """빈 데이터프레임"""
    return pd.DataFrame({
        'date': [],
        'adj_close': [],
    })


@pytest.fixture
def sample_df_single_row():
    """1개 행만 있는 데이터프레임"""
    return pd.DataFrame({
        'date': [pd.Timestamp('2020-01-01')],
        'adj_close': [100],
    })


@pytest.fixture
def sample_df_with_nan():
    """NaN 값이 포함된 데이터프레임"""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'adj_close': [100, np.nan, 102, np.nan, 104, np.nan, 106, np.nan, 108, 109],
    })


@pytest.fixture
def sample_training_data_ml():
    """
    머신러닝 모델 학습용 데이터 (순수 샘플)

    기존 sample_training_data는 시계열 지표를 포함하여 많은 NaN을 가지고 있음.
    이 fixture는 순수하게 학습에 필요한 깨끗한 데이터를 제공함.

    - 300 샘플 (TimeSeriesSplit(n_splits=5) 안정성을 위함)
    - 5개 피처
    - NaN 없음
    - 균형잡힌 레이블 (50:50)

    TimeSeriesSplit은 시계열 데이터를 위해 순차적으로 분할하므로,
    충분한 샘플 수가 필요함. 300개면 각 fold에 60개 이상의 샘플이 있어
    두 클래스 모두 충분히 표현됨.
    """
    np.random.seed(42)

    n_samples = 300
    X = pd.DataFrame({
        'adj_close': np.random.randn(n_samples) * 10 + 100,
        'ma_5': np.random.randn(n_samples) * 5 + 50,
        'ma_20': np.random.randn(n_samples) * 5 + 50,
        'ma_200': np.random.randn(n_samples) * 5 + 50,
        'macd': np.random.randn(n_samples) * 2,
    })

    # 균형잡힌 레이블 생성 (시계열을 고려하여 번갈아가며)
    y = pd.Series(np.tile([0, 1], n_samples // 2))

    return X, y
