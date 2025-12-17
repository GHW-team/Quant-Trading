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
        'date': pd.date_range('2020-01-01', periods=1000),
        'open': np.arange(100, 1100),
        'high': np.arange(101, 1101),
        'low': np.arange(99, 1099),
        'close': np.arange(100, 1100),
        'adj_close': np.arange(100, 1100),
        'volume': np.arange(1000000, 1000000 + 1000)
    })

@pytest.fixture
def sample_df_small():
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
    """모든 보조지표 컬럼이 포함된 테스트용 데이터프레임"""
    df = sample_df_basic.copy()
    
    # 데이터 길이만큼의 랜덤값 생성을 위한 난수 생성기
    n_rows = len(df)

    # 지표 : 실제 계산 대신 종가와 비슷한 값으로 채움

    # 1. 이동평균선 (MA) - 단순히 종가에 약간의 노이즈를 섞어서 할당
    ma_columns = ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60', 'ma_100', 'ma_120', 'ma_200']
    for col in ma_columns:
        window_size = int(col.split('_')[1])
        # 주의: NaN값 포함됨
        df[col] = df['adj_close'].rolling(window=window_size).mean()

    # 2. MACD 관련 - 0 근처의 작은 소수값 할당
    for col in ['macd', 'macd_hist', 'macd_signal']:
        df[col] = np.random.randn(n_rows)  # 표준정규분포 난수

    # 3. 오실레이터 (0 ~ 100 범위) - RSI, Stochastic
    for col in ['rsi', 'stoch_k', 'stoch_d']:
        df[col] = np.random.uniform(20, 80, n_rows)

    # 4. 볼린저 밴드 - 종가 기준으로 위아래 벌림
    df['bb_mid'] = df['adj_close']
    df['bb_upper'] = df['adj_close'] * 1.05  # 5% 위
    df['bb_lower'] = df['adj_close'] * 0.95  # 5% 아래
    df['bb_pct'] = np.random.uniform(0, 1, n_rows) # %B는 0~1 사이

    # 5. 기타 지표
    df['atr'] = df['adj_close'] * 0.02 # 변동성 (대략 주가의 2%)
    df['hv'] = np.random.uniform(0.1, 0.5, n_rows) # 역사적 변동성
    df['obv'] = df['volume'].cumsum() # 거래량 누적 (이건 간단해서 실제 로직 적용)

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
