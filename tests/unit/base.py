"""
단위 테스트 베이스 클래스
반복되는 설정 및 헬퍼 메서드를 제공
"""

import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


class TestBase:
    """모든 테스트의 기본 클래스"""

    @staticmethod
    def create_df(rows=10, columns=None, start_value=100):
        """
        테스트용 DataFrame 빠르게 생성

        Args:
            rows: 행 수
            columns: 컬럼명 리스트 또는 딕셔너리
            start_value: 수치 데이터 시작값

        Returns:
            pd.DataFrame
        """
        if columns is None:
            columns = ['adj_close']

        if isinstance(columns, dict):
            # {'col1': values_list, 'col2': values_list}
            return pd.DataFrame(columns)
        else:
            # ['col1', 'col2'] -> 자동 생성
            data = {}
            for i, col in enumerate(columns):
                if col == 'date':
                    data[col] = pd.date_range('2020-01-01', periods=rows)
                elif col in ['adj_close', 'open', 'high', 'low', 'close']:
                    data[col] = np.arange(start_value, start_value + rows)
                elif col == 'volume':
                    data[col] = np.arange(1000000, 1000000 + rows)
                else:
                    data[col] = np.random.randn(rows)
            return pd.DataFrame(data)

    @staticmethod
    def assert_no_nan(df, columns=None):
        """
        DataFrame에서 NaN이 없는지 확인

        Args:
            df: DataFrame
            columns: 확인할 컬럼 (None이면 전체)

        Raises:
            AssertionError: NaN이 발견되면
        """
        if columns is None:
            assert not df.isna().any().any(), f"Found NaN values in {df.columns.tolist()}"
        else:
            for col in columns:
                assert not df[col].isna().any(), f"Found NaN values in column '{col}'"

    @staticmethod
    def assert_has_columns(df, expected_columns):
        """
        DataFrame이 필요한 컬럼을 모두 가지고 있는지 확인

        Args:
            df: DataFrame
            expected_columns: 필수 컬럼 리스트

        Raises:
            AssertionError: 컬럼 부족 시
        """
        missing = set(expected_columns) - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    @staticmethod
    def assert_shape(df, expected_rows=None, expected_cols=None):
        """
        DataFrame의 shape 확인

        Args:
            df: DataFrame
            expected_rows: 기대 행 수 (None이면 확인 안 함)
            expected_cols: 기대 열 수 (None이면 확인 안 함)

        Raises:
            AssertionError: Shape 불일치 시
        """
        if expected_rows is not None:
            assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
        if expected_cols is not None:
            assert len(df.columns) == expected_cols, f"Expected {expected_cols} cols, got {len(df.columns)}"


class TestDatabaseBase(TestBase):
    """데이터베이스 관련 테스트 베이스"""

    def setup_ticker(self, manager, ticker_code='005930.KS', name='삼성전자', market='KOSPI'):
        """
        테스트용 Ticker 생성 및 ID 반환

        Args:
            manager: DatabaseManager
            ticker_code: Ticker 코드
            name: 종목명
            market: 시장명

        Returns:
            int: ticker_id
        """
        return manager._get_ticker_id(ticker_code, name=name, market=market)

    def save_sample_prices(self, manager, ticker_id, df, periods=50):
        """
        샘플 가격 데이터 저장

        Args:
            manager: DatabaseManager
            ticker_id: Ticker ID
            df: 데이터프레임 (기본값 사용 시 None)
            periods: 행 수 (df가 None일 때만 사용)

        Returns:
            list: 저장된 행 ID들
        """
        if df is None:
            df = self.create_df(rows=periods, columns=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])

        manager.save_price_data(
            {f'TICKER_{ticker_id}': df},
            update_if_exists=True
        )
        return df

    def save_sample_indicators(self, manager, ticker_id, df, indicators=None):
        """
        샘플 지표 데이터 저장

        Args:
            manager: DatabaseManager
            ticker_id: Ticker ID
            df: 데이터프레임 (기본값 사용 시 None)
            indicators: 지표명 리스트 (기본값: ['ma_5', 'ma_20', 'macd'])

        Returns:
            pd.DataFrame: 저장된 데이터
        """
        if indicators is None:
            indicators = ['ma_5', 'ma_20', 'macd']

        if df is None:
            cols = ['date'] + indicators
            df = self.create_df(rows=50, columns=cols)

        manager.save_indicators(
            {f'TICKER_{ticker_id}': df},
            version='v1.0'
        )
        return df

    def verify_saved_data(self, manager, ticker_code):
        """
        저장된 데이터 검증

        Args:
            manager: DatabaseManager
            ticker_code: Ticker 코드

        Returns:
            dict: (price_df, indicator_df) 튜플
        """
        prices = manager.load_price_data(ticker_code)
        indicators = manager.load_indicators(ticker_code)
        return prices, indicators


class TestPipelineBase(TestBase):
    """파이프라인 관련 테스트 베이스"""

    def get_sample_tickers(self):
        """테스트용 Ticker 리스트 반환"""
        return ['005930.KS', '000660.KS']

    def get_date_range(self, start='2020-01-01', end='2020-12-31'):
        """테스트용 날짜 범위 반환"""
        return pd.date_range(start, end)

    def create_mock_pipeline_data(self, tickers=None, periods=100):
        """
        파이프라인 테스트용 Mock 데이터 생성

        Args:
            tickers: Ticker 리스트
            periods: 행 수

        Returns:
            dict: {ticker: DataFrame}
        """
        if tickers is None:
            tickers = self.get_sample_tickers()

        result = {}
        for ticker in tickers:
            result[ticker] = self.create_df(
                rows=periods,
                columns=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            )
        return result


class TestMLBase(TestBase):
    """머신러닝 관련 테스트 베이스"""

    def create_training_data(self, n_samples=50, n_features=5):
        """
        모델 학습용 X, y 데이터 생성

        Args:
            n_samples: 샘플 수
            n_features: 특성 수

        Returns:
            tuple: (X: pd.DataFrame, y: pd.Series)
        """
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
        y = pd.Series(np.random.randint(0, 2, n_samples))
        return X, y

    def assert_valid_predictions(self, predictions, expected_length=None):
        """
        모델 예측값 검증

        Args:
            predictions: 예측값 배열
            expected_length: 기대 길이 (None이면 확인 안 함)

        Raises:
            AssertionError: 검증 실패 시
        """
        if expected_length is not None:
            assert len(predictions) == expected_length, \
                f"Expected {expected_length} predictions, got {len(predictions)}"

        # 이진 분류 검증
        assert set(predictions).issubset({0, 1, 0.0, 1.0}), \
            f"Predictions should be binary (0 or 1), got {set(predictions)}"

    def assert_valid_probabilities(self, probabilities, expected_length=None):
        """
        모델 확률값 검증

        Args:
            probabilities: 확률값 배열 또는 (n, 2) 형태
            expected_length: 기대 길이 (None이면 확인 안 함)

        Raises:
            AssertionError: 검증 실패 시
        """
        if expected_length is not None:
            assert len(probabilities) == expected_length, \
                f"Expected {expected_length} probabilities, got {len(probabilities)}"

        # 확률 범위 검증 (0 ~ 1)
        if isinstance(probabilities, (list, np.ndarray)):
            if len(probabilities) > 0 and isinstance(probabilities[0], (list, np.ndarray)):
                # 2D 확률 (n, 2)
                assert np.all((probabilities >= 0) & (probabilities <= 1)), \
                    "Probabilities should be between 0 and 1"
            else:
                # 1D 확률
                assert np.all((np.array(probabilities) >= 0) & (np.array(probabilities) <= 1)), \
                    "Probabilities should be between 0 and 1"
