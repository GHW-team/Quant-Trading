"""
Mock 객체 Fixture
yfinance, DatabaseManager, StockDataFetcher, IndicatorCalculator 등
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_yfinance():
    """yfinance Mock"""
    with patch('yfinance.Ticker') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # history() 메서드 설정
        mock_instance.history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Adj Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000],
        }, index=pd.date_range('2020-01-01', periods=3, name = 'Date'))

        yield mock


@pytest.fixture
def mock_database_manager(temp_db_path):
    """DatabaseManager Mock"""
    with patch('src.data.pipeline.DatabaseManager') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # 기본 메서드 설정
        mock_instance.save_prices.return_value = True
        mock_instance.save_indicators.return_value = True
        mock_instance.load_price_data.return_value = {'005930.KS': pd.DataFrame()}
        mock_instance.load_indicators.return_value = {'005930.KS': pd.DataFrame()}

        yield mock


@pytest.fixture
def mock_data_fetcher():
    """StockDataFetcher Mock"""
    with patch('src.data.pipeline.StockDataFetcher') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # fetch_multiple() 메서드 설정
        mock_instance.fetch_multiple.return_value = {
            '005930.KS': pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=100),
                'open': np.arange(100, 200),
                'close': np.arange(100, 200),
                'adj_close': np.arange(100, 200),
                'volume': np.arange(1000000, 1000000 + 100),
            })
        }

        yield mock


@pytest.fixture
def mock_indicator_calculator():
    """IndicatorCalculator Mock"""
    with patch('src.data.pipeline.IndicatorCalculator') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # calculate_indicators() 메서드 설정
        def mock_calculate(df, indicator_list):
            df_copy = df.copy()
            for indicator in (indicator_list or []):
                df_copy[indicator] = np.random.randn(len(df))
            return df_copy

        mock_instance.calculate_indicators.side_effect = mock_calculate

        yield mock
