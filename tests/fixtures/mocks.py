"""
Mock 객체 Fixture
yfinance Mock
"""

import pytest
import pandas as pd
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
