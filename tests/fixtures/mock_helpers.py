"""
Mock 헬퍼 함수들
반복되는 Mock 설정을 재사용 가능하게 추상화
"""

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def create_sample_ohlcv_dataframe(periods=100, start_date='2020-01-01',
                                  base_price=100, columns=None):
    """
    재사용 가능한 샘플 OHLCV DataFrame 생성

    Args:
        periods: 행 수
        start_date: 시작 날짜
        base_price: 기본 가격
        columns: 포함할 컬럼 ('open', 'high', 'low', 'close', 'adj_close', 'volume' 등)

    Returns:
        pd.DataFrame: OHLCV 데이터
    """
    if columns is None:
        columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']

    df_dict = {
        'date': pd.date_range(start_date, periods=periods),
        'open': np.arange(base_price, base_price + periods),
        'high': np.arange(base_price + 1, base_price + periods + 1),
        'low': np.arange(base_price - 1, base_price + periods - 1),
        'close': np.arange(base_price, base_price + periods),
        'adj_close': np.arange(base_price, base_price + periods),
        'volume': np.arange(1000000, 1000000 + periods),
    }

    return pd.DataFrame({k: v for k, v in df_dict.items() if k in columns or k == 'date'})


def setup_yfinance_mock(mock_data=None):
    """
    yfinance Mock 설정

    Args:
        mock_data: 반환할 DataFrame (None이면 기본값 사용)

    Returns:
        MagicMock: 설정된 yfinance.Ticker mock

    사용 예:
        with setup_yfinance_mock(custom_df) as mock:
            result = StockDataFetcher()._fetch_single_by_period('005930.KS')
    """
    if mock_data is None:
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Adj Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000],
        }, index=pd.date_range('2020-01-01', periods=3))

    mock = MagicMock()
    mock.history.return_value = mock_data
    return mock


def setup_pykrx_mock(tickers_kospi=None, tickers_kosdaq=None):
    """
    pykrx Mock 설정

    Args:
        tickers_kospi: KOSPI ticker 리스트
        tickers_kosdaq: KOSDAQ ticker 리스트

    Returns:
        tuple: (mock_pykrx, context_manager)

    사용 예:
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.side_effect = [
                tickers_kospi or ['005930', '000660'],
                tickers_kosdaq or ['068270', '095570']
            ]
    """
    if tickers_kospi is None:
        tickers_kospi = ['005930', '000660']
    if tickers_kosdaq is None:
        tickers_kosdaq = ['068270', '095570']

    return tickers_kospi, tickers_kosdaq


def setup_fdr_mock(data_by_exchange=None):
    """
    FinanceDataReader (fdr) Mock 설정

    Args:
        data_by_exchange: 거래소별 데이터 딕셔너리
            {'S&P500': DataFrame, 'NYSE': DataFrame, ...}

    Returns:
        MagicMock: 설정된 fdr.StockListing mock

    사용 예:
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame({'Symbol': ['AAPL', 'MSFT']})
    """
    if data_by_exchange is None:
        data_by_exchange = {
            'S&P500': pd.DataFrame({'Symbol': ['AAPL', 'MSFT', 'GOOGL']}),
            'NYSE': pd.DataFrame({'Symbol': ['IBM', 'GE']}),
            'AMEX': pd.DataFrame({'Symbol': ['XBI', 'XLY']}),
            'KRX': pd.DataFrame({'Symbol': ['005930', '000660'], 'Market': ['KOSPI', 'KOSPI']}),
        }

    mock = MagicMock()
    mock.StockListing.side_effect = lambda exchange: data_by_exchange.get(
        exchange, pd.DataFrame()
    )
    return mock


def setup_pipeline_mocks(price_data=None, indicator_data=None):
    """
    Pipeline 관련 Mock들을 한번에 설정

    Args:
        price_data: 가격 데이터 딕셔너리 {'005930.KS': DataFrame, ...}
        indicator_data: 지표 데이터 딕셔너리

    Returns:
        tuple: (mock_fetcher, mock_calculator)

    사용 예:
        fetcher_mock, calc_mock = setup_pipeline_mocks()
        with patch('...StockDataFetcher', return_value=fetcher_mock):
            ...
    """
    if price_data is None:
        price_data = {
            '005930.KS': create_sample_ohlcv_dataframe(periods=100)
        }

    mock_fetcher = MagicMock()
    mock_fetcher.fetch_multiple_by_period.return_value = price_data
    mock_fetcher.fetch_multiple_by_date.return_value = price_data

    mock_calculator = MagicMock()
    def mock_calculate(df, indicator_list):
        df_copy = df.copy()
        for indicator in (indicator_list or []):
            df_copy[indicator] = np.random.randn(len(df))
        return df_copy
    mock_calculator.calculate_indicators.side_effect = mock_calculate

    return mock_fetcher, mock_calculator


def assert_dataframe_content(df, expected_columns=None, min_rows=0, max_rows=None):
    """
    DataFrame 콘텐츠 검증

    Args:
        df: 검증할 DataFrame
        expected_columns: 필수 컬럼 리스트
        min_rows: 최소 행 수
        max_rows: 최대 행 수

    Raises:
        AssertionError: 검증 실패 시

    사용 예:
        result = labeler.label_data(df)
        assert_dataframe_content(result, expected_columns=['label'], min_rows=1)
    """
    assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"

    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"

    if max_rows is not None:
        assert len(df) <= max_rows, f"Expected at most {max_rows} rows, got {len(df)}"
