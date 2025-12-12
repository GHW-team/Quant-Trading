"""
StockDataFetcher 클래스 단위 테스트
yfinance 통합, 병렬 처리, 재시도 로직 검증
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from requests.exceptions import RequestException, Timeout, ConnectionError
import time

from src.data.data_fetcher import StockDataFetcher

#__init__
class TestStockDataFetcherInitialization:
    """StockDataFetcher 초기화 테스트"""
    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_invalid_max_workers_below_one(self):
        """max_workers < 1 시 ValueError 발생"""
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            StockDataFetcher(max_workers=0)

    def test_invalid_max_retries_negative(self):
        """max_retries < 0 시 ValueError 발생"""
        with pytest.raises(ValueError, match="max_retries cannot be negative"):
            StockDataFetcher(max_retries=-1)

#==========================================
# 헬퍼 함수 (단일 fethcer)
#==========================================

#_fetch_single_by_period
class TestStockDataFetcherSingleByPeriod:
    """Period 기반 단일 데이터 수집 테스트"""

    #==========================================
    # 함수 정상 동작 테스트
    #==========================================

    def test_fetch_single_by_period_success(self, mock_yfinance):
        """정상 수집"""
        fetcher = StockDataFetcher()
        result = fetcher._fetch_single_by_period('005930.KS', period='1y')

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'date' in result.columns

    def test_column_normalization_to_snake_case(self, mock_yfinance):
        """컬럼명 snake_case 정규화"""
        fetcher = StockDataFetcher()
        result = fetcher._fetch_single_by_period('005930.KS', period='1y')

        for col in result.columns:
            # 모든 컬럼명이 소문자
            assert col.islower()
            # 공백 유무 검증
            assert ' ' not in col
            # 특수문자 없는지 검증.
            assert col.replace('_','').isalpha()

        # 예상 컬럼명들
        expected_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        assert set(expected_cols) == set(result.columns)

    @pytest.mark.parametrize("period,interval", [
        ('1y', '1d'),
        ('6mo', '1d'),
        ('1mo', '1d'),
        ('1w', '1d'),
        ('1d', '1d'),
    ])
    def test_fetch_with_various_params(self, period, interval, mock_yfinance):
        """api 파라미터 입력 검증 (period / interval)"""
        fetcher = StockDataFetcher()
        result = fetcher._fetch_single_by_period('005930.KS', period=period, interval=interval)

        #파라미터 검증
        mock_instance = mock_yfinance.return_value
        mock_instance.history.assert_called_with(
            period = period,
            interval = interval,
            auto_adjust = False,
            actions = False,
        )

        #상태 검증
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_all_retries_exhausted(self):
        """모든 재시도 소진"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = RequestException("Network error")

            with patch('time.sleep') as mock_sleep:
                fetcher = StockDataFetcher(max_retries=5)
                result = fetcher._fetch_single_by_period('INVALID.XX')

            assert result is None
            assert mock_instance.history.call_count == 5

    def test_exponential_backoff_timing(self):
        """지수 백오프 시간 검증"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = RequestException("Network error")

            with patch('time.sleep') as mock_sleep:
                fetcher = StockDataFetcher(max_retries=4)
                fetcher._fetch_single_by_period('005930.KS')

                # 2^1 = 2, 2^2 = 4, 2^3 = 8 , .....
                mock_sleep.assert_any_call(2)
                mock_sleep.assert_any_call(4)
                mock_sleep.assert_any_call(8)
    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_empty_data_returns_none(self):
        """빈 데이터 반환 시 None"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.return_value = pd.DataFrame()

            fetcher = StockDataFetcher()
            result = fetcher._fetch_single_by_period('INVALID.XX')

            assert result is None
    
    #==========================================
    # 함수 에러 처리 테스트
    #==========================================
    def test_network_error_with_retry(self):
        """네트워크 에러 재시도"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance

            # 첫 시도는 실패, 두 번째는 성공
            mock_instance.history.side_effect = [
                RequestException("Network error"),
                pd.DataFrame({
                    'Open': [100],
                    'High': [101],
                    'Low': [99],
                    'Close': [100.5],
                    'Adj Close': [100.5],
                    'Volume': [1000000],
                }, index=pd.DatetimeIndex(['2020-01-01']))
            ]

            fetcher = StockDataFetcher(max_retries=2)
            result = fetcher._fetch_single_by_period('005930.KS')

            assert result is not None
            assert mock_instance.history.call_count == 2

    @pytest.mark.parametrize("exception_type,exception_msg", [
        (RequestException("Generic request error"), "generic error"),
        (Timeout("Connection timeout"), "timeout"),
        (ConnectionError("Connection failed"), "connection failed"),
    ])
    def test_request_exception_types(self, exception_type, exception_msg):
        """RequestException 하위 타입 처리"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = exception_type

            fetcher = StockDataFetcher(max_retries=1)
            result = fetcher._fetch_single_by_period('005930.KS')

            assert result is None
            assert mock_instance.history.call_count == 1  # 재시도 없이 실패

    def test_invalid_date_format(self):
        """유효하지 않은 날짜 형식"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = ValueError("Invalid date format")

            fetcher = StockDataFetcher()
            result = fetcher._fetch_single_by_date(
                '005930.KS',
                start_date='invalid-date',
                end_date='2020-12-31'
            )

            assert result is None

    def test_unexpected_exception_handling(self):
        """예상치 못한 예외 처리"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = ValueError("Unexpected error")

            fetcher = StockDataFetcher()
            result = fetcher._fetch_single_by_period('005930.KS')

            assert result is None

#_fetch_single_by_date
class TestStockDataFetcherSingleByDate:
    """Date 기반 단일 데이터 수집 테스트"""
    #==========================================
    # 함수 정상 동작 테스트
    #==========================================
    def test_fetch_single_by_date_success(self, mock_yfinance):
        """정상 수집"""
        fetcher = StockDataFetcher()
        result = fetcher._fetch_single_by_date(
            '005930.KS',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )

        assert result is not None
        assert len(result) == 3
        assert 'date' in result.columns
        assert isinstance(result, pd.DataFrame)
    
    def test_column_normalization_to_snake_case_by_date(self, mock_yfinance):
        """컬럼명 snake_case 정규화"""
        fetcher = StockDataFetcher()
        result = fetcher._fetch_single_by_date(
            '005930.KS',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )

        for col in result.columns:
            # 모든 컬럼명이 소문자
            assert col.islower()
            # 공백 유무 검증
            assert ' ' not in col
            # 특수문자 없는지 검증.
            assert col.replace('_','').isalpha()

        # 예상 컬럼명들
        expected_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in expected_cols:
            assert col in result.columns

    @pytest.mark.parametrize("start_date, end_date",[
        ('2020-01-01', '2020-01-31'),
        ('2022-05-01', '2023-06-20'),
        ('2024-02-10', '2024-12-10'),
    ])
    def test_date_range_applied(self, start_date, end_date, mock_yfinance):
        """날짜 범위 적용"""
        fetcher = StockDataFetcher()
        result = fetcher._fetch_single_by_date(
            '005930.KS',
            start_date=start_date,
            end_date=end_date
        )

        #파라미터 검증
        mock_instance = mock_yfinance.return_value
        mock_instance.history.assert_called_with(
            start = start_date,
            end = end_date,
            interval = "1d",
            auto_adjust = False,
            actions = False,
        )

        #결과 검증
        assert len(result) > 0
        assert isinstance(result,pd.DataFrame)
        assert result is not None

    def test_all_retries_exhausted_by_date(self):
        """모든 재시도 소진"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = RequestException("Network error")

            with patch('time.sleep') as mock_sleep:
                fetcher = StockDataFetcher(max_retries=5)
                result = fetcher._fetch_single_by_date(
                    'INVALID.XX',
                    start_date='2020-01-01',
                    end_date='2020-12-31'
                )

            assert result is None
            assert mock_instance.history.call_count == 5

    def test_exponential_backoff_timing_by_date(self):
        """지수 백오프 시간 검증"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = RequestException("Network error")

            with patch('time.sleep') as mock_sleep:
                fetcher = StockDataFetcher(max_retries=4)
                fetcher._fetch_single_by_date(
                    '005930.KS',
                    start_date='2020-01-01',
                    end_date='2020-12-31'
                )

                # 2^1 = 2, 2^2 = 4, 2^3 = 8
                mock_sleep.assert_any_call(2)
                mock_sleep.assert_any_call(4)
                mock_sleep.assert_any_call(8)
    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_empty_data_returns_none(self):
        """빈 데이터 반환 시 None"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.return_value = pd.DataFrame()

            fetcher = StockDataFetcher()
            result = fetcher._fetch_single_by_date(
                'INVALID.XX',
                start_date='2020-01-01',
                end_date='2020-12-31'
            )

            assert result is None

    #==========================================
    # 함수 에러 처리 테스트
    #==========================================
    def test_network_error_with_retry_by_date(self):
        """네트워크 에러 재시도"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance

            mock_instance.history.side_effect = [
                RequestException("Network error"),
                pd.DataFrame({
                    'Open': [100],
                    'High': [101],
                    'Low': [99],
                    'Close': [100.5],
                    'Adj Close': [100.5],
                    'Volume': [1000000],
                }, index=pd.DatetimeIndex(['2020-01-01']))
            ]

            fetcher = StockDataFetcher(max_retries=2)
            result = fetcher._fetch_single_by_date(
                '005930.KS',
                start_date='2020-01-01',
                end_date='2020-01-31'
            )

            assert result is not None
            assert mock_instance.history.call_count == 2

    @pytest.mark.parametrize("exception_type,exception_msg", [
        (RequestException("Generic request error"), "generic error"),
        (Timeout("Connection timeout"), "timeout"),
        (ConnectionError("Connection failed"), "connection failed"),
    ])
    def test_request_exception_types(self, exception_type, exception_msg):
        """RequestException 하위 타입 처리"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = exception_type

            fetcher = StockDataFetcher(max_retries=1)
            result = fetcher._fetch_single_by_date(
                '005930.KS',
                start_date='2020-01-01',
                end_date='2020-12-31'
            )

            assert result is None
            assert mock_instance.history.call_count == 1  # 재시도 없이 실패
            
    def test_unexpected_exception_handling_by_date(self):
        """예상치 못한 예외 처리"""
        with patch('yfinance.Ticker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.history.side_effect = ValueError("Unexpected error")

            fetcher = StockDataFetcher()
            result = fetcher._fetch_single_by_date(
                '005930.KS',
                start_date='2020-01-01',
                end_date='2020-12-31'
            )

            assert result is None


#==========================================
# Fetcher 함수
#==========================================

#fetch_multiple_by_period
class TestStockDataFetcherMultipleByPeriod:
    """Period 기반 병렬 수집 테스트"""
    #==========================================
    # 함수 정상 동작 테스트
    #==========================================
    def test_fetch_multiple_by_period_success(self, mock_yfinance):
        """정상 병렬 수집"""
        fetcher = StockDataFetcher(max_workers=2)
        tickers = ['005930.KS', '000660.KS']
        result = fetcher.fetch_multiple_by_period(tickers, period='1y')

        assert isinstance(result, dict)
        # mock은 모두 성공하는 데이터 반환
        assert len(result) == 2
        assert '005930.KS' in result
        assert '000660.KS' in result

    def test_parallel_execution(self):
        """병렬 실행 타이밍 검증"""
        start_time = time.time()

        def mock_fetch(ticker, period='1y', interval='1d', actions=False):
            time.sleep(0.1)  # 각 호출마다 100ms 지연
            return pd.DataFrame({'date': ['2020-01-01'], 'close': [100]})

        with patch.object(StockDataFetcher, '_fetch_single_by_period', side_effect=mock_fetch):
            fetcher = StockDataFetcher(max_workers=2)
            result = fetcher.fetch_multiple_by_period(
                ['005930.KS', '000660.KS'],
                period='1y'
            )

            elapsed = time.time() - start_time

            # 병렬 실행: 2개 × 100ms = 100ms + overhead (약 0.12~0.18초)
            # 순차 실행: 2개 × 100ms = 200ms + overhead (약 0.20초 이상)
            # 병렬 실행을 확인하기 위해 0.2초 이내로 완료되어야 함
            assert elapsed < 0.2, f"병렬 실행 예상 시간 0.2초 미만, 실제: {elapsed:.3f}초"
            assert len(result) == 2

    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_empty_ticker_list(self):
        """빈 ticker 리스트"""
        fetcher = StockDataFetcher()
        result = fetcher.fetch_multiple_by_period([], period='1y')

        assert result == {}
        assert isinstance(result, dict)

    def test_partial_success(self):
        """부분 성공"""
        with patch.object(StockDataFetcher, '_fetch_single_by_period') as mock:
            # 첫 번째는 성공, 두 번째는 실패
            mock.side_effect = [
                None,  # 실패
                pd.DataFrame({'date': ['2020-01-01'], 'close': [100]}),
                None,
            ]

            fetcher = StockDataFetcher(max_workers=2)
            result = fetcher.fetch_multiple_by_period(
                ['005930.KS', '000660.KS','002410.KS'],
                period='1y'
            )

            assert '000660.KS' in result
            assert len(result) == 1
    #==========================================
    # 함수 에러 처리 테스트
    #==========================================
    def test_thread_exception_handling(self):
        """병렬 스레드 부분 예러 처리"""
        with patch.object(StockDataFetcher, '_fetch_single_by_period') as mock:
            mock.side_effect = [
                RuntimeError("Thread exception"),
                pd.DataFrame({'date': ['2020-01-01'], 'close': [100]}),
                RuntimeError("Thread exception"),
            ]

            fetcher = StockDataFetcher(max_workers=2)
            result = fetcher.fetch_multiple_by_period(
                ['005930.KS', '000660.KS','001420.KS'],
                period='1y'
            )

            # 에러 발생하지 않은 쓰레드 정상 작동
            assert len(result) == 1

#fetch_multiple_by_date
class TestStockDataFetcherMultipleByDate:
    """Date 기반 병렬 수집 테스트"""

    #==========================================
    # 함수 정상 동작 테스트
    #==========================================
    def test_fetch_multiple_by_date_success(self, mock_yfinance):
        """정상 병렬 수집"""
        fetcher = StockDataFetcher(max_workers=2)
        tickers = ['005930.KS', '000660.KS']
        result = fetcher.fetch_multiple_by_date(
            tickers,
            start_date='2020-01-01',
            end_date='2020-12-31'
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        assert '005930.KS' in result
        assert '000660.KS' in result

    def test_parallel_execution_by_date(self):
        """병렬 실행 타이밍 검증"""
        start_time = time.time()

        def mock_fetch(ticker, start_date, end_date, interval='1d', actions=False):
            time.sleep(0.1)  # 각 호출마다 100ms 지연
            return pd.DataFrame({'date': ['2020-01-01'], 'close': [100]})

        with patch.object(StockDataFetcher, '_fetch_single_by_date', side_effect=mock_fetch):
            fetcher = StockDataFetcher(max_workers=2)
            result = fetcher.fetch_multiple_by_date(
                ['005930.KS', '000660.KS'],
                start_date='2020-01-01',
                end_date='2020-12-31'
            )

            elapsed = time.time() - start_time

            # 병렬 실행: 2개 × 100ms = 100ms + overhead (약 0.12~0.18초)
            # 순차 실행: 2개 × 100ms = 200ms + overhead (약 0.20초 이상)
            # 병렬 실행을 확인하기 위해 0.2초 이내로 완료되어야 함
            assert elapsed < 0.2, f"병렬 실행 예상 시간 0.2초 미만, 실제: {elapsed:.3f}초"
            assert len(result) == 2
    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_empty_ticker_list_by_date(self):
        """빈 ticker 리스트"""
        fetcher = StockDataFetcher()
        result = fetcher.fetch_multiple_by_date(
            [],
            start_date='2020-01-01',
            end_date='2020-12-31'
        )

        assert result == {}
        assert isinstance(result, dict)

    def test_partial_success_by_date(self):
        """부분 성공"""
        with patch.object(StockDataFetcher, '_fetch_single_by_date') as mock:
            mock.side_effect = [
                None,
                pd.DataFrame({'date': ['2020-01-01'], 'close': [100]}),
            ]

            fetcher = StockDataFetcher(max_workers=2)
            result = fetcher.fetch_multiple_by_date(
                ['005930.KS', '000660.KS'],
                start_date='2020-01-01',
                end_date='2020-12-31'
            )

            assert '000660.KS' in result
            assert len(result) == 1
    #==========================================
    # 함수 에러 처리 테스트
    #==========================================
    def test_thread_exception_handling(self):
        """병렬 스레드 부분 예러 처리"""
        with patch.object(StockDataFetcher, '_fetch_single_by_date') as mock:
            mock.side_effect = [
                RuntimeError("Thread exception"),
                pd.DataFrame({'date': ['2020-01-01'], 'close': [100]}),
                RuntimeError("Thread exception"),
            ]

            fetcher = StockDataFetcher(max_workers=2)
            result = fetcher.fetch_multiple_by_date(
                ['005930.KS', '000660.KS','001420.KS'],
                start_date='2020-01-01',
                end_date='2020-12-31'
            )

            # 에러 발생하지 않은 쓰레드 정상 작동
            assert len(result) == 1