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
# 헬퍼 함수 
#==========================================
# _convert_period_to_dates
class TestDataFetcherConvertPeriodToDates:
    """Period 문자열 파싱 테스트"""
    # ==================================
    # 함수 정상 케이스 테스트
    # ==================================
    @pytest.mark.parametrize("period,min,max",[
        ("1y",360,370),
        ("6mo",175,185),
        ("1mo",28,32),
        ("50d",48,52),
        ("1d",0,2),
        ("0y",0,1),
    ])
    def test_convert_period(self, period, min, max):
        """1년 period 변환"""
        start_date, end_date = StockDataFetcher._convert_period_to_dates(period)

        assert isinstance(start_date, str)
        assert isinstance(end_date, str)

        # 날짜 형식 검증
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        diff_days = (end_dt- start_dt).days
        assert min <= diff_days <= max
    
    def test_ytd_period(self):
        """Year-to-date 변환 검증"""
        from datetime import datetime
        
        start_date, end_date = StockDataFetcher._convert_period_to_dates("ytd")
        start_dt = pd.to_datetime(start_date)
        
        assert start_dt.month == 1
        assert start_dt.day == 1
        assert start_dt.year == datetime.now().year

    def test_period_format_yyyy_mm_dd(self):
        """반환값이 YYYY-MM-DD 형식"""
        start_date, end_date = StockDataFetcher._convert_period_to_dates('1y')

        # YYYY-MM-DD 형식 검증
        assert len(start_date) == 10
        assert len(end_date) == 10
        assert start_date[4] == '-'
        assert start_date[7] == '-'
    
    def test_max_period(self):
        """최대 기간 변환 검증"""
        start_date, end_date = StockDataFetcher._convert_period_to_dates("max")
        start_dt = pd.to_datetime(start_date)
        
        assert start_dt.year == 1970
        assert start_dt.month == 1
        assert start_dt.day == 1
    # ==================================
    # 함수 에러 케이스 테스트
    # ==================================
    @pytest.mark.parametrize("period",[
        'invalid',
        '1.5y',      # 소수점 불가
        '-1y',       # 음수 불가
        '1w',        # 지원 안 하는 단위
        '1month',    # 'mo'가 아님
        '',          # 빈 문자열
        'y1',        # 순서 뒤바뀜
        '1Y',        # 대문자 (현재 정규식은 소문자만)
    ])
    def test_invalid_period_raises_error(self,period):
        """잘못된 period 형식"""
        with pytest.raises(ValueError):
            StockDataFetcher._convert_period_to_dates(period)

# _get_exchange_code
class TestGetExchangeCode:
    #==========================================
    # 함수 정상 케이스 테스트
    #==========================================
    @pytest.mark.parametrize("ticker, expected_exchange",[
        ('011253.KS', 'XKRX'),
        ('023412.KQ', 'XKRX'),
        ('1923.T', 'XTKS'),
        ('12502.HK', 'XHKG'),
        ('GOGL', 'XNYS'),
    ])
    def test_get_exchange_code(self, ticker, expected_exchange):
        """시장 정보 추론"""
        #Arrange 
        fetcher = StockDataFetcher()

        #act
        exchange = fetcher._get_exchange_code(ticker)

        #assert
        assert exchange == expected_exchange

    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_unkwon_ticker(self):
        """부적절한 티커 입력"""
        #Arrange
        fetcher = StockDataFetcher()

        #act
        exchange = fetcher._get_exchange_code('02943.UW')

        #assert
        assert exchange == None

# _validate_and_fill_gaps
class TestValidateAndFillGaps:
    #==========================================
    # 함수 정상 케이스 테스트
    #==========================================
    def test_validate_and_fill_gaps_success(self, mock_exchange_calendars):
        """정상 동작 테스트"""
        #Arrange
        fetcher = StockDataFetcher()
        df = pd.DataFrame({
            'open': [100, 102],
            'high': [101, 103],
            'low': [99, 101],
            'close': [100.5, 102.5],
            'adj_close': [100.5, 102.5],
            'volume': [1000000, 1200000],
            'date': pd.to_datetime(['2020-01-01', '2020-01-04']),
        })
        df_dict = {'005930.KS': df}

        #Act
        result_dict = fetcher._validate_and_fill_gaps('2020-01-01', '2020-01-04', df_dict)
        result = result_dict['005930.KS']

        #Assert
        assert len(result) == 4
        assert 'date' in result.columns

        #누락된 날짜 채워졌는지 확인
        assert pd.isna(result['adj_close'].iloc[1])
        assert pd.isna(result['volume'].iloc[1])
        assert pd.isna(result['adj_close'].iloc[2])
        assert pd.isna(result['volume'].iloc[2])

        assert result['date'].iloc[1] == pd.Timestamp('2020-01-02')
        assert result['date'].iloc[2] == pd.Timestamp('2020-01-03')

        #기존 데이터 유지되는지 확인
        assert result['open'].iloc[0] == 100.0
        assert result['open'].iloc[3] == 102.0
        assert result['date'].iloc[0] == pd.Timestamp('2020-01-01')
        assert result['date'].iloc[3] == pd.Timestamp('2020-01-04')

    def test_timezone_preservation(self,mock_exchange_calendars):
        """'date'의 timezone 정보 보존 여부"""
        #Arrange
        fetcher = StockDataFetcher()
        df = pd.DataFrame({
            'open': [100, 102],
            'high': [101, 103],
            'low': [99, 101],
            'close': [100.5, 102.5],
            'adj_close': [100.5, 102.5],
            'volume': [1000000, 1200000],
            'date': pd.to_datetime(['2020-01-01', '2020-01-04']).tz_localize('Asia/Seoul'),
        })
        df_dict = {'005930.KS': df}


        #Act
        result_dict = fetcher._validate_and_fill_gaps('2020-01-01', '2020-01-04', df_dict)
        result = result_dict['005930.KS']

        #Assert
        assert len(result) == 4
        assert result['date'].iloc[0] == pd.Timestamp('2020-01-01',tz='Asia/Seoul')
        assert result['date'].iloc[1] == pd.Timestamp('2020-01-02',tz='Asia/Seoul')

    def test_no_missing_dates(self, mock_exchange_calendars):
        """누락된 날짜가 없는 경우"""
        #Arrange
        fetcher = StockDataFetcher()
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-01-02','2020-01-03', '2020-01-04']),
            'open': [100, 102, 104, 106],
            'high': [101, 103, 105, 107],
            'low': [99, 101, 103, 105],
            'close': [100.5, 102.5, 104.5, 106.5],
            'adj_close': [100.5, 102.5, 104.5, 106.5],
            'volume': [1000000, 1200000, 1100000, 1200000],
        })
        df_dict = {'005930.KS': df}
        
        #Act
        # 모든 거래일 데이터가 있으면 그대로 반환되는지 확인
        result_dict = fetcher._validate_and_fill_gaps('2020-01-01', '2020-01-04', df_dict)
        result = result_dict['005930.KS']

        #Assert
        pd.testing.assert_frame_equal(result,df)

    def test_all_columns_preserved(self, mock_exchange_calendars):
        """모든 컬럼이 유지되는지 확인"""
        #Arrange
        fetcher = StockDataFetcher()
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-01-04']).tz_localize('Asia/Seoul'),
            'open': [100, 102],
            'high': [101, 103],
            'low': [99, 101],
            'close': [100.5, 102.5],
            'adj_close': [100.5, 102.5],
            'volume': [1000000, 1200000],
        })
        df_dict = {'005930.KS': df}
        
        #Act
        # 모든 거래일 데이터가 있으면 그대로 반환되는지 확인
        result_dict = fetcher._validate_and_fill_gaps('2020-01-01', '2020-01-04', df_dict)
        result = result_dict['005930.KS']

        #Assert
        # 날짜 채우기 후에도 모든 컬럼이 존재하는지
        assert df.columns.equals(result.columns)

    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_unknown_ticker(self, sample_df_basic):
        """부적절한 ticker"""
        #Arrange
        fetcher = StockDataFetcher()
        df_dict = {'004290.UW': sample_df_basic}

        #Act
        result_dict = fetcher._validate_and_fill_gaps('2020-01-01', '2020-01-04', df_dict)

        #Assert
        assert '004290.UW' not in result_dict

        
        

#==========================================
# Fetcher 함수
#==========================================

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
    
    def test_invalid_date_format(self):
        #Arrange
        fetcher = StockDataFetcher()

        #Act
        with pytest.raises(ValueError):
            result = fetcher._fetch_single_by_date(
                ticker = "005930.KS",
                start_date="2020.01.01",
                end_date="2023.01.01"
            )
            #Assert
    
    def test_invalid_time_order(self):
        #Arrange
        fetcher = StockDataFetcher()

        #Act
        with pytest.raises(ValueError):
            result = fetcher._fetch_single_by_date(
                ticker = "005930.KS",
                start_date= "2023-01-01",
                end_date= "2020-01-01",
            )
        
#fetch_multiple_by_date
class TestStockDataFetcherMultipleByDate:
    """Date 기반 병렬 수집 테스트"""

    #==========================================
    # 함수 정상 동작 테스트
    #==========================================
    def test_fetch_multiple_by_date_success(self, mock_yfinance, mock_exchange_calendars):
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

    def test_parallel_execution_by_date(self, mock_exchange_calendars):
        """병렬 실행 타이밍 검증"""
        start_time = time.time()

        def mock_fetch(ticker, start_date, end_date, interval='1d', actions=False):
            time.sleep(0.1)  # 각 호출마다 100ms 지연
            return pd.DataFrame({'date': pd.to_datetime(['2020-01-01']), 'close': [100]})

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

    def test_partial_success_by_date(self, mock_exchange_calendars):
        """부분 성공"""
        with patch.object(StockDataFetcher, '_fetch_single_by_date') as mock:
            mock.side_effect = [
                None,
                pd.DataFrame({'date': pd.to_datetime(['2020-01-01']), 'close': [100]}),
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
    def test_thread_exception_handling(self, mock_exchange_calendars):
        """병렬 스레드 부분 예러 처리"""
        with patch.object(StockDataFetcher, '_fetch_single_by_date') as mock:
            mock.side_effect = [
                RuntimeError("Thread exception"),
                pd.DataFrame({'date': pd.to_datetime(['2020-01-01']), 'close': [100]}),
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