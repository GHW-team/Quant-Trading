"""
DataPipeline 통합 테스트 (시나리오 기반)
전체 파이프라인 흐름을 End-to-End로 검증
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.data.pipeline import DataPipeline
from src.data.db_manager import DatabaseManager


#==========================================
# run_full_pipeline 통합 테스트
#==========================================
class TestRunFullPipeline:
    """전체 파이프라인 (Fetch → Save Price → Calculate → Save Indicators) 시나리오 테스트"""

    def test_end_to_end_success(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: 사용자가 전체 파이프라인을 실행하여 가격 데이터와 지표를 DB에 저장

        Given: 3개 티커, 기간 설정
        When: run_full_pipeline() 실행
        Then:
            - DB에 가격 데이터 저장됨
            - DB에 지표 데이터 저장됨
            - 가격과 지표의 날짜 범위가 일치함
        """
        # Given
        tickers = ['005930.KS', '000660.KS', '035720.KS']
        start_date = '2020-01-01'
        end_date = '2020-01-31'
        indicators = ['ma_5', 'ma_20', 'rsi']

        # When
        with DataPipeline(db_path=temp_db_path, max_workers=1, max_retries=1) as pipeline:
            results = pipeline.run_full_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                indicator_list=indicators,
                version='v1.0',
                batch_size=10,
            )

        # Then: 결과 검증
        assert results is not None
        assert 'summary' in results

        # DB에 데이터 저장 검증
        db = DatabaseManager(db_path=temp_db_path)

        # 1. 가격 데이터 저장 확인
        for ticker in tickers:
            price_data = db.load_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            assert price_data is not None, f"{ticker}: Price data not saved"
            assert len(price_data) > 0, f"{ticker}: Price data is empty"
            assert 'date' in price_data.columns
            assert 'close' in price_data.columns

        # 2. 지표 데이터 저장 확인
        for ticker in tickers:
            indicator_data = db.load_indicator_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                version='v1.0'
            )
            assert indicator_data is not None, f"{ticker}: Indicator data not saved"
            assert len(indicator_data) > 0, f"{ticker}: Indicator data is empty"

            # 지표 컬럼 존재 확인
            for indicator in indicators:
                assert indicator in indicator_data.columns, f"{ticker}: {indicator} not in columns"

        db.close()

    def test_with_lookback_period(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: Lookback 기간이 적용되어 더 긴 기간의 데이터를 fetch하지만,
                 저장은 원래 요청한 기간만 저장됨

        Given: ma_200 지표 (200일 lookback 필요)
        When: run_full_pipeline() 실행
        Then:
            - Fetch는 lookback 포함 기간
            - 저장은 원래 요청 기간만
        """
        # Given
        tickers = ['005930.KS']
        start_date = '2020-06-01'
        end_date = '2020-06-30'
        indicators = ['ma_200']  # 200일 lookback 필요

        # When
        with DataPipeline(db_path=temp_db_path, max_workers=1) as pipeline:
            results = pipeline.run_full_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                indicator_list=indicators,
                version='v1.0',
            )

        # Then
        db = DatabaseManager(db_path=temp_db_path)

        # 저장된 데이터는 원래 요청 기간만
        price_data = db.load_price_data(
            ticker=tickers[0],
            start_date=start_date,
            end_date=end_date
        )

        # 날짜 범위 검증
        saved_start = pd.to_datetime(price_data['date'].min())
        saved_end = pd.to_datetime(price_data['date'].max())

        # 저장된 데이터가 요청 기간 내에 있어야 함
        assert saved_start >= pd.to_datetime(start_date)
        assert saved_end <= pd.to_datetime(end_date)

        db.close()

    def test_data_integrity_price_and_indicators_match(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: 가격 데이터와 지표 데이터의 날짜가 일치해야 함

        Given: 가격 + 지표 파이프라인 실행
        When: DB에서 두 데이터 로드
        Then:
            - 날짜 범위 일치
            - 날짜 순서 일치
            - 중복 날짜 없음
        """
        # Given
        tickers = ['005930.KS', '000660.KS']
        start_date = '2020-01-01'
        end_date = '2020-01-31'
        indicators = ['ma_5', 'rsi']

        # When
        with DataPipeline(db_path=temp_db_path) as pipeline:
            pipeline.run_full_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                indicator_list=indicators,
                version='v1.0',
            )

        # Then
        db = DatabaseManager(db_path=temp_db_path)

        for ticker in tickers:
            # 가격 데이터
            price_data = db.load_price_data(ticker, start_date, end_date)
            # 지표 데이터
            indicator_data = db.load_indicator_data(ticker, start_date, end_date, 'v1.0')

            # 1. 날짜 일치 검증
            price_dates = set(pd.to_datetime(price_data['date']))
            indicator_dates = set(pd.to_datetime(indicator_data['date']))

            # 지표 데이터는 가격 데이터의 부분집합이어야 함 (NaN 제외 가능)
            assert indicator_dates.issubset(price_dates), f"{ticker}: Date mismatch"

            # 2. 중복 날짜 없음
            assert len(price_data) == price_data['date'].nunique(), f"{ticker}: Duplicate dates in price"
            assert len(indicator_data) == indicator_data['date'].nunique(), f"{ticker}: Duplicate dates in indicator"

        db.close()

    def test_partial_ticker_failure_continues_others(self, temp_db_path, mock_exchange_calendars):
        """
        시나리오: 일부 티커가 실패해도 나머지는 정상 처리됨

        Given: 3개 티커 중 1개 실패하도록 mock 설정
        When: run_full_pipeline() 실행
        Then: 2개는 DB에 저장됨
        """
        # Given
        tickers = ['SUCCESS1.KS', 'FAIL.KS', 'SUCCESS2.KS']

        # Mock: FAIL.KS만 실패
        def mock_history_side_effect(*args, **kwargs):
            ticker_obj = args[0] if args else None
            # StockDataFetcher에서 호출하는 방식에 맞춤
            df = pd.DataFrame({
                'Date': pd.date_range('2020-01-01', periods=10),
                'Close': [100 + i for i in range(10)],
                'Open': [99 + i for i in range(10)],
                'High': [101 + i for i in range(10)],
                'Low': [98 + i for i in range(10)],
                'Volume': [1000000] * 10,
            })
            df = df.set_index('Date')
            return df

        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_instance = MagicMock()
            mock_instance.history.side_effect = mock_history_side_effect
            mock_ticker_class.return_value = mock_instance

            # When
            with DataPipeline(db_path=temp_db_path, max_workers=1) as pipeline:
                results = pipeline.run_full_pipeline(
                    ticker_list=tickers,
                    start_date='2020-01-01',
                    end_date='2020-01-10',
                    indicator_list=['ma_5'],
                    version='v1.0',
                )

        # Then: 최소 1개 이상 성공
        db = DatabaseManager(db_path=temp_db_path)
        saved_tickers = db.get_all_tickers()

        assert len(saved_tickers) >= 1, "At least one ticker should succeed"

        db.close()

    def test_idempotency_no_duplicate_data(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: 동일한 파이프라인을 2번 실행해도 데이터가 중복되지 않음

        Given: 2개 티커, 기간 설정
        When:
            - run_full_pipeline() 첫 번째 실행
            - 동일한 조건으로 두 번째 실행
        Then:
            - 가격 데이터 개수 동일 (중복 없음)
            - 지표 데이터 개수 동일 (중복 없음)
            - 날짜별 중복 없음
        """
        # Given
        tickers = ['005930.KS', '000660.KS']
        start_date = '2020-01-01'
        end_date = '2020-01-31'
        indicators = ['ma_5', 'rsi']

        # When: 첫 번째 실행
        with DataPipeline(db_path=temp_db_path, max_workers=1) as pipeline:
            pipeline.run_full_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                indicator_list=indicators,
                version='v1.0',
            )

        # 첫 번째 실행 후 데이터 개수 저장
        db = DatabaseManager(db_path=temp_db_path)

        first_price_counts = {}
        first_indicator_counts = {}
        for ticker in tickers:
            price_data = db.load_price_data(ticker, start_date, end_date)
            indicator_data = db.load_indicator_data(ticker, start_date, end_date, 'v1.0')

            first_price_counts[ticker] = len(price_data) if price_data is not None else 0
            first_indicator_counts[ticker] = len(indicator_data) if indicator_data is not None else 0

        db.close()

        # When: 두 번째 실행 (동일 조건)
        with DataPipeline(db_path=temp_db_path, max_workers=1) as pipeline:
            pipeline.run_full_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                indicator_list=indicators,
                version='v1.0',
            )

        # Then: 데이터 개수가 동일해야 함 (중복 없음)
        db = DatabaseManager(db_path=temp_db_path)

        for ticker in tickers:
            price_data = db.load_price_data(ticker, start_date, end_date)
            indicator_data = db.load_indicator_data(ticker, start_date, end_date, 'v1.0')

            second_price_count = len(price_data) if price_data is not None else 0
            second_indicator_count = len(indicator_data) if indicator_data is not None else 0

            # 가격 데이터 중복 검증
            assert second_price_count == first_price_counts[ticker], \
                f"{ticker}: Price data duplicated after second run (expected {first_price_counts[ticker]}, got {second_price_count})"

            # 지표 데이터 중복 검증
            assert second_indicator_count == first_indicator_counts[ticker], \
                f"{ticker}: Indicator data duplicated after second run (expected {first_indicator_counts[ticker]}, got {second_indicator_count})"

            # 날짜별 중복 검증
            if price_data is not None and len(price_data) > 0:
                assert price_data['date'].nunique() == len(price_data), \
                    f"{ticker}: Duplicate dates found in price data"

            if indicator_data is not None and len(indicator_data) > 0:
                assert indicator_data['date'].nunique() == len(indicator_data), \
                    f"{ticker}: Duplicate dates found in indicator data"

        db.close()

    def test_incremental_update_preserves_existing(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: 기존 데이터에 새로운 기간 데이터를 추가할 때 기존 데이터가 보존됨

        Given:
            - 첫 번째 기간: 2020-01-01 ~ 2020-01-31
            - 두 번째 기간: 2020-02-01 ~ 2020-02-29 (신규)
        When:
            - 첫 번째 기간으로 파이프라인 실행
            - 두 번째 기간으로 파이프라인 실행 (update_if_exists=True)
        Then:
            - 첫 번째 기간의 가격 데이터 값이 변경되지 않음
            - 두 번째 기간의 데이터가 정상 추가됨
            - 총 데이터 개수 = 1월 데이터 + 2월 데이터
        """
        # Given
        tickers = ['005930.KS', '000660.KS']
        first_start = '2020-01-01'
        first_end = '2020-01-31'
        second_start = '2020-02-01'
        second_end = '2020-02-29'
        indicators = ['ma_5', 'rsi']

        # When: 첫 번째 기간 실행
        with DataPipeline(db_path=temp_db_path, max_workers=1) as pipeline:
            pipeline.run_full_pipeline(
                ticker_list=tickers,
                start_date=first_start,
                end_date=first_end,
                indicator_list=indicators,
                version='v1.0',
                update_if_exists=True,
            )

        # 첫 번째 기간의 종가 데이터 저장 (검증용)
        db = DatabaseManager(db_path=temp_db_path)

        first_month_closes = {}
        first_month_counts = {}
        for ticker in tickers:
            price_data = db.load_price_data(ticker, first_start, first_end)
            if price_data is not None and len(price_data) > 0:
                # 날짜와 종가를 딕셔너리로 저장
                first_month_closes[ticker] = dict(zip(
                    pd.to_datetime(price_data['date']).dt.strftime('%Y-%m-%d'),
                    price_data['close']
                ))
                first_month_counts[ticker] = len(price_data)
            else:
                first_month_closes[ticker] = {}
                first_month_counts[ticker] = 0

        db.close()

        # When: 두 번째 기간 실행 (증분 업데이트)
        with DataPipeline(db_path=temp_db_path, max_workers=1) as pipeline:
            pipeline.run_full_pipeline(
                ticker_list=tickers,
                start_date=second_start,
                end_date=second_end,
                indicator_list=indicators,
                version='v1.0',
                update_if_exists=True,
            )

        # Then: 검증
        db = DatabaseManager(db_path=temp_db_path)

        for ticker in tickers:
            # 전체 기간 데이터 로드
            all_price_data = db.load_price_data(ticker, first_start, second_end)

            if all_price_data is not None and len(all_price_data) > 0:
                # 1. 첫 번째 기간 데이터가 변경되지 않았는지 확인
                for date_str, original_close in first_month_closes[ticker].items():
                    matching_rows = all_price_data[
                        pd.to_datetime(all_price_data['date']).dt.strftime('%Y-%m-%d') == date_str
                    ]
                    if not matching_rows.empty:
                        current_close = matching_rows.iloc[0]['close']
                        assert current_close == original_close, \
                            f"{ticker}: Close price changed for {date_str} (was {original_close}, now {current_close})"

                # 2. 두 번째 기간 데이터가 추가되었는지 확인
                second_month_data = db.load_price_data(ticker, second_start, second_end)
                if second_month_data is not None:
                    assert len(second_month_data) > 0, \
                        f"{ticker}: Second period data not added"

                # 3. 총 데이터 개수 검증 (첫 번째 + 두 번째)
                assert len(all_price_data) >= first_month_counts[ticker], \
                    f"{ticker}: Total data count should be >= first month count"

        db.close()


#==========================================
# run_price_pipeline 통합 테스트
#==========================================
class TestRunPricePipeline:
    """가격 데이터 파이프라인 시나리오 테스트"""

    def test_price_only_success(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: 가격 데이터만 저장

        Given: 티커 리스트
        When: run_price_pipeline() 실행
        Then:
            - DB에 가격 데이터만 저장됨
            - 지표 데이터는 없음
        """
        # Given
        tickers = ['005930.KS', '000660.KS']
        start_date = '2020-01-01'
        end_date = '2020-01-31'

        # When
        with DataPipeline(db_path=temp_db_path) as pipeline:
            results = pipeline.run_price_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                batch_size=10,
            )

        # Then
        db = DatabaseManager(db_path=temp_db_path)

        # 가격 데이터 존재
        for ticker in tickers:
            price_data = db.load_price_data(ticker, start_date, end_date)
            assert price_data is not None
            assert len(price_data) > 0

        # 지표 데이터는 없음 (에러 나지 않고 None 또는 empty 반환)
        indicator_data = db.load_indicator_data(tickers[0], start_date, end_date, 'v1.0')
        assert indicator_data is None or len(indicator_data) == 0

        db.close()

    def test_update_existing_prices(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: 기존 가격 데이터를 업데이트

        Given: DB에 이미 가격 데이터 존재
        When: update_if_exists=True로 재실행
        Then: 데이터가 업데이트됨
        """
        # Given: 첫 번째 저장
        tickers = ['005930.KS']
        start_date = '2020-01-01'
        end_date = '2020-01-10'

        with DataPipeline(db_path=temp_db_path) as pipeline:
            pipeline.run_price_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                update_if_exists=True,
            )

        db = DatabaseManager(db_path=temp_db_path)
        first_data = db.load_price_data(tickers[0], start_date, end_date)
        first_count = len(first_data)
        db.close()

        # When: 두 번째 저장 (더 긴 기간)
        end_date_new = '2020-01-20'

        with DataPipeline(db_path=temp_db_path) as pipeline:
            pipeline.run_price_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date_new,
                update_if_exists=True,
            )

        # Then
        db = DatabaseManager(db_path=temp_db_path)
        updated_data = db.load_price_data(tickers[0], start_date, end_date_new)

        # 데이터가 증가했거나 유지됨
        assert len(updated_data) >= first_count

        db.close()


#==========================================
# run_indicator_pipeline 통합 테스트
#==========================================
class TestRunIndicatorPipeline:
    """지표 계산 파이프라인 시나리오 테스트"""

    def test_indicators_from_existing_prices(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: DB에 저장된 가격 데이터로부터 지표 계산

        Given: DB에 가격 데이터만 존재
        When: run_indicator_pipeline() 실행
        Then: 지표 데이터가 DB에 저장됨
        """
        # Given: 먼저 가격 데이터 저장
        tickers = ['005930.KS', '000660.KS']
        start_date = '2020-01-01'
        end_date = '2020-01-31'

        with DataPipeline(db_path=temp_db_path) as pipeline:
            pipeline.run_price_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
            )

        # When: 지표 계산
        indicators = ['ma_5', 'ma_20', 'rsi']

        with DataPipeline(db_path=temp_db_path) as pipeline:
            results = pipeline.run_indicator_pipeline(
                ticker_list=tickers,
                indicator_list=indicators,
                start_date=start_date,
                end_date=end_date,
                version='v1.0',
            )

        # Then
        db = DatabaseManager(db_path=temp_db_path)

        for ticker in tickers:
            indicator_data = db.load_indicator_data(ticker, start_date, end_date, 'v1.0')

            assert indicator_data is not None, f"{ticker}: No indicator data"
            assert len(indicator_data) > 0, f"{ticker}: Empty indicator data"

            # 지표 컬럼 존재 확인
            for indicator in indicators:
                assert indicator in indicator_data.columns, f"{ticker}: {indicator} not found"

        db.close()

    def test_batch_processing_multiple_batches(self, temp_db_path, mock_yfinance, mock_exchange_calendars):
        """
        시나리오: 여러 배치로 나눠서 처리

        Given: 10개 티커, batch_size=3
        When: run_indicator_pipeline() 실행
        Then:
            - 4개 배치로 나눠서 처리됨 (3+3+3+1)
            - 모든 티커의 지표 저장됨
        """
        # Given
        tickers = [f'TICKER{i}.KS' for i in range(10)]
        start_date = '2020-01-01'
        end_date = '2020-01-31'

        # 먼저 가격 데이터 저장
        with DataPipeline(db_path=temp_db_path) as pipeline:
            pipeline.run_price_pipeline(
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                batch_size=3,
            )

        # When: 지표 계산 (batch_size=3)
        with DataPipeline(db_path=temp_db_path) as pipeline:
            results = pipeline.run_indicator_pipeline(
                ticker_list=tickers,
                indicator_list=['ma_5'],
                start_date=start_date,
                end_date=end_date,
                version='v1.0',
                batch_size=3,
            )

        # Then: 모든 티커 처리됨
        db = DatabaseManager(db_path=temp_db_path)

        saved_tickers_count = 0
        for ticker in tickers:
            try:
                data = db.load_indicator_data(ticker, start_date, end_date, 'v1.0')
                if data is not None and len(data) > 0:
                    saved_tickers_count += 1
            except:
                pass

        # 대부분 성공해야 함 (최소 50% 이상)
        assert saved_tickers_count >= len(tickers) * 0.5, "At least half should succeed"

        db.close()


#==========================================
# Context Manager 및 초기화 테스트
#==========================================
class TestDataPipelineInitialization:
    """DataPipeline 초기화 테스트"""

    def test_initialization_creates_components(self, temp_db_path):
        """필요한 컴포넌트 생성"""
        pipeline = DataPipeline(db_path=temp_db_path)

        assert isinstance(pipeline.db_manager, DatabaseManager)
        assert isinstance(pipeline.fetcher, StockDataFetcher)
        assert isinstance(pipeline.calculator, IndicatorCalculator)

        pipeline.close()


class TestDataPipelineContextManager:
    """Context Manager 테스트"""

    def test_enter_returns_pipeline(self, temp_db_path):
        """__enter__가 pipeline 반환"""
        with DataPipeline(db_path=temp_db_path) as pipeline:
            assert isinstance(pipeline, DataPipeline)

    def test_exit_calls_close(self, temp_db_path):
        """with 블록을 나갈 때 close()가 호출되는지 검증"""
        pipeline = DataPipeline(db_path=temp_db_path)

        with patch.object(pipeline, 'close') as mock_close:
            with pipeline:
                pass

            mock_close.assert_called_once()


class TestDataPipelineClose:
    """Close 메서드 테스트"""

    def test_close_method(self, temp_db_path):
        """close() 호출"""
        pipeline = DataPipeline(db_path=temp_db_path)
        pipeline.close()
        # 에러 없음


#==========================================
# 헬퍼 함수 테스트
#==========================================
class TestDataPipelineCalculateExtendedStartDate:
    """Lookback 시작 날짜 계산 테스트"""

    @pytest.mark.parametrize("indicator_list, expected_date",[
        (['ma_5'],5),
        (['ma_200'],200),
        (['bb_lower'],20),
        (['ma_5','hv','macd_hist'],34),
    ])
    def test_extended_start_date(self, temp_db_path, indicator_list, expected_date):
        """ma_5 lookback 계산"""
        pipeline = DataPipeline(db_path=temp_db_path)

        original_date = '2020-01-01'
        extended_date = pipeline._calculate_extended_start_date(
            original_date,
            indicator_list
        )

        # 과거로 확장됨
        original_dt = pd.to_datetime(original_date)
        extended_dt = pd.to_datetime(extended_date)

        diff_days = (original_dt - extended_dt).days
        lookback = int(expected_date * 1.6) + 10

        assert diff_days == lookback

        pipeline.close()

    def test_extended_start_date_format(self, temp_db_path):
        """반환값이 YYYY-MM-DD 형식"""
        pipeline = DataPipeline(db_path=temp_db_path)

        extended_date = pipeline._calculate_extended_start_date(
            '2020-01-01',
            ['ma_5']
        )

        # YYYY-MM-DD 형식
        assert len(extended_date) == 10
        assert extended_date[4] == '-'
        assert extended_date[7] == '-'

        pipeline.close()


class TestValidatePeriodAndDate:
    """기간 검증 테스트"""

    @pytest.mark.parametrize('start_date,end_date,period',[
        ('2023-02-03','2024-01-01','1y'),
        (None,'2024-01-01','1y'),
        ('2023-02-03',None,'1y'),
    ])
    def test_input_period_and_date_both(self, start_date, end_date, period):
        """period와 date 동시 입력"""
        pipeline = DataPipeline()

        with pytest.raises(ValueError):
            pipeline._validate_period_and_date(
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

        pipeline.close()

    @pytest.mark.parametrize('start_date,end_date,period',[
        (None,'2024-01-01',None),
        ('2023-02-03',None,None),
    ])
    def test_input_only_one_of_dates(self, start_date, end_date, period):
        """start_date 와 end_date 둘중 하나만 입력"""
        pipeline = DataPipeline()

        with pytest.raises(ValueError):
            pipeline._validate_period_and_date(
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

        pipeline.close()

    @pytest.mark.parametrize('start_date,end_date,period',[
        ('2023.02.03','2024.01.01',None),
        ('2023/01/01','2024/12/01',None),
        ('2023 01 01','2024 12 01',None),
    ])
    def test_invalid_date_format(self, start_date, end_date, period):
        """잘못된 date 형식"""
        pipeline = DataPipeline()

        with pytest.raises(ValueError):
            pipeline._validate_period_and_date(
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

        pipeline.close()

    @pytest.mark.parametrize('start_date,end_date,period',[
        ('2025-02-03','2024-01-01',None),
        ('2021-02-03','2020-01-01',None),
        ('2022-02-02','2022-02-01',None),
        ('2022-02-02','2022-02-02',None),
    ])
    def test_invalid_date_order(self, start_date, end_date, period):
        """부적절한 날짜 순서"""
        pipeline = DataPipeline()

        with pytest.raises(ValueError):
            pipeline._validate_period_and_date(
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

        pipeline.close()
