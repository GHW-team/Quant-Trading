"""
DataPipeline 클래스 메서드별 단위 테스트
정적 메서드, 초기화, context manager 등 검증
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.data.pipeline import DataPipeline
from src.data.db_manager import DatabaseManager
from src.data.data_fetcher import StockDataFetcher
from src.data.indicator_calculator import IndicatorCalculator

# _calculate_extended_start_date
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

        print(indicator_list)
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

# __init__
class TestDataPipelineInitialization:
    """DataPipeline 초기화 테스트"""

    def test_initialization_creates_components(self, temp_db_path):
        """필요한 컴포넌트 생성"""
        pipeline = DataPipeline(db_path=temp_db_path)

        assert isinstance(pipeline.db_manager, DatabaseManager)
        assert isinstance(pipeline.fetcher, StockDataFetcher)
        assert isinstance(pipeline.calculator, IndicatorCalculator)

        pipeline.close()

# __enter__ , __exit__
class TestDataPipelineContextManager:
    """Context Manager 테스트"""

    def test_enter_returns_pipeline(self, temp_db_path):
        """__enter__가 pipeline 반환"""
        with DataPipeline(db_path=temp_db_path) as pipeline:
            assert isinstance(pipeline, DataPipeline)

    def test_exit_calls_close(self, temp_db_path):
        """with 블록을 나갈 때 close()가 호출되는지 검증"""
        pipeline = DataPipeline(db_path=temp_db_path)

        # pipeline의 close 메서드를 감시하는 Mock 객체
        with patch.object(pipeline, 'close') as mock_close:
            
            # 3. Context Manager 진입 및 탈출
            with pipeline:
                pass
            
            # 4. 검증: "나갈 때 close()가 딱 한 번 호출됐니?"
            mock_close.assert_called_once()

# close
class TestDataPipelineClose:
    """Close 메서드 테스트"""

    def test_close_method(self, temp_db_path):
        """close() 호출"""
        pipeline = DataPipeline(db_path=temp_db_path)
        pipeline.close()
        # 에러 없음


class TestDataPipelineValidation:
    """입력 검증 테스트"""

    def test_period_and_date_mutual_exclusive(self, temp_db_path):
        """period와 start_date 동시 제공 오류"""
        pipeline = DataPipeline(db_path=temp_db_path)

        with pytest.raises(ValueError):
            pipeline.run_full_pipeline(
                ticker_list=['005930.KS'],
                period='1y',
                start_date='2020-01-01',  # 동시 제공
                indicator_list=['ma_5']
            )

        pipeline.close()

    def test_empty_ticker_list(self, temp_db_path):
        """빈 ticker 리스트"""
        pipeline = DataPipeline(db_path=temp_db_path)

        # 빈 리스트는 0개 처리
        result = pipeline.run_full_pipeline(
            ticker_list=[],
            period='1y',
            indicator_list=['ma_5']
        )

        assert result is not None

        pipeline.close()


class TestDataPipelineIntegration:
    """통합 테스트"""

    def test_period_to_dates_roundtrip(self, temp_db_path):
        """Period 변환의 일관성"""
        pipeline = DataPipeline(db_path=temp_db_path)

        # 같은 period 변환을 여러 번
        dates1 = DataPipeline._convert_period_to_dates('1y')
        dates2 = DataPipeline._convert_period_to_dates('1y')

        # 결과가 거의 동일 (몇 초 차이 가능)
        dt1_start = pd.to_datetime(dates1[0])
        dt2_start = pd.to_datetime(dates2[0])

        diff = abs((dt1_start - dt2_start).total_seconds())
        assert diff < 5  # 5초 이내

        pipeline.close()

    def test_complete_date_calculation_workflow(self, temp_db_path):
        """완전한 날짜 계산 워크플로우"""
        pipeline = DataPipeline(db_path=temp_db_path)

        # 1. Period 변환
        start_date, end_date = DataPipeline._convert_period_to_dates('1y')

        # 2. Lookback 확장
        extended_start = pipeline._calculate_extended_start_date(
            start_date,
            ['ma_5', 'ma_20', 'ma_200']
        )

        # 3. 검증
        assert pd.to_datetime(extended_start) < pd.to_datetime(start_date)
        assert pd.to_datetime(start_date) < pd.to_datetime(end_date)

        pipeline.close()


class TestDataPipelineGetAvailableIndicators:
    """사용 가능한 지표 테스트"""

    def test_get_available_indicators(self, temp_db_path):
        """사용 가능한 지표 목록"""
        pipeline = DataPipeline(db_path=temp_db_path)

        # calculator의 메서드 사용
        available = pipeline.calculator.get_available_indicators()

        assert isinstance(available, list)
        assert len(available) > 0

        pipeline.close()


class TestDataPipelineErrorHandling:
    """에러 처리 테스트"""

    def test_invalid_interval_parameter(self, temp_db_path):
        """유효하지 않은 interval"""
        pipeline = DataPipeline(db_path=temp_db_path)

        # interval은 무시되거나 오류
        try:
            result = pipeline.run_full_pipeline(
                ticker_list=['005930.KS'],
                period='1y',
                interval='invalid',
                indicator_list=['ma_5']
            )
        except (ValueError, Exception):
            pass  # 예상되는 에러

        pipeline.close()

    def test_invalid_ticker_codes(self, temp_db_path):
        """유효하지 않은 ticker 코드"""
        pipeline = DataPipeline(db_path=temp_db_path)

        # 잘못된 ticker는 처리되거나 스킵
        result = pipeline.run_full_pipeline(
            ticker_list=['INVALID.CODE'],
            period='1y',
            indicator_list=['ma_5']
        )

        # 결과가 있거나 비어있음
        assert result is not None

        pipeline.close()


class TestDataPipelineStaticMethodsEdgeCases:
    """정적 메서드 엣지 케이스"""

    def test_period_boundary_values(self):
        """경계값 period"""
        # 2년
        start, end = DataPipeline._convert_period_to_dates('2y')
        diff_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
        assert diff_days > 700

        # 1주
        start, end = DataPipeline._convert_period_to_dates('1w')
        diff_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
        assert 5 < diff_days < 10

    def test_period_case_insensitive(self):
        """period 단위는 소문자"""
        start1, end1 = DataPipeline._convert_period_to_dates('1y')
        start2, end2 = DataPipeline._convert_period_to_dates('1Y')  # 대문자

        # 대문자도 작동해야 함 (또는 에러)
        assert isinstance(start2, str) and isinstance(end2, str)

    def test_extended_start_with_empty_indicators(self, temp_db_path):
        """빈 지표 리스트"""
        pipeline = DataPipeline(db_path=temp_db_path)

        result = pipeline._calculate_extended_start_date(
            '2020-01-01',
            []
        )

        # 빈 리스트는 최소 lookback만 적용
        assert result is not None

        pipeline.close()


class TestDataPipelineFullWorkflow:
    """run_full_pipeline 전체 워크플로우 테스트 (핵심만)"""

    def test_run_full_pipeline_by_period(self, temp_db_path):
        """Period 기반 전체 파이프라인 기본 실행"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                # Mock 설정
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher

                # _convert_period_to_dates Mock 추가
                mock_fetcher._convert_period_to_dates.return_value = ('2023-01-01', '2023-12-31')

                mock_fetcher.fetch_multiple_by_period.return_value = {
                    '005930.KS': pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=100),
                        'open': np.random.randn(100) + 100,
                        'high': np.random.randn(100) + 101,
                        'low': np.random.randn(100) + 99,
                        'close': np.random.randn(100) + 100,
                        'adj_close': np.random.randn(100) + 100,
                        'volume': np.random.randint(1000000, 2000000, 100),
                    })
                }

                mock_calc = MagicMock()
                mock_calc_class.return_value = mock_calc
                mock_calc.calculate_indicators.side_effect = lambda df, ind_list: (
                    df.assign(**{ind: np.random.randn(len(df)) for ind in (ind_list or [])})
                )

                pipeline = DataPipeline(db_path=temp_db_path)
                result = pipeline.run_full_pipeline(
                    ticker_list=['005930.KS'],
                    period='1y',
                    indicator_list=['ma_5', 'ma_20']
                )

                assert result is not None
                pipeline.close()

    def test_run_full_pipeline_multiple_tickers(self, temp_db_path):
        """여러 ticker 동시 처리 (병렬 처리 검증)"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher

                # _convert_period_to_dates Mock 추가
                mock_fetcher._convert_period_to_dates.return_value = ('2023-01-01', '2023-12-31')

                mock_fetcher.fetch_multiple_by_period.return_value = {
                    '005930.KS': pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=100),
                        'open': np.arange(100, 200),
                        'high': np.arange(101, 201),
                        'low': np.arange(99, 199),
                        'close': np.arange(100, 200),
                        'adj_close': np.arange(100, 200),
                        'volume': np.arange(1000000, 1000000 + 100),
                    }),
                    '000660.KS': pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=100),
                        'open': np.arange(50, 150),
                        'high': np.arange(51, 151),
                        'low': np.arange(49, 149),
                        'close': np.arange(50, 150),
                        'adj_close': np.arange(50, 150),
                        'volume': np.arange(1000000, 1000000 + 100),
                    })
                }

                mock_calc = MagicMock()
                mock_calc_class.return_value = mock_calc
                mock_calc.calculate_indicators.side_effect = lambda df, ind_list: (
                    df.assign(**{ind: np.random.randn(len(df)) for ind in (ind_list or [])})
                )

                pipeline = DataPipeline(db_path=temp_db_path)
                result = pipeline.run_full_pipeline(
                    ticker_list=['005930.KS', '000660.KS'],
                    period='1y',
                    indicator_list=['ma_5', 'ma_20']
                )

                assert result is not None
                pipeline.close()
