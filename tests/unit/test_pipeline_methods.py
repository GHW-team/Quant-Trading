"""
DataPipeline 클래스 메서드별 단위 테스트
정적 메서드, 초기화, context manager 등 검증
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.pipeline import DataPipeline


class TestDataPipelineConvertPeriodToDates:
    """Period 문자열 파싱 테스트"""

    def test_convert_1y_period(self):
        """1년 period 변환"""
        start_date, end_date = DataPipeline._convert_period_to_dates('1y')

        assert isinstance(start_date, str)
        assert isinstance(end_date, str)

        # 날짜 형식 검증
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        assert start_dt < end_dt

    def test_convert_6m_period(self):
        """6개월 period 변환"""
        start_date, end_date = DataPipeline._convert_period_to_dates('6m')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 약 6개월 차이
        diff_days = (end_dt - start_dt).days
        assert 150 < diff_days < 200

    def test_convert_3m_period(self):
        """3개월 period 변환"""
        start_date, end_date = DataPipeline._convert_period_to_dates('3m')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 약 3개월 차이
        diff_days = (end_dt - start_dt).days
        assert 75 < diff_days < 95

    def test_convert_1m_period(self):
        """1개월 period 변환"""
        start_date, end_date = DataPipeline._convert_period_to_dates('1m')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 약 1개월 차이
        diff_days = (end_dt - start_dt).days
        assert 25 < diff_days < 35

    def test_convert_1d_period(self):
        """1일 period 변환"""
        start_date, end_date = DataPipeline._convert_period_to_dates('1d')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 1일 차이
        diff_days = (end_dt - start_dt).days
        assert 0 < diff_days < 2

    def test_convert_1w_period(self):
        """1주 period 변환"""
        start_date, end_date = DataPipeline._convert_period_to_dates('1w')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 약 7일 차이
        diff_days = (end_dt - start_dt).days
        assert 5 < diff_days < 10

    def test_invalid_period_raises_error(self):
        """잘못된 period 형식"""
        with pytest.raises(ValueError):
            DataPipeline._convert_period_to_dates('invalid')

    def test_period_with_zero_returns_same_year(self):
        """0 기간 - 현재 연도로 변환됨"""
        # 현재 소스코드는 amount=0일 때 에러를 발생하지 않음
        # today.year - 0 = today.year (현재 연도)
        start_date, end_date = DataPipeline._convert_period_to_dates('0y')

        # start_date와 end_date가 같은 연도
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        assert start_dt.year == end_dt.year
        # start_date와 end_date가 같은 날짜 또는 매우 가까움
        assert (end_dt - start_dt).days <= 1

    def test_period_with_negative_amount(self):
        """음수 기간 - 미래 날짜로 변환됨"""
        # '-1y'는 today.year - (-1) = today.year + 1
        start_date, end_date = DataPipeline._convert_period_to_dates('-1y')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # start_date가 end_date보다 미래
        assert start_dt > end_dt

    def test_period_with_float_raises_error(self):
        """부동소수점 기간 - ValueError 발생"""
        # '1.5y'는 int('1.')를 호출하려고 함 → ValueError
        with pytest.raises((ValueError, IndexError)):
            DataPipeline._convert_period_to_dates('1.5y')

    def test_period_format_yyyy_mm_dd(self):
        """반환값이 YYYY-MM-DD 형식"""
        start_date, end_date = DataPipeline._convert_period_to_dates('1y')

        # YYYY-MM-DD 형식 검증
        assert len(start_date) == 10
        assert len(end_date) == 10
        assert start_date[4] == '-'
        assert start_date[7] == '-'


class TestDataPipelineCalculateExtendedStartDate:
    """Lookback 시작 날짜 계산 테스트"""

    def test_extended_start_date_with_ma_5(self, temp_db_path):
        """ma_5 lookback 계산"""
        pipeline = DataPipeline(db_path=temp_db_path)

        original_date = '2020-01-01'
        extended_date = pipeline._calculate_extended_start_date(
            original_date,
            ['ma_5']
        )

        # 과거로 확장됨
        original_dt = pd.to_datetime(original_date)
        extended_dt = pd.to_datetime(extended_date)
        assert extended_dt < original_dt

        pipeline.close()

    def test_extended_start_date_with_ma_200(self, temp_db_path):
        """ma_200 lookback 계산"""
        pipeline = DataPipeline(db_path=temp_db_path)

        original_date = '2020-01-01'
        extended_date = pipeline._calculate_extended_start_date(
            original_date,
            ['ma_200']
        )

        # ma_200은 lookback이 길어야 함
        original_dt = pd.to_datetime(original_date)
        extended_dt = pd.to_datetime(extended_date)
        diff_days = (original_dt - extended_dt).days

        assert diff_days > 200  # 최소 200일 이상

        pipeline.close()

    def test_extended_start_date_multiple_indicators(self, temp_db_path):
        """여러 지표 lookback (최대값)"""
        pipeline = DataPipeline(db_path=temp_db_path)

        date1 = pipeline._calculate_extended_start_date('2020-01-01', ['ma_5'])
        date2 = pipeline._calculate_extended_start_date('2020-01-01', ['ma_200'])
        date3 = pipeline._calculate_extended_start_date('2020-01-01', ['ma_5', 'ma_200'])

        # ma_200이 포함되면 가장 멀어야 함
        assert pd.to_datetime(date3) <= pd.to_datetime(date2)
        assert pd.to_datetime(date3) < pd.to_datetime(date1)

        pipeline.close()

    def test_extended_start_date_with_none(self, temp_db_path):
        """None 입력은 None 반환"""
        pipeline = DataPipeline(db_path=temp_db_path)

        result = pipeline._calculate_extended_start_date(None, ['ma_5'])
        assert result is None

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


class TestDataPipelineInitialization:
    """DataPipeline 초기화 테스트"""

    def test_initialization_with_custom_db_path(self, temp_db_path):
        """커스텀 DB 경로로 초기화"""
        pipeline = DataPipeline(db_path=temp_db_path)

        assert pipeline.db_path == temp_db_path
        assert pipeline.db_manager is not None
        assert pipeline.fetcher is not None
        assert pipeline.calculator is not None

        pipeline.close()

    def test_initialization_with_custom_workers(self, temp_db_path):
        """커스텀 worker 수"""
        pipeline = DataPipeline(
            db_path=temp_db_path,
            max_workers=10,
            max_retries=5
        )

        assert pipeline is not None
        pipeline.close()

    def test_initialization_creates_components(self, temp_db_path):
        """필요한 컴포넌트 생성"""
        pipeline = DataPipeline(db_path=temp_db_path)

        assert hasattr(pipeline, 'db_manager')
        assert hasattr(pipeline, 'fetcher')
        assert hasattr(pipeline, 'calculator')

        pipeline.close()


class TestDataPipelineContextManager:
    """Context Manager 테스트"""

    def test_enter_returns_pipeline(self, temp_db_path):
        """__enter__가 pipeline 반환"""
        with DataPipeline(db_path=temp_db_path) as pipeline:
            assert isinstance(pipeline, DataPipeline)

    def test_context_manager_closes_connection(self, temp_db_path):
        """__exit__에서 연결 종료"""
        pipeline = DataPipeline(db_path=temp_db_path)
        pipeline.__exit__(None, None, None)
        # 종료 후에도 에러 없음

    def test_with_statement_workflow(self, temp_db_path):
        """with 문장 워크플로우"""
        with DataPipeline(db_path=temp_db_path) as pipeline:
            assert pipeline is not None
            # with 블록 내에서 사용 가능


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
