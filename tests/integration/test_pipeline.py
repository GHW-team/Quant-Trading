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

#==========================================
# 헬퍼 함수
#==========================================
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

# _validate_period_and_date
class TestValidatePeriodAndDate:
    @pytest.mark.parametrize('start_date,end_date,period',[
        ('2023-02-03','2024-01-01','1y'),
        (None,'2024-01-01','1y'),
        ('2023-02-03',None,'1y'),
    ])
    def test_input_period_and_date_both(self, start_date, end_date, period):
        """period와 date 동시 입력"""
        #Arrange
        pipeline = DataPipeline()       

        #Act
        with pytest.raises(ValueError):
            #Assert
            pipeline._validate_period_and_date(
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

    @pytest.mark.parametrize('start_date,end_date,period',[
        (None,'2024-01-01',None),
        ('2023-02-03',None,None),
    ])
    def test_input_only_one_of_dates(self, start_date, end_date, period):
        """start_date 와 end_date 둘중 하나만 입력"""
        #Arrange
        pipeline = DataPipeline()       

        #Act
        with pytest.raises(ValueError):
            #Assert
            pipeline._validate_period_and_date(
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

    @pytest.mark.parametrize('start_date,end_date,period',[
        ('2023.02.03','2024.01.01',None),
        ('2023/01/01','2024/12/01',None),
        ('2023 01 01','2024 12 01',None),
    ])
    def test_input_only_one_of_dates(self, start_date, end_date, period):
        """잘못된 date 형식"""
        #Arrange
        pipeline = DataPipeline()       

        #Act
        with pytest.raises(ValueError):
            #Assert
            pipeline._validate_period_and_date(
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

    @pytest.mark.parametrize('start_date,end_date,period',[
        ('2025-02-03','2024-01-01',None),
        ('2021-02-03','2020-01-01',None),
        ('2022-02-02','2022-02-01',None),
        ('2022-02-02','2022-02-02',None),
    ])
    def test_input_only_one_of_dates(self, start_date, end_date, period):
        """부적절한 날짜 순서"""
        #Arrange
        pipeline = DataPipeline()       

        #Act
        with pytest.raises(ValueError):
            #Assert
            pipeline._validate_period_and_date(
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

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


class TestDataPipelineStaticMethodsEdgeCases:
    """정적 메서드 엣지 케이스"""

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
