"""
DataPipeline 통합 테스트
데이터 수집, 지표 계산, DB 저장의 전체 파이프라인 검증
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.data.pipeline import DataPipeline
from src.data.db_models import Ticker, DailyPrice, TechnicalIndicator


class TestDataPipelineFullWorkflow:
    """DataPipeline 전체 워크플로우 테스트"""

    def test_run_full_pipeline_by_period(self, temp_db_path):
        """Period 기반 전체 파이프라인"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                # Mock 설정
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher
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

    def test_run_full_pipeline_by_date(self, temp_db_path):
        """Date 기반 전체 파이프라인"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                # Mock 설정
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher
                mock_fetcher.fetch_multiple_by_date.return_value = {
                    '005930.KS': pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=100),
                        'open': np.arange(100, 200),
                        'high': np.arange(101, 201),
                        'low': np.arange(99, 199),
                        'close': np.arange(100, 200),
                        'adj_close': np.arange(100, 200),
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
                    ticker_list=['005930.KS'],
                    start_date='2020-01-01',
                    end_date='2020-12-31',
                    indicator_list=['ma_5']
                )

                assert result is not None
                pipeline.close()

    def test_run_full_pipeline_multiple_tickers(self, temp_db_path):
        """여러 ticker 처리"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher
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

    def test_pipeline_with_extended_lookback(self, temp_db_path):
        """Lookback 확장이 적용되는 경우"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher
                mock_fetcher.fetch_multiple_by_period.return_value = {
                    '005930.KS': pd.DataFrame({
                        'date': pd.date_range('2019-09-01', periods=500),
                        'open': np.arange(100, 600),
                        'high': np.arange(101, 601),
                        'low': np.arange(99, 599),
                        'close': np.arange(100, 600),
                        'adj_close': np.arange(100, 600),
                        'volume': np.arange(1000000, 1500000),
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
                    indicator_list=['ma_5', 'ma_20', 'ma_200']
                )

                assert result is not None
                pipeline.close()

    def test_pipeline_period_and_date_mutual_exclusive(self, temp_db_path):
        """Period와 Date 동시 제공 불가"""
        pipeline = DataPipeline(db_path=temp_db_path)

        with pytest.raises(ValueError, match="period와 start_date"):
            pipeline.run_full_pipeline(
                ticker_list=['005930.KS'],
                period='1y',
                start_date='2020-01-01',
                indicator_list=['ma_5']
            )

        pipeline.close()

    def test_pipeline_empty_ticker_list(self, temp_db_path):
        """빈 ticker 리스트"""
        pipeline = DataPipeline(db_path=temp_db_path)

        result = pipeline.run_full_pipeline(
            ticker_list=[],
            period='1y',
            indicator_list=['ma_5']
        )

        assert result is not None
        pipeline.close()

    def test_pipeline_no_indicators(self, temp_db_path):
        """지표 없이 실행"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher
            mock_fetcher.fetch_multiple_by_period.return_value = {
                '005930.KS': pd.DataFrame({
                    'date': pd.date_range('2020-01-01', periods=100),
                    'open': np.arange(100, 200),
                    'high': np.arange(101, 201),
                    'low': np.arange(99, 199),
                    'close': np.arange(100, 200),
                    'adj_close': np.arange(100, 200),
                    'volume': np.arange(1000000, 1000000 + 100),
                })
            }

            pipeline = DataPipeline(db_path=temp_db_path)
            result = pipeline.run_full_pipeline(
                ticker_list=['005930.KS'],
                period='1y',
                indicator_list=[]
            )

            assert result is not None
            pipeline.close()

    def test_pipeline_with_lookback_none_date(self, temp_db_path):
        """start_date=None일 때"""
        pipeline = DataPipeline(db_path=temp_db_path)

        result = pipeline.run_full_pipeline(
            ticker_list=['005930.KS'],
            start_date=None,
            end_date='2020-12-31',
            indicator_list=['ma_5']
        )

        assert result is not None
        pipeline.close()

    def test_pipeline_different_period_values(self, temp_db_path):
        """다양한 period 값"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher
                mock_fetcher.fetch_multiple_by_period.return_value = {
                    '005930.KS': pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=100),
                        'open': np.arange(100, 200),
                        'high': np.arange(101, 201),
                        'low': np.arange(99, 199),
                        'close': np.arange(100, 200),
                        'adj_close': np.arange(100, 200),
                        'volume': np.arange(1000000, 1000000 + 100),
                    })
                }

                mock_calc = MagicMock()
                mock_calc_class.return_value = mock_calc
                mock_calc.calculate_indicators.side_effect = lambda df, ind_list: (
                    df.assign(**{ind: np.random.randn(len(df)) for ind in (ind_list or [])})
                )

                pipeline = DataPipeline(db_path=temp_db_path)

                for period in ['1d', '1w', '1m', '3m', '6m', '1y']:
                    result = pipeline.run_full_pipeline(
                        ticker_list=['005930.KS'],
                        period=period,
                        indicator_list=['ma_5']
                    )
                    assert result is not None

                pipeline.close()

    def test_pipeline_custom_max_workers(self, temp_db_path):
        """커스텀 worker 설정"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher
                mock_fetcher.fetch_multiple_by_period.return_value = {
                    '005930.KS': pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=100),
                        'open': np.arange(100, 200),
                        'high': np.arange(101, 201),
                        'low': np.arange(99, 199),
                        'close': np.arange(100, 200),
                        'adj_close': np.arange(100, 200),
                        'volume': np.arange(1000000, 1000000 + 100),
                    })
                }

                mock_calc = MagicMock()
                mock_calc_class.return_value = mock_calc
                mock_calc.calculate_indicators.side_effect = lambda df, ind_list: (
                    df.assign(**{ind: np.random.randn(len(df)) for ind in (ind_list or [])})
                )

                pipeline = DataPipeline(db_path=temp_db_path, max_workers=5)
                result = pipeline.run_full_pipeline(
                    ticker_list=['005930.KS'],
                    period='1y',
                    indicator_list=['ma_5']
                )

                assert result is not None
                pipeline.close()

    def test_pipeline_data_saved_to_db(self, temp_db_path):
        """DB에 데이터 저장 확인"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher
                mock_fetcher.fetch_multiple_by_period.return_value = {
                    '005930.KS': pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=50),
                        'open': np.arange(100, 150),
                        'high': np.arange(101, 151),
                        'low': np.arange(99, 149),
                        'close': np.arange(100, 150),
                        'adj_close': np.arange(100, 150),
                        'volume': np.arange(1000000, 1050000),
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
                    indicator_list=['ma_5']
                )

                # DB에서 데이터 확인
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                engine = create_engine(f'sqlite:///{temp_db_path}')
                Session = sessionmaker(bind=engine)
                session = Session()

                ticker_count = session.query(Ticker).filter_by(ticker_code='005930.KS').count()
                assert ticker_count > 0

                session.close()
                pipeline.close()

    def test_pipeline_workflow_period_lookback_indicators(self, temp_db_path):
        """Period → Lookback → Indicators 전체 흐름"""
        with patch('src.data.pipeline.StockDataFetcher') as mock_fetcher_class:
            with patch('src.data.pipeline.IndicatorCalculator') as mock_calc_class:
                mock_fetcher = MagicMock()
                mock_fetcher_class.return_value = mock_fetcher

                # Period를 Lookback까지 확장한 데이터 반환
                mock_fetcher.fetch_multiple_by_period.return_value = {
                    '005930.KS': pd.DataFrame({
                        'date': pd.date_range('2019-09-01', periods=500),
                        'open': np.arange(100, 600),
                        'high': np.arange(101, 601),
                        'low': np.arange(99, 599),
                        'close': np.arange(100, 600),
                        'adj_close': np.arange(100, 600),
                        'volume': np.arange(1000000, 1500000),
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
                    indicator_list=['ma_5', 'ma_20', 'ma_200', 'macd']
                )

                assert result is not None
                pipeline.close()


class TestDataPipelineContextManager:
    """Context Manager 통합 테스트"""

    def test_pipeline_with_statement(self, temp_db_path):
        """with 문장 사용"""
        with DataPipeline(db_path=temp_db_path) as pipeline:
            assert pipeline is not None
            assert pipeline.db_manager is not None

    def test_pipeline_resource_cleanup(self, temp_db_path):
        """리소스 정리"""
        with DataPipeline(db_path=temp_db_path) as pipeline:
            pass
        # with 블록을 벗어나면 자동으로 close()
