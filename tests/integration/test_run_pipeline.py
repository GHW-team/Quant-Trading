"""
run_pipeline.py 통합 테스트
파이프라인 설정, 로더, 실행 검증
"""

import pytest
import sys
import yaml
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_pipeline import ConfigLoader, setup_logging, _resolve_tickers


class TestConfigLoader:
    """ConfigLoader 테스트"""

    @pytest.fixture
    def temp_config_file(self):
        """임시 config.yaml 파일"""
        config = {
            'data': {
                'tickers': ['005930.KS', '000660.KS'],
                'exchanges': [],
                'database_path': 'data/database/stocks.db',
                'period': '1y',
                'start_date': None,
                'end_date': None,
                'update_if_exists': True,
                'interval': '1d',
            },
            'indicators': {
                'list': ['ma_5', 'ma_20', 'ma_200', 'macd'],
                'version': 'v1.0',
            },
            'batch': {
                'size': 100,
            },
            'fetcher': {
                'max_workers': 5,
                'max_retries': 3,
            },
            'logging': {
                'level': 'INFO',
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            yield config_file

    def test_config_loader_initialization(self, temp_config_file):
        """ConfigLoader 초기화"""
        loader = ConfigLoader(config_file=str(temp_config_file))
        assert loader.config_file == temp_config_file

    def test_config_loader_load_yaml(self, temp_config_file):
        """YAML 로드"""
        loader = ConfigLoader(config_file=str(temp_config_file))
        assert loader.yaml_config is not None
        assert 'data' in loader.yaml_config

    def test_config_loader_build_config(self, temp_config_file):
        """Config 구축"""
        loader = ConfigLoader(config_file=str(temp_config_file))
        config = loader._build_config_from_yaml()

        assert config['tickers'] == ['005930.KS', '000660.KS']
        assert config['period'] == '1y'
        assert config['max_workers'] == 5

    def test_config_loader_merge_cli_args(self, temp_config_file):
        """CLI 인자 병합"""
        loader = ConfigLoader(config_file=str(temp_config_file))
        config = loader._build_config_from_yaml()

        # Mock CLI args
        cli_args = argparse.Namespace(
            tickers=['012345.KS'],
            exchanges=[],
            period='6m',
            start_date=None,
            end_date=None,
            indicators=['ma_5'],
            batch_size=50,
            update_if_exists=None,
            log_level='INFO'
        )

        merged = loader._merge_cli_args(config, cli_args)
        assert merged['tickers'] == ['012345.KS']
        assert merged['period'] == '6m'
        assert merged['batch_size'] == 50

    def test_config_loader_get_config(self, temp_config_file):
        """최종 config 획득"""
        loader = ConfigLoader(config_file=str(temp_config_file))

        cli_args = argparse.Namespace(
            tickers=None,
            exchanges=None,
            period=None,
            start_date=None,
            end_date=None,
            indicators=None,
            batch_size=None,
            update_if_exists=None,
            log_level='INFO'
        )

        config = loader.get_config(cli_args)
        assert config is not None
        assert 'tickers' in config


class TestConfigLoaderEdgeCases:
    """ConfigLoader 엣지 케이스"""

    def test_config_loader_missing_file(self):
        """파일 없음"""
        loader = ConfigLoader(config_file='/nonexistent/config.yaml')
        assert loader.yaml_config == {}

    def test_config_loader_empty_yaml(self):
        """빈 YAML"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / 'empty.yaml'
            with open(config_file, 'w') as f:
                f.write('')

            loader = ConfigLoader(config_file=str(config_file))
            assert loader.yaml_config == {}

    def test_config_loader_missing_sections(self):
        """섹션 누락"""
        config = {'data': {}}  # indicators, batch 없음

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f)

            loader = ConfigLoader(config_file=str(config_file))
            built = loader._build_config_from_yaml()
            # 기본값 사용
            assert built['indicators'] == ['ma_5', 'ma_20', 'ma_200', 'macd']

    def test_config_loader_partial_data_section(self):
        """부분 데이터 섹션"""
        config = {
            'data': {
                'tickers': ['005930.KS'],
                # period 없음
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f)

            loader = ConfigLoader(config_file=str(config_file))
            built = loader._build_config_from_yaml()
            assert built['tickers'] == ['005930.KS']
            assert built['period'] == '1y'  # 기본값


class TestResolveTickers:
    """_resolve_tickers 함수 테스트"""

    def test_resolve_tickers_from_list(self):
        """Ticker 리스트 사용"""
        config = {
            'tickers': ['005930.KS', '000660.KS'],
            'exchanges': None
        }

        result = _resolve_tickers(config)
        assert result == ['005930.KS', '000660.KS']

    def test_resolve_tickers_from_exchanges(self):
        """거래소 코드 사용"""
        config = {
            'tickers': None,
            'exchanges': ['KOSPI']
        }

        with patch('scripts.run_pipeline.TickerUniverse') as mock_universe:
            mock_instance = MagicMock()
            mock_universe.return_value = mock_instance
            expected_tickers = ['005930.KS', '000660.KS']
            mock_instance.get.return_value = expected_tickers

            result = _resolve_tickers(config)
            assert result == expected_tickers
            mock_instance.get.assert_called_once_with(['KOSPI'])

    def test_resolve_tickers_conflict_error(self):
        """Ticker와 Exchange 동시 사용 에러"""
        config = {
            'tickers': ['005930.KS'],
            'exchanges': ['KOSPI']
        }

        with pytest.raises(ValueError, match="동시에 지정할 수 없습니다"):
            _resolve_tickers(config)

    def test_resolve_tickers_empty_error(self):
        """비어있는 경우 에러"""
        config = {
            'tickers': None,
            'exchanges': None
        }

        with pytest.raises(ValueError, match="비어 있습니다"):
            _resolve_tickers(config)

    def test_resolve_tickers_empty_list(self):
        """빈 리스트"""
        config = {
            'tickers': [],
            'exchanges': []
        }

        with pytest.raises(ValueError):
            _resolve_tickers(config)


class TestSetupLogging:
    """setup_logging 함수 테스트"""

    @pytest.mark.parametrize("log_level,expected_level", [
        ("DEBUG", "DEBUG"),
        ("INFO", "INFO"),
        ("WARNING", "WARNING"),
        ("ERROR", "ERROR"),
    ])
    def test_setup_logging_levels(self, log_level, expected_level):
        """다양한 로그 레벨 설정"""
        logger = setup_logging(log_level=log_level)
        assert logger is not None
        assert logger.name == '__main__'


class TestRunPipelineIntegration:
    """파이프라인 실행 통합 테스트"""

    def test_run_full_pipeline_workflow(self):
        """전체 파이프라인 워크플로우"""
        from scripts.run_pipeline import run_full_pipeline

        config = {
            'database_path': 'test.db',
            'tickers': ['005930.KS'],
            'exchanges': [],
            'period': '1y',
            'start_date': None,
            'end_date': None,
            'interval': '1d',
            'update_if_exists': True,
            'indicators': ['ma_5', 'ma_20'],
            'indicator_version': 'v1.0',
            'batch_size': 100,
            'max_workers': 3,
            'max_retries': 3,
        }

        logger = setup_logging()

        with patch('scripts.run_pipeline.DataPipeline') as mock_pipeline:
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_instance.run_full_pipeline.return_value = {'summary': 'success'}

            run_full_pipeline(config, logger)

            # 파이프라인이 올바른 인자로 호출되었는지 확인
            mock_instance.run_full_pipeline.assert_called_once()
            call_args = mock_instance.run_full_pipeline.call_args
            assert call_args[1]['ticker_list'] == ['005930.KS']
            assert call_args[1]['period'] == '1y'
            assert call_args[1]['indicator_list'] == ['ma_5', 'ma_20']

    def test_run_price_pipeline_workflow(self):
        """가격 파이프라인 워크플로우"""
        from scripts.run_pipeline import run_price_pipeline

        config = {
            'database_path': 'test.db',
            'tickers': ['005930.KS'],
            'exchanges': [],
            'period': '1y',
            'start_date': None,
            'end_date': None,
            'interval': '1d',
            'update_if_exists': True,
            'batch_size': 100,
            'max_workers': 3,
            'max_retries': 3,
        }

        logger = setup_logging()

        with patch('scripts.run_pipeline.DataPipeline') as mock_pipeline:
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_instance.run_price_pipeline.return_value = {'summary': 'success'}

            run_price_pipeline(config, logger)

            # 파이프라인이 올바른 인자로 호출되었는지 확인
            mock_instance.run_price_pipeline.assert_called_once()
            call_args = mock_instance.run_price_pipeline.call_args
            assert call_args[1]['ticker_list'] == ['005930.KS']
            assert call_args[1]['period'] == '1y'
            assert call_args[1]['interval'] == '1d'
            assert call_args[1]['update_if_exists'] is True
            assert call_args[1]['batch_size'] == 100

    def test_run_indicator_pipeline_workflow(self):
        """지표 파이프라인 워크플로우"""
        from scripts.run_pipeline import run_indicator_pipeline

        config = {
            'database_path': 'test.db',
            'tickers': ['005930.KS'],
            'exchanges': [],
            'indicator_version': 'v1.0',
            'indicators': ['ma_5', 'ma_20'],
            'start_date': None,
            'end_date': None,
            'batch_size': 100,
            'max_workers': 3,
            'max_retries': 3,
        }

        logger = setup_logging()

        with patch('scripts.run_pipeline.DataPipeline') as mock_pipeline:
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            mock_instance.run_indicator_pipeline.return_value = {'summary': 'success'}

            run_indicator_pipeline(config, logger)

            # 파이프라인이 올바른 인자로 호출되었는지 확인
            mock_instance.run_indicator_pipeline.assert_called_once()
            call_args = mock_instance.run_indicator_pipeline.call_args
            assert call_args[1]['ticker_list'] == ['005930.KS']
            assert call_args[1]['indicator_list'] == ['ma_5', 'ma_20']
            assert call_args[1]['version'] == 'v1.0'
            assert call_args[1]['batch_size'] == 100


class TestConfigLoaderActualYAML:
    """ConfigLoader가 실제 YAML 파일을 제대로 로드하는지 테스트"""

    def test_config_loader_loads_tickers(self, temp_config_file):
        """YAML에서 Ticker를 올바르게 로드"""
        loader = ConfigLoader(config_file=str(temp_config_file))
        config = loader._build_config_from_yaml()

        assert 'tickers' in config
        assert config['tickers'] == ['005930.KS', '000660.KS']

    def test_config_loader_loads_period(self, temp_config_file):
        """YAML에서 Period를 올바르게 로드"""
        loader = ConfigLoader(config_file=str(temp_config_file))
        config = loader._build_config_from_yaml()

        assert 'period' in config
        assert config['period'] == '1y'

    def test_config_loader_loads_indicators(self, temp_config_file):
        """YAML에서 Indicators를 올바르게 로드"""
        loader = ConfigLoader(config_file=str(temp_config_file))
        config = loader._build_config_from_yaml()

        assert 'indicators' in config
        assert isinstance(config['indicators'], list)
        assert len(config['indicators']) > 0

    def test_config_loader_loads_batch_size(self, temp_config_file):
        """YAML에서 Batch Size를 올바르게 로드"""
        loader = ConfigLoader(config_file=str(temp_config_file))
        config = loader._build_config_from_yaml()

        assert 'batch_size' in config
        assert config['batch_size'] == 100


class TestRunPipelineErrorHandling:
    """파이프라인 에러 처리 테스트"""

    @pytest.mark.parametrize("exception,exit_code", [
        (KeyboardInterrupt(), 0),
        (Exception("Database error"), 1),
    ])
    def test_pipeline_error_handling(self, exception, exit_code):
        """파이프라인 에러 처리"""
        from scripts.run_pipeline import main

        with patch('scripts.run_pipeline.create_parser') as mock_parser:
            with patch('scripts.run_pipeline.setup_logging'):
                with patch('scripts.run_pipeline.ConfigLoader') as mock_loader:
                    mock_args = argparse.Namespace(
                        full=True, price=False, indicators=False,
                        tickers=None, exchanges=None, period=None,
                        start_date=None, end_date=None, indicators_list=None,
                        batch_size=None, update_if_exists=None,
                        log_level='INFO', config='config.yaml'
                    )
                    mock_parser_inst = MagicMock()
                    mock_parser.return_value = mock_parser_inst
                    mock_parser_inst.parse_args.return_value = mock_args

                    mock_loader_inst = MagicMock()
                    mock_loader.return_value = mock_loader_inst
                    mock_loader_inst.get_config.side_effect = exception

                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    # 올바른 exit code 확인
                    assert exc_info.value.code == exit_code
