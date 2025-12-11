"""
설정 Fixture
샘플 config 딕셔너리 등
"""

import pytest
import tempfile
import yaml
from pathlib import Path


@pytest.fixture
def sample_config():
    """샘플 config 딕셔너리"""
    return {
        'data': {
            'database_path': 'data/database/stocks.db',
            'tickers': ['005930.KS', '000660.KS'],
            'period': None,
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
        },
        'indicators': {
            'list': ['ma_5', 'ma_20', 'ma_200', 'macd'],
            'version': 'v1.0',
        },
        'model': {
            'training': {
                'tickers': ['005930.KS'],
                'period': None,
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
            },
            'labeling': {
                'horizon': 5,
                'threshold': 0.02,
            },
            'features': {
                'columns': ['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd'],
                'indicators': ['ma_5', 'ma_20', 'ma_200', 'macd'],
            },
            'output': {
                'dir': 'models',
                'file': 'test_model.pkl',
            },
        },
    }


@pytest.fixture
def temp_config_file(tmp_path):
    """임시 YAML 설정 파일 생성"""
    config_data = {
        'data': {
            'database_path': 'data/database/stocks.db',
            'tickers': ['AAPL', 'MSFT'],
            'period': '1y',
        },
        'indicators': {
            'list': ['ma_5', 'ma_20', 'rsi'],
        },
        'batch': {
            'size': 10,
        },
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    return str(config_file)
