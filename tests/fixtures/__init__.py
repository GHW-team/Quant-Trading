"""
Pytest fixtures 패키지
테스트에서 사용하는 모든 fixture를 import하여 conftest.py에서 사용
"""

from .sample_data import (
    sample_df_basic,
    sample_df_small,
    sample_df_with_indicators,
    sample_df_labeled,
    sample_training_data,
    sample_training_data_ml,
    sample_df_empty,
    sample_df_single_row,
    sample_df_with_nan,
)

from .database import (
    temp_db_path,
    temp_model_dir,
)

from .mocks import (
    mock_yfinance,
    mock_database_manager,
    mock_data_fetcher,
    mock_indicator_calculator,
)

from .config import (
    sample_config,
)

from .utils import (
    assert_df_equal,
    assert_series_equal,
)

from .managers import (
    db_manager,
    pipeline_context,
    db_manager_with_data,
)

__all__ = [
    # Sample data
    'sample_df_basic',
    'sample_df_with_indicators',
    'sample_df_labeled',
    'sample_training_data',
    'sample_training_data_ml',
    'sample_df_empty',
    'sample_df_single_row',
    'sample_df_with_nan',
    # Database
    'temp_db_path',
    'temp_model_dir',
    # Mocks
    'mock_yfinance',
    'mock_database_manager',
    'mock_data_fetcher',
    'mock_indicator_calculator',
    # Config
    'sample_config',
    # Utils
    'assert_df_equal',
    'assert_series_equal',
    # Context Managers
    'db_manager',
    'pipeline_context',
    'db_manager_with_data',
]
