"""
pytest 설정 파일
모든 테스트에서 공통으로 사용하는 fixtures와 설정

Fixtures는 tests/fixtures/ 폴더에 분리되어 있습니다:
  - fixtures/sample_data.py - 샘플 데이터 (DataFrame 등)
  - fixtures/database.py - DB 관련 (임시 경로 등)
  - fixtures/mocks.py - Mock 객체 (yfinance, DatabaseManager 등)
  - fixtures/config.py - 설정 (config 딕셔너리 등)
  - fixtures/utils.py - 유틸리티 (DataFrame/Series 비교 등)
"""

import pytest
import logging

# 모든 fixtures를 import하여 conftest.py에서 사용 가능하게 함
from tests.fixtures import (
    # Sample data
    sample_df_basic,
    sample_df_small,
    sample_df_with_indicators,
    sample_df_labeled,
    sample_training_data,
    sample_training_data_ml,
    sample_df_empty,
    sample_df_single_row,
    sample_df_with_nan,
    # Database
    temp_db_path,
    temp_model_dir,
    # Mocks
    mock_yfinance,
    mock_database_manager,
    mock_data_fetcher,
    mock_indicator_calculator,
    # Config
    sample_config,
    # Utils
    assert_df_equal,
    assert_series_equal,
    # Context Managers (자동으로 close 호출)
    db_manager,
    pipeline_context,
    db_manager_with_data,
)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)


# ==========================================
# 로깅 설정 Fixture
# ==========================================

@pytest.fixture(autouse=True)
def suppress_logging():
    """테스트 중 불필요한 로깅 제거"""
    logging.getLogger('src.data.pipeline').setLevel(logging.WARNING)
    logging.getLogger('src.data.db_manager').setLevel(logging.WARNING)
    logging.getLogger('src.ml.labeler').setLevel(logging.WARNING)
    logging.getLogger('src.ml.logistic_regression').setLevel(logging.WARNING)
    yield
