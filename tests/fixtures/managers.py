"""
자동 정리 Context Manager Fixtures
DatabaseManager, DataPipeline 등의 리소스를 자동으로 정리
"""

import pytest
from src.data.db_manager import DatabaseManager
from src.data.pipeline import DataPipeline


@pytest.fixture
def db_manager(temp_db_path):
    """
    자동으로 close()되는 DatabaseManager

    사용 예:
        def test_save_price(db_manager, sample_df_basic):
            db_manager.save_price_data({'005930.KS': sample_df_basic})
            # 자동으로 close() 호출됨
    """
    manager = DatabaseManager(db_path=temp_db_path)
    yield manager
    manager.close()


@pytest.fixture
def pipeline_context(temp_db_path):
    """
    자동으로 close()되는 DataPipeline

    사용 예:
        def test_pipeline(pipeline_context):
            result = pipeline_context.run_full_pipeline(...)
            # 자동으로 close() 호출됨
    """
    pipeline = DataPipeline(db_path=temp_db_path)
    yield pipeline
    pipeline.close()


@pytest.fixture
def db_manager_with_data(temp_db_path, sample_df_basic):
    """
    초기 데이터가 있는 DatabaseManager

    사용 예:
        def test_query(db_manager_with_data):
            prices = db_manager_with_data.load_price_data('005930.KS')
    """
    manager = DatabaseManager(db_path=temp_db_path)
    manager.save_price_data({'005930.KS': sample_df_basic})
    yield manager
    manager.close()
