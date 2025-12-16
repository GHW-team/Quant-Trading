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
