"""
데이터베이스 관련 Fixture
임시 DB 경로, 모델 저장 디렉토리 등
"""

import pytest
import tempfile
import os


@pytest.fixture
def temp_db_path():
    """임시 데이터베이스 경로"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        yield db_path
        # teardown: 파일 자동 삭제


@pytest.fixture
def temp_model_dir():
    """임시 모델 저장 디렉토리"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
        # teardown: 디렉토리 자동 삭제
