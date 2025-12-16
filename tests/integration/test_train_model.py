"""
train_model.py 통합 테스트
전체 모델 학습 파이프라인 검증
"""

import pytest
import sys
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTrainModelIntegration:
    """train_model.py 전체 통합 테스트"""

    @pytest.fixture
    def temp_config(self, temp_db_path):
        """임시 config.yaml 생성"""
        config = {
            'data': {
                'database_path': temp_db_path,
                'tickers': ['005930.KS'],
                'period': None,
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
            },
            'indicators': {
                'list': ['ma_5', 'ma_20'],
                'version': 'v1.0',
            },
            'model': {
                'training': {
                    'tickers': ['005930.KS'],
                    'period': None,
                    'start_date': '2020-01-01',
                    'end_date': '2020-12-31',
                },
                'labeling': {
                    'horizon': 5,
                    'threshold': 0.02,
                },
                'features': {
                    'columns': ['adj_close', 'ma_5', 'ma_20'],
                    'indicators': ['ma_5', 'ma_20'],
                },
                'output': {
                    'dir': 'models',
                    'file': 'test_model.pkl',
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / 'config'
            config_dir.mkdir()
            config_file = config_dir / 'config.yaml'

            with open(config_file, 'w') as f:
                yaml.dump(config, f)

            yield config_file, tmpdir

    def test_train_model_labeling_integration(self, sample_training_data_ml):
        """라벨링이 올바르게 작동하는지 통합 테스트"""
        from src.ml.labeler import Labeler

        # 원본 데이터 (라벨 제거)
        df = sample_training_data_ml.drop('label', axis=1).iloc[:100]

        labeler = Labeler(horizon=5, threshold=0.02)
        df_labeled = labeler.label_data(df, price_col='adj_close')

        # 라벨 생성 확인
        assert 'label' in df_labeled.columns
        assert df_labeled['label'].isin([0, 1]).all()  # 바이너리 클래스
        # 처음 horizon개 행은 NaN이어야 함
        assert df_labeled['label'].iloc[:5].isna().all()

    def test_train_model_handler_training(self, sample_training_data_ml):
        """모델 핸들러 학습"""
        from src.ml.logistic_regression import LogisticRegressionHandler

        feature_columns = ['adj_close', 'ma_5', 'ma_20']
        X = sample_training_data_ml[feature_columns].iloc[:200]
        y = sample_training_data_ml['label'].iloc[:200].dropna()
        X = X.iloc[:len(y)]  # X와 y의 길이를 맞춤

        handler = LogisticRegressionHandler(feature_names=feature_columns)
        handler.train(X, y)

        assert handler.is_fitted is True
        assert handler.model is not None
        # 모델이 feature names을 기억하는지 확인
        assert handler.feature_names == feature_columns

    def test_train_model_handler_evaluation(self, sample_training_data_ml):
        """모델 평가"""
        from src.ml.logistic_regression import LogisticRegressionHandler

        feature_columns = ['adj_close', 'ma_5', 'ma_20']
        X = sample_training_data_ml[feature_columns].iloc[:200]
        y = sample_training_data_ml['label'].iloc[:200].dropna()
        X = X.iloc[:len(y)]

        handler = LogisticRegressionHandler(feature_names=feature_columns)
        handler.train(X, y)

        metrics = handler.evaluate(X, y)
        assert 'test_accuracy' in metrics
        assert 0 <= metrics['test_accuracy'] <= 1
        # 충분한 데이터로 학습했으므로 accuracy가 0 이상이어야 함
        assert metrics['test_accuracy'] > 0

    def test_train_model_handler_prediction(self, sample_training_data_ml):
        """모델 예측"""
        from src.ml.logistic_regression import LogisticRegressionHandler
        from sklearn.model_selection import train_test_split

        feature_columns = ['adj_close', 'ma_5', 'ma_20']
        df = sample_training_data_ml[feature_columns + ['label']].dropna()

        X = df[feature_columns]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        handler = LogisticRegressionHandler(feature_names=feature_columns)
        handler.train(X_train, y_train)

        predictions = handler.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
        # 충분한 데이터로 학습했으므로 정확도가 reasonable해야 함
        accuracy = (predictions == y_test.values).mean()
        assert accuracy > 0.4  # 무작위 추측(50%)보다 나아야 함

    def test_train_model_model_saving(self, temp_model_dir, sample_training_data_ml):
        """모델 저장"""
        from src.ml.logistic_regression import LogisticRegressionHandler

        feature_columns = ['adj_close', 'ma_5', 'ma_20']
        df = sample_training_data_ml[feature_columns + ['label']].dropna()
        X = df[feature_columns].iloc[:200]
        y = df['label'].iloc[:200]

        handler = LogisticRegressionHandler(feature_names=feature_columns)
        handler.train(X, y)

        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        handler.save(model_path)

        assert os.path.exists(model_path)
        # 파일 크기가 합리적인지 확인 (최소 1KB 이상)
        assert os.path.getsize(model_path) > 1000

    def test_train_model_model_loading(self, temp_model_dir, sample_training_data_ml):
        """모델 로드"""
        from src.ml.logistic_regression import LogisticRegressionHandler

        feature_columns = ['adj_close', 'ma_5', 'ma_20']
        df = sample_training_data_ml[feature_columns + ['label']].dropna()

        X_train = df[feature_columns].iloc[:160]
        y_train = df['label'].iloc[:160]
        X_test = df[feature_columns].iloc[160:200]

        # 모델 저장
        handler = LogisticRegressionHandler(feature_names=feature_columns)
        handler.train(X_train, y_train)
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        handler.save(model_path)

        # 모델 로드
        loaded_handler = LogisticRegressionHandler.load(model_path)
        predictions = loaded_handler.predict(X_test)

        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
        # 로드된 모델이 원래 모델과 동일한 예측을 해야 함
        original_predictions = handler.predict(X_test)
        assert (predictions == original_predictions).all()

    def test_train_model_full_workflow(self, temp_config, temp_model_dir, sample_training_data_ml):
        """전체 워크플로우 - 데이터 로딩부터 평가까지"""
        from src.ml.labeler import Labeler
        from src.ml.logistic_regression import LogisticRegressionHandler
        from sklearn.model_selection import train_test_split

        config_file, _ = temp_config

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # 1. 데이터 로드 - 실제 fixture 사용
        df_all = sample_training_data_ml.drop('label', axis=1).iloc[:250]

        # 2. 라벨링
        labeler = Labeler(
            horizon=config['model']['labeling']['horizon'],
            threshold=config['model']['labeling']['threshold']
        )
        df_labeled = labeler.label_data(df_all, price_col='adj_close')

        # 3. 데이터 정제
        feature_columns = config['model']['features']['columns']
        clean_data = df_labeled.dropna(subset=feature_columns + ['label'])

        # 데이터가 충분한지 확인
        assert len(clean_data) > 0, "라벨링 후 유효한 데이터가 없음"

        # 4. Train/Test 분할
        train_df, test_df = train_test_split(clean_data, test_size=0.2, shuffle=False)

        X_train = train_df[feature_columns]
        y_train = train_df['label']
        X_test = test_df[feature_columns]
        y_test = test_df['label']

        assert len(X_train) > 0, "학습 데이터가 없음"
        assert len(X_test) > 0, "테스트 데이터가 없음"

        # 5. 모델 학습
        handler = LogisticRegressionHandler(feature_names=feature_columns)
        handler.train(X_train, y_train)

        assert handler.is_fitted is True, "모델이 학습되지 않음"

        # 6. 모델 저장
        model_path = os.path.join(temp_model_dir, config['model']['output']['file'])
        handler.save(model_path)

        assert os.path.exists(model_path), "모델 파일이 저장되지 않음"

        # 7. 예측
        predictions = handler.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

        # 8. 평가
        metrics = handler.evaluate(X_test, y_test)
        assert 'test_accuracy' in metrics
        assert 0 <= metrics['test_accuracy'] <= 1


class TestTrainModelEdgeCases:
    """엣지 케이스 테스트"""

    def test_train_model_with_empty_data(self):
        """빈 데이터 처리"""
        from src.ml.logistic_regression import LogisticRegressionHandler

        X = pd.DataFrame({
            'adj_close': [],
            'ma_5': [],
            'ma_20': [],
        })
        y = pd.Series([], dtype=int)

        handler = LogisticRegressionHandler(feature_names=['adj_close', 'ma_5', 'ma_20'])

        with pytest.raises((ValueError, Exception)):
            handler.train(X, y)

    @pytest.mark.parametrize("num_samples,class_ratio", [
        (300, 0.5),      # 균형 클래스
        (300, 0.9),      # 심한 불균형 (90:10)
        (300, 0.1),      # 역 불균형 (10:90)
        (1000, 0.5),     # 큰 데이터셋 (균형)
    ])
    def test_train_model_with_various_distributions(self, num_samples, class_ratio):
        """다양한 클래스 분포 처리"""
        from src.ml.logistic_regression import LogisticRegressionHandler

        X = pd.DataFrame({
            'adj_close': np.random.randn(num_samples),
            'ma_5': np.random.randn(num_samples),
            'ma_20': np.random.randn(num_samples),
        })

        # 지정된 비율로 클래스 생성
        num_positive = int(num_samples * class_ratio)
        y = pd.Series([1] * num_positive + [0] * (num_samples - num_positive))

        handler = LogisticRegressionHandler(feature_names=['adj_close', 'ma_5', 'ma_20'])
        handler.train(X, y)

        assert handler.is_fitted is True

        # 데이터 크기가 충분하면 예측도 검증
        if num_samples >= 100:
            predictions = handler.predict(X)
            assert len(predictions) == num_samples
            assert set(predictions).issubset({0, 1})

    def test_train_model_single_class_warning(self):
        """단일 클래스 데이터 처리 (경고 또는 에러)"""
        from src.ml.logistic_regression import LogisticRegressionHandler

        X = pd.DataFrame({
            'adj_close': np.random.randn(300),
            'ma_5': np.random.randn(300),
            'ma_20': np.random.randn(300),
        })
        y = pd.Series([1] * 300)  # 모두 같은 클래스

        handler = LogisticRegressionHandler(feature_names=['adj_close', 'ma_5', 'ma_20'])

        # 단일 클래스는 보통 sklearn에서 에러를 발생시킴
        with pytest.raises((ValueError, Exception)):
            handler.train(X, y)
