"""
LogisticRegressionHandler 클래스 단위 테스트
GridSearchCV + TimeSeriesSplit을 사용한 모델 학습 및 예측 검증
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.ml.logistic_regression import LogisticRegressionHandler


class TestLogisticRegressionHandlerInitialization:
    """모델 핸들러 초기화 테스트"""

    def test_initialization_with_features(self):
        """feature names와 함께 초기화"""
        features = ['ma_5', 'ma_20', 'ma_200', 'macd']
        handler = LogisticRegressionHandler(feature_names=features)

        assert handler.feature_names == features
        assert handler.model is None
        assert handler.is_fitted is False

    def test_initialization_empty_features(self):
        """빈 feature list로 초기화"""
        handler = LogisticRegressionHandler(feature_names=[])

        assert handler.feature_names == []
        assert handler.model is None
        assert handler.is_fitted is False

    def test_scaler_initialized(self):
        """StandardScaler 초기화"""
        handler = LogisticRegressionHandler(feature_names=['col1'])

        assert handler.scaler is not None
        from sklearn.preprocessing import StandardScaler
        assert isinstance(handler.scaler, StandardScaler)


class TestLogisticRegressionHandlerTrain:
    """모델 학습 테스트"""

    def test_train_creates_model(self, sample_training_data_ml):
        """학습 후 model이 생성되는가?"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )

        handler.train(X, y)

        assert handler.model is not None

    def test_train_sets_fitted_flag(self, sample_training_data_ml):
        """is_fitted 플래그가 설정됨"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )

        assert handler.is_fitted is False
        handler.train(X, y)
        assert handler.is_fitted is True

    def test_scaler_is_fitted(self, sample_training_data_ml):
        """Scaler가 학습되는가?"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )

        handler.train(X, y)

        assert handler.scaler.mean_ is not None
        assert len(handler.scaler.mean_) == X.shape[1]

    def test_train_metrics_stored(self, sample_training_data_ml):
        """학습 메트릭이 저장됨"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )

        handler.train(X, y)

        assert 'best_cv_score' in handler.train_metrics
        assert 'best_params' in handler.train_metrics
        assert handler.train_metrics['best_cv_score'] > 0
        assert handler.train_metrics['best_cv_score'] <= 1

    def test_train_returns_self(self, sample_training_data_ml):
        """train()이 self를 반환 (메서드 체이닝)"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )

        result = handler.train(X, y)

        assert result is handler

    def test_train_with_single_feature(self):
        """단일 feature로 학습"""
        X = pd.DataFrame({
            'feature1': np.random.randn(50)
        })
        y = pd.Series(np.random.randint(0, 2, 50))

        handler = LogisticRegressionHandler(feature_names=['feature1'])
        handler.train(X, y)

        assert handler.is_fitted is True


class TestLogisticRegressionHandlerEvaluate:
    """모델 평가 테스트"""

    @pytest.fixture
    def trained_handler(self, sample_training_data_ml):
        """미리 학습된 모델"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )
        handler.train(X, y)
        return handler, X, y

    def test_evaluate_returns_dict(self, trained_handler):
        """evaluate 결과가 dict"""
        handler, X, y = trained_handler
        result = handler.evaluate(X, y)

        assert isinstance(result, dict)

    def test_evaluate_has_accuracy(self, trained_handler):
        """결과에 test_accuracy 키가 있음"""
        handler, X, y = trained_handler
        result = handler.evaluate(X, y)

        assert 'test_accuracy' in result

    def test_evaluate_accuracy_in_valid_range(self, trained_handler):
        """정확도가 0-1 범위"""
        handler, X, y = trained_handler
        result = handler.evaluate(X, y)

        accuracy = result['test_accuracy']
        assert 0 <= accuracy <= 1

    def test_evaluate_without_training_raises_error(self):
        """학습되지 않은 모델로 평가하면 에러"""
        handler = LogisticRegressionHandler(feature_names=['col1'])
        X = pd.DataFrame({'col1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])

        with pytest.raises(ValueError, match="학습되지 않았습니다"):
            handler.evaluate(X, y)

    def test_evaluate_has_count(self, trained_handler):
        """결과에 count 키가 있음"""
        handler, X, y = trained_handler
        result = handler.evaluate(X, y)

        assert 'count' in result
        assert result['count'] == len(X)

    def test_evaluate_validates_features(self, trained_handler):
        """feature 검증"""
        handler, X, y = trained_handler

        # 잘못된 컬럼
        X_wrong = pd.DataFrame({
            'wrong_col': [1, 2, 3]
        })

        with pytest.raises(ValueError):
            handler.evaluate(X_wrong, pd.Series([0, 1, 0]))


class TestLogisticRegressionHandlerPredict:
    """모델 예측 테스트"""

    @pytest.fixture
    def trained_handler(self, sample_training_data_ml):
        """미리 학습된 모델"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )
        handler.train(X, y)
        return handler, X, y

    def test_predict_returns_array(self, trained_handler):
        """predict가 배열 반환"""
        handler, X, y = trained_handler
        predictions = handler.predict(X[:5])

        assert isinstance(predictions, (np.ndarray, list))

    def test_predict_length_matches_input(self, trained_handler):
        """예측값 개수가 입력 개수와 일치"""
        handler, X, y = trained_handler
        predictions = handler.predict(X[:5])

        assert len(predictions) == 5

    def test_predict_binary_classification(self, trained_handler):
        """예측값이 0 또는 1 (이진 분류)"""
        handler, X, y = trained_handler
        predictions = handler.predict(X[:10])

        assert set(predictions).issubset({0, 1})

    def test_predict_without_training_raises_error(self):
        """학습되지 않은 모델로 예측하면 에러"""
        handler = LogisticRegressionHandler(feature_names=['col1'])
        X = pd.DataFrame({'col1': [1, 2, 3]})

        with pytest.raises(ValueError, match="학습되지 않았습니다"):
            handler.predict(X)

    def test_predict_validates_features(self, trained_handler):
        """feature 검증"""
        handler, X, y = trained_handler

        X_wrong = pd.DataFrame({
            'wrong_col': [1, 2, 3]
        })

        with pytest.raises(ValueError):
            handler.predict(X_wrong)

    def test_predict_handles_different_input_sizes(self, trained_handler):
        """다양한 크기의 입력 처리"""
        handler, X, y = trained_handler

        for size in [1, 5, 10, len(X)]:
            predictions = handler.predict(X[:size])
            assert len(predictions) == size

    def test_predict_consistent_across_calls(self, trained_handler):
        """같은 입력에 대해 일관된 예측"""
        handler, X, y = trained_handler
        X_test = X[:5]

        pred1 = handler.predict(X_test)
        pred2 = handler.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


class TestLogisticRegressionHandlerSaveLoad:
    """모델 저장/로드 테스트"""

    @pytest.fixture
    def trained_handler(self, sample_training_data_ml):
        """미리 학습된 모델"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )
        handler.train(X, y)
        return handler, X, y

    def test_save_creates_file(self, trained_handler, temp_model_dir):
        """모델이 저장됨"""
        handler, X, y = trained_handler
        model_path = os.path.join(temp_model_dir, 'model.pkl')

        handler.save(model_path)

        assert os.path.exists(model_path)

    def test_load_returns_handler(self, trained_handler, temp_model_dir):
        """로드된 객체가 LogisticRegressionHandler"""
        handler, X, y = trained_handler
        model_path = os.path.join(temp_model_dir, 'model.pkl')

        handler.save(model_path)
        loaded_handler = LogisticRegressionHandler.load(model_path)

        assert isinstance(loaded_handler, LogisticRegressionHandler)

    def test_loaded_model_can_predict(self, trained_handler, temp_model_dir):
        """로드된 모델로 예측 가능"""
        handler, X, y = trained_handler
        model_path = os.path.join(temp_model_dir, 'model.pkl')

        handler.save(model_path)
        loaded_handler = LogisticRegressionHandler.load(model_path)

        predictions = loaded_handler.predict(X[:5])
        assert len(predictions) == 5

    def test_predictions_consistent_after_load(self, trained_handler, temp_model_dir):
        """로드 후 예측이 동일"""
        handler, X, y = trained_handler
        model_path = os.path.join(temp_model_dir, 'model.pkl')

        original_predictions = handler.predict(X[:5])

        handler.save(model_path)
        loaded_handler = LogisticRegressionHandler.load(model_path)
        loaded_predictions = loaded_handler.predict(X[:5])

        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_save_includes_metadata(self, trained_handler, temp_model_dir):
        """메타데이터도 저장됨"""
        handler, X, y = trained_handler
        model_path = os.path.join(temp_model_dir, 'model.pkl')

        handler.save(model_path)
        loaded_handler = LogisticRegressionHandler.load(model_path)

        assert 'best_cv_score' in loaded_handler.train_metrics
        assert 'saved_at' in loaded_handler.train_metrics


class TestLogisticRegressionHandlerFeatureValidation:
    """feature 검증 테스트"""

    @pytest.fixture
    def trained_handler(self, sample_training_data_ml):
        """미리 학습된 모델"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )
        handler.train(X, y)
        return handler

    def test_missing_columns_raise_error(self, trained_handler):
        """필요한 컬럼 누락 시 에러"""
        X_missing = pd.DataFrame({
            'adj_close': [1, 2, 3],
            # ma_5 누락
            'ma_20': [1, 2, 3],
            'ma_200': [1, 2, 3],
            'macd': [1, 2, 3],
        })

        with pytest.raises(ValueError, match="누락된 컬럼"):
            trained_handler.predict(X_missing)

    def test_extra_columns_raise_error(self, trained_handler):
        """불필요한 컬럼이 있으면 에러"""
        X_extra = pd.DataFrame({
            'adj_close': [1, 2, 3],
            'ma_5': [1, 2, 3],
            'ma_20': [1, 2, 3],
            'ma_200': [1, 2, 3],
            'macd': [1, 2, 3],
            'extra_col': [1, 2, 3],  # 불필요한 컬럼
        })

        with pytest.raises(ValueError, match="불필요한 컬럼"):
            trained_handler.predict(X_extra)

    def test_column_order_independent(self, trained_handler):
        """컬럼 순서는 상관없음"""
        X_reordered = pd.DataFrame({
            'macd': np.random.randn(10),
            'adj_close': np.random.randn(10),
            'ma_200': np.random.randn(10),
            'ma_5': np.random.randn(10),
            'ma_20': np.random.randn(10),
        })

        # 순서가 다르지만 컬럼이 모두 있음
        predictions = trained_handler.predict(X_reordered)
        assert len(predictions) == 10

    def test_validates_dataframe_type(self, trained_handler):
        """입력이 DataFrame이 아니면 에러"""
        X_numpy = np.random.randn(5, 5)

        with pytest.raises(ValueError, match="not pd.DataFrame"):
            trained_handler.predict(X_numpy)


class TestLogisticRegressionHandlerEdgeCases:
    """엣지 케이스 테스트"""

    def test_train_with_balanced_classes(self):
        """균형 잡힌 클래스로 학습"""
        np.random.seed(42)
        # TimeSeriesSplit(n_splits=5) 안정성을 위해 충분한 샘플 필요
        # 각 fold가 최소 30개 샘플을 가지려면 최소 300개 필요
        n_samples = 300
        X = pd.DataFrame({
            'f1': np.random.randn(n_samples),
            'f2': np.random.randn(n_samples),
        })
        # 시계열 균형 배치
        y = pd.Series(np.tile([0, 1], n_samples // 2))

        handler = LogisticRegressionHandler(feature_names=['f1', 'f2'])
        handler.train(X, y)

        assert handler.is_fitted is True

    def test_train_with_imbalanced_classes(self):
        """불균형 클래스로 학습"""
        np.random.seed(42)
        # TimeSeriesSplit(n_splits=5)이 각 fold에서 두 클래스를 모두 볼 수 있도록
        # 클래스를 interleave하여 배치
        n_samples = 300
        X = pd.DataFrame({
            'f1': np.random.randn(n_samples),
            'f2': np.random.randn(n_samples),
        })
        # 70:30 불균형 (60과 140이 아닌 더 안정적인 분포)
        # 클래스를 번갈아 배치하되 0을 더 많이
        y_list = []
        for i in range(n_samples):
            # 70% 확률로 0, 30% 확률로 1 (하지만 순차적으로 배치)
            y_list.append(0 if i % 10 < 7 else 1)
        y = pd.Series(y_list)

        handler = LogisticRegressionHandler(feature_names=['f1', 'f2'])
        handler.train(X, y)

        assert handler.is_fitted is True

    def test_predict_with_extreme_values(self):
        """극단값으로 예측"""
        X_train = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))

        handler = LogisticRegressionHandler(feature_names=['f1', 'f2'])
        handler.train(X_train, y_train)

        # 극단값
        X_extreme = pd.DataFrame({
            'f1': [1e6, -1e6],
            'f2': [1e6, -1e6],
        })

        predictions = handler.predict(X_extreme)
        assert len(predictions) == 2

    def test_train_eval_predict_workflow(self, sample_training_data_ml):
        """완전한 워크플로우"""
        X_all, y_all = sample_training_data_ml

        # Train/Test 분할
        split_idx = int(len(X_all) * 0.8)
        X_train = X_all[:split_idx]
        y_train = y_all[:split_idx]
        X_test = X_all[split_idx:]
        y_test = y_all[split_idx:]

        # 학습
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )
        handler.train(X_train, y_train)

        # 평가
        metrics = handler.evaluate(X_test, y_test)
        assert 'test_accuracy' in metrics

        # 예측
        predictions = handler.predict(X_test)
        assert len(predictions) == len(X_test)
