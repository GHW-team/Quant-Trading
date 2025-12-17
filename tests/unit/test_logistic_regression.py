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
from sklearn.preprocessing import StandardScaler

# train
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

# evaluate
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
    # ==================================
    # 함수 정상 케이스 테스트
    # ==================================

    def test_evaluate_has_accuracy(self, trained_handler):
        """결과에 test_accuracy, count 키가 있음"""
        handler, X, y = trained_handler
        result = handler.evaluate(X, y)

        assert isinstance(result, dict)
        assert 'test_accuracy' in result
        assert 'count' in result
        assert result['count'] == len(X)

    # ==================================
    # 함수 에러 케이스 테스트
    # ==================================

    def test_evaluate_without_training_raises_error(self):
        """학습되지 않은 모델로 평가하면 에러"""
        handler = LogisticRegressionHandler(feature_names=['col1'])
        X = pd.DataFrame({'col1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])

        with pytest.raises(ValueError, match="학습되지 않았습니다"):
            handler.evaluate(X, y)

    def test_evaluate_validates_features(self, trained_handler):
        """feature 검증"""
        handler, X, y = trained_handler

        # 잘못된 컬럼
        X_wrong = pd.DataFrame({
            'wrong_col': [1, 2, 3]
        })

        with pytest.raises(ValueError):
            handler.evaluate(X_wrong, pd.Series([0, 1, 0]))

# predict
class TestLogisticRegressionHandlerPredict:
    """모델 예측 테스트"""
    # ==================================
    # 함수 정상 케이스 테스트
    # ==================================
    @pytest.fixture
    def trained_handler(self, sample_training_data_ml):
        """미리 학습된 모델"""
        X, y = sample_training_data_ml
        handler = LogisticRegressionHandler(
            feature_names=['adj_close', 'ma_5', 'ma_20', 'ma_200', 'macd']
        )
        handler.train(X, y)
        return handler, X, y

    # ==================================
    # 함수 정상 케이스 테스트
    # ==================================
    def test_predict_returns_array(self, trained_handler):
        """predict가 배열 반환"""
        handler, X, y = trained_handler
        predictions = handler.predict(X[:5])

        assert isinstance(predictions, np.ndarray)

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

    # ==================================
    # 함수 에러 케이스 테스트
    # ==================================
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

# save / load
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

    # ==================================
    # 함수 정상 케이스 테스트
    # ==================================
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
        """메타데이터 저장/로드 전 후 상태가 동일한지 검증"""
        handler, X, y = trained_handler
        model_path = os.path.join(temp_model_dir, 'model.pkl')

        handler.save(model_path)
        loaded_handler = LogisticRegressionHandler.load(model_path)

        # 1. 기본 속성(설정값, 상태) 비교 - Python 기본 타입
        assert handler.feature_names == loaded_handler.feature_names
        assert handler.is_fitted == loaded_handler.is_fitted
        assert handler.train_metrics == loaded_handler.train_metrics

        # 2. Scaler 학습 상태 비교 (평균, 분산 등) - Numpy 배열
        # 스케일러가 기억하는 '평균'과 '표준편차'가 소수점 단위까지 같은지 확인
        np.testing.assert_array_almost_equal(handler.scaler.mean_, loaded_handler.scaler.mean_)
        np.testing.assert_array_almost_equal(handler.scaler.scale_, loaded_handler.scaler.scale_)

        # 3. Model 학습 상태 비교 (가중치, 절편) - Numpy 배열
        # 모델의 '뇌'에 해당하는 가중치(Coefficient)가 같은지 확인
        np.testing.assert_array_almost_equal(handler.model.coef_, loaded_handler.model.coef_)
        np.testing.assert_array_almost_equal(handler.model.intercept_, loaded_handler.model.intercept_)

# _validate_features
class TestLogisticRegressionHandlerFeatureValidation:
    """feature 검증 테스트"""

    # ==================================
    # 함수 정상 케이스 테스트
    # ==================================
    
    def test_valid_features_pass(self):
        """정확한 컬럼이면 에러 없이 통과"""
        handler = LogisticRegressionHandler(
            feature_names=['col1', 'col2', 'col3']
        )
        
        X_valid = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
        })
        
        # 에러 없이 통과해야 함
        handler._validate_features(X_valid)
    
    def test_missing_columns_raise_error(self):
        """필요한 컬럼 누락 시 ValueError"""
        handler = LogisticRegressionHandler(
            feature_names=['col1', 'col2', 'col3']
        )
        
        X_missing = pd.DataFrame({
            'col1': [1, 2, 3],
            # col2 누락
            'col3': [7, 8, 9],
        })
        
        with pytest.raises(ValueError, match="누락된 컬럼"):
            handler._validate_features(X_missing)
    
    def test_extra_columns_raise_error(self):
        """불필요한 컬럼이 있으면 ValueError"""
        handler = LogisticRegressionHandler(
            feature_names=['col1', 'col2']
        )
        
        X_extra = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'extra_col': [7, 8, 9],  # 불필요한 컬럼
        })
        
        with pytest.raises(ValueError, match="불필요한 컬럼"):
            handler._validate_features(X_extra)
    # ==================================
    # 함수 에러 케이스 테스트
    # ==================================
    
    def test_not_dataframe_raises_error(self):
        """입력이 DataFrame이 아니면 ValueError"""
        handler = LogisticRegressionHandler(
            feature_names=['col1']
        )
        
        # Numpy array 입력
        X_numpy = np.array([[1, 2, 3]])
        
        with pytest.raises(ValueError, match="not pd.DataFrame"):
            handler._validate_features(X_numpy)
            