import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Dict, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

class LogisticRegressionHandler:
    """
    [모델 통합 관리자] - 최종 경량화 버전
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.model = None       # 학습 후 결정됨
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.train_metrics = {} 

    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
    ) -> 'LogisticRegressionHandler':
        """
        [통합 학습 메서드]
        스케일링 -> GridSearch(자동 튜닝) -> CV(교차 검증) -> 최적 모델 학습
        """
        # 1. 데이터 검증
        self._validate_features(X)
        #column 순서 맞추기
        X = X[self.feature_names]
        
        # 2. 스케일링 (Fit & Transform)
        X_scaled = self.scaler.fit_transform(X)

        # 3. 학습 진행
        logger.info("🚀 Training with (GridSearch + CV)...")
        
        # 실험할 설정들 (이 범위 안에서 제일 좋은 걸 찾음)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100], 
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        }
        
        # 시계열 교차 검증 (미래 데이터 참조 방지)
        cv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            estimator=LogisticRegression(random_state=42),
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy', #모델 평가 기준
            n_jobs=-1, #cpu 코어 수 조절
            verbose=1 #로그 깊이
        )
        grid_search.fit(X_scaled, y)
        
        self.model = grid_search.best_estimator_
        self.train_metrics = {
            'best_cv_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        logger.info(f"✅ Train Complete. Best CV Score: {grid_search.best_score_:.4f}")

        self.is_fitted = True
        return self

    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series) -> Dict[str, float]:
        """
        [최종 평가] 학습에 쓰지 않은 테스트 데이터(수능)로 성능 확인
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 데이터 검증
        self._validate_features(X)
        # 컬럼 순서 맞추기 
        X = X[self.feature_names]
        
        # 학습된 스케일러로 변환만 수행 (Transform Only)
        X_scaled = self.scaler.transform(X)
        
        score = self.model.score(X_scaled, y)
        return {'test_accuracy': score, 'count': len(X)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """[실전/백테스트] 예측 (자동 스케일링)"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        #데이터 검증
        self._validate_features(X)
        #컬럼 순서 맞추기
        X = X[self.feature_names]

        # 자동 스케일링 후 예측
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: str):
        """통합 저장"""
        self.train_metrics['saved_at'] = pd.Timestamp.now().isoformat()
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str) -> 'LogisticRegressionHandler':
        """불러오기"""
        handler = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return handler

    def _validate_features(self, X: pd.DataFrame):
        """데이터 컬럼 구성 검증 (안전장치)"""
        if not isinstance(X, pd.DataFrame): 
            raise ValueError(f"X is not pd.DataFrame! X type: {type(X)}")
        
        required = set(self.feature_names)
        input = set(X.columns)
        if required != input:
            miss_columns = required - input
            extra_columns = input - required

            error_msg = f"입력된 데이터의 컬럼이 모델학습에 필요한 컬럼과 일치하지 않습니다!"
            if miss_columns:
                error_msg += f"\n - 누락된 컬럼: {miss_columns}"
            if extra_columns:
                error_msg += f"\n - 불필요한 컬럼(제거 필요): {extra_columns}"
            raise ValueError(error_msg)