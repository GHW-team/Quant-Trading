"""
레짐 분류 모델 (Logistic Regression 기반)

- 기존 LogisticRegressionHandler를 상속/래핑하여 사용
- 레짐 전용 피처 목록 관리
- predict_proba를 통해 p_trend 확률 반환
"""
import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Dict, Optional

from src.ml.logistic_regression import LogisticRegressionHandler

logger = logging.getLogger(__name__)


# 레짐 분류에 사용할 피처 목록
REGIME_FEATURES = [
    'ma_spread',       # x1: (MA50 - MA200) / MA200
    'ma_50_slope',     # x2: MA50의 10일 변화율
    'price_vs_ma200',  # x3: 가격 대비 MA200 괴리율
    'ret_20',          # x4: 1개월 수익률
    'ret_60',          # x5: 3개월 수익률
    'vol_20',          # x6: 20일 실현 변동성
    'bb_width',        # x7: 볼린저 밴드 폭
    'adx',             # x8: 추세 강도
]


class RegimeModel:
    """
    레짐 분류 전용 모델
    
    내부적으로 LogisticRegressionHandler를 사용합니다.
    predict_regime()으로 p_trend(추세 확률)를 반환합니다.
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or REGIME_FEATURES
        self.handler = LogisticRegressionHandler(feature_names=self.feature_names)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> 'RegimeModel':
        """
        레짐 분류기 학습
        
        Args:
            X: 피처 데이터프레임 (REGIME_FEATURES 컬럼 필요)
            y: 라벨 (1=Trend, 0=Range)
        """
        self.handler.train(X, y)
        logger.info("레짐 분류 모델 학습 완료")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        추세 확률(p_trend) 반환
        
        Returns:
            np.ndarray: 각 행의 P(Trend) 확률값 (0~1)
        """
        if not self.handler.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        X_valid = X[self.feature_names].copy()
        X_scaled = self.handler.scaler.transform(X_valid)
        
        # predict_proba[:,1] = P(class=1) = P(Trend)
        return self.handler.model.predict_proba(X_scaled)[:, 1]
    
    def predict_regime(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        레짐 예측 (이진 분류)
        
        Args:
            threshold: 분류 임계값 (기본 0.5)
            
        Returns:
            np.ndarray: 1=Trend, 0=Range
        """
        p_trend = self.predict_proba(X)
        return (p_trend >= threshold).astype(int)
    
    def save(self, path: str):
        """모델 저장"""
        joblib.dump(self, path)
        logger.info(f"레짐 모델 저장: {path}")
    
    @staticmethod
    def load(path: str) -> 'RegimeModel':
        """모델 불러오기"""
        model = joblib.load(path)
        logger.info(f"레짐 모델 로드: {path}")
        return model
