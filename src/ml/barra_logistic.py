"""
barra_logistic.py
=================
교차항(interaction terms) 포함 로지스틱 회귀 핸들러.

Feature 구성:
  - 기본: 시장(1) + 산업 더미(24) + 스타일 Z-score(8) = 33개
  - 교차항:
    - style × style: C(8,2) = 28개
    - industry × style: 24 × 8 = 192개
    - market × style: 1 × 8 = 8개
    - market × industry: 1 × 24 = 24개
  - 총: ~285개

정규화: L1, L2, ElasticNet 중 GridSearchCV로 최적 선택.

사용법:
    from src.ml.barra_logistic import BarraLogisticHandler

    handler = BarraLogisticHandler(
        style_features=STYLE_Z_COLS,
        industry_features=INDUSTRY_COLS,
    )
    handler.train(X_train, y_train, date_col_for_cv=dates_train)
    metrics = handler.evaluate(X_test, y_test)
"""

import logging
from itertools import combinations
from typing import List, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

# Feature CSV의 기본 컬럼 목록
DEFAULT_STYLE_Z_COLS = [
    "value_bp_z", "value_ep_z", "size_z", "profitability_z",
    "investment_z", "low_vol_z", "stability_z", "momentum_z",
]

DEFAULT_INDUSTRY_COLS = [
    "industry_Comm_Media_Entertainment", "industry_Comm_Telecom",
    "industry_Cons_Disc_Auto", "industry_Cons_Disc_Distribution",
    "industry_Cons_Disc_Durables", "industry_Cons_Disc_Services",
    "industry_Cons_Staples_Food", "industry_Cons_Staples_Personal",
    "industry_Cons_Staples_Retail", "industry_Energy",
    "industry_Fin_Banks", "industry_Fin_Financial_Services",
    "industry_Fin_Insurance", "industry_HC_Equipment_Services",
    "industry_HC_Pharma_Bio", "industry_IT_Hardware_Equipment",
    "industry_IT_Semiconductors", "industry_IT_Software_Services",
    "industry_Ind_Capital_Goods", "industry_Ind_Commercial_Services",
    "industry_Ind_Transportation", "industry_Materials",
    "industry_RE_Development", "industry_RE_REIT",
    # industry_Utilities 제거 (다중공선성 방지를 위해 하나 drop)
]

DEFAULT_MARKET_COL = "market_excess_return"


class BarraLogisticHandler:
    """교차항 포함 로지스틱 회귀 핸들러."""

    def __init__(
        self,
        style_features: Optional[List[str]] = None,
        industry_features: Optional[List[str]] = None,
        market_feature: str = DEFAULT_MARKET_COL,
        interaction_types: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        style_features : 스타일 Z-score 컬럼 리스트
        industry_features : 산업 더미 컬럼 리스트
        market_feature : 시장 팩터 컬럼명
        interaction_types : 교차항 유형 리스트
            ["style_style", "industry_style", "market_style", "market_industry"]
        """
        self.style_features = style_features or DEFAULT_STYLE_Z_COLS
        self.industry_features = industry_features or DEFAULT_INDUSTRY_COLS
        self.market_feature = market_feature
        self.interaction_types = interaction_types or [
            "style_style", "industry_style", "market_style", "market_industry",
        ]

        self.base_features: List[str] = []
        self.interaction_feature_names: List[str] = []
        self.all_feature_names: List[str] = []
        self._build_feature_names()

        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.train_metrics: Dict = {}

    def _build_feature_names(self):
        """기본 + 교차항 feature 이름 목록을 구성한다."""
        self.base_features = (
            [self.market_feature] + self.industry_features + self.style_features
        )

        interaction_names = []

        if "style_style" in self.interaction_types:
            for a, b in combinations(self.style_features, 2):
                interaction_names.append(f"{a}_x_{b}")

        if "industry_style" in self.interaction_types:
            for ind in self.industry_features:
                for sty in self.style_features:
                    interaction_names.append(f"{ind}_x_{sty}")

        if "market_style" in self.interaction_types:
            for sty in self.style_features:
                interaction_names.append(f"{self.market_feature}_x_{sty}")

        if "market_industry" in self.interaction_types:
            for ind in self.industry_features:
                interaction_names.append(f"{self.market_feature}_x_{ind}")

        self.interaction_feature_names = interaction_names
        self.all_feature_names = self.base_features + interaction_names

        logger.info(
            f"Feature 구성: 기본 {len(self.base_features)}개 + "
            f"교차항 {len(interaction_names)}개 = 총 {len(self.all_feature_names)}개"
        )

    def build_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        기본 feature DataFrame에 교차항 컬럼을 추가한다.

        Parameters
        ----------
        X : 기본 feature를 포함하는 DataFrame

        Returns
        -------
        교차항이 추가된 DataFrame
        """
        interaction_cols = {}

        if "style_style" in self.interaction_types:
            for a, b in combinations(self.style_features, 2):
                interaction_cols[f"{a}_x_{b}"] = X[a].values * X[b].values

        if "industry_style" in self.interaction_types:
            for ind in self.industry_features:
                ind_vals = X[ind].values
                for sty in self.style_features:
                    interaction_cols[f"{ind}_x_{sty}"] = ind_vals * X[sty].values

        if "market_style" in self.interaction_types:
            mkt_vals = X[self.market_feature].values
            for sty in self.style_features:
                interaction_cols[f"{self.market_feature}_x_{sty}"] = mkt_vals * X[sty].values

        if "market_industry" in self.interaction_types:
            mkt_vals = X[self.market_feature].values
            for ind in self.industry_features:
                interaction_cols[f"{self.market_feature}_x_{ind}"] = mkt_vals * X[ind].values

        # pd.concat로 한 번에 합치기 (단편화 방지)
        if interaction_cols:
            interaction_df = pd.DataFrame(interaction_cols, index=X.index)
            return pd.concat([X, interaction_df], axis=1)
        return X.copy()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        rebalance_dates: Optional[pd.Series] = None,
        penalty_search: Optional[List[str]] = None,
        C_search: Optional[List[float]] = None,
        l1_ratio_search: Optional[List[float]] = None,
        cv_splits: int = 5,
        max_iter: int = 5000,
    ) -> "BarraLogisticHandler":
        """
        교차항 생성 + StandardScaler + GridSearchCV로 학습한다.

        Parameters
        ----------
        X : 기본 feature DataFrame (교차항은 내부에서 생성)
        y : 라벨 Series
        rebalance_dates : 각 행의 리밸런싱 날짜 (TimeSeriesSplit 정렬용)
        penalty_search : 정규화 유형 탐색 리스트
        C_search : C 파라미터 탐색 리스트
        l1_ratio_search : ElasticNet l1_ratio 탐색 리스트
        cv_splits : TimeSeriesSplit fold 수
        max_iter : 최대 반복 횟수
        """
        penalties = penalty_search or ["l1", "l2", "elasticnet"]
        C_values = C_search or [0.001, 0.01, 0.1, 1.0, 10.0]
        l1_ratios = l1_ratio_search or [0.1, 0.3, 0.5, 0.7, 0.9]

        # 교차항 생성
        X_full = self.build_interaction_features(X)

        # NaN 처리: Z-score NaN → 0 (평균 노출)
        X_full = X_full[self.all_feature_names].fillna(0)

        # 시계열 순서 정렬 (rebalance_dates 기준)
        if rebalance_dates is not None:
            sort_idx = rebalance_dates.argsort()
            X_full = X_full.iloc[sort_idx].reset_index(drop=True)
            y = y.iloc[sort_idx].reset_index(drop=True)

        # 스케일링
        X_scaled = self.scaler.fit_transform(X_full)

        # GridSearchCV 파라미터 그리드 구성
        # saga solver는 L1, L2, ElasticNet 모두 지원
        param_grid = []

        if "l1" in penalties:
            param_grid.append({
                "penalty": ["l1"],
                "C": C_values,
                "solver": ["saga"],
                "max_iter": [max_iter],
                "class_weight": [None, "balanced"],
            })

        if "l2" in penalties:
            param_grid.append({
                "penalty": ["l2"],
                "C": C_values,
                "solver": ["saga"],
                "max_iter": [max_iter],
                "class_weight": [None, "balanced"],
            })

        if "elasticnet" in penalties:
            param_grid.append({
                "penalty": ["elasticnet"],
                "C": C_values,
                "solver": ["saga"],
                "l1_ratio": l1_ratios,
                "max_iter": [max_iter],
                "class_weight": [None, "balanced"],
            })

        cv = TimeSeriesSplit(n_splits=cv_splits)

        logger.info(
            f"GridSearchCV 시작: {len(param_grid)}개 그리드, "
            f"CV={cv_splits} folds, 데이터={X_scaled.shape}"
        )

        grid_search = GridSearchCV(
            estimator=LogisticRegression(random_state=42),
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_scaled, y)

        self.model = grid_search.best_estimator_
        self.train_metrics = {
            "best_cv_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "n_features": X_scaled.shape[1],
            "n_samples": X_scaled.shape[0],
        }
        self.is_fitted = True

        logger.info(
            f"학습 완료. Best CV Accuracy: {grid_search.best_score_:.4f}, "
            f"Best Params: {grid_search.best_params_}"
        )
        return self

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """테스트 데이터로 성능 평가."""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X_full = self.build_interaction_features(X)
        X_full = X_full[self.all_feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_full)

        score = self.model.score(X_scaled, y)
        return {"test_accuracy": score, "count": len(X)}

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """예측 (자동 교차항 생성 + 스케일링)."""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X_full = self.build_interaction_features(X)
        X_full = X_full[self.all_feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_full)

        if threshold == 0.5:
            return self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """클래스 확률 반환."""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X_full = self.build_interaction_features(X)
        X_full = X_full[self.all_feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_full)

        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """학습된 coefficient를 반환한다. L1으로 0이 된 feature도 포함."""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        coefs = self.model.coef_[0]
        df = pd.DataFrame({
            "feature": self.all_feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }).sort_values("abs_coefficient", ascending=False)

        n_nonzero = (df["coefficient"] != 0).sum()
        logger.info(
            f"Feature importance: {n_nonzero}/{len(df)} features have non-zero coefficients"
        )
        return df.reset_index(drop=True)

    def save(self, path: str):
        """모델 저장."""
        self.train_metrics["saved_at"] = pd.Timestamp.now().isoformat()
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str) -> "BarraLogisticHandler":
        """모델 로드."""
        handler = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return handler
