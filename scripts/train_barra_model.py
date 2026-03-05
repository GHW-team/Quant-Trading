"""
train_barra_model.py
====================
Barra 기반 ML 퀀트 파이프라인 오케스트레이션 스크립트.

실행 순서:
    1. 설정 로드 (barra_config.yaml)
    2. Feature CSV 로드
    3. ReturnCalculator로 월별/분기별 수익률 계산
    4. BarraRegression으로 종목별 market beta 추정
    5. BarraLabeler로 알파 기반 라벨 생성
    6. 날짜 기준 Train/Test 분할
    7. BarraLogisticHandler로 학습 + GridSearchCV
    8. 테스트셋 평가
    9. 모델 및 결과 저장

사용법:
    python scripts/train_barra_model.py
"""

import sys
import logging
from pathlib import Path

import yaml
import pandas as pd
from sklearn.metrics import classification_report

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.return_calculator import ReturnCalculator
from src.ml.barra_regression import BarraRegression
from src.ml.barra_labeler import BarraLabeler
from src.ml.barra_logistic import BarraLogisticHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ══════════════════════════════════════════════════════════════════════
    # 1. 설정 로드
    # ══════════════════════════════════════════════════════════════════════
    config_path = Path(__file__).parent.parent / "config" / "barra_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["barra_pipeline"]

    feature_csv = config["feature_csv"]
    db_path = config["db_path"]
    market_proxy = config["market_proxy"]

    ols_cfg = config["ols"]
    label_cfg = config["labeling"]
    logistic_cfg = config["logistic"]
    split_cfg = config["split"]
    output_cfg = config["output"]

    logger.info(f"설정 로드 완료: {config_path}")
    logger.info(f"  OLS lookback: {ols_cfg['lookback_months']}개월")
    logger.info(f"  라벨링 target_ratio: {label_cfg['target_ratio']}")
    logger.info(f"  Train/Test 분할: ~{split_cfg['train_end']} / {split_cfg['test_start']}~")

    # ══════════════════════════════════════════════════════════════════════
    # 2. Feature CSV 로드
    # ══════════════════════════════════════════════════════════════════════
    logger.info("Feature CSV 로드 중...")
    features_df = pd.read_csv(feature_csv)
    rebalance_dates = sorted(features_df["rebalance_date"].unique().tolist())
    symbols = features_df["symbol"].unique().tolist()

    logger.info(
        f"  {len(features_df)}행, {len(symbols)}종목, "
        f"{len(rebalance_dates)}개 리밸런싱 날짜"
    )
    logger.info(f"  날짜 범위: {rebalance_dates[0]} ~ {rebalance_dates[-1]}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. 수익률 계산 (한 번의 DB 로드로 통합)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("수익률 계산 중 (월별 + 분기별 + 무위험 통합 로드)...")
    rc = ReturnCalculator(db_path=db_path)

    monthly_returns, quarterly_returns, monthly_rf = rc.compute_all(
        symbols=symbols,
        rebalance_dates=rebalance_dates,
        lookback_months=ols_cfg["lookback_months"],
        market_proxy=market_proxy,
    )

    logger.info(
        f"  월별 수익률: {len(monthly_returns)}행, "
        f"무위험: {len(monthly_rf)}행, "
        f"분기별: {len(quarterly_returns)}행"
    )

    # ══════════════════════════════════════════════════════════════════════
    # 4. OLS 베타 추정
    # ══════════════════════════════════════════════════════════════════════
    logger.info("OLS 베타 추정 중...")
    barra_reg = BarraRegression(
        lookback_months=ols_cfg["lookback_months"],
        min_observations=ols_cfg["min_observations"],
        vif_threshold=ols_cfg["vif_threshold"],
        ridge_alpha=ols_cfg["ridge_alpha"],
    )

    betas = barra_reg.estimate_all_dates(
        monthly_returns=monthly_returns,
        monthly_risk_free=monthly_rf,
        rebalance_dates=rebalance_dates,
        market_proxy=market_proxy,
    )

    logger.info(f"  베타 추정 결과: {len(betas)}행")
    logger.info(f"  베타 통계: mean={betas['beta_market'].mean():.3f}, "
                f"std={betas['beta_market'].std():.3f}, "
                f"median={betas['beta_market'].median():.3f}")

    # 베타 결과 저장
    beta_path = Path(output_cfg["beta_csv"])
    beta_path.parent.mkdir(parents=True, exist_ok=True)
    betas.to_csv(beta_path, index=False)
    logger.info(f"  베타 저장: {beta_path}")

    # ══════════════════════════════════════════════════════════════════════
    # 5. 알파 기반 라벨링
    # ══════════════════════════════════════════════════════════════════════
    logger.info("알파 기반 라벨링 중...")
    labeler = BarraLabeler(target_ratio=label_cfg["target_ratio"])

    labeled_df = labeler.label(
        forward_returns=quarterly_returns,
        betas=betas,
        market_proxy=market_proxy,
    )

    # 라벨 결과 저장
    label_path = Path(output_cfg["label_csv"])
    label_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(label_path, index=False)
    logger.info(f"  라벨 저장: {label_path}")

    # ══════════════════════════════════════════════════════════════════════
    # 6. Feature + Label 병합 및 Train/Test 분할
    # ══════════════════════════════════════════════════════════════════════
    logger.info("Feature + Label 병합 및 분할 중...")

    # Feature와 Label 병합
    train_data = features_df.merge(
        labeled_df[["rebalance_date", "symbol", "label"]],
        on=["rebalance_date", "symbol"],
        how="inner",
    )

    # NaN 라벨 제거
    train_data = train_data.dropna(subset=["label"])
    train_data["label"] = train_data["label"].astype(int)

    logger.info(f"  병합 결과: {len(train_data)}행")
    logger.info(f"  라벨 분포: 1={train_data['label'].mean():.1%}, "
                f"0={1 - train_data['label'].mean():.1%}")

    # 날짜 기준 분할
    train_mask = train_data["rebalance_date"] <= split_cfg["train_end"]
    test_mask = train_data["rebalance_date"] >= split_cfg["test_start"]

    train_set = train_data[train_mask].copy()
    test_set = train_data[test_mask].copy()

    logger.info(f"  Train: {len(train_set)}행 ({train_set['rebalance_date'].nunique()}분기)")
    logger.info(f"  Test:  {len(test_set)}행 ({test_set['rebalance_date'].nunique()}분기)")

    if train_set.empty or test_set.empty:
        logger.error("Train 또는 Test 데이터가 비어있습니다. 분할 날짜를 확인하세요.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # 7. 로지스틱 회귀 학습
    # ══════════════════════════════════════════════════════════════════════
    logger.info("로지스틱 회귀 학습 중...")

    handler = BarraLogisticHandler(
        interaction_types=logistic_cfg["interaction_types"],
    )

    X_train = train_set
    y_train = train_set["label"]
    dates_train = train_set["rebalance_date"]

    handler.train(
        X=X_train,
        y=y_train,
        rebalance_dates=dates_train,
        penalty_search=logistic_cfg["penalty_search"],
        C_search=logistic_cfg["C_search"],
        l1_ratio_search=logistic_cfg["l1_ratio_search"],
        cv_splits=logistic_cfg["cv_splits"],
        max_iter=logistic_cfg["max_iter"],
    )

    # 모델 저장
    model_path = Path(output_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    handler.save(str(model_path))
    logger.info(f"  모델 저장: {model_path}")

    # ══════════════════════════════════════════════════════════════════════
    # 8. 테스트셋 평가
    # ══════════════════════════════════════════════════════════════════════
    logger.info("테스트셋 평가 중...")

    X_test = test_set
    y_test = test_set["label"]

    metrics = handler.evaluate(X_test, y_test)
    y_pred = handler.predict(X_test)

    print(f"\n{'='*60}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"{'='*60}")
    print(f"\n[Classification Report]")
    print(classification_report(y_test, y_pred))

    # ══════════════════════════════════════════════════════════════════════
    # 9. Feature Importance
    # ══════════════════════════════════════════════════════════════════════
    importance = handler.get_feature_importance()
    print(f"\n[Top 20 Features by |coefficient|]")
    print(importance.head(20).to_string(index=False))

    n_nonzero = (importance["coefficient"] != 0).sum()
    n_total = len(importance)
    print(f"\nNon-zero coefficients: {n_nonzero}/{n_total}")

    # Train 결과 요약
    print(f"\n{'='*60}")
    print(f"[학습 결과 요약]")
    print(f"  Best CV Score: {handler.train_metrics['best_cv_score']:.4f}")
    print(f"  Best Params: {handler.train_metrics['best_params']}")
    print(f"  Features: {handler.train_metrics['n_features']}")
    print(f"  Train Samples: {handler.train_metrics['n_samples']}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
