"""
레짐 분류 모델 학습 스크립트

실행: python scripts/train_regime_model.py
"""
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_fetcher import StockDataFetcher
from src.data.indicator_calculator import IndicatorCalculator
from src.data.feature_engineer import FeatureEngineer
from src.ml.regime_labeler import RegimeLabeler
from src.ml.regime_model import RegimeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # ---- 1. SPY 데이터 수집 ----
    logger.info("Step 1: SPY 데이터 수집")
    fetcher = StockDataFetcher(max_workers=1)
    spy_dict = fetcher.fetch_multiple_by_date(
        ticker_list=['SPY'],
        start_date='2015-01-01',
        end_date='2025-12-31',
    )
    spy_df = spy_dict['SPY']
    logger.info(f"  수집 완료: {len(spy_df)} rows")
    
    # ---- 2. 기본 지표 계산 ----
    logger.info("Step 2: 기본 지표 계산")
    calc = IndicatorCalculator()
    spy_df = calc.calculate_indicators(spy_df, indicator_list=[
        'ma_50', 'ma_200', 'rsi', 'bb_upper', 'bb_mid', 'bb_lower', 'hv'
        # 'adx'도 indicator_calculator에 추가했으면 여기에 넣기
    ])
    
    # ---- 3. ML 피처 생성 ----
    logger.info("Step 3: ML 피처 생성")
    fe = FeatureEngineer()
    spy_df = fe.create_features(spy_df)
    
    # ---- 4. 라벨링 ----
    logger.info("Step 4: 레짐 라벨링")
    labeler = RegimeLabeler(horizon=20, threshold=0.01)
    spy_df = labeler.label_data(spy_df)
    spy_df = spy_df.dropna(subset=['label'])
    
    # ---- 5. 학습/테스트 분할 (Walk-forward) ----
    logger.info("Step 5: 데이터 분할 (시간순)")
    split_date = '2023-01-01'
    train_df = spy_df[spy_df['date'] < split_date]
    test_df = spy_df[spy_df['date'] >= split_date]
    
    feature_names = fe.get_feature_names()
    X_train = train_df[feature_names]
    y_train = train_df['label'].astype(int)
    X_test = test_df[feature_names]
    y_test = test_df['label'].astype(int)
    
    logger.info(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    
    # ---- 6. 모델 학습 ----
    logger.info("Step 6: 모델 학습")
    model = RegimeModel(feature_names=feature_names)
    model.train(X_train, y_train)
    
    # ---- 7. 평가 ----
    logger.info("Step 7: 모델 평가")
    eval_result = model.handler.evaluate(X_test, y_test)
    logger.info(f"  테스트 정확도: {eval_result['test_accuracy']:.4f}")
    
    # ---- 8. 저장 ----
    save_path = 'data/models/regime_classifier.pkl'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    logger.info(f"  모델 저장 완료: {save_path}")


if __name__ == '__main__':
    main()
