import sys
import yaml
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 프로젝트 루트 경로 추가 (src 모듈을 찾기 위함)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pipeline import DataPipeline
from src.ml.labeler import Labeler
from src.ml.logistic_regression import LogisticRegressionHandler

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. config.yaml 읽기
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 설정값 추출 (model 섹션만 사용 - train_model.py용)
    # database_path는 data 섹션에서만 가져옴 (DB 로드용)
    database_path = config['data']['database_path']

    # model.training 설정
    tickers = config['model']['training']['tickers']
    period = config['model']['training'].get('period')
    start_date = config['model']['training'].get('start_date')
    end_date = config['model']['training'].get('end_date')

    horizon = config['model']['labeling']['horizon']
    threshold = config['model']['labeling']['threshold']

    feature_columns = config['model']['features']['columns']
    indicator_list = config['model']['features']['indicators']

    output_dir = config['model']['output']['dir']
    output_file = config['model']['output']['file']
    output_path = f"{output_dir}/{output_file}"

    # 변수 정의 (편의상 대문자 유지)
    FEATURE_COLUMNS = feature_columns
    INDICATOR_LIST = indicator_list

    logger.info(f"✅ Config loaded from {config_path}")
    logger.info(f"   - Tickers: {tickers}")
    if period:
        logger.info(f"   - Period: {period}")
    else:
        logger.info(f"   - Date Range: {start_date} ~ {end_date}")
    logger.info(f"   - Horizon: {horizon}, Threshold: {threshold:.1%}")
    logger.info(f"   - Features: {FEATURE_COLUMNS}\n")

    # ==========================================
    # 2. 데이터 파이프라인으로 데이터 로드
    # ==========================================
    logger.info("2. Loading Data from Pipeline...")

    try:
        with DataPipeline(db_path=database_path) as pipeline:
            # DataPipeline의 run_full_pipeline 사용 (model.training 설정값 적용)
            pipeline_kwargs = {
                'ticker_list': tickers,
                'indicator_list': INDICATOR_LIST,
                'update_if_exists': True,
            }

            df_dict = pipeline.run_full_pipeline(
                ticker_list=tickers,
                indicator_list=INDICATOR_LIST,
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

        if not df_dict:
            logger.error("❌ DataPipeline에서 데이터를 로드할 수 없습니다. 데이터 설정을 확인하세요.")
            sys.exit(1)

        # 모든 종목의 데이터를 하나로 병합
        dfs = list(df_dict.values())
        df_all = pd.concat(dfs, ignore_index=True)
        logger.info(f"   ✓ Loaded {len(df_all)} records for {tickers}")

    except Exception as e:
        logger.error(f"❌ DataPipeline 실행 중 오류: {e}")
        sys.exit(1)

    # ==========================================
    # 3. 라벨링 (Labeling)
    # ==========================================
    logger.info(f"3. Labeling Data (Horizon={horizon}, Threshold={threshold:.1%})...")

    labeler = Labeler(horizon=horizon, threshold=threshold)
    df_labeled = labeler.label_data(df_all, price_col='adj_close')

    # ==========================================
    # 4. 데이터 전처리 및 분할 (Preprocessing)
    # ==========================================
    logger.info("4. Splitting & Cleaning Data...")

    # (1) NaN 제거: Feature나 Label이 없는 행은 학습 불가
    # ModelHandler에게 깨끗한 데이터를 주기 위한 필수 과정
    clean_data = df_labeled.dropna(subset=FEATURE_COLUMNS + ['label'])

    if clean_data.empty:
        logger.error("❌ 유효한 데이터가 없습니다 (모두 NaN). 기간을 늘려보세요.")
        sys.exit(1)

    # (2) Train / Test 분할 (시계열이므로 섞지 않음)
    train_df, test_df = train_test_split(clean_data, test_size=0.2, shuffle=False)

    # (3) X, y 분리 (ModelHandler.train 입력용)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['label']

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['label']

    logger.info(f"   - Train Samples: {len(train_df)}")
    logger.info(f"   - Test Samples:  {len(test_df)}")

    # ==========================================
    # 5. 모델 학습 (Training with Handler)
    # ==========================================
    logger.info("5. Initializing & Training ModelHandler...")

    handler = LogisticRegressionHandler(feature_names=FEATURE_COLUMNS)

    # [핵심] GridSearch + TimeSeriesSplit 교차검증 수행
    handler.train(X_train, y_train)

    # 모델 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    handler.save(output_path)
    logger.info(f"✅ Model Saved: {output_path}")

    # ==========================================
    # 6. 최종 평가 (Evaluation)
    # ==========================================
    logger.info("6. Evaluating on Test Set...")

    metrics = handler.evaluate(X_test, y_test)
    print(f"\n📊 Test Accuracy: {metrics['test_accuracy']:.4f}")
    
    # 상세 리포트 출력 (Precision, Recall 등 확인)
    # predict() 메서드는 Backtrader용이지만 여기서도 검증용으로 사용 가능
    y_pred = handler.predict(test_df[FEATURE_COLUMNS])
    y_true = test_df['label']
    
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()