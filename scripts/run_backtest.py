# scripts/run_backtest.py
# 백테스트 실행용 CLI (config.yaml + CLI override + ML 모델 연동)

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import yaml
import joblib

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.runner import BacktestRunner
from src.backtest.strategy import MLSignalStrategy, BuyAndHoldStrategy
from src.data.db_manager import DatabaseManager
from src.ml.logistic_regression import LogisticRegressionHandler
from src.data.pipeline import DataPipeline
from src.data.all_ticker import TickerUniverse

CONFIG_PATH = "config/backtest.yaml"


# ============================================
# 로깅 설정
# ============================================
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


# ============================================
# 설정 로더
# ============================================
def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """config.yaml 로드"""
    config_file = Path(config_path)
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ============================================
# ML 신호 생성
# ============================================
def generate_ml_signals(
    df_dict: Dict[str,pd.DataFrame],
    model_path: str,
) -> Dict[str, pd.Series]:
    """
    학습된 ML 모델로 매수 신호 생성
    
    Args:
        df_dict: Dict[ticker, df of ticker]
            티커와 그 티커에 대한 feature값들이 포함된 DataFrame
        model_path: 모델 파일 경로
    
    Returns:
        {ticker: pd.Series(index=date, values=0/1)} 형태의 신호 딕셔너리
    """
    logger = logging.getLogger(__name__)
    signals = {}
    
    # 모델 로드
    if not Path(model_path).exists():
        logger.warning(f"모델 파일이 없습니다: {model_path}")
        logger.warning("신호 없이 백테스트를 진행합니다 (모든 날짜 signal=0)")
        return signals
    
    try:
        handler = joblib.load(model_path)
        logger.info(f"✓ 모델 로드 완료: {model_path}")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return signals
    
    #피쳐 컬럼 불러오기
    feature_columns = handler.feature_names

    #티커리스트 불러오기
    ticker_list = list(df_dict.keys())

    logger.info(f"📊 총 {len(ticker_list)}개 티커에 대해 ML 신호 생성 시작")
    success_count = 0
    fail_count = 0

    for ticker in ticker_list:
        try:
            df = df_dict[ticker]

            # ML 입력값 선택
            X = df[feature_columns].copy()

            # volume = 0 인 행은 신호를 NaN으로 설정
            volume_zero_mask = (df['volume']== 0)

            # NaN 이 있는 행은 신호를 NaN으로 설정
            nan_mask = X.isnull().any(axis=1)
            
            # 예측 가능한 행만 선택
            valid_mask = ~(volume_zero_mask | nan_mask)
            X_valid = X[valid_mask]

            # 전체 인덱스에 대한 신호 시리즈 생성 (기본값 NaN)
            signal_series = pd.Series(
                np.nan,
                index=pd.to_datetime(df['date']),
                name='signal'
            )

            # 예측 가능한 행이 있는 경우에만 예측 수행
            if len(X_valid) > 0:
                # 예측
                predictions = handler.predict(X_valid, threshold=0.57)

                # 예측 가능한 행에만 예측값 할당
                valid_dates = df.loc[valid_mask, 'date']
                signal_series.loc[valid_dates] = predictions
            else:
                # 예측 가능한 행이 0개인 경우 - 모든 행이 NaN (이미 기본값)
                logger.warning(f"{ticker}: 예측 가능한 행이 0개 - 모든 signal=NaN")

            signals[ticker] = signal_series

            # NaN 로깅
            nan_count = signal_series.isna().sum()
            if nan_count > 0:
                logger.info(f"{ticker}: {nan_count}개 행에 signal=NaN (volume=0 또는 feature NaN)")
            
            # 통계 출력
            buy_signals = (signal_series == 1).sum()
            total_days = len(signal_series)
            logger.info(f"{ticker}: {buy_signals}/{total_days} 매수 신호 ({buy_signals/total_days:.1%})")
            success_count += 1

        except Exception as e:
            logger.warning(f"{ticker}: 신호 생성 실패 - {e}")
            fail_count += 1
            continue

    logger.info(f"✅ 신호 생성 완료 - 성공: {success_count}개, 실패: {fail_count}개")
    return signals

# ============================================
# 메인 실행
# ============================================
def main():
    # ============================
    # 변수 설정 (Config 로드)
    # ============================

    # Config 로드
    config = load_config(CONFIG_PATH)

    # 로그 설정
    logger = setup_logging(config['log_level'])

    # 1.데이터 로드
    data_config = config['data']

    db_path = data_config['db_path']
    ticker_codes = TickerUniverse().get(['KOSPI'])

    #임시
    #data_config['ticker_codes']
    start_date = data_config['start_date']
    end_date = data_config['end_date']
    indicator_list = data_config['indicator_list']

    # 2.ML 신호 생성
    ml_config = config['ml']

    model_path = ml_config['model_path']

    # 3.backtest 변수
        # (1) 기본 변수
    basic_params = config['backtest']['basic_params']

    initial_cash = basic_params['initial_cash']
    commission = basic_params['commission']
    slippage = basic_params['slippage']
        # (2) Strategy 전용 변수
    strategy_params = config['backtest']['strategy_params']

    #기타
    output_csv = config['output_csv']
    
    logger.info(f"\n{'='*60}")
    logger.info("🚀 백테스트 설정")
    logger.info(f"{'='*60}")
    logger.info(f"종목: {ticker_codes}")
    logger.info(f"기간: {start_date} ~ {end_date}")
    logger.info(f"초기 자본: {initial_cash:,.0f}")
    logger.info(f"수수료: {commission:.4%}")
    logger.info(f"슬리피지: {slippage:.4%}")
    
    try:
        # ============ DataFrame 로드 =============
        pipeline = DataPipeline(db_path=db_path)
        df_dict = pipeline.run_full_pipeline(
            ticker_list=ticker_codes,
            start_date=start_date,
            end_date=end_date,
            indicator_list=indicator_list,
        )
        ticker_codes = list(df_dict.keys())

        # ============ ML 신호 생성 ============
        signals = None
        strategy_class = MLSignalStrategy
        
        logger.info(f"\n🤖 ML 모델로 신호 생성 중...")
        signals = generate_ml_signals(
            model_path=model_path,
            df_dict=df_dict,
        )
            
        if not signals:
            logger.warning("ML신호가 생성되지 않았습니다.")
            raise ValueError(
                f"ML 신호 생성 과정에서 문제가 생겼습니다!!"
            )

        # ============ df와 signal 통합 =============
        logger.info("\n🔗 DataFrame과 ML 신호 병합 중...")

        # 신호가 있는 티커만 처리할 새로운 df_dict 생성
        updated_df_dict = {}

        for ticker in ticker_codes:
            # df_dict에 없는 티커는 오류 발생
            if ticker not in df_dict:
                raise ValueError(f"{ticker}: df_dict에 없습니다")

            df = df_dict[ticker]
            signal = signals.get(ticker)

            # 신호가 없는 티커는 오류 발생
            if signal is None:
                raise ValueError(f"{ticker}: 신호가 없습니다")

            # signal은 DatetimeIndex를 가진 Series
            # df는 'date' 컬럼을 가진 DataFrame

            # signal을 DataFrame으로 변환
            signal_df = signal.reset_index()

            # date 타입 맞추기
            df['date'] = pd.to_datetime(df['date'])
            signal_df['date'] = pd.to_datetime(signal_df['date'])

            # 병합 (left join - df의 모든 날짜 유지)
            df = pd.merge(df, signal_df, on='date', how='left')

            # 업데이트된 df_dict에 저장
            updated_df_dict[ticker] = df

        # 원래 df_dict를 업데이트된 버전으로 교체
        df_dict = updated_df_dict

        logger.info(f"신호 병합 완료: {len(df_dict)}개 종목")
        
        # ============ 백테스트 실행 ============
        runner = BacktestRunner(
            db_path=db_path,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
        )
        
        metrics = runner.run(
            df_dict = df_dict,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
        )
        
        # ============ 결과 저장 ============
        if output_csv:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results_df = pd.DataFrame([metrics.to_dict()])
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"\n✓ 결과 저장: {output_path}")
        
        logger.info("\n✅ 백테스트 완료!")
        
    except KeyboardInterrupt:
        logger.warning("\n⛔ 사용자 중단")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    ticker_univ = TickerUniverse()
    kospi_all_ticker = ticker_univ.get(["KOSPI"])
    print(f"코스피 티커 개수: {len(kospi_all_ticker)}")
    print(f"코스피 티커 리스트 head: {kospi_all_ticker[50:]}")
    print(f"코스피 티커 리스트 tail: {kospi_all_ticker[:-50]}")
    main()
