# scripts/run_backtest.py
# 백테스트 실행용 CLI (config.yaml + CLI override + ML 모델 연동)

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.runner import BacktestRunner, run_backtest
from src.backtest.strategy import MLSignalStrategy, BuyAndHoldStrategy
from src.data.db_manager import DatabaseManager
from src.ml.logistic_regression import LogisticRegressionHandler

CONFIG_PATH = "/app/config/backtest.yaml"


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
        feature_columns: 피처 컬럼 리스트
    
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
        handler = LogisticRegressionHandler.load(model_path)
        logger.info(f"✓ 모델 로드 완료: {model_path}")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return signals
    
    #피쳐 컬럼 불러오기
    feature_columns = handler.feature_names

    #티커리스트 불러오기
    ticker_list = df_dict.keys()
    for ticker in ticker_list:
        try:
            df = df_dict[ticker]

            # 피처 컬럼 확인
            available_features = [c for c in feature_columns if c in df.columns]
            if set(available_features) != set(feature_columns):
                missing_columns = set(feature_columns) - set(available_features)
                raise ValueError(
                    f"{missing_columns} columns should be in DataFrame(df_dict)"
                )
            
            # NaN 제거 (예측에 필요)
            df_clean = df.dropna(subset=feature_columns)
            
            if df_clean.empty:
                logger.warning(f"{ticker}: 유효한 데이터 없음")
                continue
            
            # 예측
            X = df_clean[feature_columns]
            predictions = handler.predict(X)
            
            # 신호 시리즈 생성
            signal_series = pd.Series(
                predictions,
                index=pd.to_datetime(df_clean['date']),
                name='signal'
            )
            
            signals[ticker] = signal_series
            
            # 통계 출력
            buy_signals = (signal_series == 1).sum()
            total_days = len(signal_series)
            logger.info(f"{ticker}: {buy_signals}/{total_days} 매수 신호 ({buy_signals/total_days:.1%})")
            
        except Exception as e:
            logger.error(f"{ticker}: 신호 생성 실패 - {e}")
            continue
    
    return signals


# ============================================
# DB에서 DataFrame 로드
# ============================================

# ============================================
# 메인 실행
# ============================================
def main():
    # Config 로드
    config = load_config(CONFIG_PATH)
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})

    #로그 설정
    logger = setup_logging(config.get('log_level'))
    
    # 파라미터 결정 (CLI > config > 기본값)
    ticker_codes = model_cfg.get('training', {}).get('tickers', 
                   data_cfg.get('tickers', ['005930.KS', '000660.KS', '051910.KS']))
    
    start_date = model_cfg.get('training', {}).get('start_date', '2023-01-01')
    end_date = model_cfg.get('training', {}).get('end_date', '2023-12-31')
    
    db_path = data_cfg.get('database_path', 'data/database/stocks.db')
    
    feature_columns = model_cfg.get('features', {}).get('columns', [
        'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60', 
        'ma_100', 'ma_120', 'ma_200', 
        'macd', 'macd_signal', 'macd_hist'
    ])
    
    logger.info(f"\n{'='*60}")
    logger.info("🚀 백테스트 설정")
    logger.info(f"{'='*60}")
    logger.info(f"종목: {ticker_codes}")
    logger.info(f"기간: {start_date} ~ {end_date}")
    logger.info(f"초기 자본: {args.initial_cash:,.0f}")
    logger.info(f"보유 기간: {args.holding_period}일")
    logger.info(f"수수료: {args.commission:.4%}")
    
    try:
        # ============ DataFrame 로드 =============

        # ============ ML 신호 생성 ============
        signals = None
        strategy_class = MLSignalStrategy
        
        if args.no_model:
            logger.info("\n📊 모델 미사용 (Buy & Hold 전략)")
            strategy_class = BuyAndHoldStrategy
        else:
            logger.info(f"\n🤖 ML 모델로 신호 생성 중...")
            signals = generate_ml_signals(
                model_path=args.model_path,
                db_path=db_path,
                ticker_codes=ticker_codes,
                start_date=start_date,
                end_date=end_date,
                feature_columns=feature_columns,
            )
            
            if not signals:
                logger.warning("생성된 신호가 없습니다. 모든 날짜에 signal=0으로 진행합니다.")
        
        # ============ 백테스트 실행 ============
        runner = BacktestRunner(
            db_path=db_path,
            initial_cash=args.initial_cash,
            commission=args.commission,
        )
        
        strategy_params = {
            'holding_period': args.holding_period,
            'use_stop_loss': args.stop_loss is not None,
            'stop_loss_pct': args.stop_loss or 0.05,
            'use_take_profit': args.take_profit is not None,
            'take_profit_pct': args.take_profit or 0.10,
            'printlog': args.log_level == "DEBUG",
        }
        
        if args.compare_benchmark:
            results = runner.run_with_benchmark(
                ticker_codes=ticker_codes,
                start_date=start_date,
                end_date=end_date,
                signals=signals,
                strategy_class=strategy_class,
                strategy_params=strategy_params,
            )
            metrics = results['strategy']
        else:
            metrics = runner.run(
                ticker_codes=ticker_codes,
                start_date=start_date,
                end_date=end_date,
                signals=signals,
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                plot=args.plot,
                plot_path=args.plot_path,
            )
        
        # ============ 결과 저장 ============
        if args.output_csv:
            output_path = Path(args.output_csv)
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
    main()
