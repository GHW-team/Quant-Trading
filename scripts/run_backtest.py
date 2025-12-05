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
    model_path: str,
    db_path: str,
    ticker_codes: List[str],
    start_date: str,
    end_date: str,
    feature_columns: List[str],
) -> Dict[str, pd.Series]:
    """
    학습된 ML 모델로 매수 신호 생성
    
    Args:
        model_path: 저장된 모델 경로
        db_path: 데이터베이스 경로
        ticker_codes: 종목 코드 리스트
        start_date: 시작일
        end_date: 종료일
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
    
    # 데이터 로드 및 예측
    with DatabaseManager(db_path=db_path) as db:
        # 가격 데이터 로드
        price_dict = db.load_price_data(
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
        )
        
        # 지표 데이터 로드
        indicator_dict = db.load_indicators(
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
        )
    
    for ticker in ticker_codes:
        try:
            price_df = price_dict.get(ticker)
            indicator_df = indicator_dict.get(ticker)
            
            if price_df is None or price_df.empty:
                logger.warning(f"{ticker}: 가격 데이터 없음")
                continue
            
            # 가격 + 지표 병합
            df = price_df.copy()
            if indicator_df is not None and not indicator_df.empty:
                df = pd.merge(df, indicator_df, on='date', how='left')
            
            # 피처 컬럼 확인
            available_features = [c for c in feature_columns if c in df.columns]
            if len(available_features) < len(feature_columns):
                missing = set(feature_columns) - set(available_features)
                logger.warning(f"{ticker}: 누락된 피처 {missing}")
                continue
            
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
# CLI 파서
# ============================================
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ML 퀀트 트레이딩 백테스트 실행기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (config.yaml 사용)
  python scripts/run_backtest.py
  
  # 종목/기간 지정
  python scripts/run_backtest.py --tickers 005930.KS 000660.KS 051910.KS \\
                                  --start-date 2023-01-01 --end-date 2023-12-31
  
  # 손절/익절 설정
  python scripts/run_backtest.py --stop-loss 0.05 --take-profit 0.10
  
  # 벤치마크 비교
  python scripts/run_backtest.py --compare-benchmark
        """
    )
    
    # 종목/기간
    parser.add_argument("--tickers", nargs="+", 
                       help="종목 코드 (예: 005930.KS 000660.KS)")
    parser.add_argument("--start-date", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="종료일 (YYYY-MM-DD)")
    
    # 전략 파라미터
    parser.add_argument("--holding-period", type=int, default=5,
                       help="보유 기간 (기본: 5일)")
    parser.add_argument("--initial-cash", type=float, default=100_000_000,
                       help="초기 자본금 (기본: 1억원)")
    parser.add_argument("--commission", type=float, default=0.00015,
                       help="거래 수수료 (기본: 0.015%%)")
    
    # 손절/익절
    parser.add_argument("--stop-loss", type=float,
                       help="손절 비율 (예: 0.05 = 5%%)")
    parser.add_argument("--take-profit", type=float,
                       help="익절 비율 (예: 0.10 = 10%%)")
    
    # 모델
    parser.add_argument("--model-path", default="models/momentum_model.pkl",
                       help="학습된 모델 경로")
    parser.add_argument("--no-model", action="store_true",
                       help="모델 사용 안함 (Buy & Hold)")
    
    # 벤치마크
    parser.add_argument("--compare-benchmark", action="store_true",
                       help="Buy & Hold 벤치마크와 비교")
    
    # 출력
    parser.add_argument("--plot", action="store_true",
                       help="차트 출력")
    parser.add_argument("--plot-path", 
                       help="차트 저장 경로 (예: output/backtest.png)")
    parser.add_argument("--output-csv",
                       help="결과 CSV 저장 경로")
    
    # 기타
    parser.add_argument("--config", default="config/config.yaml",
                       help="설정 파일 경로")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return parser


# ============================================
# 메인 실행
# ============================================
def main():
    parser = create_parser()
    args = parser.parse_args()
    
    logger = setup_logging(args.log_level)
    
    # Config 로드
    config = load_config(args.config)
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    
    # 파라미터 결정 (CLI > config > 기본값)
    ticker_codes = args.tickers or model_cfg.get('training', {}).get('tickers', 
                   data_cfg.get('tickers', ['005930.KS', '000660.KS', '051910.KS']))
    
    start_date = args.start_date or model_cfg.get('training', {}).get('start_date', '2023-01-01')
    end_date = args.end_date or model_cfg.get('training', {}).get('end_date', '2023-12-31')
    
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
