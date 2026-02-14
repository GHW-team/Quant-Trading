"""
레짐 스위칭 백테스트 실행 스크립트

실행: python scripts/run_regime_backtest.py

흐름:
1. SPY + S&P500 종목 데이터 수집
2. 학습된 모델로 SPY에 p_trend 추가
3. Backtrader로 RegimeSwitchingStrategy 실행
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import backtrader as bt
import pandas as pd

from src.data.data_fetcher import StockDataFetcher
from src.data.indicator_calculator import IndicatorCalculator
from src.data.feature_engineer import FeatureEngineer
from src.data.all_ticker import TickerUniverse
from src.ml.regime_model import RegimeModel
from src.backtest.regime_data_feed import create_regime_feeds
from src.backtest.regime_strategy import RegimeSwitchingStrategy
from src.backtest.analyzer import PerformanceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- 설정 ----
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
INITIAL_CASH = 1_000_000   # 100만 달러
MODEL_PATH = 'data/models/regime_classifier.pkl'
MAX_STOCKS = 500  # 테스트용 (전체 S&P500은 시간이 오래 걸림)


def main():
    # 1. 종목 리스트 준비
    logger.info("Step 1: 종목 리스트 로드")
    universe = TickerUniverse()
    sp500_tickers = universe.get(['S&P500'])[:MAX_STOCKS]  # 테스트용 제한
    all_tickers = ['SPY'] + sp500_tickers
    
    # 2. 데이터 수집
    logger.info(f"Step 2: 데이터 수집 ({len(all_tickers)} 종목)")
    fetcher = StockDataFetcher(max_workers=5)
    raw_data = fetcher.fetch_multiple_by_date(
        ticker_list=all_tickers,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    
    # 3. 지표 계산
    logger.info("Step 3: 지표 계산")
    calc = IndicatorCalculator()
    indicator_list = ['ma_50', 'ma_200', 'rsi', 'bb_upper', 'bb_mid', 'bb_lower']
    
    processed = {}
    for ticker, df in raw_data.items():
        try:
            processed[ticker] = calc.calculate_indicators(df, indicator_list)
        except Exception as e:
            logger.warning(f"{ticker}: 지표 계산 실패 - {e}")
    
    # 4. SPY에 p_trend 추가
    logger.info("Step 4: SPY에 ML 예측값(p_trend) 추가")
    model = RegimeModel.load(MODEL_PATH)
    fe = FeatureEngineer()
    
    spy_df = fe.create_features(processed['SPY'])
    feature_names = fe.get_feature_names()
    spy_df['p_trend'] = model.predict_proba(spy_df[feature_names])
    processed['SPY'] = spy_df
    
    # 5. 유니버스 필터링
    logger.info("Step 5: 유니버스 필터링")
    stock_dfs = {}
    for ticker, df in processed.items():
        if ticker == 'SPY':
            continue
        # 가격 필터: Close >= $5
        if df['close'].iloc[-1] < 5:
            continue
        # 데이터 길이 필터: 최소 504일
        if len(df) < 504:
            continue
        stock_dfs[ticker] = df
    
    logger.info(f"  필터링 후 종목 수: {len(stock_dfs)}")
    
    # 6. Backtrader 실행
    logger.info("Step 6: 백테스트 실행")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)  # 10bps
    
    # 데이터 피드 추가 (SPY가 반드시 첫 번째)
    feeds = create_regime_feeds(spy_df=processed['SPY'], stock_dfs=stock_dfs)
    for ticker, feed in feeds.items():
        cerebro.adddata(feed, name=ticker)
    
    # 전략 추가
    cerebro.addstrategy(RegimeSwitchingStrategy)
    
    # 분석기 추가
    PerformanceAnalyzer.add_analyzers(cerebro)
    
    # 실행
    results = cerebro.run()
    
    # 7. 결과 분석
    logger.info("Step 7: 성과 분석")
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(cerebro, results, INITIAL_CASH)
    logger.info(metrics.summary())


if __name__ == '__main__':
    main()
