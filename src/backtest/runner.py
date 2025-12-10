# src/backtest/runner.py
# 백테스트 실행기 - 데이터 로드, 전략 실행, 성과 분석 통합

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

import backtrader as bt
import pandas as pd
import matplotlib
#화면 차트 출력 막기
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.backtest.data_feed import create_feeds_from_dataframe
from src.backtest.strategy import MLSignalStrategy, BuyAndHoldStrategy
from src.backtest.analyzer import PerformanceAnalyzer, PerformanceMetrics

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    백테스트 통합 실행기
    
    Usage:
        runner = BacktestRunner(
            db_path='data/database/stocks.db',
            initial_cash=100_000_000,
            commission=0.00015,
            slippage=0.001,
        )
        
        metrics = runner.run(
            ticker_codes=['005930.KS', '000660.KS', '051910.KS'],
            df_dict = df_dict,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
        )
        
        print(metrics.summary())
    """
    
    def __init__(
        self,
        initial_cash: float,
        commission: float,
        slippage: float,
        db_path: str = 'data/database/stocks.db',
    ):
        """
        Args:
            db_path: 데이터베이스 경로
            initial_cash: 초기 자본금
            commission: 거래 수수료
            slippage: 슬리피지
        """
        self.db_path = db_path
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        
        self.cerebro = None
        self.results = None
        self.metrics = None
    
    def run(
        self,
        df_dict: Dict[str,pd.DataFrame],
        strategy_class: Type[bt.Strategy],
        strategy_params: Dict[str, Any],
        plot_path: Optional[str] = "data/backtest/plot",
    ) -> PerformanceMetrics:
        """
        백테스트 실행
        
        Args:
            df_dict: {종목코드: 종목정도(데이터프레임)}
                -df = OHLCV + 지표 + ML신호
                -'date'는 컬럼.
            strategy_class: 전략 클래스 (기본: MLSignalStrategy)
            strategy_params: 전략 파라미터 딕셔너리
            plot_path: 차트 저장 경로 (None이면 화면 출력)
        
        Returns:
            PerformanceMetrics 객체
        """
        ticker_codes = df_dict.keys()
        first_df = next(iter(df_dict.values()))
        start_date = pd.to_datetime(first_df['date']).min().strftime('%Y-%m-%d')
        end_date = pd.to_datetime(first_df['date']).max().strftime('%Y-%m-%d')
        

        logger.info(f"\n{'='*60}")
        logger.info("🚀 백테스트 시작")
        logger.info(f"{'='*60}")
        logger.info(f"종목: {ticker_codes}")
        logger.info(f"기간: {start_date} ~ {end_date}")
        logger.info(f"초기 자본: {self.initial_cash:,.0f}")
        logger.info(f"수수료: {self.commission:.4%}, 슬리피지: {self.slippage:.3%}")
        
        # ============ 1. Cerebro 초기화 ============
        self.cerebro = bt.Cerebro()
        
        # 브로커 설정
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # 슬리피지 설정
        self.cerebro.broker.set_slippage_perc(
            perc=self.slippage,
            slip_open=True,
            slip_limit=True,
            slip_match=True,
            slip_out=False,
        )

        # 종가로 강제 주문 체결
        self.cerebro.broker.set_coc(True)
        
        # ============ 2. 데이터 피드 추가 ============
        logger.info("\n📊 데이터 로딩 중...")
        
        feeds = create_feeds_from_dataframe(
            df_dict=df_dict,
        )
        
        if not feeds:
            raise ValueError("No valid data feeds created. Check your data.")
        
        for ticker, feed in feeds.items():
            self.cerebro.adddata(feed, name=ticker)
            logger.info(f"  ✓ {ticker} 데이터 추가")
        
        # ============ 3. 전략 추가 ============
        self.cerebro.addstrategy(strategy_class, **strategy_params)

        logger.info(f"\n📈 전략: {strategy_class.__name__}")
        logger.info(f"  파라미터: {strategy_params}")
        
        # ============ 4. 분석기 추가 ============
        PerformanceAnalyzer.add_analyzers(self.cerebro)
        
        # ============ 5. 백테스트 실행 ============
        logger.info(f"\n⏳ 백테스트 실행 중...")
        logger.info(f"초기 포트폴리오 가치: {self.cerebro.broker.getvalue():,.0f}")
        
        self.results = self.cerebro.run()
        
        final_value = self.cerebro.broker.getvalue()
        logger.info(f"최종 포트폴리오 가치: {final_value:,.0f}")
        
        # ============ 6. 성과 분석 ============
        analyzer = PerformanceAnalyzer()
        self.metrics = analyzer.analyze(
            cerebro=self.cerebro,
            results=self.results,
            initial_cash=self.initial_cash,
        )
        
        logger.info(self.metrics.summary())
        
        # ============ 7. 차트 출력 ============
        self._plot(plot_path)
        
        return self.metrics
    
    def run_with_benchmark(
        self,
        df_dict: Dict[str,pd.DataFrame],
        strategy_class: Type[bt.Strategy],
        strategy_params: Dict[str, Any],
    ) -> Dict[str, PerformanceMetrics]:
        """
        전략과 벤치마크(Buy & Hold) 비교 실행
        
        Returns:
            {'strategy': PerformanceMetrics, 'benchmark': PerformanceMetrics}
        """
        results = {}
        
        # 전략 실행
        logger.info("\n" + "="*60)
        logger.info("📈 ML 전략 백테스트")
        logger.info("="*60)
        results['strategy'] = self.run(
            df_dict=df_dict,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
        )
        
        # 벤치마크 (Buy & Hold) 실행
        logger.info("\n" + "="*60)
        logger.info("📊 벤치마크 (Buy & Hold) 백테스트")
        logger.info("="*60)
        results['benchmark'] = self.run(
            df_dict=df_dict,
            strategy_class=BuyAndHoldStrategy,
            strategy_params={'printlog': True},
        )
        
        # 비교 출력
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, PerformanceMetrics]):
        """전략 vs 벤치마크 비교 출력"""
        strategy = results['strategy']
        benchmark = results['benchmark']
        
        print(f"\n{'='*60}")
        print("📊 전략 vs 벤치마크 비교")
        print(f"{'='*60}")
        print(f"{'지표':<20} {'ML전략':>15} {'Buy&Hold':>15} {'차이':>15}")
        print("-" * 65)
        
        metrics_to_compare = [
            ('총 수익률', 'total_return', '{:+.2%}'),
            ('연환산 수익률', 'annual_return', '{:+.2%}'),
            ('변동성', 'volatility', '{:.2%}'),
            ('샤프 비율', 'sharpe_ratio', '{:.2f}'),
            ('최대 낙폭', 'max_drawdown', '{:.2%}'),
            ('승률', 'win_rate', '{:.1%}'),
            ('총 거래', 'total_trades', '{:.0f}'),
        ]
        
        for name, attr, fmt in metrics_to_compare:
            s_val = getattr(strategy, attr, 0) or 0
            b_val = getattr(benchmark, attr, 0) or 0
            diff = s_val - b_val
            
            s_str = fmt.format(s_val)
            b_str = fmt.format(b_val)
            d_str = fmt.format(diff) if '%' in fmt else f'{diff:+.2f}'
            
            print(f"{name:<20} {s_str:>15} {b_str:>15} {d_str:>15}")
        
        print(f"{'='*60}")
        
        # 알파 계산 (전략 수익률 - 벤치마크 수익률)
        alpha = strategy.total_return - benchmark.total_return
        print(f"\n🎯 알파 (초과 수익률): {alpha:+.2%}")
    
    def _plot(self, save_path: Optional[str] = None):
        """차트 출력/저장"""
        try:
            plt.rcParams['figure.figsize'] = [14, 25]  # 가로 14인치, 세로 25인치로 설정
            # Backtrader 기본 플롯
            figs = self.cerebro.plot(
                style='candlestick',
                barup='lightcoral',
                bardown='blue',
                volup='lightcoral',
                voldown='blue',
                volume=True,
                subplot=True,
                iplot=False, 
            )
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"차트 저장: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"차트 생성 실패: {e}")