# src/backtest/runner.py
# 백테스트 실행기 - 데이터 로드, 전략 실행, 성과 분석 통합

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.data_feed import create_feeds_from_db, create_feeds_from_dataframe
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
        )
        
        metrics = runner.run(
            ticker_codes=['005930.KS', '000660.KS', '051910.KS'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            signals=predicted_signals,  # ML 모델 예측 결과
        )
        
        print(metrics.summary())
    """
    
    def __init__(
        self,
        db_path: str = 'data/database/stocks.db',
        initial_cash: float = 100_000_000,
        commission: float = 0.00015,
        slippage: float = 0.001,
    ):
        """
        Args:
            db_path: 데이터베이스 경로
            initial_cash: 초기 자본금 (기본 1억원)
            commission: 거래 수수료 (기본 0.015%)
            slippage: 슬리피지 (기본 0.1%)
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
        ticker_codes: List[str],
        start_date: str,
        end_date: str,
        signals: Optional[Dict[str, pd.Series]] = None,
        strategy_class: Type[bt.Strategy] = MLSignalStrategy,
        strategy_params: Optional[Dict[str, Any]] = None,
        plot: bool = False,
        plot_path: Optional[str] = None,
    ) -> PerformanceMetrics:
        """
        백테스트 실행
        
        Args:
            ticker_codes: 종목 코드 리스트
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            signals: ML 예측 신호 {ticker: pd.Series(index=date, values=0/1)}
            strategy_class: 전략 클래스 (기본: MLSignalStrategy)
            strategy_params: 전략 파라미터 딕셔너리
            plot: 차트 출력 여부
            plot_path: 차트 저장 경로 (None이면 화면 출력)
        
        Returns:
            PerformanceMetrics 객체
        """
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
        
        # ============ 2. 데이터 피드 추가 ============
        logger.info("\n📊 데이터 로딩 중...")
        
        feeds = create_feeds_from_db(
            db_path=self.db_path,
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
            signals=signals,
        )
        
        if not feeds:
            raise ValueError("No valid data feeds created. Check your data.")
        
        for ticker, feed in feeds.items():
            self.cerebro.adddata(feed, name=ticker)
            logger.info(f"  ✓ {ticker} 데이터 추가")
        
        # ============ 3. 전략 추가 ============
        strategy_params = strategy_params or {}
        
        # 기본 파라미터 설정
        default_params = {
            'holding_period': 5,
            'commission_pct': self.commission,
            'printlog': True,
        }
        default_params.update(strategy_params)
        
        self.cerebro.addstrategy(strategy_class, **default_params)
        logger.info(f"\n📈 전략: {strategy_class.__name__}")
        logger.info(f"  파라미터: {default_params}")
        
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
        
        # ============ 7. 차트 출력 (옵션) ============
        if plot:
            self._plot(plot_path)
        
        return self.metrics
    
    def run_with_benchmark(
        self,
        ticker_codes: List[str],
        start_date: str,
        end_date: str,
        signals: Optional[Dict[str, pd.Series]] = None,
        strategy_class: Type[bt.Strategy] = MLSignalStrategy,
        strategy_params: Optional[Dict[str, Any]] = None,
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
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
            signals=signals,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
        )
        
        # 벤치마크 (Buy & Hold) 실행
        logger.info("\n" + "="*60)
        logger.info("📊 벤치마크 (Buy & Hold) 백테스트")
        logger.info("="*60)
        results['benchmark'] = self.run(
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
            signals=None,  # 신호 없이 단순 보유
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
            # Backtrader 기본 플롯
            figs = self.cerebro.plot(
                style='candlestick',
                barup='red',
                bardown='blue',
                volup='red',
                voldown='blue',
                volume=True,
                subplot=True,
            )
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"차트 저장: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"차트 생성 실패: {e}")


def run_backtest(
    db_path: str,
    ticker_codes: List[str],
    start_date: str,
    end_date: str,
    signals: Optional[Dict[str, pd.Series]] = None,
    initial_cash: float = 100_000_000,
    holding_period: int = 5,
    commission: float = 0.00015,
    use_stop_loss: bool = False,
    stop_loss_pct: float = 0.05,
    use_take_profit: bool = False,
    take_profit_pct: float = 0.10,
    compare_benchmark: bool = True,
) -> PerformanceMetrics:
    """
    백테스트 편의 함수
    
    Args:
        db_path: 데이터베이스 경로
        ticker_codes: 종목 코드 리스트
        start_date: 시작일
        end_date: 종료일
        signals: ML 예측 신호
        initial_cash: 초기 자본금
        holding_period: 보유 기간 (일)
        commission: 수수료
        use_stop_loss: 손절 사용 여부
        stop_loss_pct: 손절 비율
        use_take_profit: 익절 사용 여부
        take_profit_pct: 익절 비율
        compare_benchmark: 벤치마크 비교 여부
    
    Returns:
        PerformanceMetrics 객체
    """
    runner = BacktestRunner(
        db_path=db_path,
        initial_cash=initial_cash,
        commission=commission,
    )
    
    strategy_params = {
        'holding_period': holding_period,
        'use_stop_loss': use_stop_loss,
        'stop_loss_pct': stop_loss_pct,
        'use_take_profit': use_take_profit,
        'take_profit_pct': take_profit_pct,
    }
    
    if compare_benchmark:
        results = runner.run_with_benchmark(
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
            signals=signals,
            strategy_params=strategy_params,
        )
        return results['strategy']
    else:
        return runner.run(
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
            signals=signals,
            strategy_params=strategy_params,
        )


# ============ CLI용 메인 ============
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 프로젝트 루트 추가
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("BacktestRunner 테스트")
    print("="*70)
    
    # 테스트 실행
    try:
        runner = BacktestRunner(
            db_path='data/database/stocks.db',
            initial_cash=100_000_000,
        )
        
        # 간단한 테스트 (신호 없이 Buy & Hold)
        metrics = runner.run(
            ticker_codes=['005930.KS'],
            start_date='2024-01-01',
            end_date='2024-06-30',
            signals=None,
            strategy_class=BuyAndHoldStrategy,
        )
        
        print(metrics.summary())
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()
