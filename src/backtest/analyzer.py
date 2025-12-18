# src/backtest/analyzer.py
# 백테스트 성과 분석 모듈

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import backtrader as bt
import backtrader.analyzers as btanalyzers

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """백테스트 성과 지표 데이터 클래스"""
    
    # 기본 수익률
    total_return: float = 0.0           # 총 수익률
    annual_return: float = 0.0          # 연환산 수익률 (CAGR)
    
    # 리스크 지표
    volatility: float = 0.0             # 연환산 변동성
    sharpe_ratio: float = 0.0           # 샤프 비율
    sortino_ratio: float = 0.0          # 소르티노 비율
    max_drawdown: float = 0.0           # 최대 낙폭 (MDD)
    max_drawdown_duration: int = 0      # MDD 지속 기간 (일)
    calmar_ratio: float = 0.0           # 칼마 비율 (CAGR / MDD)
    
    # 거래 통계
    total_trades: int = 0               # 총 거래 횟수
    winning_trades: int = 0             # 승리 거래 수
    losing_trades: int = 0              # 패배 거래 수
    win_rate: float = 0.0               # 승률
    
    # 손익 통계
    avg_trade_pnl: float = 0.0          # 평균 거래 손익
    avg_winning_pnl: float = 0.0        # 평균 승리 손익
    avg_losing_pnl: float = 0.0         # 평균 패배 손익
    profit_factor: float = 0.0          # 수익 팩터 (총이익/총손실)
    expectancy: float = 0.0             # 기대값 (승률*평균이익 - 패배율*평균손실)
    
    # 포트폴리오
    initial_value: float = 0.0          # 초기 자산
    final_value: float = 0.0            # 최종 자산
    
    # 기간
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0
    
    # 상세 데이터 (선택적)
    daily_returns: Optional[pd.Series] = field(default=None, repr=False)
    equity_curve: Optional[pd.Series] = field(default=None, repr=False)
    drawdown_series: Optional[pd.Series] = field(default=None, repr=False)
    trade_log: Optional[List[Dict]] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (시각화/저장용)"""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_trade_pnl': self.avg_trade_pnl,
            'avg_winning_pnl': self.avg_winning_pnl,
            'avg_losing_pnl': self.avg_losing_pnl,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'trading_days': self.trading_days,
        }
    
    def summary(self) -> str:
        """성과 요약 문자열"""
        return f"""
{'='*60}
📊 백테스트 성과 요약
{'='*60}
📅 기간: {self.start_date} ~ {self.end_date} ({self.trading_days} 거래일)

💰 수익률
  - 총 수익률:     {self.total_return:>+10.2%}
  - 연환산 수익률: {self.annual_return:>+10.2%}
  - 초기 자산:     {self.initial_value:>15,.0f}
  - 최종 자산:     {self.final_value:>15,.0f}

📉 리스크
  - 연환산 변동성: {self.volatility:>10.2%}
  - 최대 낙폭:     {self.max_drawdown:>10.2%}
  - MDD 지속기간:  {self.max_drawdown_duration:>10} 일

📈 리스크 조정 수익률
  - 샤프 비율:     {self.sharpe_ratio:>10.2f}
  - 소르티노 비율: {self.sortino_ratio:>10.2f}
  - 칼마 비율:     {self.calmar_ratio:>10.2f}

🔄 거래 통계
  - 총 거래:       {self.total_trades:>10} 회
  - 승리/패배:     {self.winning_trades:>5} / {self.losing_trades:<5}
  - 승률:          {self.win_rate:>10.1%}
  - 수익 팩터:     {self.profit_factor:>10.2f}
  - 기대값:        {self.expectancy:>+10.2%}
{'='*60}
"""


class PerformanceAnalyzer:
    """
    Backtrader 결과를 분석하여 성과 지표 계산
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Args:
            risk_free_rate: 무위험 이자율 (연간) - 기본값 3%
        """
        self.risk_free_rate = risk_free_rate
    
    def analyze(
        self,
        cerebro: bt.Cerebro,
        results: List,
        initial_cash: float,
    ) -> PerformanceMetrics:
        """
        Backtrader 실행 결과에서 성과 지표 추출
        
        Args:
            cerebro: Backtrader Cerebro 인스턴스
            results: cerebro.run() 결과
            initial_cash: 초기 자본금
        
        Returns:
            PerformanceMetrics 객체
        """
        metrics = PerformanceMetrics()
        metrics.initial_value = initial_cash
        
        if not results:
            logger.warning("No backtest results to analyze")
            return metrics
        
        strategy = results[0]
        
        # ============ 기본 정보 추출 ============
        metrics.final_value = cerebro.broker.getvalue()
        metrics.total_return = (metrics.final_value - metrics.initial_value) / metrics.initial_value
        
        # ============ Analyzer 결과 추출 ============
        # TimeReturn Analyzer
        if hasattr(strategy.analyzers, 'timereturn'):
            time_return = strategy.analyzers.timereturn.get_analysis()
            daily_returns = pd.Series(time_return)
            metrics.daily_returns = daily_returns
            
            if len(daily_returns) > 0:
                metrics.trading_days = len(daily_returns)
                metrics.start_date = str(daily_returns.index[0])
                metrics.end_date = str(daily_returns.index[-1])
                
                # 연환산 수익률 (CAGR)
                years = metrics.trading_days / 252
                if years > 0:
                    metrics.annual_return = (1 + metrics.total_return) ** (1/years) - 1
                
                # 변동성 (연환산)
                metrics.volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio Analyzer
        if hasattr(strategy.analyzers, 'sharpe'):
            sharpe = strategy.analyzers.sharpe.get_analysis()
            metrics.sharpe_ratio = sharpe.get('sharperatio', 0.0) or 0.0
        
        # DrawDown Analyzer
        if hasattr(strategy.analyzers, 'drawdown'):
            dd = strategy.analyzers.drawdown.get_analysis()
            metrics.max_drawdown = dd.get('max', {}).get('drawdown', 0.0) / 100
            metrics.max_drawdown_duration = dd.get('max', {}).get('len', 0)
        
        # Trade Analyzer
        if hasattr(strategy.analyzers, 'trades'):
            trades = strategy.analyzers.trades.get_analysis()
            
            metrics.total_trades = trades.get('total', {}).get('closed', 0)
            
            won = trades.get('won', {})
            lost = trades.get('lost', {})
            
            metrics.winning_trades = won.get('total', 0)
            metrics.losing_trades = lost.get('total', 0)
            
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades
            
            # PnL 통계
            pnl = trades.get('pnl', {})
            metrics.avg_trade_pnl = pnl.get('net', {}).get('average', 0.0)
            
            if metrics.winning_trades > 0:
                metrics.avg_winning_pnl = won.get('pnl', {}).get('average', 0.0)
            if metrics.losing_trades > 0:
                metrics.avg_losing_pnl = lost.get('pnl', {}).get('average', 0.0)
            
            # Profit Factor
            gross_profit = won.get('pnl', {}).get('total', 0.0)
            gross_loss = abs(lost.get('pnl', {}).get('total', 0.0))
            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss
        
        # ============ 추가 계산 ============
        # 소르티노 비율 (하방 변동성 기준)
        if metrics.daily_returns is not None and len(metrics.daily_returns) > 0:
            negative_returns = metrics.daily_returns[metrics.daily_returns < 0]
            downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            
            if downside_std > 0:
                excess_return = metrics.annual_return - self.risk_free_rate
                metrics.sortino_ratio = excess_return / downside_std
        
        # 칼마 비율 (CAGR / MDD)
        if metrics.max_drawdown != 0:
            metrics.calmar_ratio = metrics.annual_return / abs(metrics.max_drawdown)
        
        # 기대값 (Expectancy)
        if metrics.initial_value > 0:
            avg_win_pct = metrics.avg_winning_pnl / metrics.initial_value if metrics.avg_winning_pnl else 0
            avg_loss_pct = abs(metrics.avg_losing_pnl) / metrics.initial_value if metrics.avg_losing_pnl else 0
            
            metrics.expectancy = (metrics.win_rate * avg_win_pct) - ((1 - metrics.win_rate) * avg_loss_pct)
        
        # ============ 전략 거래 로그 (있는 경우) ============
        if hasattr(strategy, 'trade_log'):
            metrics.trade_log = strategy.trade_log
        
        return metrics
    
    @staticmethod
    def add_analyzers(cerebro: bt.Cerebro):
        """
        Cerebro에 표준 분석기 추가
        
        Args:
            cerebro: Backtrader Cerebro 인스턴스
        """
        # 일별 수익률
        cerebro.addanalyzer(btanalyzers.TimeReturn, _name='timereturn')
        
        # 샤프 비율
        cerebro.addanalyzer(btanalyzers.SharpeRatio, 
                          _name='sharpe',
                          riskfreerate=0.03,
                          annualize=True,
                          timeframe=bt.TimeFrame.Days)
        
        # 낙폭 분석
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        
        # 거래 분석
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        
        # 수익률 분석
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        
        # 거래 횟수
        cerebro.addanalyzer(btanalyzers.Transactions, _name='transactions')
        
        # SQN (System Quality Number)
        cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
        
        logger.info("Added standard analyzers to Cerebro")


def compare_strategies(
    metrics_list: List[PerformanceMetrics],
    names: List[str],
) -> pd.DataFrame:
    """
    여러 전략의 성과 비교 테이블 생성
    
    Args:
        metrics_list: PerformanceMetrics 리스트
        names: 전략 이름 리스트
    
    Returns:
        비교 DataFrame
    """
    data = []
    for metrics, name in zip(metrics_list, names):
        row = metrics.to_dict()
        row['strategy'] = name
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.set_index('strategy')
    
    # 주요 지표만 선택하여 보기 좋게 정렬
    key_cols = [
        'total_return', 'annual_return', 'volatility',
        'sharpe_ratio', 'max_drawdown', 'calmar_ratio',
        'total_trades', 'win_rate', 'profit_factor'
    ]
    
    available_cols = [c for c in key_cols if c in df.columns]
    return df[available_cols]
