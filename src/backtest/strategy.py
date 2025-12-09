# src/backtest/strategy.py
# ML 신호 기반 매매 전략 (5일 보유, 동일 비중)

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import backtrader as bt

logger = logging.getLogger(__name__)
_REQUIRED = object()

class MLSignalStrategy(bt.Strategy):
    """
    ML 예측 신호 기반 매매 전략
    
    로직:
    1. signal=1 발생 시 매수 (동일 비중)
    2. N일(기본 5일) 보유 후 자동 매도
    3. 이미 포지션 있으면 추가 매수 안함
    4. 여러 종목 동시 운용 (동일 비중)
    
    Parameters:
        holding_period: 보유 기간 (일) - 기본값 5
        equal_weight: 종목당 투자 비중 - 기본값 자동 계산 (1/종목수)
        use_stop_loss: 손절 사용 여부 - 기본값 False
        stop_loss_pct: 손절 비율 - 기본값 0.05 (5%)
        use_take_profit: 익절 사용 여부 - 기본값 False  
        take_profit_pct: 익절 비율 - 기본값 0.10 (10%)
        max_positions: 최대 동시 포지션 수 - 기본값 None (제한 없음)
    """
    
    params = (
        ('holding_period', _REQUIRED),
        ('equal_weight', _REQUIRED),  
        ('use_stop_loss', _REQUIRED),
        ('stop_loss_pct', _REQUIRED),
        ('use_take_profit', _REQUIRED),
        ('take_profit_pct', _REQUIRED),
        ('max_positions', _REQUIRED),
        ('printlog', _REQUIRED),
    )
    
    def __init__(self):
        # 종목별 보유일 카운터: {data_name: 매수 후 경과일}
        self.holding_days = defaultdict(int)
        
        # 종목별 매수가 기록: {data_name: 매수 평균가}
        self.buy_prices = {}
        
        # 종목별 주문 추적: {data_name: order}
        self.orders = {}
        
        # 데이터 피드 이름 → 인덱스 매핑
        self.data_map = {d._name: i for i, d in enumerate(self.datas)}
        
        # 동일 비중 계산
        num_assets = len(self.datas)
        if self.params.equal_weight is None:
            self.weight = 1.0 / num_assets if num_assets > 0 else 1.0
        else:
            self.weight = self.params.equal_weight
        
        # 거래 기록
        self.trade_log = []
        
        logger.info(f"Strategy initialized: {num_assets} assets, "
                   f"weight={self.weight:.2%}, holding={self.params.holding_period}d")
        
        #입력 파라미터 검증
        self._check_required_params()

    def _check_required_params(self):
        """필수 파라미터가 입력되었는지 검증"""
        missing_params = []

        for p_name in self.params._getkeys():
            val = getattr(self.params,p_name)

            if val is _REQUIRED:
                missing_params.append(p_name)
        
        if missing_params:
            raise ValueError(
                f"\n{'='*40}\n"
                f"🛑 전략 설정 오류 ({self.__class__.__name__})\n"
                f"필수 파라미터 누락: '{missing_params}'.\n"
                f"cerebro.addstrategy() 호출 시 값을 전달해주세요.\n"
                f"{'='*40}"
            )
    
    def log(self, txt: str, dt=None):
        """로그 출력"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            logger.info(f'{dt.isoformat()} | {txt}')
    
    def notify_order(self, order):
        """주문 상태 변경 콜백"""
        data_name = order.data._name
        
        if order.status in [order.Submitted, order.Accepted]:
            return  # 대기 중
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'{data_name} BUY EXECUTED | '
                        f'Price: {order.executed.price:,.0f} | '
                        f'Size: {order.executed.size:.2f} | '
                        f'Cost: {order.executed.value:,.0f} | '
                        f'Comm: {order.executed.comm:,.0f}')
                
                # 매수가 기록
                self.buy_prices[data_name] = order.executed.price
                self.holding_days[data_name] = 0
                
            elif order.issell():
                self.log(f'{data_name} SELL EXECUTED | '
                        f'Price: {order.executed.price:,.0f} | '
                        f'Size: {order.executed.size:.2f} | '
                        f'Cost: {order.executed.value:,.0f} | '
                        f'Comm: {order.executed.comm:,.0f}')
                
                # PnL 계산
                if data_name in self.buy_prices:
                    buy_price = self.buy_prices[data_name]
                    sell_price = order.executed.price
                    pnl_pct = (sell_price - buy_price) / buy_price
                    
                    self.trade_log.append({
                        'ticker': data_name,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'pnl_pct': pnl_pct,
                        'holding_days': self.holding_days.get(data_name, 0),
                        'date': self.datas[0].datetime.date(0),
                    })
                    
                    self.log(f'{data_name} TRADE CLOSED | PnL: {pnl_pct:+.2%}')
                    
                    del self.buy_prices[data_name]
                
                self.holding_days[data_name] = 0
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'{data_name} Order {order.Status[order.status]}')
        
        # 주문 완료 후 추적 해제
        self.orders[data_name] = None
    
    def notify_trade(self, trade):
        """거래 완료 콜백"""
        if not trade.isclosed:
            return
        
        self.log(f'{trade.data._name} TRADE PROFIT | '
                f'Gross: {trade.pnl:,.0f} | Net: {trade.pnlcomm:,.0f}')
    
    def next(self):
        """매 봉마다 호출되는 메인 로직"""
        
        # 현재 포지션 수 계산
        current_positions = sum(1 for d in self.datas if self.getposition(d).size > 0)
        
        for i, data in enumerate(self.datas):
            data_name = data._name
            pos = self.getposition(data)
            
            # 보유일 증가
            if pos.size > 0:
                self.holding_days[data_name] += 1
            
            # 대기 중인 주문이 있으면 스킵
            if self.orders.get(data_name):
                continue
            
            # ============ 매도 로직 ============
            if pos.size > 0:
                should_sell = False
                sell_reason = ""
                
                # 1. 보유 기간 만료
                if self.holding_days[data_name] >= self.params.holding_period:
                    should_sell = True
                    sell_reason = f"HOLDING_PERIOD({self.params.holding_period}d)"
                
                # 2. 손절 (옵션)
                elif self.params.use_stop_loss and data_name in self.buy_prices:
                    current_price = data.close[0]
                    buy_price = self.buy_prices[data_name]
                    loss_pct = (current_price - buy_price) / buy_price
                    
                    if loss_pct <= -self.params.stop_loss_pct:
                        should_sell = True
                        sell_reason = f"STOP_LOSS({loss_pct:.2%})"
                
                # 3. 익절 (옵션)
                elif self.params.use_take_profit and data_name in self.buy_prices:
                    current_price = data.close[0]
                    buy_price = self.buy_prices[data_name]
                    profit_pct = (current_price - buy_price) / buy_price
                    
                    if profit_pct >= self.params.take_profit_pct:
                        should_sell = True
                        sell_reason = f"TAKE_PROFIT({profit_pct:.2%})"
                
                if should_sell:
                    self.log(f'{data_name} SELL ORDER | Reason: {sell_reason}')
                    self.orders[data_name] = self.sell(data=data)
                    continue
            
            # ============ 매수 로직 ============
            if pos.size == 0:
                # 최대 포지션 제한 체크
                if self.params.max_positions and current_positions >= self.params.max_positions:
                    continue
                
                # ML 신호 확인 (signal 라인이 있는 경우)
                signal = 0
                if hasattr(data.lines, 'signal'):
                    signal = data.lines.signal[0]
                
                if signal == 1:
                    # 동일 비중으로 매수 금액 계산
                    cash = self.broker.getcash()
                    target_value = self.broker.getvalue() * self.weight
                    
                    # 현재 가격
                    price = data.close[0]
                    
                    if price > 0:
                        # 매수 가능 수량 (정수)
                        size = int(target_value / price)
                        
                        if size > 0 and cash >= price * size:
                            self.log(f'{data_name} BUY ORDER | '
                                    f'Signal=1 | Price: {price:,.0f} | Size: {size} | Total: {price*size}')
                            self.orders[data_name] = self.buy(data=data, size=size)
                            current_positions += 1
    
    def stop(self):
        """백테스트 종료 시 호출"""
        # 최종 결과 출력
        final_value = self.broker.getvalue()
        self.log(f'Final Portfolio Value: {final_value:,.0f}')
        
        # 거래 통계
        if self.trade_log:
            total_trades = len(self.trade_log)
            winning_trades = sum(1 for t in self.trade_log if t['pnl_pct'] > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_pnl = sum(t['pnl_pct'] for t in self.trade_log) / total_trades
            avg_holding = sum(t['holding_days'] for t in self.trade_log) / total_trades
            
            self.log(f'Trade Stats: {total_trades} trades | '
                    f'Win Rate: {win_rate:.1%} | '
                    f'Avg PnL: {avg_pnl:+.2%} | '
                    f'Avg Holding: {avg_holding:.1f}d')


class BuyAndHoldStrategy(bt.Strategy):
    """
    벤치마크용 Buy & Hold 전략
    
    시작 시점에 동일 비중으로 매수 후 보유
    """
    
    params = (
        ('printlog', True),
    )
    
    def __init__(self):
        self.bought = False
        self.weight = 1.0 / len(self.datas) if len(self.datas) > 0 else 1.0

    def log(self, txt: str, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            logger.info(f'{dt.isoformat()} | {txt}')
    
    def next(self):
        if self.bought:
            return
        
        for data in self.datas:
            target_value = self.broker.getvalue() * self.weight
            price = data.close[0]
            
            if price > 0:
                size = int(target_value / price)
                if size > 0:
                    self.buy(data=data, size=size)
                    self.log(f'{data._name} BUY & HOLD | Price: {price:,.0f} | Size: {size}')
        
        self.bought = True
    
    def stop(self):
        final_value = self.broker.getvalue()
        self.log(f'Buy & Hold Final Value: {final_value:,.0f}')
