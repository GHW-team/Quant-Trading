"""
레짐 스위칭 전략 (Regime Switching Strategy)

- TREND 상태: 모멘텀 + MA 추세 추종 (상위 20종목, 월 1회 리밸런싱)
- RANGE 상태: 볼린저 + RSI 평균 회귀 (최대 10종목, 매일 평가)
- 스위칭: p_trend 확률 기반 히스테리시스 (3일 연속 조건)
"""
import backtrader as bt
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RegimeSwitchingStrategy(bt.Strategy):
    """
    ML 기반 레짐 스위칭 전략
    
    datas[0] = SPY (p_trend 라인 포함)
    datas[1:] = S&P500 개별 종목들
    """
    
    params = (
        # 스위칭 파라미터
        ('trend_enter_threshold', 0.60),  # p_trend >= 이 값이면 추세 신호
        ('trend_exit_threshold', 0.40),   # p_trend <= 이 값이면 횡보 신호
        ('confirmation_days', 3),          # 3일 연속 조건 충족 시 전환
        
        # 추세장 파라미터
        ('trend_top_n', 20),               # 추세장: 상위 N개 종목 매수
        ('trend_rebal_day', 1),            # 리밸런싱: 매달 첫 거래일
        
        # 횡보장 파라미터
        ('range_top_n', 10),               # 횡보장: 최대 N개 종목 매수
        ('range_exposure', 0.60),          # 횡보장: 총 주식 익스포저 상한 60%
        ('range_rsi_entry', 30),           # RSI 매수 기준
        ('range_rsi_exit', 50),            # RSI 매도 기준
        ('range_max_hold', 10),            # 최대 보유 기간 (거래일)
        ('range_stop_loss', 0.03),         # 손절선 3%
        
        # 공통
        ('commission_bps', 10),            # 거래비용 10bps
        ('printlog', True),
    )
    
    def __init__(self):
        # ---- 데이터 피드 분류 ----
        self.spy = self.datas[0]           # 첫 번째 = SPY
        self.stocks = self.datas[1:]       # 나머지 = 개별 종목
        
        # ---- 레짐 상태 ----
        self.regime = 'RANGE'              # 초기 상태: 횡보장
        self.trend_streak = 0              # 연속 추세 신호 일수
        self.range_streak = 0              # 연속 횡보 신호 일수
        
        # ---- 보유 정보 추적 ----
        self.buy_prices = {}               # {data_name: 매수가}
        self.holding_days = defaultdict(int)  # {data_name: 보유 일수}
        self.orders = {}                   # {data_name: 주문 객체}
        self.last_rebal_month = -1         # 마지막 리밸런싱 월
        
        logger.info(f"전략 초기화: SPY + {len(self.stocks)}개 종목")
    
    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.spy.datetime.date(0)
            logger.info(f"{dt} | {txt}")
    
    # ================================================================
    # 메인 루프
    # ================================================================
    def next(self):
        # ---- 1. 레짐 판단 (매일) ----
        p = self.spy.p_trend[0]
        self._update_regime(p)
        
        # ---- 2. 하위 전략 실행 ----
        if self.regime == 'TREND':
            self._execute_trend()
        else:
            self._execute_range()
    
    # ================================================================
    # 레짐 스위칭 로직
    # ================================================================
    def _update_regime(self, p_trend):
        """
        히스테리시스 스위칭

        RANGE → TREND: p_trend >= 0.60 이 3일 연속
        TREND → RANGE: p_trend <= 0.40 이 3일 연속
        """
        if p_trend >= self.params.trend_enter_threshold:
            self.trend_streak += 1
            self.range_streak = 0
        elif p_trend <= self.params.trend_exit_threshold:
            self.range_streak += 1
            self.trend_streak = 0
        else:
            # 애매한 구간: 카운트 리셋, 현재 상태 유지
            self.trend_streak = 0
            self.range_streak = 0
        
        # 전환 판단
        if self.regime == 'RANGE' and self.trend_streak >= self.params.confirmation_days:
            self._switch_regime('TREND')
        elif self.regime == 'TREND' and self.range_streak >= self.params.confirmation_days:
            self._switch_regime('RANGE')
    
    def _switch_regime(self, new_regime):
        """레짐 전환 시 포지션 전량 청산"""
        self.log(f"⚡ 레짐 전환: {self.regime} → {new_regime}")
        
        # 기존 포지션 전량 청산
        for d in self.stocks:
            pos = self.getposition(d)
            if pos.size > 0:
                self.sell(data=d, size=pos.size)
                self.log(f"  청산: {d._name} (size={pos.size})")
        
        self.regime = new_regime
        self.holding_days.clear()
        self.buy_prices.clear()
        self.trend_streak = 0
        self.range_streak = 0
    
    # ================================================================
    # 추세장 전략: 모멘텀 + MA 추세 추종
    # ================================================================
    def _execute_trend(self):
        """
        [추세장 로직]
        1. Close > MA200 인 종목만 후보
        2. 12-1 모멘텀(= 252일 수익률 - 21일 수익률)으로 랭킹
        3. 상위 N개를 동일가중으로 매수
        4. 월 1회 리밸런싱
        """
        current_date = self.spy.datetime.date(0)
        current_month = current_date.month
        
        # 리밸런싱 조건: 월이 바뀌었을 때만
        if current_month == self.last_rebal_month:
            # 리밸런싱 아닌 날: 긴급 청산만 체크
            self._trend_risk_check()
            return
        
        self.last_rebal_month = current_month
        self.log(f"📊 [TREND] 월간 리밸런싱 실행")
        
        # ---- 후보 선정 ----
        candidates = []
        for d in self.stocks:
            if len(d) < 252:
                continue  # 데이터 부족
            
            # 조건: Close > MA200
            if d.close[0] <= 0 or d.ma_200[0] <= 0:
                continue
            if d.close[0] <= d.ma_200[0]:
                continue
            
            # 모멘텀 점수: 12-1 momentum
            try:
                ret_252 = (d.close[0] - d.close[-252]) / d.close[-252]
                ret_21 = (d.close[0] - d.close[-21]) / d.close[-21]
                momentum = ret_252 - ret_21
            except (IndexError, ZeroDivisionError):
                continue
            
            if momentum > 0:  # 모멘텀 양수인 종목만
                candidates.append((d, momentum))
        
        # ---- 랭킹 및 매수/청산 ----
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_n_datas = set(d for d, _ in candidates[:self.params.trend_top_n])
        
        # 탈락 종목 청산
        for d in self.stocks:
            if self.getposition(d).size > 0 and d not in top_n_datas:
                self.sell(data=d, size=self.getposition(d).size)
                self.log(f"  청산(탈락): {d._name}")
        
        # 신규/유지 종목 비중 조절
        n = len(top_n_datas) if top_n_datas else 1
        weight = 1.0 / n
        portfolio_value = self.broker.getvalue()
        
        for d in top_n_datas:
            target_value = portfolio_value * weight
            current_value = self.getposition(d).size * d.close[0]
            diff = target_value - current_value
            
            if diff > d.close[0]:  # 추가 매수
                size = int(diff / d.close[0])
                if size > 0:
                    self.buy(data=d, size=size)
            elif diff < -d.close[0]:  # 일부 매도
                size = int(abs(diff) / d.close[0])
                if size > 0:
                    self.sell(data=d, size=size)
    
    def _trend_risk_check(self):
        """추세장 긴급 리스크 관리 (MA200 하회 시 즉시 청산)"""
        for d in self.stocks:
            pos = self.getposition(d)
            if pos.size > 0:
                if d.close[0] < d.ma_200[0]:
                    self.sell(data=d, size=pos.size)
                    self.log(f"  긴급 청산(MA200 이탈): {d._name}")
    
    # ================================================================
    # 횡보장 전략: 볼린저 + RSI 평균 회귀
    # ================================================================
    def _execute_range(self):
        """
        [횡보장 로직]
        매수: Close < BB하단 AND RSI < 30
        매도: Close >= BB중심 OR RSI >= 50 OR 10일 경과 OR -3% 손절
        """
        # ---- 보유 종목 매도 체크 (매일) ----
        for d in self.stocks:
            pos = self.getposition(d)
            if pos.size <= 0:
                continue
            
            name = d._name
            self.holding_days[name] += 1
            should_sell = False
            reason = ""
            
            # 청산 조건 1: BB 중심선 회귀
            if d.close[0] >= d.bb_mid[0]:
                should_sell, reason = True, "BB중심 도달"
            
            # 청산 조건 2: RSI 회복
            elif d.rsi[0] >= self.params.range_rsi_exit:
                should_sell, reason = True, f"RSI={d.rsi[0]:.0f}"
            
            # 청산 조건 3: 최대 보유기간
            elif self.holding_days[name] >= self.params.range_max_hold:
                should_sell, reason = True, f"보유기간 {self.params.range_max_hold}일"
            
            # 청산 조건 4: 손절
            elif name in self.buy_prices:
                loss = (d.close[0] - self.buy_prices[name]) / self.buy_prices[name]
                if loss <= -self.params.range_stop_loss:
                    should_sell, reason = True, f"손절 {loss:.2%}"
            
            if should_sell:
                self.sell(data=d, size=pos.size)
                self.log(f"  [RANGE] 청산: {name} ({reason})")
                self.holding_days[name] = 0
                self.buy_prices.pop(name, None)
        
        # ---- 신규 매수 ----
        current_positions = sum(1 for d in self.stocks if self.getposition(d).size > 0)
        if current_positions >= self.params.range_top_n:
            return  # 최대 포지션 도달
        
        # 매수 후보: BB하단 이탈 + RSI 과매도
        candidates = []
        for d in self.stocks:
            if self.getposition(d).size > 0:
                continue
            if d.close[0] <= 0 or d.bb_lower[0] <= 0:
                continue
            
            if d.close[0] < d.bb_lower[0] and d.rsi[0] < self.params.range_rsi_entry:
                # 이탈 깊이(depth) = (BB_lower - Close) / STD20 으로 점수화
                # 간단히 (BB_lower - Close) / Close 사용
                depth = (d.bb_lower[0] - d.close[0]) / d.close[0]
                candidates.append((d, depth))
        
        # depth 큰 순서대로 매수
        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = self.params.range_top_n - current_positions
        
        portfolio_value = self.broker.getvalue()
        weight = self.params.range_exposure / self.params.range_top_n  # 0.6/10 = 6%
        
        for d, depth in candidates[:slots]:
            target_value = portfolio_value * weight
            size = int(target_value / d.close[0])
            
            if size > 0 and self.broker.getcash() >= size * d.close[0]:
                self.buy(data=d, size=size)
                self.buy_prices[d._name] = d.close[0]
                self.holding_days[d._name] = 0
                self.log(f"  [RANGE] 매수: {d._name} (depth={depth:.4f}, RSI={d.rsi[0]:.0f})")
    
    # ================================================================
    # 종료
    # ================================================================
    def stop(self):
        final = self.broker.getvalue()
        self.log(f"최종 포트폴리오: {final:,.0f}")
