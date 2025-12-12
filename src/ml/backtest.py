"""백테스트 러너(backtrader) - 변동성 타겟 전략 전용."""

import logging
import math
import sqlite3
from typing import Iterable, List, Optional

import backtrader as bt
import numpy as np
import pandas as pd

from src.ml.strategy import VolTargetLogic

logger = logging.getLogger(__name__)


class DataLoader:
    """daily_prices + technical_indicators 조인 후 backtrader feed 생성."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def load(self, symbols: Iterable[str], start: str, end: str) -> List[bt.feeds.PandasData]:
        symbols = list(symbols)
        if not symbols:
            raise ValueError("symbols is empty")

        placeholders = ",".join("?" for _ in symbols)
        query = f"""
            SELECT
                t.ticker_code,
                p.date,
                p.open,
                p.high,
                p.low,
                p.close,
                p.volume,
                ti.vol_20d,
                ti.ret_126d,
                ti.mom_rank_pct
            FROM daily_prices AS p
            JOIN tickers AS t ON p.ticker_id = t.ticker_id
            LEFT JOIN technical_indicators AS ti
                   ON ti.ticker_id = p.ticker_id AND ti.date = p.date
            WHERE t.ticker_code IN ({placeholders})
              AND p.date BETWEEN ? AND ?
            ORDER BY t.ticker_code, p.date
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[*symbols, start, end],
                    parse_dates=["date"],
                )
        except Exception as e:
            raise RuntimeError(f"data load failed: {e}")

        if df.empty:
            raise ValueError("no data returned")

        feeds: List[bt.feeds.PandasData] = []
        for sym, g in df.groupby("ticker_code"):
            required = {"date", "open", "high", "low", "close"}
            missing = required - set(g.columns)
            if missing:
                raise ValueError(f"{sym} missing columns: {missing}")

            g = g.copy()
            # 필요한 컬럼만 포함된 feed 추가
            feeds.append(self._to_feed(g, name=sym))

        return feeds

    def _to_feed(self, df: pd.DataFrame, name: str) -> bt.feeds.PandasData:
        class CustomFeed(bt.feeds.PandasData):
            lines = ("vol_20d", "ret_126d", "mom_rank_pct")
            params = (
                ("datetime", "date"),
                ("vol_20d", "vol_20d"),
                ("ret_126d", "ret_126d"),
                ("mom_rank_pct", "mom_rank_pct"),
            )

        return CustomFeed(dataname=df, name=name)


class VolTargetAdapter(bt.Strategy):
    """VolTargetLogic을 backtrader에서 실행하기 위한 어댑터."""

    params = dict(
        target_vol=0.10, # 포트폴리오 연간 목표 변동성
        lookback_vol=20, # 변동성 계산 시 사용하는 거래일
        lookback_mom=126, # 모멘텀 계산시 사용하는 거래일
        mom_top_pct=0.15, # 모멘텀 상위 비중
        rebalance_freq=20, # 리밸런싱 주기
        vol_cap=0.10, # 종목별 변동성 상한
        vol_floor=0.0, # 종목별 변동성 하한
        k_vol_stop=2.0, # 변동성 기반 스탑로스 계수
        max_gross=2.0, # 총 노출 상한 (1.0=100%, 0.8=80%, 1.5=150% 허용)
        overlay_lookback=20, # 포트폴리오 변동성 산출 룩백(오버레이 스케일링)
    )

    def __init__(self):
        self.logic = VolTargetLogic(
            target_vol=self.p.target_vol,
            lookback_vol=self.p.lookback_vol,
            lookback_mom=self.p.lookback_mom,
            mom_top_pct=self.p.mom_top_pct,
            vol_cap=self.p.vol_cap,
            vol_floor=self.p.vol_floor,
            k_vol_stop=self.p.k_vol_stop,
        )
        self.stops = {}  # data_name -> stop price
        self.bar_count = 0
        self.portfolio_values = []  # 포트폴리오 변동성 오버레이용 값 시계열

    def prenext(self):
        """일부 데이터만 live여도 활성 데이터로부터 스텝 실행."""
        active = [d for d in self.datas if len(d)]
        if not active:
            return
        self._run_step(active)

    def nextstart(self):
        """모든 데이터가 live 되기 시작하는 구간도 동일 처리."""
        active = [d for d in self.datas if len(d)]
        if not active:
            return
        self._run_step(active)

    def _get_closes(self, data: bt.feeds.PandasData, count: int):
        """종가 시퀀스(가장 오래된 값이 앞)를 count+1 길이로 반환."""
        if len(data) <= count:
            return None
        closes = []
        try:
            for i in range(count, -1, -1):
                closes.append(data.close[-i])
        except IndexError:
            return None
        return closes

    def next(self):
        self._run_step(self.datas)

    def _run_step(self, datas):
        self.bar_count += 1
        ref = datas[0]
        logger.debug("next called %s bar=%s dt=%s", ref._name, self.bar_count, bt.num2date(ref.datetime[0]))

        vol_map = {}
        mom_map = {}
        rank_map = {}

        # 포트폴리오 현재 가치 기록 (오버레이 변동성 계산용)
        try:
            port_val = float(self.broker.getvalue())
            self.portfolio_values.append(port_val)
            if len(self.portfolio_values) > (self.p.overlay_lookback + 1):
                self.portfolio_values.pop(0)
        except Exception:
            pass

        for d in datas:
            # Prefer precomputed DB columns; fallback to on-the-fly calc
            vol_raw = d.vol_20d[0]
            vol_val = None if vol_raw is None or np.isnan(vol_raw) else vol_raw
            if vol_val is None:
                closes_vol = self._get_closes(d, self.p.lookback_vol)
                vol_val = self.logic.calc_vol(closes_vol) if closes_vol else None

            mom_raw = d.ret_126d[0]
            mom_val = None if mom_raw is None or np.isnan(mom_raw) else mom_raw
            if mom_val is None:
                closes_mom = self._get_closes(d, self.p.lookback_mom)
                mom_val = self.logic.calc_momentum(closes_mom) if closes_mom else None

            rank_raw = d.mom_rank_pct[0]
            rank_val = None if rank_raw is None or np.isnan(rank_raw) else rank_raw

            vol_map[d._name] = vol_val
            mom_map[d._name] = mom_val
            rank_map[d._name] = rank_val

        # Determine momentum winners: use mom_rank_pct if available, else percentile from momentum
        if any(v is not None for v in rank_map.values()):
            cutoff = 1 - self.p.mom_top_pct
            top_set = {sym for sym, r in rank_map.items() if r is not None and r >= cutoff}
        else:
            top_set = self.logic.select_top_momentum(mom_map)

        # 스탑 업데이트 및 청산
        for d in datas:
            pos = self.getposition(d)
            if not pos.size:
                continue
            vol = vol_map.get(d._name)
            prev_stop = self.stops.get(d._name)
            updated_stop = self.logic.update_stop(prev_stop, d.high[0], vol)
            if updated_stop is not None:
                self.stops[d._name] = updated_stop
            if self.logic.should_exit(d.close[0], self.stops.get(d._name)):
                self.close(data=d)
                self.stops.pop(d._name, None)

        # 리밸런스: 상위 모멘텀 종목의 INVERSE-VOL 비중을 합계 1로 정규화
        if self.bar_count % self.p.rebalance_freq == 0:
            raw_weights = {}
            total_raw = 0.0
            for name in top_set:
                w_raw = self.logic.weight(vol_map.get(name))
                if w_raw > 0:
                    raw_weights[name] = w_raw
                    total_raw += w_raw

            base_scale = (1.0 / total_raw) if total_raw > 0 else 0.0
            overlay_scale = self._portfolio_overlay_scale()
            # 총 노출 = base_scale * overlay_scale를 max_gross로 제한 (overlay가 1을 넘으면 레버리지 허용)
            total_scale = min(self.p.max_gross, base_scale * overlay_scale)

            for d in datas:
                name = d._name
                # 포트폴리오 변동성이 목표보다 높으면 overlay_scale<1로 전체 비중을 줄여 현금 확보
                w = raw_weights.get(name, 0.0) * total_scale

                self.order_target_percent(data=d, target=w)
                if w > 0:
                    stop = self.logic.init_stop(d.close[0], vol_map.get(name))
                    if stop is not None:
                        self.stops[name] = stop
                else:
                    self.stops.pop(name, None)

    def _portfolio_overlay_scale(self) -> float:
        """포트폴리오 실현 변동성 기반 스케일링.

        realized_vol > target_vol -> scale < 1 (노출 축소, 현금 확보)
        realized_vol < target_vol -> scale > 1 허용(레버리지), 단 max_gross 이하로 캡
        """
        if len(self.portfolio_values) <= 1:
            return 1.0
        try:
            log_rets = []
            for i in range(1, len(self.portfolio_values)):
                v0 = self.portfolio_values[i - 1]
                v1 = self.portfolio_values[i]
                if v0 and v1 and v0 > 0 and v1 > 0:
                    log_rets.append(math.log(v1 / v0))
            if len(log_rets) < max(2, self.p.overlay_lookback // 2):
                return 1.0
            std = float(np.std(log_rets, ddof=0))
            realized_vol = std * math.sqrt(252)
            if realized_vol <= 0:
                return 1.0
            scale = self.p.target_vol / realized_vol
            # 레버리지 허용: scale을 max_gross까지 허용, 음수/0은 방어
            if scale <= 0:
                return 1.0
            return min(self.p.max_gross, scale)
        except Exception:
            return 1.0


class BacktestRunner:
    """Initialize cerebro, add feeds/strategy/analyzers, and run."""

    def __init__(self, commission: float = 0.00015, slippage_perc: float = 0.0015):
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcommission(commission=commission)
        # 슬리피지: 체결가 대비 slippage_perc 비율만큼 적용 (기본 15bp, 중형주 가정)
        if slippage_perc and slippage_perc > 0:
            self.cerebro.broker.set_slippage_perc(perc=slippage_perc, slip_open=False)
        self.initial_cash = 100_000_000
        self.cerebro.broker.setcash(self.initial_cash)
        # 체결 보장을 위해 cheat-on-close 사용
        self.cerebro.broker.set_coc(True)

    def add_feeds(self, feeds: List[bt.feeds.PandasData]) -> None:
        if not feeds:
            raise ValueError("no feeds to add")
        for feed in feeds:
            self.cerebro.adddata(feed)

    def run(self):
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns", timeframe=bt.TimeFrame.Days, fund=True)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addstrategy(VolTargetAdapter)
        results = self.cerebro.run()
        self._sharpe = None
        self._sortino = None
        self._mdd = None
        self._mdd_duration = None
        self._trades = None
        self._returns = None

        if results and hasattr(results[0], "analyzers"):
            analyzers = results[0].analyzers
            self._sharpe = analyzers.sharpe.get_analysis().get("sharperatio")

            returns = analyzers.returns.get_analysis()
            self._returns = returns
            self._sortino = self._calc_sortino(returns.values())

            drawdown = analyzers.drawdown.get_analysis()
            if drawdown:
                self._mdd = drawdown.max.drawdown
                self._mdd_duration = drawdown.max.len
            self._trades = analyzers.trades.get_analysis()

        return results

    def plot(self, **kwargs):
        """백테스트 결과를 캔들/거래량 차트로 시각화."""
        plot_kwargs = dict(
            style="candlestick",
            barup="lightcoral",
            bardown="blue",
            volup="lightcoral",
            voldown="blue",
            volume=True,
            subplot=True,
            iplot=False,
        )
        plot_kwargs.update(kwargs)
        return self.cerebro.plot(**plot_kwargs)

    def get_value(self) -> float:
        return self.cerebro.broker.getvalue()

    def summary(self) -> float:
        return self.get_value()

    def sharpe(self):
        return getattr(self, "_sharpe", None)

    def sortino(self):
        return getattr(self, "_sortino", None)

    def max_drawdown(self):
        return getattr(self, "_mdd", None)

    def max_drawdown_duration(self):
        return getattr(self, "_mdd_duration", None)

    def metrics(self) -> dict:
        return {
            "sharpe": self.sharpe(),
            "sortino": self.sortino(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration": self.max_drawdown_duration(),
        }

    def trades(self):
        return getattr(self, "_trades", None)

    def equity_curve(self) -> Optional[pd.Series]:
        """누적 자산 곡선 (returns analyzer 기반)."""
        returns = getattr(self, "_returns", None)
        if not returns:
            return None
        ser = pd.Series(returns)
        eq = (1 + ser).cumprod() * self.initial_cash
        return eq

    def _calc_sortino(self, returns, periods_per_year: int = 252) -> Optional[float]:
        """Calculate annualized Sortino ratio from iterable of period returns."""
        rets = [r for r in returns if r is not None]
        if not rets:
            return None

        downside = [r for r in rets if r < 0]
        if not downside:
            return None

        mean = float(np.mean(rets))
        downside_dev = math.sqrt(np.mean(np.square(downside)))
        if downside_dev == 0:
            return None

        return (mean * math.sqrt(periods_per_year)) / downside_dev
