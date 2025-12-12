import math
import statistics
from typing import Dict, Iterable, Optional, Sequence, Set


class VolStopEngine:
    """변동성(연율화) 기반 스탑 관리."""

    def __init__(self, k: float = 2.0):
        self.k = k

    def init_stop(self, entry: float, vol: float) -> float:
        if entry is None or vol is None or entry <= 0 or vol <= 0:
            return None
        return entry * (1 - self.k * vol)

    def update_stop(self, prev_stop: float, high: float, vol: float) -> float:
        if high is None or vol is None or high <= 0 or vol <= 0:
            return prev_stop
        new_stop = high * (1 - self.k * vol)
        if prev_stop is None:
            return new_stop
        return max(prev_stop, new_stop)

    def should_exit(self, price: float, stop: float) -> bool:
        if stop is None or price is None:
            return False
        return price <= stop


class PositionSizer:
    """인버스 볼 기반 비중 계산."""

    def __init__(self, target_vol: float = 0.15, vol_cap: float = 0.15, vol_floor: float = 0.0):
        self.target_vol = target_vol
        self.vol_cap = vol_cap
        self.vol_floor = vol_floor

    def weight(self, vol: float) -> float:
        if vol is None or vol <= 0:
            return 0.0
        w = self.target_vol / vol
        w = min(w, self.vol_cap) if self.vol_cap is not None else w
        w = max(w, self.vol_floor) if self.vol_floor is not None else w
        return max(w, 0.0)


class VolTargetLogic:
    """
    모멘텀 필터(lookback_mom 상위 퍼센타일) + 인버스 볼 사이징 + 변동성 스탑의
    순수 로직 모듈. 실행 엔진(backtrader 등)은 외부에서 래핑한다.
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        lookback_vol: int = 20,
        lookback_mom: int = 126,
        mom_top_pct: float = 0.5,
        vol_cap: float = 0.15,
        vol_floor: float = 0.0,
        k_vol_stop: float = 2.0,
    ):
        
        self.target_vol = target_vol
        self.lookback_vol = lookback_vol
        self.lookback_mom = lookback_mom
        self.mom_top_pct = mom_top_pct
        self.vol_cap = vol_cap
        self.vol_floor = vol_floor
        self.stop_engine = VolStopEngine(k=k_vol_stop)
        self.sizer_engine = PositionSizer(target_vol=target_vol, vol_cap=vol_cap, vol_floor=vol_floor)

    def calc_vol(self, closes: Sequence[float]) -> Optional[float]:
        """최근 lookback_vol 연율화 변동성."""
        if closes is None or len(closes) <= self.lookback_vol:
            return None
        try:
            rets = []
            for i in range(1, self.lookback_vol + 1):
                c0 = closes[-i]
                c1 = closes[-i - 1]
                if c0 is None or c1 is None or c0 <= 0 or c1 <= 0:
                    continue
                rets.append(math.log(c0 / c1))
        except Exception:
            return None
        if len(rets) < self.lookback_vol:
            return None
        std = statistics.pstdev(rets)
        return std * math.sqrt(252)

    def calc_momentum(self, closes: Sequence[float]) -> Optional[float]:
        """lookback_mom 기준 모멘텀(단순 수익률)."""
        if closes is None or len(closes) <= self.lookback_mom:
            return None
        old = closes[-self.lookback_mom - 1]
        cur = closes[-1]
        if old is None or cur is None or old <= 0:
            return None
        return cur / old - 1

    def select_top_momentum(self, mom_map: Dict[str, Optional[float]]) -> Set[str]:
        """모멘텀 상위 퍼센타일 티커 선택."""
        valid = [(sym, m) for sym, m in mom_map.items() if m is not None]
        if not valid:
            return set()
        valid.sort(key=lambda x: x[1], reverse=True)
        top_n = max(1, int(math.ceil(len(valid) * self.mom_top_pct)))
        cutoff = valid[top_n - 1][1]
        return {sym for sym, m in valid if m >= cutoff}

    def weight(self, vol: Optional[float]) -> float:
        return self.sizer_engine.weight(vol)

    def init_stop(self, entry: float, vol: Optional[float]) -> Optional[float]:
        return self.stop_engine.init_stop(entry, vol)

    def update_stop(self, prev_stop: Optional[float], high: float, vol: Optional[float]) -> Optional[float]:
        return self.stop_engine.update_stop(prev_stop, high, vol)

    def should_exit(self, price: float, stop: Optional[float]) -> bool:
        return self.stop_engine.should_exit(price, stop)
