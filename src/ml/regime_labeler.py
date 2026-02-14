"""
레짐(Trend vs Range) 라벨링 모듈

y(t) = 1 (추세장) 조건:
  (a) 미래 H일 중 MA50 > MA200 상태가 70% 이상
  (b) |미래 수익률| >= 1% (노이즈 제거)
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RegimeLabeler:
    """
    시장 국면(Trend vs Range) 라벨링 클래스
    기준 자산: SPY (S&P500 ETF)
    """
    
    def __init__(self, horizon: int = 20, threshold: float = 0.01, trend_ratio: float = 0.7):
        self.horizon = horizon          # 미래 관측 기간 (영업일), 약 1개월
        self.threshold = threshold      # 최소 변동폭 (노이즈 필터), 1%
        self.trend_ratio = trend_ratio  # MA 정배열 비율 기준, 70%

    def label_data(self, df: pd.DataFrame, price_col: str = 'adj_close') -> pd.DataFrame:
        """
        SPY 데이터프레임에 'label' 컬럼을 추가합니다.
        
        Args:
            df: SPY OHLCV + 지표 데이터 (ma_50, ma_200 컬럼 필요)
            price_col: 가격 컬럼명
            
        Returns:
            label 컬럼이 추가된 DataFrame
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # 이평선이 없으면 계산
        if 'ma_50' not in df.columns:
            df['ma_50'] = df[price_col].rolling(50).mean()
        if 'ma_200' not in df.columns:
            df['ma_200'] = df[price_col].rolling(200).mean()
        
        # ---- 조건 (a): 미래 H일 중 MA50 > MA200 비율 >= 70% ----
        # 정배열 여부를 0/1로 표시
        is_golden = (df['ma_50'] > df['ma_200']).astype(int)
        
        # "미래" H일을 앞으로 내다보는 rolling sum
        # FixedForwardWindowIndexer: 현재 시점부터 앞으로 window_size만큼 참조
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon)
        future_golden_count = is_golden.rolling(window=indexer, min_periods=self.horizon).sum()
        
        golden_ratio = future_golden_count / self.horizon
        condition_a = golden_ratio >= self.trend_ratio
        
        # ---- 조건 (b): |미래 수익률| >= 1% ----
        #future_price = df[price_col].shift(-self.horizon)
        #future_return = (future_price / df[price_col]) - 1
        #condition_b = future_return.abs() >= self.threshold
        
        # ---- 라벨 결합 ----
        #df['label'] = np.where(condition_a & condition_b, 1, 0) # 조건(condition_a & condition_b)이 모두 만족하면 1, 아니면 0
        df['label'] = condition_a

        # 마지막 horizon 기간은 미래를 모르므로 NaN 처리
        df.loc[df.index[-self.horizon:], 'label'] = np.nan
        
        # 통계 로그
        valid = df.dropna(subset=['label'])
        trend_pct = valid['label'].mean()
        logger.info(f"레짐 라벨링 완료 (H={self.horizon}, 정배열 비율={self.trend_ratio:.0%}, 최소 변동={self.threshold:.1%})")
        logger.info(f"  - 전체: {len(valid)}일, Trend: {trend_pct:.1%}, Range: {1-trend_pct:.1%}")
        
        return df
