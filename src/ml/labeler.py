import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


#주가가 20일 신고가 돌파시 매수하는 모델. 매수 조건은 단순화시키고 리스크 관리에 집중해서 손실을 줄이는게 목적
class Labeler:
    """
    20일 신고가 돌파 라벨링

    Parameters:
    - breakout_window : 신고가 판단 기간(기본 20일)
    """
    def __init__(self, breakout_window: int=20):
        self.breakout_window = breakout_window

    def label_data(self, df: pd.DataFrame, price_col: str = 'adj_close') -> pd.DataFrame:
        """
        데이터프레임에 'label' 컬럼을 추가하여 반환합니다.
        """
        #가격 존재하는지 검증
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in DataFrame")

        df = df.copy()
        
        # 정렬 보장
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        elif not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # 1. 최근 20일 내 신고가 계산 (shift(1)로 오늘 값이 들어가서 예견편향이 발생하는걸 방지함.)
        prev_high = (
            df[price_col]
            .rolling(window = self.breakout_window, min_periods = self.breakout_window)
            .max()
            .shift(1) #lookahead bias 방지
        )
        
        # 2. 라벨링: 주가가 20일 신고가를 갱신하면 1, 아니면 0
        # 가격 데이터가 20개 미만인 경우는 NaN처리
        df['label'] = np.where(df[price_col] >= prev_high, 1.0, 0.0)
        df.loc[prev_high.isna(), "label"] = np.nan


        #통계 로그
        valid = df.dropna(subset=['label']) # NaN처리된 데이터를 제외한 유효 데이터
        pos_ratio = valid['label'].mean() if not valid.empty else 0.0
        logger.info(
            "Labeling Complete (breakout_window=%d)\n  - Total: %d rows\n  - Breakout(1): %.1f%%",
            self.breakout_window,
            len(valid),
            pos_ratio * 100,
        )

        return df