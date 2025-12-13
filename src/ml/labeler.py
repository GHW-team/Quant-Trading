import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Labeler:
    """
    고정 기간 수익률 기반 라벨링 클래스

    Parameters:
    - horizon: 미래 기간 (일) - 기본값 5
    - threshold: 수익률 임계값 - 기본값 0.02 (2%)
    """
    def __init__(self, horizon: int = 5, threshold: float = 0.02):
        self.horizon = horizon      # N일 뒤 (예: 5일)
        self.threshold = threshold  # 목표 수익률 (예: 2%)

    def label_data(self, df: pd.DataFrame, price_col: str = 'adj_close') -> pd.DataFrame:
        """
        데이터프레임에 'label' 컬럼을 추가하여 반환합니다.
        """
        #가격 존재하는지 검증
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in DataFrame")

        #날짜 존재하는지 검증
        if 'date' not in df.columns:
            raise ValueError(f"Column 'date' not found in DataFrame")

        df = df.copy()
        
        # 정렬 보장
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        elif not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # 1. 미래 가격 가져오기 (shift 음수는 위로 당김)
        future_close = df[price_col].shift(-self.horizon)
        
        # 2. 수익률 계산: (미래가격 - 현재가격) / 현재가격
        df['return_n'] = (future_close - df[price_col]) / df[price_col]
        
        # 3. 라벨링: 수익률 >= 2% 이면 1, 아니면 0
        # NaN(미래 데이터 없음)은 0으로 처리하거나 나중에 dropna() 해야 함.
        # 여기서는 계산 가능한 구간만 1/0 처리하고 나머지는 NaN 유지
        df['label'] = (df['return_n'] >= self.threshold).astype(int)
        
        # 마지막 horizon 기간은 미래를 알 수 없으므로 NaN 처리
        df.loc[df.index[-self.horizon:], 'label'] = np.nan

        #['retrun_n']수익률 컬럼 제거
        df.drop(columns=['return_n'], inplace=True)
        
        # 통계 로그
        valid_data = df.dropna(subset=['label'])
        pos_ratio = valid_data['label'].mean()
        logger.info(f"Labeling Complete (N={self.horizon}, K={self.threshold:.1%})")
        logger.info(f"  - Total: {len(valid_data)} rows")
        logger.info(f"  - Up(1): {pos_ratio:.1%} | Down/Neutral(0): {1-pos_ratio:.1%}")

        return df