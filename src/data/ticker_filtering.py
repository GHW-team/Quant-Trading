import pandas as pd
from src.data.db_manager import DatabaseManager

#db에 저장된 티커를 유니버스별로 분리





#티커 필터링 조건: 최근 20일 동안 수익률 상위 20%, 거래량 상위 50%
class TickerSelector:
    """설정된 분위수 기준을 초과하는 티커만 선별한다."""
    def __init__(self, return_quantile: float = 0.8, volume_quantile: float = 0.5) -> None:
        """성과·유동성을 평가할 분위수 임계값을 보관한다."""
        self.return_quantile = return_quantile
        self.volume_quantile = volume_quantile
    def select(self, metrics: List[TickerMetrics]) -> pd.DataFrame:
        """수익률·거래량 조건을 모두 만족하는 티커만 골라 반환한다."""
        if not metrics:
            return pd.DataFrame()
        metrics_df = pd.DataFrame([metric.__dict__ for metric in metrics])
        top_return = metrics_df['total_return'].quantile(self.return_quantile)
        top_volume = metrics_df['avg_volume'].quantile(self.volume_quantile)
        return metrics_df[
            (metrics_df['total_return'] >= top_return)
            & (metrics_df['avg_volume'] >= top_volume)
        ].reset_index(drop=True)

