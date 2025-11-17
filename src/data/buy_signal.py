import pandas as pd
import numpy as np
from src.data.indicator_calculator import IndicatorCalculator
from

class DifferencingMACD:
    def __init__(self, indicator_calc = None):
        self.calc = indicator_calc or IndicatorCalculator() #디폴트 인스턴스를 제공하면서도 필요하면 다른 계산기를 넣을 수 있음
    
    def run(self, price_df: pd.DataFrame) ->pd.DataFrame:
        # price_df는 OHLCV 데이터, 이 클래스는 delta MACD만을 계산하므로 워크플로우에서 price_df를 받아와야함
        frames = []
        for ticker, ticker_df in price_df.groupby("Ticker"):
            ohlcv = (
                ticker_df.sort_values("Date")
                .rename(columns=str.lower)
                .set_index("date")["open", "high", "low", "close", "volume"]
            )
            macd_df = self.calc.calculate_indicators(ohlcv, ["macd"])


