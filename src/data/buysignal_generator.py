from src.data.indicator_calculator import IndicatorCalculator
import pandas as pd
import numpy as np


class DifferencingMACD:
    def __init__(self, indicator_calc: IndicatorCalculator | None = None):
        self.calc = indicator_calc or IndicatorCalculator()

    def run(self, price_df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for ticker, ticker_df in price_df.groupby("Ticker"):
            ohlcv = (
                ticker_df.sort_values("Date")
                .rename(columns=lambda c: c.lower().replace(" ", "_"))
                .set_index("date")[["open", "high", "low", "close", "volume", "adj_close"]]
            )
            macd_df = self.calc.calculate_indicators(ohlcv, ["macd"])
            if macd_df is None or "macd" not in macd_df:
                continue
            ticker_macd = macd_df[["macd"]].copy()
            ticker_macd["macd_delta"] = ticker_macd["macd"].diff()
            ticker_macd["Ticker"] = ticker
            ticker_macd = ticker_macd.reset_index().rename(columns={"date": "Date"})
            frames.append(ticker_macd[["Ticker", "Date", "macd", "macd_delta"]])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=40, freq="D")

    def build_block(ticker: str, base_price: float, volume_start: int) -> pd.DataFrame:
        prices = np.linspace(base_price, base_price + 5, len(dates))
        volumes = np.linspace(volume_start, volume_start + 500, len(dates))
        block = pd.DataFrame(
            {
                "Ticker": [ticker] * len(dates),
                "Date": dates,
                "Open": prices + 0.1,
                "High": prices + 0.5,
                "Low": prices - 0.5,
                "Close": prices,
                "Volume": volumes,
            }
        )
        block["Adj Close"] = block["Close"] * 0.99
        return block

    sample = pd.concat(
        [
            build_block("AAA", 10, 1000),
            build_block("BBB", 20, 2000),
        ],
        ignore_index=True,
    )
    macd = DifferencingMACD()
    print(len(sample))
    print(macd.run(sample).tail(30))
