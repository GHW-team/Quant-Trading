from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import yfinance as yf
try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager
@dataclass
class TickerMetrics:
    ticker: str
    log_volatility: float
    total_return: float
    avg_volume: float

@dataclass
class FilteredUniverse: #
    tickers_by_index: Dict[str, List[str]]
    metrics: pd.DataFrame
    @property
    def flattened(self) -> List[str]:
        merged: List[str] = []
        for tickers in self.tickers_by_index.values():
            merged.extend(tickers)
        return merged
    
class TickerUniverse:
    def __init__(
        self,
        db_manager: DatabaseManager,
        listing_indexes: Optional[Sequence[str]] = None,
    ) -> None:
        self.db_manager = db_manager
        self.listing_indexes = list(listing_indexes) if listing_indexes else None
    def load_grouped(self) -> Dict[str, List[str]]:
        return self.db_manager.fetch_ticker_codes_by_indexes(self.listing_indexes)
    def load(self) -> List[str]:
        tickers: List[str] = []
        for codes in self.load_grouped().values():
            tickers.extend(codes)
        return tickers
    
class MarketDataService:
    def __init__(self, downloader: Optional[Callable[..., pd.DataFrame]] = None) -> None:
        self.downloader = downloader or yf.download
    def fetch(self, tickers: Sequence[str], lookback: int) -> pd.DataFrame:
        tickers_list = list(tickers)
        if not tickers_list:
            return pd.DataFrame()
        period = f"{lookback + 1}d"
        raw_data = self.downloader(tickers_list, period=period)
        if raw_data.empty:
            return pd.DataFrame()
        if isinstance(raw_data.columns, pd.MultiIndex):
            return raw_data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
        single_df = raw_data.reset_index().copy()
        single_df['Ticker'] = tickers_list[0]
        cols = ['Date', 'Ticker'] + [col for col in single_df.columns if col not in ['Date', 'Ticker']]
        return single_df[cols]
    
class TickerStatisticsCalculator:
    def __init__(self, lookback: int) -> None:
        self.lookback = lookback
    def build_metrics(self, data: pd.DataFrame) -> List[TickerMetrics]:
        metrics: List[TickerMetrics] = []
        if data.empty:
            return metrics
        for ticker, ticker_df in data.groupby('Ticker'):
            ticker_df = ticker_df.sort_values('Date').copy()
            if len(ticker_df) <= self.lookback:
                continue
            ticker_df['Return'] = np.log(ticker_df['Close'] / ticker_df['Close'].shift(1))
            rolling_std = ticker_df['Return'].rolling(window=self.lookback).std(ddof=1)
            log_volatility = rolling_std.iloc[-1] * np.sqrt(252)
            total_return = ticker_df['Close'].iloc[-1] / ticker_df['Close'].iloc[-self.lookback - 1] - 1
            avg_volume = ticker_df['Volume'].iloc[-self.lookback:].mean()
            metrics.append(
                TickerMetrics(
                    ticker=ticker,
                    log_volatility=log_volatility,
                    total_return=total_return,
                    avg_volume=avg_volume,
                )
            )
        return metrics
    
class TickerSelector:
    def __init__(self, return_quantile: float = 0.8, volume_quantile: float = 0.5) -> None:
        self.return_quantile = return_quantile
        self.volume_quantile = volume_quantile
    def select(self, metrics: List[TickerMetrics]) -> pd.DataFrame:
        if not metrics:
            return pd.DataFrame()
        metrics_df = pd.DataFrame([asdict(metric) for metric in metrics])
        top_return = metrics_df['total_return'].quantile(self.return_quantile)
        top_volume = metrics_df['avg_volume'].quantile(self.volume_quantile)
        return metrics_df[
            (metrics_df['total_return'] >= top_return)
            & (metrics_df['avg_volume'] >= top_volume)
        ].reset_index(drop=True)
    
class TickerFilteringWorkflow:
    def __init__(
        self,
        db_path: str = "data/database/stocks.db",
        listing_indexes: Optional[Sequence[str]] = None,
        lookback: int = 20,
        return_quantile: float = 0.8,
        volume_quantile: float = 0.5,
        downloader: Optional[Callable[..., pd.DataFrame]] = None,
    ) -> None:
        self.db_manager = DatabaseManager(db_path=db_path)
        self.universe = TickerUniverse(self.db_manager, listing_indexes=listing_indexes)
        self.market_data = MarketDataService(downloader=downloader)
        self.statistics = TickerStatisticsCalculator(lookback=lookback)
        self.selector = TickerSelector(
            return_quantile=return_quantile,
            volume_quantile=volume_quantile,
        )
    def run(self) -> FilteredUniverse:
        grouped_tickers = self.universe.load_grouped()
        if not grouped_tickers:
            print("No tickers found for the requested listing indexes.")
            return FilteredUniverse(tickers_by_index={}, metrics=pd.DataFrame())
        filtered_by_index: Dict[str, List[str]] = {}
        selected_frames: List[pd.DataFrame] = []
        for index_name, tickers in grouped_tickers.items():
            if not tickers:
                continue
            market_data = self.market_data.fetch(tickers, self.statistics.lookback)
            metrics = self.statistics.build_metrics(market_data)
            selected_df = self.selector.select(metrics)
            if selected_df.empty:
                continue
            selected_df.insert(0, 'listing_index', index_name)
            filtered_by_index[index_name] = selected_df['ticker'].tolist()
            selected_frames.append(selected_df)
        combined_df = (
            pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
        )
        if not combined_df.empty:
            print(combined_df)
        print("Filtered tickers by index:", filtered_by_index)
        return FilteredUniverse(tickers_by_index=filtered_by_index, metrics=combined_df)
    
if __name__ == '__main__':
    workflow = TickerFilteringWorkflow()
    workflow.run()