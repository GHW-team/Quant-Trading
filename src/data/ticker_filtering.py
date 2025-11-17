from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import yfinance as yf
try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager

class TickerMetrics:
    """Container for volatility/return/volume statistics per ticker."""
    def __init__(
        self,
        ticker: str,
        log_volatility: float,
        total_return: float,
        avg_volume: float,
    ) -> None:
        self.ticker = ticker
        self.log_volatility = log_volatility
        self.total_return = total_return
        self.avg_volume = avg_volume

class FilteredUniverse: #
    """Container for tickers grouped by index and their metrics."""
    def __init__(self, tickers_by_index: Dict[str, List[str]], metrics: pd.DataFrame) -> None:
        self.tickers_by_index = tickers_by_index
        self.metrics = metrics
    @property
    def flattened(self) -> List[str]:
        """모든 지수의 티커를 하나의 리스트로 합쳐 반환한다."""
        merged: List[str] = []
        for tickers in self.tickers_by_index.values():
            merged.extend(tickers)
        return merged
    
class TickerUniverse:
    """데이터베이스에서 티커 목록을 불러오는 역할을 한다."""
    def __init__(
        self,
        db_manager: DatabaseManager,
        listing_indexes: Optional[Sequence[str]] = None,
    ) -> None:
        """DB 매니저와 선택적 상장 지수 필터를 저장한다."""
        self.db_manager = db_manager
        self.listing_indexes = list(listing_indexes) if listing_indexes else None
    def load_grouped(self) -> Dict[str, List[str]]:
        """DB에서 상장 지수별로 묶인 티커를 가져온다."""
        return self.db_manager.fetch_ticker_codes_by_indexes(self.listing_indexes)
    def load(self) -> List[str]:
        """선택된 전체 유니버스의 티커를 평탄화해 반환한다."""
        tickers: List[str] = []
        for codes in self.load_grouped().values():
            tickers.extend(codes)
        return tickers
    
class MarketDataService:
    """여러 티커의 과거 시세 데이터를 내려받는다."""
    def __init__(self, downloader: Optional[Callable[..., pd.DataFrame]] = None) -> None:
        """다운로드 함수를 주입받을 수 있게 하며 기본값은 yfinance다."""
        self.downloader = downloader or yf.download
    def fetch(self, tickers: Sequence[str], lookback: int) -> pd.DataFrame:
        """지정한 기간의 OHLCV 데이터를 내려받아 평탄화된 DataFrame으로 변환한다."""
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
    """변동성·수익률·평균 거래량 등 파생 지표를 계산한다."""
    def __init__(self, lookback: int) -> None:
        """롤링 통계를 계산할 때 사용할 기간을 저장한다."""
        self.lookback = lookback
    def build_metrics(self, data: pd.DataFrame) -> List[TickerMetrics]:
        """각 티커에 대해 변동성, 누적 수익률, 평균 거래량을 계산한다."""
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
    
class TickerFilteringWorkflow:
    """티커 로딩·분석·필터링 단계를 하나의 워크플로로 묶는다."""
    def __init__(
        self,
        db_path: str = "data/database/stocks.db",
        listing_indexes: Optional[Sequence[str]] = None,
        lookback: int = 20,
        return_quantile: float = 0.8,
        volume_quantile: float = 0.5,
        downloader: Optional[Callable[..., pd.DataFrame]] = None,
    ) -> None:
        """설정값에 맞춰 파이프라인 구성 요소를 초기화한다."""
        self.db_manager = DatabaseManager(db_path=db_path)
        self.universe = TickerUniverse(self.db_manager, listing_indexes=listing_indexes)
        self.market_data = MarketDataService(downloader=downloader)
        self.statistics = TickerStatisticsCalculator(lookback=lookback)
        self.selector = TickerSelector(
            return_quantile=return_quantile,
            volume_quantile=volume_quantile,
        )
    def run(self) -> FilteredUniverse:
        """전체 파이프라인을 실행해 필터링된 유니버스를 반환한다."""
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
