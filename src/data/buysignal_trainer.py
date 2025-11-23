import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.db_manager import DatabaseManager
from src.data.buysignal_generator import LogisticBuySignalModel


class BuySignalTrainer:
    """DB 가격·지표와 벤치마크 지수를 병합해 매수 신호 모델을 학습·예측하는 헬퍼."""

    def __init__(
        self,
        db_path: str = "data/database/stocks.db",
        start_date: str = "2017-01-01",
        end_date: str = "2022-12-31",
        indicators: Optional[List[str]] = None,
        benchmark_col: str = "kospi_close",
        ema_col: str = "ma_60",
    ) -> None:
        # DB 연결 + 모델, 학습 구간, 사용할 지표/벤치마크 컬럼을 설정
        self.db = DatabaseManager(db_path=db_path)
        self.model = LogisticBuySignalModel(ema_col=ema_col, benchmark_col=benchmark_col)
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_col = benchmark_col
        self.indicators = indicators or [
            "ma_5",
            "ma_10",
            "ma_20",
            "ma_50",
            "ma_60",
            "ma_120",
            "ma_200",
            "macd",
            "macd_signal",
            "macd_hist",
        ]

    @staticmethod
    def _normalize_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        # 날짜 컬럼을 문자열(YYYY-MM-DD)로 통일
        if date_col in df.columns:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col]).dt.date.astype(str)
        return df

    @staticmethod
    def _normalize_benchmark_df(bench_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        # 벤치마크 데이터 컬럼명/날짜 형식을 정규화
        df = bench_df.copy()
        df.columns = df.columns.str.strip().str.lower()
        if date_col not in df.columns:
            raise ValueError(f"Required column '{date_col}' missing in benchmark df")
        df[date_col] = pd.to_datetime(df[date_col]).dt.date.astype(str)
        return df

    def load_dataset(self, tickers: List[str], benchmark_df: pd.DataFrame) -> pd.DataFrame:
        # 티커별 가격+지표+벤치마크를 병합해 학습용 데이터셋 생성
        if not tickers:
            raise ValueError("Ticker list is empty")

        benchmark_df = self._normalize_benchmark_df(benchmark_df)
        if self.benchmark_col not in benchmark_df.columns:
            raise ValueError(f"Required column '{self.benchmark_col}' missing in benchmark df")

        prices = self.db.load_price_data(tickers, start_date=self.start_date, end_date=self.end_date)
        indicators = self.db.load_indicators(tickers, start_date=self.start_date, end_date=self.end_date)

        frames: List[pd.DataFrame] = []
        for ticker in tickers:
            price_df = self._normalize_dates(prices.get(ticker, pd.DataFrame()))
            ind_df = self._normalize_dates(indicators.get(ticker, pd.DataFrame()))
            # 가격/지표가 비어 있거나, EMA/핵심 지표가 NaN이라면 스킵
            if price_df.empty or ind_df.empty:
                logging.warning("%s: missing price or indicator data; price_rows=%d, ind_rows=%d", ticker, len(price_df), len(ind_df))
                continue
            needed = ["ma_60", "macd", "macd_hist"]
            if not set(needed).issubset(ind_df.columns):
                logging.warning("%s: required indicator columns missing; have=%s", ticker, list(ind_df.columns))
                continue
            if ind_df[needed].isna().any(axis=None):
                logging.warning("%s: core indicators contain NaN; skipping", ticker)
                continue

            merged = price_df.merge(ind_df, on="date", how="left")
            merged["ticker"] = ticker
            merged = merged.merge(benchmark_df[["date", self.benchmark_col]], on="date", how="left")
            frames.append(merged)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fit(self, dataset: pd.DataFrame) -> dict:
        # 데이터셋으로 모델 학습 및 CV 점수 리턴
        train_df = self.model.build_dataset(dataset)
        fit_info = self.model.fit(train_df)
        fit_info.update({"samples": len(train_df)})
        return fit_info

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # 데이터셋에 대해 매수 확률 예측을 추가한 결과 반환
        feature_df = self.model.build_dataset(dataset)
        proba = self.model.predict_proba(feature_df)
        return feature_df.assign(buy_proba=proba)

    def generate_signals(
        self,
        tickers: List[str],
        benchmark_df: pd.DataFrame,
        threshold: Optional[float] = None,
        include_prices: bool = True,
    ) -> pd.DataFrame:
        """
        backtrader 등에서 바로 사용할 수 있는 신호/확률 테이블을 만든다.
        threshold 미지정 시 모델 기본 임계값을 사용하며, 필요 시 가격 컬럼까지 병합한다.
        """
        threshold = threshold if threshold is not None else getattr(self.model, "proba_threshold", 0.5)

        # 데이터셋 생성 및 확률/신호 산출
        dataset = self.load_dataset(tickers, benchmark_df)
        feature_df = self.model.build_dataset(dataset)
        proba = self.model.predict_proba(feature_df)
        signals = feature_df.assign(buy_proba=proba, buy_signal=lambda df: df["buy_proba"] >= threshold)

        if not include_prices:
            return signals.sort_values(["date", "ticker"]).reset_index(drop=True)

        prices = self.db.load_price_data(tickers, start_date=self.start_date, end_date=self.end_date)
        frames: List[pd.DataFrame] = []
        for ticker in tickers:
            price_df = self._normalize_dates(prices.get(ticker, pd.DataFrame()))
            sig_df = signals[signals["ticker"] == ticker]
            if price_df.empty or sig_df.empty:
                continue
            merged = sig_df.merge(price_df, on="date", how="left", suffixes=("", "_price"))
            frames.append(merged)

        if not frames:
            return pd.DataFrame()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )


__all__ = ["BuySignalTrainer"]
