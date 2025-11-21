import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.indicator_calculator import IndicatorCalculator


class DifferencingMACD:
    """ticker별 MACD와 기울기(macd_delta)를 계산한다."""

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


class LogisticBuySignalModel:
    """
    절대모멘텀 + EMA60 필터를 통과한 종목에 대해
    MACD 델타(음→양)를 주요 신호로 사용하는 로지스틱 회귀 헬퍼.
    """

    def __init__(
        self,
        horizon_days: int = 63,  # 약 3개월 영업일
        abs_mom_window: int = 252,  # 약 12개월 영업일
        ema_col: str = "ma_60",
        min_rows: int | None = None,
        bond_rate_col: str = "bond_3m_rate",  # 일별 3개월물 금리(연율, %)
    ) -> None:
        self.horizon_days = horizon_days
        self.abs_mom_window = abs_mom_window
        self.ema_col = ema_col
        # 최소 필요 데이터 길이: 모멘텀 창 + 미래 수익 계산 구간 + 여유
        self.min_rows = min_rows or (self.abs_mom_window + self.horizon_days + 10)
        self.bond_rate_col = bond_rate_col
        # 스케일러+로지스틱 회귀 파이프라인(불균형 대응 class_weight)
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=300,
                        class_weight="balanced",
                        n_jobs=-1,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.feature_cols = [
            "abs_mom_6m",
            "ema60_ratio",
            "macd",
            "macd_delta",
            "macd_hist",
        ]

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        # 입력 컬럼 이름을 내부 표준으로 통일
        renamed = df.rename(
            columns={
                "Ticker": "ticker",
                "Date": "date",
                "Adj Close": "adj_close",
                "AdjClose": "adj_close",
                "Close": "close",
                "bond": "bond_3m_rate",
            }
        )
        return renamed

    def _build_features_for_group(self, df: pd.DataFrame) -> pd.DataFrame:
        # 티커별 시계열 정렬 후 피처/라벨 생성
        df = df.sort_values("date").copy()
        if len(df) < self.min_rows:
            return pd.DataFrame()

        df["abs_mom_6m"] = df["adj_close"].pct_change(self.abs_mom_window)
        df["ema60_ratio"] = df["adj_close"] / df[self.ema_col] - 1
        df["macd_delta"] = df["macd"].diff()
        df["fwd_stock_ret"] = df["adj_close"].shift(-self.horizon_days) / df["adj_close"] - 1
        # 단순이자 근사: 연율 금리를 영업일 기준 연환산 일수로 나눠 적용
        df["fwd_bond_ret"] = (df[self.bond_rate_col] / 100.0) * (self.horizon_days / 252)
        df["label"] = (df["fwd_stock_ret"] - df["fwd_bond_ret"] > 0).astype(int)

        # 필터: 절대모멘텀>0, 가격>EMA60
        universe_mask = (df["abs_mom_6m"] > 0) & (df["adj_close"] > df[self.ema_col])
        df = df[universe_mask].copy()

        cols_needed = self.feature_cols + ["label", "date", "ticker"]
        return df[cols_needed].dropna()

    def build_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """ticker별로 피처/라벨을 만든 뒤 하나로 합친다."""
        df = self._normalize_columns(df)
        required = {"ticker", "date", "adj_close", self.ema_col, "macd", self.bond_rate_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

        frames = []
        for _, g in df.groupby("ticker"):
            enriched = self._build_features_for_group(g)
            if not enriched.empty:
                frames.append(enriched)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fit(self, dataset: pd.DataFrame) -> dict:
        # 시계열 CV로 대략적인 점수 확인 후 전체 데이터로 재학습
        if dataset.empty:
            raise ValueError("학습할 데이터가 없습니다.")
        X = dataset[self.feature_cols].values
        y = dataset["label"].values

        # 시계열 분할로 검증 지표 대략 확인
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, test_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
            cv_scores.append(self.model.score(X[test_idx], y[test_idx]))

        # 전체로 재학습
        self.model.fit(X, y)
        return {"cv_scores": cv_scores, "n_samples": len(dataset)}

    def predict_proba(self, dataset: pd.DataFrame) -> pd.Series:
        # 매수 성공 확률(buy_proba) 반환
        if dataset.empty:
            return pd.Series(dtype=float)
        X = dataset[self.feature_cols].values
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=dataset.index, name="buy_proba")


# 테스트용 데이터
if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=300, freq="B")

    def build_block(ticker: str, base_price: float, volume_start: int) -> pd.DataFrame:
        prices = np.linspace(base_price, base_price + 15, len(dates))
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

    calc = IndicatorCalculator()
    macd = DifferencingMACD(calc)
    macd_df = macd.run(sample)

    merged = (
        sample.merge(macd_df, on=["Ticker", "Date"], how="left")
        .rename(columns=lambda c: c.lower().replace(" ", "_"))
    )
    merged["ma_60"] = merged.groupby("ticker")["adj_close"].transform(
        lambda s: s.ewm(span=60, adjust=False).mean()
    )

    model = LogisticBuySignalModel()
    dataset = model.build_dataset(merged)
    if not dataset.empty:
        summary = model.fit(dataset)
        print("cv scores:", summary["cv_scores"])
        proba = model.predict_proba(dataset.head(5))
        print("sample proba:", proba.tolist())
    else:
        print("empty dataset")

