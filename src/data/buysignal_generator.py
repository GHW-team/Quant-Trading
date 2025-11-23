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
    MACD 계열 피처로 단기(KOSPI 대비) 초과수익 확률을 추정한다.
    """

    def __init__(
        self,
        horizon_days: int = 20,  # 약 1개월 영업일(단기 홀딩, 민감도 분석시 조정)
        abs_mom_window: int = 63,  # 약 3개월 영업일(단기 모멘텀 창, 민감도 분석시 조정)
        ema_col: str = "ma_60",
        min_rows: int | None = None,
        benchmark_col: str = "kospi_close",  # 기준 지수 종가
        proba_threshold: float = 0.55,  # 신호 평가 기본 임계값
    ) -> None:
        self.horizon_days = horizon_days
        self.abs_mom_window = abs_mom_window
        self.ema_col = ema_col
        # 최소 필요 데이터 길이: 모멘텀 창 + 미래 수익 계산 구간
        self.min_rows = min_rows or (self.abs_mom_window + self.horizon_days)
        self.benchmark_col = benchmark_col
        self.proba_threshold = proba_threshold
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
            "abs_mom",
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
            }
        )
        return renamed

    def _build_features_for_group(self, df: pd.DataFrame) -> pd.DataFrame:
        # 티커별 시계열 정렬 후 피처/라벨 생성
        df = df.sort_values("date").copy()
        if len(df) < self.min_rows:
            return pd.DataFrame()

        df["abs_mom"] = df["adj_close"].pct_change(self.abs_mom_window)
        df["ema60_ratio"] = df["adj_close"] / df[self.ema_col] - 1
        df["macd_delta"] = df["macd"].diff()
        df["fwd_stock_ret"] = df["adj_close"].shift(-self.horizon_days) / df["adj_close"] - 1
        df["fwd_bench_ret"] = df[self.benchmark_col].shift(-self.horizon_days) / df[self.benchmark_col] - 1
        df["label"] = (df["fwd_stock_ret"] - df["fwd_bench_ret"] > 0).astype(int)

        # 필터: 절대모멘텀>0, 가격>EMA60
        universe_mask = (df["abs_mom"] > 0) & (df["adj_close"] > df[self.ema_col])
        df = df[universe_mask].copy()

        cols_needed = self.feature_cols + ["label", "date", "ticker"]
        return df[cols_needed].dropna()

    def build_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """ticker별로 피처/라벨을 만든 뒤 하나로 합친다."""
        df = self._normalize_columns(df)
        # EMA가 없는 경우 adj_close로 계산해 채워준다(데이터 누락 보완용)
        if self.ema_col not in df.columns and {"ticker", "date", "adj_close"} <= set(df.columns):
            try:
                span = int(self.ema_col.split("_")[-1])
                df[self.ema_col] = (
                    df.sort_values(["ticker", "date"])
                    .groupby("ticker")["adj_close"]
                    .transform(lambda s: s.ewm(span=span, adjust=False).mean())
                )
            except Exception:
                pass

        required = {"ticker", "date", "adj_close", self.ema_col, "macd", self.benchmark_col}
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
        """시계열 CV로 점수/PR 지표를 계산한 뒤 전체 데이터로 재학습한다."""
        if dataset.empty:
            raise ValueError("학습할 데이터가 없습니다.")
        X = dataset[self.feature_cols].values
        y = dataset["label"].values

        # 시계열 분할로 정확도 + PR 지표 대략 확인
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        y_true_all: list[float] = []
        y_score_all: list[float] = []
        for train_idx, test_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
            cv_scores.append(self.model.score(X[test_idx], y[test_idx]))
            proba_fold = self.model.predict_proba(X[test_idx])[:, 1]
            y_true_all.extend(y[test_idx])
            y_score_all.extend(proba_fold)

        pr_auc = None
        precision_at_thr = None
        recall_at_thr = None
        try:
            from sklearn.metrics import average_precision_score, precision_recall_curve

            pr_auc = float(average_precision_score(y_true_all, y_score_all))
            precision, recall, thresholds = precision_recall_curve(y_true_all, y_score_all)
            thr = self.proba_threshold
            mask = thresholds >= thr
            if mask.any():
                idx = mask.argmax()
                precision_at_thr = float(precision[idx])
                recall_at_thr = float(recall[idx])
        except Exception:
            pass

        # 전체로 재학습
        self.model.fit(X, y)
        return {
            "cv_scores": cv_scores,
            "pr_auc": pr_auc,
            "precision_at_threshold": precision_at_thr,
            "recall_at_threshold": recall_at_thr,
            "threshold": self.proba_threshold,
            "n_samples": len(dataset),
        }

    def predict_proba(self, dataset: pd.DataFrame) -> pd.Series:
        # 매수 성공 확률(buy_proba) 반환
        if dataset.empty:
            return pd.Series(dtype=float)
        X = dataset[self.feature_cols].values
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=dataset.index, name="buy_proba")
