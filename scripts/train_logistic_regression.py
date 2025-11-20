import argparse
import pathlib
import sqlite3

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import get_db_path, get_config


def load_price_and_features(db_path: pathlib.Path) -> pd.DataFrame:
    sql = """
    SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume, f.ma_5, f.ma_20, f.ma_200
    FROM price_daily AS p JOIN features_daily AS f ON p.ticker = f.ticker AND p.date = f.date
    ORDER BY p.ticker, p.date
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, parse_dates=["date"])
    return df

def add_abs_momentum_label(df: pd.DataFrame, horizon, threshold) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    def _per_ticker(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"]
        fwd_close = close.shift(-horizon)
        fwd_ret = fwd_close / close - 1.0
        group["fwd_ret"] = fwd_ret
        group["label"] = (fwd_ret > threshold).astype("Int64")
        return group

    df = df.groupby("ticker", group_keys=False).apply(_per_ticker)
    # 마지막 horizon일은 미래 수익률이 없어서 NaN → 제거
    df = df.dropna(subset=["fwd_ret", "label"])
    return df

def train_logreg_abs_mom(db_path: pathlib.Path, horizon, threshold):
    # 1) 데이터 로딩
    df = load_price_and_features(db_path)

    # 2) 라벨 생성 (절대 모멘텀)
    df = add_abs_momentum_label(df, horizon=horizon, threshold=threshold)

    # if len(df) < 100:
    #     raise ValueError(
    #         f"학습에 사용할 샘플이 너무 적습니다: {len(df)}행. "
    #         "config/settings.json의 fetch.start / fetch.end 구간을 더 길게 잡거나, "
    #         "horizon(모멘텀 기간)을 더 짧게 줄여보세요."
    #     )

    # 3) 피처 / 타겟 분리
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma_5",
        "ma_20",
        "ma_200",
    ]
    df = df.dropna(subset=feature_cols + ["fwd_ret", "label"]).reset_index(drop=True)
    X = df[feature_cols].values
    y = df["label"].astype(int).values

    # 4) 시간 기준 train/test split (앞 80% train, 뒤 20% test)
    n = len(df)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5) 스케일러 + 로지스틱 회귀 파이프라인
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # 6) 평가
    y_pred = model.predict(X_test)
    print("=== Classification Report (Absolute Momentum, Logistic Regression) ===")
    print(classification_report(y_test, y_pred, digits=3))

    return model, df, feature_cols

def export_labels_to_csv(df_used: pd.DataFrame, horizon, threshold) -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "logistic_regression_labels.csv"

    out_df = df_used[["ticker", "date", "fwd_ret", "label"]].copy()
    out_df["horizon"] = horizon

    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    print("라벨 CSV 저장 완료:", csv_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="절대 모멘텀 기간(일 수). 예: 5 → 5일 후 수익률 기준",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="수익률 기준. 예: 0.0 → 0% 초과면 1로 라벨링",
    )
    args = parser.parse_args()

    db_path = get_db_path()
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parents[1]

    print(f"Horizon: {args.horizon}일, Threshold: {args.threshold:.2%}")

    model, df_used, feature_cols = train_logreg_abs_mom(db_path=db_path, horizon=args.horizon, threshold=args.threshold)
    export_labels_to_csv(df_used, horizon=args.horizon, threshold=args.threshold)

if __name__ == "__main__":
    main()
