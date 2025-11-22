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
    # df = df.sort_values(["ticker", "date"]).copy()    # 정렬 안되어 있으면 하면 됨

    def _per_ticker(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"]
        fwd_close = close.shift(-horizon)   # 5칸 앞으로 맨 뒤 5행은 NaN
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
    X = df[feature_cols].values         # (샘플 개수, 8(피처개수)) matrix
    y = df["label"].astype(int).values  # (샘플 개수) vector

    # 4) 시간 기준 train/test split (앞 80% train, 뒤 20% test)
    n = len(df)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5) 스케일러 + 로지스틱 회귀
    scaler = StandardScaler()   # data 표준화
    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
    X_test_scaled = scaler.transform(X_test)
    
    logreg = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")   # solver="lbfgs" -> 경사하강법보다 좋대요
    # logreg = SGDClassifier(
    #             loss="log_loss",      # 로지스틱 회귀 (logistic loss)
    #             penalty="l2",         # L2 규제 (LogisticRegression이랑 비슷)
    #             alpha=0.0001,         # 규제 강도 (C의 역수 느낌)
    #             learning_rate="optimal",  # 러닝레이트 스케줄
    #             eta0=0.01,            # 초기 learning rate
    #             max_iter=1000,        # 에폭 수
    #             tol=1e-3,             # 수렴 기준
    #             random_state=42,
    #         ) -> 확률적 경사하강법(SGD)

    logreg.fit(X_train_scaled, y_train)
    
    # 6) 평가
    y_pred = logreg.predict(X_test_scaled)
    print("=== Classification Report (Absolute Momentum, Logistic Regression) ===")
    print(classification_report(y_test, y_pred, digits=3))

    return logreg, df, feature_cols

def export_labels_to_csv(df_used: pd.DataFrame, horizon, threshold):
    root = pathlib.Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "logistic_regression_labels.csv"

    out_df = df_used[["ticker", "date", "fwd_ret", "label"]].copy()
    out_df["horizon"] = horizon

    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    print("label CSV 저장 완료:", csv_path)

def main():
    db_path = get_db_path()
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parents[1]

    logreg, df_used, feature_cols = train_logreg_abs_mom(db_path=db_path, horizon=5, threshold=0.02)
    export_labels_to_csv(df_used, horizon=5, threshold=0.02)
    
if __name__ == "__main__":
    main()
