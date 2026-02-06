import sqlite3
import numpy as np
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.data.tc_filter import EESD2

LAMBDA = 1600.0
PLOT_START = "2016-01-01"
PLOT_END = "2020-12-31"

# 1) 설정 읽기
cfg = yaml.safe_load(Path("config/run_pipeline.yaml").read_text(encoding="utf-8"))
tickers = cfg["data"]["tickers"]
db_path = cfg["data"]["database_path"]

# 2) DB에서 가격 로드 함수
def load_prices(ticker):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT dp.date, dp.adj_close
            FROM daily_prices dp
            JOIN tickers t ON t.ticker_id = dp.ticker_id
            WHERE t.ticker_code = ?
            ORDER BY dp.date
            """,
            conn,
            params=(ticker,),
            parse_dates=["date"],
        )
    df.set_index("date", inplace=True)
    return df

# 3) 필터 적용 + 플롯
def plot_tc(ticker, lam=LAMBDA, use_log=True):
    df = load_prices(ticker)
    if df.empty:
        print(f"{ticker}: no data")
        return
    df = df.loc[PLOT_START:PLOT_END]
    if df.empty:
        print(f"{ticker}: no data in {PLOT_START} ~ {PLOT_END}")
        return
    filt = EESD2(lam=lam, use_log=use_log)
    res = filt.fit(df["adj_close"])
    if use_log:
        price_level = pd.Series(
            np.exp(res["X"].values), index=df.index, name="price"
        )
        trend_level = pd.Series(
            np.exp(res["XT"].values), index=df.index, name="trend"
        )
        price_log = pd.Series(res["X"].values, index=df.index, name="price_log")
        trend_log = pd.Series(res["XT"].values, index=df.index, name="trend_log")
        resid_series = pd.Series(res["resid"].values, index=df.index, name="resid_log")
    else:
        price_level = pd.Series(res["X"].values, index=df.index, name="price")
        trend_level = pd.Series(res["XT"].values, index=df.index, name="trend")
        price_log = pd.Series(np.log(res["X"].values), index=df.index, name="price_log")
        trend_log = pd.Series(np.log(res["XT"].values), index=df.index, name="trend_log")
        resid_series = pd.Series(res["resid"].values, index=df.index, name="resid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    log_formatter = FuncFormatter(lambda y, _: f"log({y:.0f})" if y > 0 else "log(0)")
    axes[0].plot(price_level.index, price_level, label="price (level)", alpha=0.6)
    axes[0].plot(trend_level.index, trend_level, label=f"trend (level, λ={lam:.3e})", linewidth=1.5)
    axes[0].set_title(f"{ticker} TC filter (d=1, drift)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(price_log.index, price_log, label="price (log)", alpha=0.6)
    axes[1].plot(trend_log.index, trend_log, label="trend (log)", linewidth=1.5)
    axes[1].legend()
    axes[1].grid(True)
    axes[1].yaxis.set_major_formatter(log_formatter)

    axes[2].plot(resid_series.index, resid_series, label="resid (log)" if use_log else "resid", color="gray")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# 4) 모든 티커 순회 예시 (원하면 lam 조정)
for tk in tickers:
    plot_tc(tk, use_log=True)
