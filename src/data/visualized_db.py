import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

db_path = "data/database/stocks.db"
ticker_code = "005930.KS"
output_path = Path("data/database/chart.png")  # 저장 경로

with sqlite3.connect(db_path) as conn:
    # DB 스키마: tickers(ticker_code) -> daily_prices(ticker_id) -> technical_indicators(ticker_id, date)
    df = pd.read_sql(
        """
        SELECT p.date,
               p.close,
               ti.ma_20,
               ti.ma_60,
               ti.macd,
               ti.macd_signal,
               ti.macd_hist,
               ti.rsi,
               ti.bb_upper,
               ti.bb_mid,
               ti.bb_lower,
               ti.bb_pct,
               ti.atr,
               ti.hv,
               ti.stoch_k,
               ti.stoch_d,
               ti.obv
        FROM daily_prices AS p
        JOIN tickers AS t ON p.ticker_id = t.ticker_id
        LEFT JOIN technical_indicators AS ti
               ON ti.ticker_id = t.ticker_id AND ti.date = p.date
        WHERE t.ticker_code = ?
        ORDER BY p.date
        """,
        conn,
        params=[ticker_code],
        parse_dates=["date"],
    ).set_index("date")

numeric_cols = [
    "close",
    "ma_20",
    "ma_60",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "bb_pct",
    "atr",
    "hv",
    "stoch_k",
    "stoch_d",
    "obv",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if df.empty:
    raise SystemExit(f"No data found for ticker {ticker_code}. Check if the pipeline populated technical_indicators.")

# 가격과 지표를 함께 보기 좋은 순서로 배치
fig, axes = plt.subplots(7, 1, figsize=(12, 20), sharex=True)

# 가격 + 이동평균 + 볼린저 밴드
df[["close", "ma_20", "ma_60", "bb_upper", "bb_mid", "bb_lower"]].plot(ax=axes[0])
axes[0].set_title(f"{ticker_code} Price / MAs / BBands")

# MACD
df[["macd", "macd_signal"]].plot(ax=axes[1])
axes[1].bar(df.index, df["macd_hist"], color="gray", alpha=0.3, label="macd_hist")
axes[1].axhline(0, color="gray", lw=0.5)
axes[1].legend()
axes[1].set_title("MACD")

# RSI
df[["rsi"]].plot(ax=axes[2])
axes[2].axhline(70, color="red", lw=0.8, linestyle="--")
axes[2].axhline(30, color="green", lw=0.8, linestyle="--")
axes[2].set_title("RSI")

# Stochastic
df[["stoch_k", "stoch_d"]].plot(ax=axes[3])
axes[3].axhline(80, color="red", lw=0.8, linestyle="--")
axes[3].axhline(20, color="green", lw=0.8, linestyle="--")
axes[3].set_title("Stochastic %K / %D")

# ATR (원래 스케일)
df[["atr"]].plot(ax=axes[4])
axes[4].set_title("ATR")

# HV (원래 스케일, 예: 0.24 = 24% 연환산)
df[["hv"]].plot(ax=axes[5])
axes[5].set_title("Historical Volatility (annualized)")

# OBV
df[["obv"]].plot(ax=axes[6])
axes[6].set_title("OBV")

plt.tight_layout()
try:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved chart to: {output_path.resolve()}")
except PermissionError:
    print(f"Permission denied saving to {output_path}. Change output_path to a writable location.")
plt.show()
