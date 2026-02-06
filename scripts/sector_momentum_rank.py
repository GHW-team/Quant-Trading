import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DAILY_DB = "data/database/stocks.db"


def load_prices(db_path: str, tickers: list[str]) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT dp.date, t.ticker_code, dp.adj_close
            FROM daily_prices dp
            JOIN tickers t ON t.ticker_id = dp.ticker_id
            WHERE t.ticker_code IN ({})
            ORDER BY dp.date
            """.format(",".join(["?"] * len(tickers))),
            conn,
            params=tuple(tickers),
            parse_dates=["date"],
        )
    if df.empty:
        raise ValueError("No price data returned.")
    return df.pivot(index="date", columns="ticker_code", values="adj_close").sort_index()


def load_ticker_list(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT ticker_code FROM tickers ORDER BY ticker_code", conn)
    return df["ticker_code"].tolist()


def compute_above_ma_ratio(prices: pd.Series, ma_window: int, lookback: int) -> float:
    ma = prices.rolling(ma_window).mean()
    above = prices > ma
    return float(above.tail(lookback).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank tickers by time-above-MA ratio")
    parser.add_argument("--db", default=DAILY_DB)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--ma-window", type=int, default=200)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--out-csv", default="data/backtest/sector_momentum_rank.csv")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = load_ticker_list(args.db)
    if not tickers:
        raise ValueError("No tickers found.")

    prices = load_prices(args.db, tickers)
    if args.start:
        prices = prices.loc[pd.Timestamp(args.start):]
    if args.end:
        prices = prices.loc[:pd.Timestamp(args.end)]
    if prices.empty:
        raise ValueError("No price data after date filtering.")

    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if len(s) < args.ma_window + args.lookback:
            continue
        ratio = compute_above_ma_ratio(s, args.ma_window, args.lookback)
        rows.append({"ticker": ticker, "above_ma_ratio": ratio})

    out = pd.DataFrame(rows).sort_values("above_ma_ratio", ascending=False)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.head(args.top_n))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
