import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DAILY_DB = "data/database/stocks.db"


def load_opens(db_path: str, tickers: list[str]) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT dp.date, t.ticker_code AS ticker, dp.open
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
        raise ValueError("No open prices returned.")
    return df.pivot(index="date", columns="ticker", values="open").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strength vs forward return analysis")
    parser.add_argument("--db", default=DAILY_DB)
    parser.add_argument("--events", default="data/backtest/breakout_any_events.csv")
    parser.add_argument("--horizons", default="5,20,60")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--output", default="data/backtest/breakout_strength_analysis.csv")
    args = parser.parse_args()

    events = pd.read_csv(args.events, parse_dates=["date"])
    events = events[events["event"] == "entry"].copy()
    events = events.dropna(subset=["strength", "ticker"])
    if events.empty:
        raise ValueError("No entry events with strength found.")

    tickers = sorted(events["ticker"].unique())
    opens = load_opens(args.db, tickers)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    results = []

    for h in horizons:
        # forward open-to-open return from entry date to entry+H
        fwd = (opens.shift(-h) / opens) - 1.0
        fwd = fwd.stack().rename("fwd_ret").reset_index()
        fwd.columns = ["date", "ticker", "fwd_ret"]

        merged = events.merge(fwd, on=["date", "ticker"], how="left")
        merged = merged.dropna(subset=["fwd_ret", "strength"])
        if merged.empty:
            continue

        merged["strength_bin"] = pd.qcut(merged["strength"], args.bins, labels=False, duplicates="drop")
        grouped = merged.groupby("strength_bin")
        for bin_id, g in grouped:
            results.append(
                {
                    "horizon": h,
                    "bin": int(bin_id),
                    "count": int(g.shape[0]),
                    "mean_ret": float(g["fwd_ret"].mean()),
                    "median_ret": float(g["fwd_ret"].median()),
                    "win_rate": float((g["fwd_ret"] > 0).mean()),
                }
            )

    out = pd.DataFrame(results).sort_values(["horizon", "bin"])
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
