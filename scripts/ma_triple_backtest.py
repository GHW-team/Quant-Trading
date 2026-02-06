import argparse
import math
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DAILY_DB = "data/database/stocks.db"


def load_ohlc(db_path: str, tickers: list[str]) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT dp.date, t.ticker_code, dp.open, dp.close
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
    return df


def load_ticker_list(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT ticker_code FROM tickers ORDER BY ticker_code", conn)
    return df["ticker_code"].tolist()


def performance_stats(equity: pd.Series) -> dict:
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {"CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    mdd = drawdown.min()
    sharpe = (returns.mean() / returns.std()) * math.sqrt(252) if returns.std() else 0.0
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe}


def annual_trades(pos: pd.DataFrame) -> float:
    changes = pos.ne(pos.shift(1)).sum().sum()
    years = (pos.index[-1] - pos.index[0]).days / 365.25
    return changes / years if years > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Triple MA trend strategy (20>50>100 entry, 20<50 exit)")
    parser.add_argument("--db", default=DAILY_DB)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--all-universe", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--ma-short", type=int, default=20)
    parser.add_argument("--ma-mid", type=int, default=50)
    parser.add_argument("--ma-long", type=int, default=100)
    parser.add_argument("--output-dir", default="data/backtest")
    args = parser.parse_args()

    if args.all_universe:
        tickers = load_ticker_list(args.db)
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = [args.ticker] if args.ticker else load_ticker_list(args.db)

    raw = load_ohlc(args.db, tickers)
    if args.start:
        raw = raw[raw["date"] >= pd.Timestamp(args.start)]
    if args.end:
        raw = raw[raw["date"] <= pd.Timestamp(args.end)]
    if raw.empty:
        raise ValueError("No data after date filtering.")

    opens = raw.pivot(index="date", columns="ticker_code", values="open").sort_index()
    closes = raw.pivot(index="date", columns="ticker_code", values="close").sort_index()

    ma_s = closes.rolling(args.ma_short).mean()
    ma_m = closes.rolling(args.ma_mid).mean()
    ma_l = closes.rolling(args.ma_long).mean()

    dates = closes.index
    pos = pd.DataFrame(0, index=dates, columns=closes.columns, dtype=int)
    open_ret = opens.shift(-1) / opens - 1.0

    for i in range(1, len(dates)):
        prev = dates[i - 1]
        dt = dates[i]
        # entry condition at prev close
        entry = (ma_s.loc[prev] > ma_m.loc[prev]) & (ma_m.loc[prev] > ma_l.loc[prev])
        # exit condition at prev close
        exit_ = (ma_m.loc[prev] < ma_l.loc[prev])
        pos.loc[dt] = pos.loc[prev]
        pos.loc[dt][exit_] = 0
        pos.loc[dt][entry] = 1

    # portfolio return: equal weight across active tickers
    active = pos.astype(bool)
    active_count = active.sum(axis=1).replace(0, np.nan)
    daily_ret = (open_ret.where(active).sum(axis=1) / active_count).fillna(0.0)
    equity = (1 + daily_ret).cumprod()

    stats = performance_stats(equity)
    trades_per_year = annual_trades(pos)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, label="equity")
    ax.set_title("Triple MA trend (20>50>100, exit 20<50)")
    ax.grid(True)
    ax.legend()
    fig_path = out_dir / "ma_triple_equity.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    stats_path = out_dir / "ma_triple_stats.csv"
    pd.DataFrame([{
        "strategy": "ma_triple",
        "CAGR": stats["CAGR"],
        "MDD": stats["MDD"],
        "Sharpe": stats["Sharpe"],
        "Trades_per_year": trades_per_year,
        "ma_short": args.ma_short,
        "ma_mid": args.ma_mid,
        "ma_long": args.ma_long,
        "start": dates.min().date(),
        "end": dates.max().date(),
    }]).to_csv(stats_path, index=False)

    print("Backtest stats")
    print(
        f"CAGR: {stats['CAGR']:.4f}\nMDD: {stats['MDD']:.4f}\nSharpe: {stats['Sharpe']:.4f}\nTrades/year: {trades_per_year:.2f}"
    )
    print(f"Saved equity plot: {fig_path}")
    print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
