import argparse
import math
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
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
    pivot = df.pivot(index="date", columns="ticker_code", values="adj_close").sort_index()
    return pivot


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


def annual_trades(weights: pd.DataFrame) -> float:
    changes = weights.diff().abs().sum(axis=1)
    trades = (changes > 0).sum()
    years = (weights.index[-1] - weights.index[0]).days / 365.25
    return trades / years if years > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate top-N mean-reversion ETFs with vol targeting")
    parser.add_argument("--db", default=DAILY_DB)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebalance", type=str, default="M")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--ret-window", type=int, default=20)
    parser.add_argument("--z-window", type=int, default=60)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--no-leverage", action="store_true")
    parser.add_argument("--market-filter", action="store_true")
    parser.add_argument("--market-ticker", default="SPY")
    parser.add_argument("--trend-window", type=int, default=120)
    parser.add_argument("--output-dir", default="data/backtest")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = load_ticker_list(args.db)
    if not tickers:
        raise ValueError("No tickers found.")
    if args.market_ticker not in tickers:
        tickers.append(args.market_ticker)

    prices = load_prices(args.db, tickers)
    if args.start:
        prices = prices.loc[pd.Timestamp(args.start):]
    if args.end:
        prices = prices.loc[:pd.Timestamp(args.end)]
    if prices.empty:
        raise ValueError("No price data after date filtering.")

    logp = np.log(prices)
    r_ret = logp - logp.shift(args.ret_window)
    z_std = r_ret.rolling(args.z_window).std()
    z = r_ret / (z_std + 1e-12)

    daily_ret = prices.pct_change()
    rebal_dates = prices.resample(args.rebalance).last().index

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    picks = []
    target_daily = args.target_vol / math.sqrt(252)

    market_trend = None
    if args.market_filter:
        if args.market_ticker not in prices.columns:
            raise ValueError(f"Market ticker {args.market_ticker} not in price data.")
        mlog = np.log(prices[args.market_ticker])
        mtrend = mlog.rolling(args.trend_window).mean()
        market_trend = mtrend.diff()

    for dt in rebal_dates:
        if dt not in z.index:
            continue
        if args.market_filter:
            mt = market_trend.loc[dt] if market_trend is not None else np.nan
            if np.isnan(mt) or mt <= 0:
                next_day_idx = prices.index.searchsorted(dt) + 1
                if next_day_idx < len(prices.index):
                    next_day = prices.index[next_day_idx]
                    weights.loc[next_day:] = 0.0
                    picks.append({"date": dt, "tickers": "CASH", "leverage": 0.0})
                continue

        z_t = z.loc[dt]
        z_t = z_t.replace([np.inf, -np.inf], np.nan).dropna()
        if z_t.empty:
            continue

        # mean-reversion: pick most negative z
        selected = z_t.nsmallest(args.top_n).index.tolist()
        if len(selected) < args.top_n:
            continue

        # estimate basket vol from trailing window
        hist = daily_ret[selected].loc[:dt].tail(args.vol_window).dropna()
        if len(hist) < args.vol_window:
            continue
        cov = hist.cov()
        w_eq = np.array([1.0 / args.top_n] * args.top_n)
        port_var = float(w_eq @ cov.values @ w_eq)
        port_vol = math.sqrt(port_var) if port_var > 0 else 0.0
        if port_vol == 0:
            continue

        lev = min(args.max_leverage, target_daily / port_vol)
        if args.no_leverage:
            lev = 1.0
        w = pd.Series(0.0, index=prices.columns)
        w[selected] = (lev / args.top_n)

        # weights apply from next day to avoid lookahead
        next_day_idx = prices.index.searchsorted(dt) + 1
        if next_day_idx >= len(prices.index):
            continue
        next_day = prices.index[next_day_idx]
        weights.loc[next_day:] = w.values
        picks.append({"date": dt, "tickers": ",".join(selected), "leverage": lev})

    port_ret = (weights.shift(1) * daily_ret).sum(axis=1)
    equity = (1 + port_ret.fillna(0)).cumprod()

    stats = performance_stats(equity)
    trades_per_year = annual_trades(weights)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, label="equity")
    ax.set_title("Rotating MR (Top-N) with vol targeting")
    ax.grid(True)
    ax.legend()
    fig_path = out_dir / "rotating_mr_vol_equity.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    stats_path = out_dir / "rotating_mr_vol_stats.csv"
    pd.DataFrame([{
        "CAGR": stats["CAGR"],
        "MDD": stats["MDD"],
        "Sharpe": stats["Sharpe"],
        "Trades_per_year": trades_per_year,
        "top_n": args.top_n,
        "ret_window": args.ret_window,
        "z_window": args.z_window,
        "vol_window": args.vol_window,
        "target_vol": args.target_vol,
        "max_leverage": args.max_leverage,
        "start": equity.index.min(),
        "end": equity.index.max(),
    }]).to_csv(stats_path, index=False)

    picks_path = out_dir / "rotating_mr_vol_picks.csv"
    pd.DataFrame(picks).to_csv(picks_path, index=False)

    print("Backtest stats")
    print(f"CAGR: {stats['CAGR']:.4f}")
    print(f"MDD: {stats['MDD']:.4f}")
    print(f"Sharpe: {stats['Sharpe']:.4f}")
    print(f"Trades/year: {trades_per_year:.2f}")
    print(f"Saved equity plot: {fig_path}")
    print(f"Saved stats: {stats_path}")
    print(f"Saved picks: {picks_path}")


if __name__ == "__main__":
    main()
