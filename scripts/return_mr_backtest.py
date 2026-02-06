import argparse
import math
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DAILY_DB = "data/database/stocks.db"


def load_daily_ohlc(db_path: str, ticker: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT dp.date, dp.open, dp.adj_close
            FROM daily_prices dp
            JOIN tickers t ON t.ticker_id = dp.ticker_id
            WHERE t.ticker_code = ?
            ORDER BY dp.date
            """,
            conn,
            params=(ticker,),
            parse_dates=["date"],
        )
    if df.empty:
        raise ValueError(f"No daily data for {ticker}")
    return df.set_index("date").sort_index()


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


def annual_trades(pos: pd.Series) -> float:
    changes = pos.diff().abs().fillna(0)
    trades = (changes > 0).sum()
    years = (pos.index[-1] - pos.index[0]).days / 365.25
    return trades / years if years > 0 else 0.0


def backtest_window(
    df: pd.DataFrame,
    horizon: int,
    z_std_window: int,
    z_in: float,
    z_out: float,
    wmax: float,
    long_only: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    logp = np.log(df["adj_close"])
    r = logp - logp.shift(horizon)
    z_std = r.rolling(z_std_window).std()
    z = r / (z_std + 1e-12)

    pos = pd.Series(0.0, index=df.index)
    for i, dt in enumerate(df.index):
        if i == 0:
            continue
        zt = z.loc[dt]
        if np.isnan(zt):
            pos.loc[dt] = pos.loc[df.index[i - 1]]
            continue

        if abs(zt) <= z_out:
            pos.loc[dt] = 0.0
            continue

        if abs(zt) >= z_in:
            if zt < 0:
                pos.loc[dt] = wmax
            else:
                pos.loc[dt] = 0.0 if long_only else -wmax
        else:
            pos.loc[dt] = pos.loc[df.index[i - 1]]

    open_px = df["open"]
    open_ret = open_px.shift(-1) / open_px - 1.0
    exec_pos = pos.shift(1).fillna(0)
    strat_ret = exec_pos * open_ret
    equity = (1 + strat_ret.fillna(0)).cumprod()
    return equity, exec_pos, z, pos


def main() -> None:
    parser = argparse.ArgumentParser(description="Return mean-reversion backtest")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--windows", default="5,10,20")
    parser.add_argument("--z-std-window", type=int, default=60)
    parser.add_argument("--z-in", type=float, default=1.5)
    parser.add_argument("--z-out", type=float, default=0.3)
    parser.add_argument("--wmax", type=float, default=1.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--plot-signals", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default="data/backtest")
    args = parser.parse_args()

    df = load_daily_ohlc(DAILY_DB, args.ticker)
    if args.start:
        df = df.loc[pd.Timestamp(args.start):]
    if args.end:
        df = df.loc[:pd.Timestamp(args.end)]
    if df.empty:
        raise ValueError("No data after date filtering.")

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, ax = plt.subplots(figsize=(10, 4))
    signal_payload = None
    for h in windows:
        equity, exec_pos, z, pos = backtest_window(
            df=df,
            horizon=h,
            z_std_window=args.z_std_window,
            z_in=args.z_in,
            z_out=args.z_out,
            wmax=args.wmax,
            long_only=args.long_only,
        )
        stats = performance_stats(equity)
        trades_per_year = annual_trades(exec_pos)
        rows.append({
            "ticker": args.ticker,
            "window": h,
            "CAGR": stats["CAGR"],
            "MDD": stats["MDD"],
            "Sharpe": stats["Sharpe"],
            "Trades_per_year": trades_per_year,
            "start": equity.index.min(),
            "end": equity.index.max(),
            "z_in": args.z_in,
            "z_out": args.z_out,
            "z_std_window": args.z_std_window,
            "wmax": args.wmax,
            "long_only": args.long_only,
        })
        ax.plot(equity.index, equity.values, label=f"{h}d")
        if args.plot_signals and len(windows) == 1:
            signal_payload = (h, z, pos)

    ax.set_title(f"{args.ticker} return MR equity")
    ax.grid(True)
    ax.legend()
    fig_path = output_dir / f"{args.ticker}_return_mr_equity.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    stats_path = output_dir / f"{args.ticker}_return_mr_stats.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)

    print("Backtest stats")
    for row in rows:
        print(
            f"window={row['window']}: "
            f"CAGR={row['CAGR']:.4f} MDD={row['MDD']:.4f} "
            f"Sharpe={row['Sharpe']:.4f} Trades/year={row['Trades_per_year']:.2f}"
        )
    print(f"Saved equity plot: {fig_path}")
    print(f"Saved stats: {stats_path}")

    if args.plot_signals and len(windows) == 1 and signal_payload is not None:
        h, z, pos = signal_payload
        price = df["adj_close"]
        entry = (pos.shift(1) <= 0) & (pos > 0)
        exit_ = (pos.shift(1) > 0) & (pos <= 0)

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax[0].plot(price.index, price.values, label="price")
        ax[0].scatter(price.index[entry], price[entry], marker="^", color="green", label="entry")
        ax[0].scatter(price.index[exit_], price[exit_], marker="v", color="red", label="exit")
        ax[0].set_title(f"{args.ticker} price with signals ({h}d)")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(z.index, z.values, label="z")
        ax[1].axhline(-args.z_in, color="gray", ls="--", lw=1)
        ax[1].axhline(args.z_in, color="gray", ls="--", lw=1)
        ax[1].axhline(args.z_out, color="gray", ls=":", lw=1)
        ax[1].axhline(-args.z_out, color="gray", ls=":", lw=1)
        ax[1].set_title("z-score")
        ax[1].legend()
        ax[1].grid(True)

        sig_path = output_dir / f"{args.ticker}_return_mr_signals.png"
        plt.tight_layout()
        plt.savefig(sig_path)
        plt.close(fig)
        print(f"Saved signal plot: {sig_path}")


if __name__ == "__main__":
    main()
