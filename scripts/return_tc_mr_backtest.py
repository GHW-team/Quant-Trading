import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean-reversion on return TC (fixed lag)")
    parser.add_argument("--tc-csv", default="data/return_tc_fixedlag.csv")
    parser.add_argument("--xc-in", type=float, default=0.003)
    parser.add_argument("--xc-out", type=float, default=0.0005)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--output-dir", default="data/backtest")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.tc_csv, parse_dates=["date"]).set_index("date")
    for col in ["ret", "XC_lag"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in {args.tc_csv}")

    if args.start:
        df = df.loc[pd.Timestamp(args.start):]
    if args.end:
        df = df.loc[:pd.Timestamp(args.end)]
    if df.empty:
        raise ValueError("No data after date filtering.")

    xc = df["XC_lag"]

    pos = pd.Series(0.0, index=df.index)
    for i, dt in enumerate(df.index):
        if i == 0:
            continue
        xct = xc.loc[dt]
        if np.isnan(xct):
            pos.loc[dt] = pos.loc[df.index[i - 1]]
            continue

        if abs(xct) <= args.xc_out:
            pos.loc[dt] = 0.0
            continue

        if abs(xct) >= args.xc_in:
            if xct < 0:
                pos.loc[dt] = 1.0
            else:
                pos.loc[dt] = 0.0 if args.long_only else -1.0
        else:
            pos.loc[dt] = pos.loc[df.index[i - 1]]

    ret = df["ret"].shift(-1)
    exec_pos = pos.shift(1).fillna(0)
    strat_ret = exec_pos * ret
    equity = (1 + strat_ret.fillna(0)).cumprod()

    stats = performance_stats(equity)
    trades_per_year = annual_trades(exec_pos)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, label="equity")
    ax.set_title("Return TC MR equity")
    ax.grid(True)
    ax.legend()
    fig_path = out_dir / "return_tc_mr_equity.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    stats_path = out_dir / "return_tc_mr_stats.csv"
    pd.DataFrame([{
        "CAGR": stats["CAGR"],
        "MDD": stats["MDD"],
        "Sharpe": stats["Sharpe"],
        "Trades_per_year": trades_per_year,
        "xc_in": args.xc_in,
        "xc_out": args.xc_out,
        "long_only": args.long_only,
        "start": equity.index.min(),
        "end": equity.index.max(),
    }]).to_csv(stats_path, index=False)

    print("Backtest stats")
    print(f"CAGR: {stats['CAGR']:.4f}")
    print(f"MDD: {stats['MDD']:.4f}")
    print(f"Sharpe: {stats['Sharpe']:.4f}")
    print(f"Trades/year: {trades_per_year:.2f}")
    print(f"Saved equity plot: {fig_path}")
    print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
