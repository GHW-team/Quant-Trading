import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_tc(tc_csv: str) -> pd.DataFrame:
    df = pd.read_csv(tc_csv, parse_dates=["date"]).set_index("date")
    for col in ["XT", "XC", "dXT"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in {tc_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Label trend regime using TC")
    parser.add_argument("--tc-csv", default="data/tc_price_daily.csv")
    parser.add_argument("--dxt-window", type=int, default=252)
    parser.add_argument("--xc-window", type=int, default=252)
    parser.add_argument("--dxt-quantile", type=float, default=0.80)
    parser.add_argument("--xc-quantile", type=float, default=0.80)
    parser.add_argument("--out-csv", default="data/backtest/tc_regime.csv")
    args = parser.parse_args()

    df = load_tc(args.tc_csv)
    dxt = df["dXT"].abs()
    xc_amp = df["XC"].abs()

    dxt_th = dxt.rolling(args.dxt_window).quantile(args.dxt_quantile)
    xc_th = xc_amp.rolling(args.xc_window).quantile(args.xc_quantile)

    strong_trend = (dxt > dxt_th) & (xc_amp > xc_th)
    mild_trend = (df["dXT"] > 0) & ~strong_trend
    flat = df["dXT"] <= 0

    regime = pd.Series("flat_or_down", index=df.index)
    regime[mild_trend] = "mild_up"
    regime[strong_trend] = "strong_up"

    out = pd.DataFrame({
        "date": df.index,
        "XT": df["XT"].values,
        "XC": df["XC"].values,
        "dXT": df["dXT"].values,
        "regime": regime.values,
        "dXT_th": dxt_th.values,
        "XC_th": xc_th.values,
    })
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
