import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_adj_close(db_path: Path, ticker: str) -> pd.Series:
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
    if df.empty:
        raise ValueError(f"No data for ticker: {ticker}")
    df = df.dropna(subset=["adj_close"]).set_index("date")
    return df["adj_close"]


def band_mask(freqs: np.ndarray, f0: float, rho: float) -> np.ndarray:
    # Bandwidth scaled by rho (higher rho -> narrower band)
    bw_ratio = max(0.1, min(0.5, (1 - rho) * 5))
    low = max(0.0, f0 * (1 - bw_ratio))
    high = f0 * (1 + bw_ratio)
    return (freqs >= low) & (freqs <= high)


def cycle_from_bandpass(x: np.ndarray, f0: float, rho: float) -> np.ndarray:
    n = len(x)
    x_centered = x - x.mean()
    spec = np.fft.rfft(x_centered)
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = band_mask(freqs, f0, rho)
    spec_f = spec * mask
    xc = np.fft.irfft(spec_f, n=n)
    return xc


def spectral_concentration(x: np.ndarray, f0: float, rho: float) -> float:
    n = len(x)
    x_centered = x - x.mean()
    spec = np.fft.rfft(x_centered)
    power = np.abs(spec) ** 2
    total = power.sum()
    if total == 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = band_mask(freqs, f0, rho)
    return float(power[mask].sum() / total)


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 3:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(abs(np.corrcoef(a, b)[0, 1]))


def zero_cross_score(xc: np.ndarray, L: float) -> float:
    if len(xc) < 3:
        return 0.0
    zc = np.sum(np.sign(xc[1:]) != np.sign(xc[:-1]))
    zc_rate = zc / len(xc)
    expected = 2.0 / L
    if expected <= 0:
        return 0.0
    penalty = abs(zc_rate - expected) / expected
    return float(max(0.0, 1.0 - min(1.0, penalty)))


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def main():
    parser = argparse.ArgumentParser(description="Rolling TC decomposition with mu grid (daily)")
    parser.add_argument("--config", default="config/run_pipeline.yaml")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--window-days", type=int, default=756)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--L-grid", default="20,30,40,60,80,120,160,240")
    parser.add_argument("--w1", type=float, default=0.50)
    parser.add_argument("--w2", type=float, default=0.35)
    parser.add_argument("--w3", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--out-csv", default="data/tc_price_daily.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    db_path = Path(cfg["data"]["database_path"])
    close = load_adj_close(db_path, args.ticker)
    logp = np.log(close)

    L_grid = [int(x) for x in args.L_grid.split(",") if x.strip()]
    mu_grid = [2 * math.pi / L for L in L_grid]

    results = []
    prev_L = None
    for i in range(args.window_days - 1, len(logp)):
        end_dt = logp.index[i]
        window = logp.iloc[i - args.window_days + 1 : i + 1].values

        s1_list = []
        s2_list = []
        s3_list = []
        xc_cache = {}
        for L, mu in zip(L_grid, mu_grid):
            f0 = 1.0 / L
            s1 = spectral_concentration(window, f0, args.rho)
            xc = cycle_from_bandpass(window, f0, args.rho)
            xt = window - xc
            dxt = np.diff(xt)
            s2 = 1.0 - corr_abs(dxt, xc[1:])
            s3 = zero_cross_score(xc, L)
            s1_list.append(s1)
            s2_list.append(s2)
            s3_list.append(s3)
            xc_cache[L] = xc

        s1n = minmax_norm(np.array(s1_list))
        s2n = minmax_norm(np.array(s2_list))
        s3n = minmax_norm(np.array(s3_list))
        scores = args.w1 * s1n + args.w2 * s2n + args.w3 * s3n

        if prev_L is not None:
            for j, L in enumerate(L_grid):
                scores[j] -= args.gamma * abs(math.log(L / prev_L))

        best_idx = int(np.nanargmax(scores))
        best_L = L_grid[best_idx]
        best_mu = mu_grid[best_idx]
        best_xc = xc_cache[best_L]
        best_xt = window - best_xc
        dxt = best_xt[-1] - best_xt[-2] if len(best_xt) > 1 else np.nan
        zxc = (best_xc[-1] - best_xc.mean()) / (best_xc.std(ddof=1) + 1e-12)

        results.append(
            {
                "date": end_dt,
                "close": float(close.loc[end_dt]),
                "logp": float(logp.loc[end_dt]),
                "L_selected": best_L,
                "mu_selected": best_mu,
                "XT": float(best_xt[-1]),
                "XC": float(best_xc[-1]),
                "dXT": float(dxt),
                "score_best": float(scores[best_idx]),
                "zXC": float(zxc),
            }
        )
        prev_L = best_L

    out = pd.DataFrame(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
