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


def select_tc(window: np.ndarray, L_grid: list[int], rho: float, w1: float, w2: float, w3: float):
    s1_list, s2_list, s3_list = [], [], []
    xc_cache = {}
    for L in L_grid:
        f0 = 1.0 / L
        s1 = spectral_concentration(window, f0, rho)
        xc = cycle_from_bandpass(window, f0, rho)
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
    scores = w1 * s1n + w2 * s2n + w3 * s3n
    best_idx = int(np.nanargmax(scores))
    best_L = L_grid[best_idx]
    best_xc = xc_cache[best_L]
    best_xt = window - best_xc
    return best_L, best_xt, best_xc


def compute_end_values(logp: pd.Series, window_days: int, L_grid: list[int], rho: float, w1: float, w2: float, w3: float):
    idx = logp.index
    xc_end = pd.Series(index=idx, dtype=float)
    xt_end = pd.Series(index=idx, dtype=float)
    for i in range(window_days - 1, len(logp)):
        window = logp.iloc[i - window_days + 1 : i + 1].values
        _, xt, xc = select_tc(window, L_grid, rho, w1, w2, w3)
        xc_end.iloc[i] = xc[-1]
        xt_end.iloc[i] = xt[-1]
    return xt_end, xc_end


def metrics_for_range(
    logp: pd.Series,
    dates: pd.DatetimeIndex,
    window_days: int,
    L_grid: list[int],
    rho: float,
    w1: float,
    w2: float,
    w3: float,
    k_list: list[int],
    step: int,
    component: str,
    xt_end: pd.Series,
    xc_end: pd.Series,
):
    diffs = {k: [] for k in k_list}
    sign_hits = {k: [] for k in k_list}
    std_vals = {k: [] for k in k_list}

    for end_dt in dates[::step]:
        i = logp.index.get_indexer([end_dt], method="nearest")[0]
        if i < window_days - 1:
            continue

        window = logp.iloc[i - window_days + 1 : i + 1].values
        _, xt_off, xc_off = select_tc(window, L_grid, rho, w1, w2, w3)

        for k in k_list:
            if i - k < window_days - 1:
                continue
            if i - k >= len(logp):
                continue

            offline_val = xc_off[-(k + 1)] if component == "XC" else xt_off[-(k + 1)]
            realtime_val = xc_end.iloc[i - k] if component == "XC" else xt_end.iloc[i - k]
            if pd.isna(realtime_val) or pd.isna(offline_val):
                continue
            diff = offline_val - realtime_val
            diffs[k].append(diff)
            sign_hits[k].append(np.sign(offline_val) == np.sign(realtime_val))
            std_vals[k].append(offline_val)

    rows = []
    for k in k_list:
        if not diffs[k]:
            continue
        mae = float(np.mean(np.abs(diffs[k])))
        sign_acc = float(np.mean(sign_hits[k]))
        std_ref = float(np.std(std_vals[k])) if len(std_vals[k]) > 1 else np.nan
        distortion = mae / (std_ref + 1e-12) if not np.isnan(std_ref) else np.nan
        rows.append({
            "k": k,
            "mae": mae,
            "sign_acc": sign_acc,
            "distortion": distortion,
            "n": len(diffs[k]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Endpoint revision distortion (k-lag) for TC")
    parser.add_argument("--config", default="config/run_pipeline.yaml")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--component", choices=["XC", "XT"], default="XC")
    parser.add_argument("--window-days", type=int, default=756)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--L-grid", default="20,30,40,60,80,120,160,240")
    parser.add_argument("--w1", type=float, default=0.50)
    parser.add_argument("--w2", type=float, default=0.35)
    parser.add_argument("--w3", type=float, default=0.15)
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=60)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--train-start", default="2009-01-02")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--val-start", default="2019-01-01")
    parser.add_argument("--val-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2026-01-16")
    parser.add_argument("--out-csv", default="data/backtest/tc_revision_k.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    db_path = Path(cfg["data"]["database_path"])
    close = load_adj_close(db_path, args.ticker)
    logp = np.log(close)

    L_grid = [int(x) for x in args.L_grid.split(",") if x.strip()]
    if args.k_min < 0:
        raise ValueError("--k-min must be >= 0")
    if args.k_max < args.k_min:
        raise ValueError("--k-max must be >= --k-min")
    k_list = list(range(args.k_min, args.k_max + 1))

    xt_end, xc_end = compute_end_values(logp, args.window_days, L_grid, args.rho, args.w1, args.w2, args.w3)

    def date_range(start: str, end: str) -> pd.DatetimeIndex:
        return logp.loc[pd.Timestamp(start):pd.Timestamp(end)].index

    train_dates = date_range(args.train_start, args.train_end)
    val_dates = date_range(args.val_start, args.val_end)
    test_dates = date_range(args.test_start, args.test_end)

    train_df = metrics_for_range(
        logp, train_dates, args.window_days, L_grid, args.rho,
        args.w1, args.w2, args.w3, k_list, args.step, args.component, xt_end, xc_end
    )
    train_df["split"] = "train"

    val_df = metrics_for_range(
        logp, val_dates, args.window_days, L_grid, args.rho,
        args.w1, args.w2, args.w3, k_list, args.step, args.component, xt_end, xc_end
    )
    val_df["split"] = "val"

    test_df = metrics_for_range(
        logp, test_dates, args.window_days, L_grid, args.rho,
        args.w1, args.w2, args.w3, k_list, args.step, args.component, xt_end, xc_end
    )
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
