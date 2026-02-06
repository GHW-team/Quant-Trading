import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse.linalg import spsolve


def ees_d1_with_drift(price: pd.Series, lam: float, use_log: bool = True) -> np.ndarray:
    s = price.dropna().astype(float)
    if len(s) < 5:
        raise ValueError("Need at least 5 observations.")

    x = np.log(s.values) if use_log else s.values
    t = len(x)

    e = np.ones(t)
    d = sparse.diags([e, -e], [0, -1], shape=(t - 1, t), format="csc")

    u = np.ones((t - 1, 1))
    utu = float((u.T @ u).ravel()[0])
    i_diff = sparse.identity(t - 1, format="csc")
    p = i_diff - (sparse.csc_matrix(u) @ sparse.csc_matrix(u.T)) / utu

    i_t = sparse.identity(t, format="csc")
    a = i_t + lam * (d.T @ (p @ d))

    xt = spsolve(a, x)
    return xt


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
    df.set_index("date", inplace=True)
    return df["adj_close"]


def cycle_length_from_gap(gap: np.ndarray, max_lag: int = 120) -> int:
    gap = np.asarray(gap, dtype=float)
    gap = gap - np.nanmean(gap)
    if np.allclose(gap, 0):
        return max_lag
    var = np.nanvar(gap)
    if var == 0:
        return max_lag
    max_lag = min(max_lag, len(gap) - 2)
    if max_lag < 2:
        return max_lag

    acf = []
    for lag in range(1, max_lag + 1):
        a = gap[:-lag]
        b = gap[lag:]
        acf.append(np.nanmean(a * b) / var)
    acf = np.array(acf)

    neg = np.where(acf <= 0)[0]
    if len(neg) > 0:
        return int(neg[0] + 1)
    return int(np.argmin(acf) + 1)


def lambda_revision_scores(
    price: pd.Series,
    lambdas,
    lag_days: int = 20,
    delta_days: int = 5,
    min_obs: int = 200,
    step: int = 5,
    use_log: bool = True,
) -> pd.Series:
    s = price.dropna().astype(float)
    idx = s.index
    scores = {}

    for lam in lambdas:
        rev_vals = []
        for i in range(min_obs + lag_days, len(idx) - delta_days, step):
            t = idx[i]
            t_future = idx[i + delta_days]
            t_lag = idx[i - lag_days]

            window_now = s.loc[:t]
            window_future = s.loc[:t_future]
            if len(window_now) < min_obs or len(window_future) < min_obs:
                continue

            try:
                tc_now = ees_d1_with_drift(window_now, lam=lam, use_log=use_log)
                tc_future = ees_d1_with_drift(window_future, lam=lam, use_log=use_log)
            except ValueError:
                continue

            if t_lag not in window_now.index or t_lag not in window_future.index:
                continue

            idx_now = window_now.index.get_loc(t_lag)
            idx_future = window_future.index.get_loc(t_lag)
            rev_vals.append(abs(tc_future[idx_future] - tc_now[idx_now]))

        scores[lam] = float(np.median(rev_vals)) if rev_vals else np.nan

    return pd.Series(scores, name="revision").dropna()


def select_lambda(
    window: pd.Series,
    lambdas,
    target_days: int,
    max_lag: int,
    rev_quantile: float,
    use_log: bool,
    lag_days: int,
    delta_days: int,
    min_obs: int,
    step: int,
) -> dict:
    rows = []
    for lam in lambdas:
        xt = ees_d1_with_drift(window, lam=lam, use_log=use_log)
        x = np.log(window.values) if use_log else window.values
        gap = x - xt
        cycle_len = cycle_length_from_gap(gap, max_lag=max_lag)
        rows.append({"lambda": lam, "cycle_len": cycle_len, "cycle_err": abs(cycle_len - target_days)})

    df = pd.DataFrame(rows).set_index("lambda").sort_index()
    rev = lambda_revision_scores(
        window,
        lambdas,
        lag_days=lag_days,
        delta_days=delta_days,
        min_obs=min_obs,
        step=step,
        use_log=use_log,
    )
    df = df.join(rev, how="left")

    min_err = df["cycle_err"].min()
    candidates = df[df["cycle_err"] == min_err]
    if "revision" in df.columns and candidates["revision"].notna().any():
        rev_cut = float(df["revision"].quantile(rev_quantile))
        candidates = candidates[candidates["revision"] <= rev_cut]
        if candidates.empty:
            candidates = df[df["cycle_err"] == min_err]

    chosen = float(candidates.index.values[len(candidates) // 2])
    chosen_row = df.loc[chosen]
    return {
        "lambda": chosen,
        "cycle_len": float(chosen_row["cycle_len"]),
        "cycle_err": float(chosen_row["cycle_err"]),
        "revision": float(chosen_row["revision"]) if "revision" in df.columns else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling lambda calibration (cycle length + revision)")
    parser.add_argument("--config", default="config/run_pipeline.yaml", help="Path to run_pipeline.yaml")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--target-days", type=int, default=60)
    parser.add_argument("--max-lag", type=int, default=120)
    parser.add_argument("--lam-start", type=float, default=10.0)
    parser.add_argument("--lam-mult", type=float, default=1.3)
    parser.add_argument("--lam-count", type=int, default=20)
    parser.add_argument("--rev-quantile", type=float, default=0.3)
    parser.add_argument("--lag-days", type=int, default=20)
    parser.add_argument("--delta-days", type=int, default=5)
    parser.add_argument("--min-obs", type=int, default=200)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--use-log", action="store_true", default=True)
    parser.add_argument("--no-log", dest="use_log", action="store_false")
    parser.add_argument("--out-csv", default="data/backtest/rolling_lambda.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    db_path = Path(cfg["data"]["database_path"])
    price = load_adj_close(db_path, args.ticker)

    lambdas = [args.lam_start * (args.lam_mult ** i) for i in range(args.lam_count)]

    results = []
    prev_lambda = None
    for year in range(args.start_year, args.end_year + 1):
        anchor = pd.Timestamp(f"{year}-12-31")
        if anchor not in price.index:
            anchor = price.index[price.index.get_indexer([anchor], method="ffill")[0]]

        start = anchor - pd.DateOffset(years=args.lookback_years)
        window = price.loc[start:anchor]
        if len(window) < args.min_obs:
            continue

        sel = select_lambda(
            window=window,
            lambdas=lambdas,
            target_days=args.target_days,
            max_lag=args.max_lag,
            rev_quantile=args.rev_quantile,
            use_log=args.use_log,
            lag_days=args.lag_days,
            delta_days=args.delta_days,
            min_obs=args.min_obs,
            step=args.step,
        )
        lam = sel["lambda"]
        lam_change = None if prev_lambda is None else lam - prev_lambda
        results.append(
            {
                "year": year,
                "anchor": anchor.date(),
                "lambda": lam,
                "lambda_change": lam_change,
                "cycle_len": sel["cycle_len"],
                "cycle_err": sel["cycle_err"],
                "revision": sel["revision"],
            }
        )
        prev_lambda = lam

    out = pd.DataFrame(results)
    pd.set_option("display.precision", 6)
    print(out)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved rolling lambda table: {out_path}")


if __name__ == "__main__":
    main()
