import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse.linalg import spsolve


def ees_d1_with_drift(price: pd.Series, lam: float, use_log: bool = True):
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


def lambda_sweep_plateau(price: pd.Series, lambdas, eps_quantile: float = 0.2, min_run: int = 2, use_log: bool = True):
    lambdas = sorted(list(lambdas))
    xt_list = []
    for lam in lambdas:
        xt_list.append(ees_d1_with_drift(price, lam=lam, use_log=use_log))

    d_vals = []
    for i in range(len(lambdas) - 1):
        a = xt_list[i]
        b = xt_list[i + 1]
        num = np.linalg.norm(b - a)
        den = np.linalg.norm(a) + 1e-12
        d_vals.append(num / den)
    d_series = pd.Series(d_vals, index=lambdas[:-1], name="D_change")

    eps = float(d_series.quantile(eps_quantile))
    is_small = (d_series <= eps).values

    best = (None, None, 0)
    i = 0
    while i < len(is_small):
        if not is_small[i]:
            i += 1
            continue
        j = i
        while j < len(is_small) and is_small[j]:
            j += 1
        length = j - i
        if length >= min_run and length > best[2]:
            best = (i, j, length)
        i = j

    plateau = []
    if best[0] is not None:
        start, end = best[0], best[1]
        plateau = lambdas[start : end + 1]

    return d_series, plateau


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


def lambda_mle_local_trend(price: pd.Series, use_log: bool = True) -> float:
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
    except ImportError as e:
        raise RuntimeError(
            "statsmodels is required for MLE lambda estimation. "
            "Install with: pip install statsmodels"
        ) from e

    s = price.dropna().astype(float)
    if len(s) < 50:
        raise ValueError("Need at least 50 observations for MLE.")

    y = np.log(s.values) if use_log else s.values
    model = UnobservedComponents(y, level="local level", trend=True)
    res = model.fit(disp=False)

    # Map to variance ratio: lambda = sigma_eps^2 / sigma_eta^2
    # For local level + trend, use level variance as eta proxy.
    param_map = dict(zip(res.param_names, res.params))
    sigma_eps = param_map.get("sigma2.irregular")
    sigma_level = param_map.get("sigma2.level")
    if sigma_eps is None or sigma_level is None or sigma_level == 0:
        raise ValueError("Failed to estimate variance components for lambda.")

    return float(sigma_eps / sigma_level)


def main():
    parser = argparse.ArgumentParser(description="Lambda sweep (EES d=1) using a single ticker")
    parser.add_argument("--config", default="config/run_pipeline.yaml", help="Path to run_pipeline.yaml")
    parser.add_argument("--ticker", default="SPY", help="Ticker to use for lambda selection")
    parser.add_argument("--method", choices=["plateau", "mle", "revision"], default="plateau")
    parser.add_argument("--lam-start", type=float, default=10.0)
    parser.add_argument("--lam-mult", type=float, default=1.3)
    parser.add_argument("--lam-count", type=int, default=15)
    parser.add_argument("--eps-quantile", type=float, default=0.2)
    parser.add_argument("--min-run", type=int, default=2)
    parser.add_argument("--lag-days", type=int, default=20)
    parser.add_argument("--delta-days", type=int, default=5)
    parser.add_argument("--min-obs", type=int, default=200)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--rev-quantile", type=float, default=0.3)
    parser.add_argument("--use-log", action="store_true", default=True)
    parser.add_argument("--no-log", dest="use_log", action="store_false")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    db_path = Path(cfg["data"]["database_path"])

    price = load_adj_close(db_path, args.ticker)

    if args.method == "mle":
        chosen = lambda_mle_local_trend(price, use_log=args.use_log)
        print(f"MLE lambda (variance ratio): {chosen:.6f}")
    elif args.method == "revision":
        lambdas = [args.lam_start * (args.lam_mult ** i) for i in range(args.lam_count)]
        scores = lambda_revision_scores(
            price,
            lambdas,
            lag_days=args.lag_days,
            delta_days=args.delta_days,
            min_obs=args.min_obs,
            step=args.step,
            use_log=args.use_log,
        )
        print("Revision scores:")
        print(scores)
        cutoff = float(scores.quantile(args.rev_quantile))
        candidates = scores[scores <= cutoff]
        if not candidates.empty:
            chosen = float(np.median(candidates.index.values))
            print(f"\nRevision plateau (<= q{args.rev_quantile:.2f}): {list(candidates.index.values)}")
        else:
            chosen = float(scores.idxmin())
            print("\nRevision plateau empty; falling back to min score.")
        print(f"\nSelected lambda = {chosen:.6f}")
    else:
        lambdas = [args.lam_start * (args.lam_mult ** i) for i in range(args.lam_count)]
        d_series, plateau = lambda_sweep_plateau(
            price,
            lambdas,
            eps_quantile=args.eps_quantile,
            min_run=args.min_run,
            use_log=args.use_log,
        )

        print("D change:")
        print(d_series)
        print("\nPlateau:", plateau)
        if plateau:
            chosen = float(np.sqrt(plateau[0] * plateau[-1]))
            print(f"\nPlateau geometric mean lambda = {chosen:.6f}")
        else:
            chosen = lambdas[len(lambdas) // 2]
            print(f"\nSelected lambda = {chosen:.6f}")


if __name__ == "__main__":
    main()

