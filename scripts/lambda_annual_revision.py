import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tc_filter import EESD2


def load_daily_adj_close(db_path: Path, ticker: str) -> pd.Series:
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


def lambda_mle(price: pd.Series, use_log: bool = True) -> float:
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    s = price.dropna().astype(float)
    y = np.log(s.values) if use_log else s.values
    model = UnobservedComponents(y, level="local level", trend=True)
    res = model.fit(disp=False)

    param_names = res.param_names
    params = dict(zip(param_names, res.params))
    sigma_eps = params.get("sigma2.irregular")
    sigma_level = params.get("sigma2.level")
    if sigma_eps is None or sigma_level is None or sigma_level == 0:
        raise ValueError("Failed to estimate variance components.")
    return float(sigma_eps / sigma_level)


def tc_value_at(price: pd.Series, lam: float, dt: pd.Timestamp) -> float:
    window = price.loc[:dt].dropna()
    res = EESD2(lam=lam, use_log=True).fit(window)
    return float(res["XT"].iloc[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Annual lambda roll + revision impact")
    parser.add_argument("--db", default="data/database/stocks.db")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--use-log", action="store_true", default=True)
    parser.add_argument("--no-log", dest="use_log", action="store_false")
    args = parser.parse_args()

    price = load_daily_adj_close(Path(args.db), args.ticker)

    results = []
    prev_lambda = None

    for year in range(args.start_year, args.end_year + 1):
        anchor = pd.Timestamp(f"{year}-12-31")
        if anchor not in price.index:
            anchor = price.index[price.index.get_indexer([anchor], method="ffill")[0]]

        start = anchor - pd.DateOffset(years=args.lookback_years)
        window = price.loc[start:anchor]
        if len(window) < 200:
            continue

        lam = lambda_mle(window, use_log=args.use_log)

        revision = None
        if prev_lambda is not None:
            tc_old = tc_value_at(price, prev_lambda, anchor)
            tc_new = tc_value_at(price, lam, anchor)
            revision = abs(tc_new - tc_old)

        results.append(
            {
                "year": year,
                "anchor": anchor.date(),
                "lambda": lam,
                "revision": revision,
            }
        )
        prev_lambda = lam

    out = pd.DataFrame(results)
    pd.set_option("display.precision", 12)
    print(out)

    if not out.empty:
        last = out.iloc[-1]
        lam = last["lambda"]
        anchor = pd.Timestamp(last["anchor"])
        # plot a short window around the anchor to see smoothing
        window_start = anchor - pd.DateOffset(years=2)
        window = price.loc[window_start:anchor]
        if len(window) > 50:
            res = EESD2(lam=lam, use_log=args.use_log).fit(window)
            tc_level = np.exp(res["XT"]) if args.use_log else res["XT"]
            df_plot = pd.DataFrame(
                {"price": window, "tc": tc_level}
            ).dropna()

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_plot.index, df_plot["price"], label="price", alpha=0.6)
            ax.plot(df_plot.index, df_plot["tc"], label=f"TC (λ={lam:.6f})", linewidth=1.5)
            ax.set_title(f"{args.ticker} TC around {anchor.date()}")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
