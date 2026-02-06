import argparse
import math
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DAILY_DB = "data/database/stocks.db"


def load_prices(db_path: str, tickers: list[str]) -> pd.DataFrame:
    # DB에서 일봉 OHLC를 한 번에 로드
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT dp.date, t.ticker_code, dp.open, dp.high, dp.low, dp.close
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
    # DB에 저장된 전체 티커 목록
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT ticker_code FROM tickers ORDER BY ticker_code", conn)
    return df["ticker_code"].tolist()


def sector_universe(tickers: list[str]) -> list[str]:
    # 미국 섹터/산업 ETF 기본 유니버스
    sectors = {
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
        "XAR", "XBI", "XHB", "XME", "XOP", "XRT", "XSD", "KRE",
    }
    return [t for t in tickers if t in sectors]


def sector_plus_country_universe(tickers: list[str]) -> list[str]:
    # 섹터 + 국가 ETF 유니버스
    sectors = set(sector_universe(tickers))
    countries = {"EWJ", "VGK", "FXI", "INDA", "EWZ"}
    return [t for t in tickers if t in sectors or t in countries]

def commodity_universe() -> set[str]:
    # 원자재(커모디티) 그룹: 포지션 사이징 캡에 사용
    return {
        "GLD", "SLV", "PPLT", "PALL",
        "USO", "UNG",
        "DBA", "SOYB", "WEAT", "CORN", "COW",
        "CPER",
    }


def load_open_series(db_path: str, ticker: str) -> pd.Series:
    # SPY buy&hold 비교용 시가 시계열
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT dp.date, dp.open
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
        return pd.Series(dtype=float)
    return df.set_index("date")["open"].sort_index()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    # ATR 계산을 위한 True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def performance_stats(equity: pd.Series) -> dict:
    # 성과 요약 지표(CAGR/MDD/Sharpe)
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
    # 연간 거래 횟수(포지션 변화 횟수 기준)
    changes = pos.ne(pos.shift(1)).sum()
    years = (pos.index[-1] - pos.index[0]).days / 365.25
    return changes / years if years > 0 else 0.0


def main() -> None:
    # 핵심 전략: 20/10(또는 60/20) 돌파 신호 중 강도가 가장 큰 종목 1개 매수
    parser = argparse.ArgumentParser(description="Pick strongest breakout across all tickers")
    parser.add_argument("--db", default=DAILY_DB)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--buy1", type=int, default=20)
    parser.add_argument("--sell1", type=int, default=10)
    parser.add_argument("--buy2", type=int, default=60)
    parser.add_argument("--sell2", type=int, default=20)
    parser.add_argument("--use-20-10-only", action="store_true")
    parser.add_argument("--use-regime", action="store_true")
    parser.add_argument("--regime-csv", default="data/backtest/tc_regime.csv")
    parser.add_argument("--low-window", type=int, default=20)
    parser.add_argument("--low-rise-only", action="store_true")
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--atr-mult", type=float, default=2.0)
    parser.add_argument("--switch-threshold", type=float, default=0.0)
    parser.add_argument("--switch-cooldown", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--min-strength", type=float, default=0.0)
    parser.add_argument("--signal-delay", type=int, default=0)
    parser.add_argument("--sector-only", action="store_true")
    parser.add_argument("--sector-plus-country", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--position-size", type=float, default=1.0)
    parser.add_argument("--commodity-cap", type=float, default=1.0)
    parser.add_argument("--output-dir", default="data/backtest")
    args = parser.parse_args()

    # 유니버스 결정
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = load_ticker_list(args.db)
    exclude = {t.strip() for t in args.exclude.split(",") if t.strip()}
    if exclude:
        tickers = [t for t in tickers if t not in exclude]
    if args.sector_only:
        tickers = sector_universe(tickers)
    if args.sector_plus_country:
        tickers = sector_plus_country_universe(tickers)
    if not tickers:
        raise ValueError("No tickers found.")

    df = load_prices(args.db, tickers)
    if args.start:
        df = df[df["date"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["date"] <= pd.Timestamp(args.end)]
    if df.empty:
        raise ValueError("No price data after date filtering.")

    # 가격 테이블(일자 x 티커)
    closes = df.pivot(index="date", columns="ticker_code", values="close").sort_index()
    highs = df.pivot(index="date", columns="ticker_code", values="high").sort_index()
    lows = df.pivot(index="date", columns="ticker_code", values="low").sort_index()
    opens = df.pivot(index="date", columns="ticker_code", values="open").sort_index()

    # 돌파 기준: 전일 기준 N일 최고가/최저가
    rh1 = closes.rolling(args.buy1).max().shift(1)
    rl1 = closes.rolling(args.sell1).min().shift(1)
    rh2 = closes.rolling(args.buy2).max().shift(1)
    rl2 = closes.rolling(args.sell2).min().shift(1)
    low20 = lows.rolling(args.low_window).min().shift(1)
    low20_prev = low20.shift(1)

    # ATR(변동성) 계산: 손절(ATR 배수) 기준
    atr = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)
    for t in closes.columns:
        tr = true_range(highs[t], lows[t], closes[t])
        atr[t] = tr.rolling(args.atr_window).mean()

    dates = closes.index
    pos_ticker = pd.Series(index=dates, dtype=object)
    pos_rule = pd.Series(index=dates, dtype=object)
    entry_price = np.nan
    entry_strength = np.nan
    last_switch_idx = -10**9
    events = []

    # 레짐 필터(옵션)
    regime = None
    if args.use_regime:
        reg_df = pd.read_csv(args.regime_csv, parse_dates=["date"]).set_index("date")
        if "regime" not in reg_df.columns:
            raise ValueError("regime column missing in regime csv.")
        regime = reg_df["regime"].reindex(dates).ffill()

    if args.top_n > 1 and args.signal_delay > 0:
        raise ValueError("signal-delay is only supported for top-n=1.")

    # 체결은 다음날 시가 기준(룩어헤드 방지)
    open_ret = opens.shift(-1) / opens - 1.0
    active_ret = pd.Series(0.0, index=dates)
    commodities = commodity_universe()

    if args.top_n <= 1:
        for i, dt in enumerate(dates):
            if i == 0:
                continue
            prev = dates[i - 1]

            # carry forward by default
            pos_ticker.loc[dt] = pos_ticker.loc[prev]
            pos_rule.loc[dt] = pos_rule.loc[prev]

            current = pos_ticker.loc[prev]
            rule = pos_rule.loc[prev]

            # 보유 종목 청산 조건: ATR 손절 또는 N일 최저가 이탈
            if current is not None and not pd.isna(current):
                if not np.isnan(entry_price):
                    atr_prev = atr.loc[prev, current]
                    close_prev = closes.loc[prev, current]
                    if not np.isnan(atr_prev) and close_prev <= entry_price - args.atr_mult * atr_prev:
                        pos_ticker.loc[dt] = None
                        pos_rule.loc[dt] = None
                        events.append(
                            {
                                "date": dt,
                                "event": "exit_stop",
                                "ticker": current,
                                "rule": rule,
                                "price": float(close_prev),
                            }
                        )
                        entry_price = np.nan
                        current = None
                if current is None or pd.isna(current):
                    continue
                if rule in ("20_10", "20low_up", "20low_rise"):
                    if closes.loc[prev, current] <= rl1.loc[prev, current]:
                        pos_ticker.loc[dt] = None
                        pos_rule.loc[dt] = None
                        events.append(
                            {
                                "date": dt,
                                "event": "exit_rule",
                                "ticker": current,
                                "rule": rule,
                                "price": float(closes.loc[prev, current]),
                            }
                        )
                        entry_price = np.nan
                        current = None
                elif rule == "60_20":
                    if closes.loc[prev, current] <= rl2.loc[prev, current]:
                        pos_ticker.loc[dt] = None
                        pos_rule.loc[dt] = None
                        events.append(
                            {
                                "date": dt,
                                "event": "exit_rule",
                                "ticker": current,
                                "rule": rule,
                                "price": float(closes.loc[prev, current]),
                            }
                        )
                        entry_price = np.nan
                        current = None

            # 진입/스위치 판단(돌파 강도 최대 선택)
            c_prev = closes.loc[prev]

            if args.low_rise_only:
                rise = (low20.loc[prev] > low20_prev.loc[prev])
                cand1 = rise
                cand2 = pd.Series(False, index=c_prev.index)
            elif args.use_regime and regime is not None and regime.loc[prev] == "mild_up":
                cand1 = pd.Series(False, index=c_prev.index)
                cand2 = pd.Series(False, index=c_prev.index)
            else:
                cand1 = c_prev >= rh1.loc[prev]
                cand2 = c_prev >= rh2.loc[prev] if not args.use_20_10_only else pd.Series(False, index=c_prev.index)

            # 돌파 강도: (전일 종가 / 전일 N일 최고가) - 1
            strengths = {}
            for t in c_prev.index:
                if cand1.get(t, False):
                    if args.low_rise_only:
                        prev_low = low20_prev.loc[prev, t]
                        cur_low = low20.loc[prev, t]
                        if pd.notna(prev_low) and prev_low != 0:
                            strengths[(t, "20low_rise")] = (cur_low / prev_low) - 1.0
                    else:
                        strengths[(t, "20_10")] = (c_prev[t] / rh1.loc[prev, t]) - 1.0
                if cand2.get(t, False):
                    strengths[(t, "60_20")] = (c_prev[t] / rh2.loc[prev, t]) - 1.0

            if strengths:
                (best_t, best_rule), best_strength = max(strengths.items(), key=lambda x: x[1])
                if best_strength < args.min_strength:
                    continue
                if current is None or pd.isna(current):
                    pos_ticker.loc[dt] = best_t
                    pos_rule.loc[dt] = best_rule
                    entry_price = closes.loc[prev, best_t]
                    entry_strength = best_strength
                    events.append(
                        {
                            "date": dt,
                            "event": "entry",
                            "ticker": best_t,
                            "rule": best_rule,
                            "price": float(entry_price),
                            "strength": float(best_strength),
                        }
                    )
                else:
                    # 스위치: 기존 포지션 대비 강도가 충분히 크면 교체
                    threshold = args.switch_threshold
                    if np.isnan(entry_strength):
                        entry_strength = 0.0
                    if best_strength >= entry_strength * (1 + threshold):
                        if args.switch_cooldown > 0 and (i - last_switch_idx) < args.switch_cooldown:
                            continue
                        pos_ticker.loc[dt] = best_t
                        pos_rule.loc[dt] = best_rule
                        entry_price = closes.loc[prev, best_t]
                        entry_strength = best_strength
                        last_switch_idx = i
                        events.append(
                            {
                                "date": dt,
                                "event": "switch",
                                "ticker": best_t,
                                "rule": best_rule,
                                "price": float(entry_price),
                                "strength": float(best_strength),
                            }
                        )
        # 신호 지연(옵션): signal-delay만큼 체결을 늦춤
        exec_pos = pos_ticker.shift(1 + args.signal_delay)
        for dt in dates:
            t = exec_pos.loc[dt]
            if t is None or pd.isna(t):
                continue
            if t in open_ret.columns and dt in open_ret.index:
                active_ret.loc[dt] = open_ret.loc[dt, t]
    else:
        # top-n 모드: 여러 종목 동시 보유
        positions: dict[str, dict] = {}
        for i, dt in enumerate(dates):
            if i == 0:
                continue
            prev = dates[i - 1]

            # 보유 종목 전체 청산 체크
            for t in list(positions.keys()):
                rule = positions[t]["rule"]
                entry = positions[t]["entry_price"]
                atr_prev = atr.loc[prev, t]
                close_prev = closes.loc[prev, t]
                if pd.notna(atr_prev) and close_prev <= entry - args.atr_mult * atr_prev:
                    positions.pop(t, None)
                    events.append(
                        {
                            "date": dt,
                            "event": "exit_stop",
                            "ticker": t,
                            "rule": rule,
                            "price": float(close_prev),
                        }
                    )
                    continue
                if rule in ("20_10", "20low_up", "20low_rise"):
                    if close_prev <= rl1.loc[prev, t]:
                        positions.pop(t, None)
                        events.append(
                            {
                                "date": dt,
                                "event": "exit_rule",
                                "ticker": t,
                                "rule": rule,
                                "price": float(close_prev),
                            }
                        )
                elif rule == "60_20":
                    if close_prev <= rl2.loc[prev, t]:
                        positions.pop(t, None)
                        events.append(
                            {
                                "date": dt,
                                "event": "exit_rule",
                                "ticker": t,
                                "rule": rule,
                                "price": float(close_prev),
                            }
                        )

            # 진입 후보 정렬 후 상위 N개만 보유
            c_prev = closes.loc[prev]
            if args.low_rise_only:
                rise = (low20.loc[prev] > low20_prev.loc[prev])
                cand1 = rise
                cand2 = pd.Series(False, index=c_prev.index)
            elif args.use_regime and regime is not None and regime.loc[prev] == "mild_up":
                cand1 = pd.Series(False, index=c_prev.index)
                cand2 = pd.Series(False, index=c_prev.index)
            else:
                cand1 = c_prev >= rh1.loc[prev]
                cand2 = c_prev >= rh2.loc[prev] if not args.use_20_10_only else pd.Series(False, index=c_prev.index)

            strengths = []
            for t in c_prev.index:
                if cand1.get(t, False):
                    if args.low_rise_only:
                        prev_low = low20_prev.loc[prev, t]
                        cur_low = low20.loc[prev, t]
                        if pd.notna(prev_low) and prev_low != 0:
                            strengths.append((t, "20low_rise", (cur_low / prev_low) - 1.0))
                    else:
                        strengths.append((t, "20_10", (c_prev[t] / rh1.loc[prev, t]) - 1.0))
                if cand2.get(t, False):
                    strengths.append((t, "60_20", (c_prev[t] / rh2.loc[prev, t]) - 1.0))

            if strengths:
                strengths.sort(key=lambda x: x[2], reverse=True)
                slots = max(args.top_n - len(positions), 0)
                for t, rule, strength in strengths:
                    if slots == 0:
                        break
                    if strength < args.min_strength:
                        continue
                    if t in positions:
                        continue
                    positions[t] = {"rule": rule, "entry_price": closes.loc[prev, t]}
                    events.append(
                        {
                            "date": dt,
                            "event": "entry",
                            "ticker": t,
                            "rule": rule,
                            "price": float(closes.loc[prev, t]),
                            "strength": float(strength),
                        }
                    )
                    slots -= 1

            pos_ticker.loc[dt] = ";".join(sorted(positions.keys())) if positions else None
            if positions:
                base_w = 1.0 / args.top_n
                ret_sum = 0.0
                for t in positions:
                    if t in open_ret.columns and dt in open_ret.index:
                        r = open_ret.loc[dt, t]
                        if pd.isna(r):
                            continue
                        w = base_w * (args.commodity_cap if t in commodities else 1.0)
                        ret_sum += float(r) * w
                active_ret.loc[dt] = ret_sum

    open_ret = opens.shift(-1) / opens - 1.0
    active_ret = pd.Series(0.0, index=dates)
    for dt in dates:
        t = pos_ticker.loc[dt]
        if t is None or pd.isna(t):
            continue
        if t in open_ret.columns and dt in open_ret.index:
            active_ret.loc[dt] = open_ret.loc[dt, t]

    exec_pos = pos_ticker.shift(1 + args.signal_delay)
    size = pd.Series(args.position_size, index=dates)
    for dt in dates:
        t = exec_pos.loc[dt]
        if t in commodities:
            size.loc[dt] = args.position_size * args.commodity_cap
    # 포지션 사이즈와 원자재 캡을 반영한 누적 수익
    equity = (1 + (active_ret.fillna(0) * size)).cumprod()

    stats = performance_stats(equity)
    if args.top_n <= 1:
        trades_per_year = annual_trades(exec_pos.fillna("CASH"))
    else:
        entry_count = sum(1 for e in events if e["event"] == "entry")
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        trades_per_year = entry_count / years if years > 0 else 0.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, label="equity")
    ax.set_title("Breakout strongest signal (20/10 + 60/20)")
    ax.grid(True)
    ax.legend()
    fig_path = out_dir / "breakout_any_equity.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    # 비교용 SPY buy&hold(시가 기준)
    stats_rows = [{
        "strategy": "breakout_any",
        "CAGR": stats["CAGR"],
        "MDD": stats["MDD"],
        "Sharpe": stats["Sharpe"],
        "Trades_per_year": trades_per_year,
        "buy1": args.buy1,
        "sell1": args.sell1,
        "buy2": args.buy2,
        "sell2": args.sell2,
        "start": equity.index.min(),
        "end": equity.index.max(),
    }]
    spy_open = opens["SPY"] if "SPY" in opens.columns else load_open_series(args.db, "SPY")
    if not spy_open.empty:
        spy_open = spy_open.reindex(equity.index).dropna()
        spy_ret = spy_open.shift(-1) / spy_open - 1.0
        spy_equity = (1 + spy_ret.fillna(0)).cumprod()
        spy_stats = performance_stats(spy_equity)
        stats_rows.append({
            "strategy": "SPY_buy_hold",
            "CAGR": spy_stats["CAGR"],
            "MDD": spy_stats["MDD"],
            "Sharpe": spy_stats["Sharpe"],
            "Trades_per_year": 0.0,
            "buy1": None,
            "sell1": None,
            "buy2": None,
            "sell2": None,
            "start": spy_equity.index.min(),
            "end": spy_equity.index.max(),
        })

    stats_path = out_dir / "breakout_any_stats.csv"
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    picks_path = out_dir / "breakout_any_trades.csv"
    if args.top_n <= 1:
        pd.DataFrame({
            "date": pos_ticker.index,
            "ticker": pos_ticker.values,
            "rule": pos_rule.values,
        }).to_csv(picks_path, index=False)
    else:
        pd.DataFrame({
            "date": pos_ticker.index,
            "tickers": pos_ticker.values,
        }).to_csv(picks_path, index=False)

    events_path = out_dir / "breakout_any_events.csv"
    pd.DataFrame(events).to_csv(events_path, index=False)

    print("Backtest stats")
    print(f"CAGR: {stats['CAGR']:.4f}")
    print(f"MDD: {stats['MDD']:.4f}")
    print(f"Sharpe: {stats['Sharpe']:.4f}")
    print(f"Trades/year: {trades_per_year:.2f}")
    print(f"Saved equity plot: {fig_path}")
    print(f"Saved stats: {stats_path}")
    print(f"Saved trades: {picks_path}")
    print(f"Saved events: {events_path}")


if __name__ == "__main__":
    main()
