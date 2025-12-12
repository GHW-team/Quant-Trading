import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import sqlalchemy


def calc_zweig_breadth(df: pd.DataFrame) -> pd.Series:
    total = df["advancing"] + df["declining"]
    adr = df["advancing"] / total.replace(0, np.nan)
    return ta.ema(adr, length=10)


def detect_zbt_signal(zbt: pd.Series, oversold=0.40, thrust=0.615, window=10):
    if zbt.empty:
        return {"signal": False, "remaining_points": None, "days_left": None}

    below = zbt < oversold
    if not below.any():
        return {
            "signal": False,
            "remaining_points": max(thrust - zbt.iloc[-1], 0),
            "days_left": -1,
        }

    last_dip_idx = below[::-1].idxmax()
    post = zbt.loc[last_dip_idx:].iloc[1 : window + 1]
    hit = post[post >= thrust]
    if not hit.empty:
        return {"signal": True, "remaining_points": 0.0, "days_left": 0}

    days_left = max(window - len(post), 0)
    remaining_points = thrust - oversold if len(post) == 0 else max(thrust - post.iloc[-1], 0)
    return {"signal": False, "remaining_points": remaining_points, "days_left": days_left}


def required_advancers_for_thrust(zbt_last, total_issues, thrust=0.615, length=10):
    alpha = 2 / (length + 1)
    adr_needed = (thrust - (1 - alpha) * zbt_last) / alpha
    adr_clamped = min(max(adr_needed, 0), 1)
    adv_needed = int(np.ceil(adr_clamped * total_issues))
    feasible = adr_needed <= 1
    return {"adr_needed": adr_needed, "adv_needed": adv_needed, "feasible": feasible}


def required_advancers_over_window(zbt_last, total_issues, days_left, thrust=0.615, length=10):
    """
    남은 days_left 동안 매일 같은 ADR로 움직인다고 가정할 때 필요한 평균 ADR과 상승 종목 수.
    """
    if days_left is None or days_left <= 0:
        return None

    alpha = 2 / (length + 1)
    decay = (1 - alpha) ** days_left
    denom = 1 - decay
    if denom == 0:
        return None

    adr_needed = (thrust - decay * zbt_last) / denom
    adr_clamped = min(max(adr_needed, 0), 1)
    adv_needed = int(np.ceil(adr_clamped * total_issues))
    feasible = adr_needed <= 1
    return {"adr_needed": adr_needed, "adv_needed": adv_needed, "feasible": feasible, "days": days_left}


def load_prices(engine, days=252, tickers=None):
    base_query = f"""
    SELECT dp.date, t.ticker_code AS ticker, dp.close
    FROM daily_prices dp
    JOIN tickers t ON dp.ticker_id = t.ticker_id
    WHERE dp.date >= (SELECT DATE(MAX(date), '-{days} day') FROM daily_prices)
    """
    params = {}
    if tickers is not None and len(tickers) > 0:
        placeholders = ",".join(f":t{i}" for i in range(len(tickers)))
        base_query += f" AND t.ticker_code IN ({placeholders})"
        params = {f"t{i}": sym for i, sym in enumerate(tickers)}
    base_query += " ORDER BY date, ticker;"
    return pd.read_sql(base_query, engine, params=params, parse_dates=["date"])


def compute_breadth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"])
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["chg"] = df["close"] - df["prev_close"]
    daily = (
        df.groupby("date")
        .agg(advancing=("chg", lambda x: (x > 0).sum()), declining=("chg", lambda x: (x < 0).sum()))
        .reset_index()
    )
    daily["total"] = daily["advancing"] + daily["declining"]
    return daily


def main():
    engine = sqlalchemy.create_engine("sqlite:///data/database/stocks.db")

    # NYSE 티커 로드
    ticker_df = None
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            ticker_df = pd.read_csv("data/tickers/nyse.csv", encoding=enc)
            break
        except UnicodeDecodeError:
            ticker_df = None
            continue
    if ticker_df is None:
        raise RuntimeError("nyse.csv를 읽지 못했습니다. 인코딩을 확인하세요.")
    if "Symbol" not in ticker_df.columns:
        raise RuntimeError("nyse.csv에 Symbol 컬럼이 없습니다.")
    tickers = ticker_df["Symbol"].dropna().str.strip().tolist()
    if not tickers:
        raise RuntimeError("nyse.csv에서 읽은 티커가 비어 있습니다.")

    prices = load_prices(engine, days=365, tickers=tickers)
    breadth = compute_breadth(prices)

    breadth["adr"] = breadth["advancing"] / breadth["total"]
    breadth["zbt"] = calc_zweig_breadth(breadth)
    status = detect_zbt_signal(breadth["zbt"])

    dips = breadth[breadth["zbt"] < 0.40]
    last_dip_date = dips.iloc[-1]["date"] if not dips.empty else None

    latest = breadth.iloc[-1]
    today_adr = latest["adr"] if latest["total"] else None
    avg_adr_all = breadth["adr"].mean()
    avg_adr_since_dip = (
        breadth.loc[breadth["date"] >= last_dip_date, "adr"].mean() if last_dip_date is not None else None
    )

    need = required_advancers_for_thrust(zbt_last=latest["zbt"], total_issues=latest["total"])
    window_need = required_advancers_over_window(
        zbt_last=latest["zbt"], total_issues=latest["total"], days_left=status.get("days_left")
    )

    # 마지막 딥 이후 평균 ADR로 남은 기간을 채웠을 때 도달 여부 추정
    avg_dip_feasible = None
    if avg_adr_since_dip is not None and window_need:
        avg_dip_feasible = avg_adr_since_dip >= window_need["adr_needed"]

    print("ZBT 상태:", status)
    print("오늘 ZBT:", round(latest["zbt"], 4))
    print(
        "오늘 ADR(상승 비율):",
        round(today_adr, 4) if today_adr is not None else "N/A",
        f"({int(latest['advancing'])} / {int(latest['total'])})",
    )
    print("최근 1년 평균 ADR:", round(avg_adr_all, 4))
    if avg_adr_since_dip is not None:
        print("마지막 딥 이후 평균 ADR:", round(avg_adr_since_dip, 4))
    print("latest dip(<0.40):", last_dip_date.date() if last_dip_date is not None else "없음")
    print("다음 날 0.615 도달하기 위해 필요한 ADR:", round(need["adr_needed"], 4))
    print("다음 날 0.615 도달하기 위해 필요한 상승 종목:", need["adv_needed"], f"/ 총 {int(latest['total'])}")
    if window_need:
        print(f"남은 {window_need['days']}일 동안 매일 필요한 ADR(평균):", round(window_need["adr_needed"], 4))
        print("하루 평균 필요한 상승 종목:", window_need["adv_needed"], f"/ 총 {int(latest['total'])}")
        feasible_text = "가능" if window_need["feasible"] else "불가능(ADR>1 필요)"
        if avg_dip_feasible is not None:
            print("latest dip 이후 평균 ADR으로 계속 가면 window 내 달성여부", "가능" if avg_dip_feasible else "불가")


if __name__ == "__main__":
    main()
