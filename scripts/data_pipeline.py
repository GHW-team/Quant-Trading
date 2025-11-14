import pathlib
import pandas as pd
from src.download_yf import download_yf
from src.to_long import to_long
# from src.securities import insert_security
from src.price import upsert_price_daily, upsert_features_daily
from src.features import add_ma

from src.config import get_config, get_db_path
cfg = get_config()
DB_PATH = get_db_path()

FETCH_CFG = cfg["fetch"]
TICKERS = FETCH_CFG["tickers"]          # ["005930.KS", "000660.KS", ...]
START = FETCH_CFG["start"]
END = FETCH_CFG["end"]
AUTO_ADJUST = FETCH_CFG.get("auto_adjust", True)

FEATURE_CFG = cfg.get("features", {})
FEATURE_VERSION = FEATURE_CFG.get("version", "v1")
MA_WINDOWS = FEATURE_CFG.get("ma_windows", [5, 20, 200])
# python -m scripts.fetch_price

def run_for_ticker(ticker):
    raw = download_yf(ticker, START, END, auto_adjust=True)
    long_df = to_long(raw, ticker)
    # print(long_df)
    # print(long_df["date"])
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.date.astype(str) # 문자열로 바꿈
    # print(long_df)
    # print(long_df["date"])

    rows_price = []
    for r in long_df.itertuples(index=False):
        # print(r)
        rows_price.append((
            r.ticker,
            r.date,
            r.open,
            r.high,
            r.low,
            r.close,
            r.adj_close,
            r.volume
        ))
        # print(rows)
    # print(rows)

    # 5) price_daily에 upsert
    print("[STEP 3] price_daily 테이블에 가격 데이터 upsert...")
    upsert_price_daily(DB_PATH, rows_price)
    print(f"[INFO] price_daily에 {len(rows_price)}행 upsert 완료")

    # 6) 피처 계산 (MA, 수익률, 갭 등)
    print("[STEP 4] 피처 계산 (MA, 수익률, 갭)...")
    feat_df = long_df.copy()

    # MA 5, 20, 200
    feat_df = add_ma(feat_df, windows=(5, 20, 200))

    # 수익률: 1일, 5일
    feat_df["ret_1d"] = feat_df["close"].pct_change(1)
    feat_df["ret_5d"] = feat_df["close"].pct_change(5)

    # 갭: 어제 종가 대비 오늘 시가
    # gap_o_c = (오늘 시가 - 어제 종가) / 어제 종가
    feat_df["gap_o_c"] = (
        (feat_df["open"] - feat_df["close"].shift(1)) / feat_df["close"].shift(1)
    )

    # 7) features_daily용 rows 만들기
    rows_feat = []
    for r in feat_df.itertuples(index=False):
        rows_feat.append(
            (
                r.ticker,
                r.date,
                getattr(r, "ma_5", None),
                getattr(r, "ma_20", None),
                getattr(r, "ma_200", None),
                r.ret_1d,
                r.ret_5d,
                r.gap_o_c,
                FEATURE_VERSION,
            )
        )

    # 8) features_daily에 upsert
    print("[STEP 5] features_daily 테이블에 피처 upsert...")
    upsert_features_daily(DB_PATH, rows_feat)


for tk in TICKERS:
    run_for_ticker(tk)
print("successsssssssssssssssssss")






root = pathlib.Path(__file__).resolve().parents[1]
db_path = root / "data" / "market.db"

#insert_security(db_path, TICKERS, NAME, EXCHANGE)
#upsert_price_daily(db_path, rows)