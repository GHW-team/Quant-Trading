import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = "data/ohlcv"
TICKER_DIR = "data/tickers"

os.makedirs(DATA_DIR, exist_ok=True)

def load_tickers():
    """S&P500, NASDAQ100, KOSPI 티커 리스트 로드"""
    tickers = []

    files = ["sp500.csv", "nasdaq100.csv", ]
    for f in files:
        path = os.path.join(TICKER_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "Symbol" in df.columns:
                tickers.extend(df["Symbol"].tolist())
            elif "티커" in df.columns:
                tickers.extend(df["티커"].tolist())

    tickers = list(set(tickers))  # 중복 제거
    print(f"📌 총 {len(tickers)}개의 티커 로드 완료")
    return tickers


def download_new_data(ticker):
    """티커의 신규 데이터를 다운받아 parquet로 저장"""
    file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")

    # 기존 데이터 있으면 로드
    if os.path.exists(file_path):
        old_df = pd.read_parquet(file_path)
        last_date = old_df.index.max()
        start_date = last_date + timedelta(days=1)
        print(f"🔄 {ticker}: {start_date.date()} 이후 데이터 업데이트 중...")
    else:
        old_df = None
        start_date = "2000-01-01"  # 전체 다운로드
        print(f"⬇️ {ticker}: 신규 다운로드 시작...")

    try:
        new_df = yf.download(ticker, start=start_date)

        if new_df.empty:
            print(f"⚠️ {ticker}: 신규 데이터 없음")
            return

        # 새로운 데이터 병합
        if old_df is not None:
            combined = pd.concat([old_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df

        combined.to_parquet(file_path)
        print(f"✅ {ticker}: 저장 완료")
    except Exception as e:
        print(f"❌ {ticker}: 다운로드 실패 -> {e}")


def update_all_prices():
    tickers = load_tickers()

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}]")
        download_new_data(ticker)


if __name__ == "__main__":
    print("=== 가격 데이터 업데이트 시작 ===")
    update_all_prices()
    print("=== 모든 티커 업데이트 완료 ===")
