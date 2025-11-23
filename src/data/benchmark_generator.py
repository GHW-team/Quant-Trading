import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

def parse_args():
    p = argparse.ArgumentParser(description="Download benchmark index to CSV")
    p.add_argument("--ticker", default="^KS11", help="벤치마크 지수 티커 (기본 KOSPI: ^KS11)")
    p.add_argument("--start", default="2017-01-01", help="시작일 YYYY-MM-DD")
    p.add_argument("--end", default="2022-12-31", help="종료일 YYYY-MM-DD")
    p.add_argument("--out", default="data/kospi_benchmark.csv", help="저장 경로 (CSV, 컬럼: date,kospi_close)")
    return p.parse_args()

def main():
    args = parse_args()
    df = yf.download(args.ticker, start=args.start, end=args.end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise SystemExit(f"Download failed or empty for {args.ticker}")
    out_df = df.reset_index()[["Date", "Close"]]
    out_df.columns = ["date", "kospi_close"]
    out_df["date"] = pd.to_datetime(out_df["date"]).dt.strftime("%Y-%m-%d")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"saved {len(out_df)} rows to {out_path}")

if __name__ == "__main__":
    main()

#실행방법: python -m src.data.benchmark_generator --ticker ^KS11 --start 2017-01-01 --end 2022-12-31 --out data/kospi_benchmark.csv