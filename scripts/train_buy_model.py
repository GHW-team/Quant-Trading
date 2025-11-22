import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 파이썬 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.buysignal_trainer import BuySignalTrainer


def setup_logging(level: str) -> None:
    """로깅 레벨/포맷 설정"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """CLI 인자 정의"""
    parser = argparse.ArgumentParser(description="매수 신호 모델 학습 실행")
    parser.add_argument("--tickers", nargs="+", help="사용할 티커 코드 (예: 005930.KS 000660.KS)")
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="DB에 있는 모든 티커 대상으로 학습/예측 수행",
    )
    parser.add_argument("--benchmark-csv", required=True, help="벤치마크 CSV 경로 (컬럼: date,kospi_close)")
    parser.add_argument("--start-date", default="2017-01-01", help="학습 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2022-12-31", help="학습 종료일 (YYYY-MM-DD)")
    parser.add_argument("--db-path", default="data/database/stocks.db", help="SQLite DB 경로")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="로그 레벨")
    parser.add_argument("--save-preds", help="예측 CSV 저장 경로")
    parser.add_argument("--save-dataset", help="피처/라벨 데이터셋 CSV 저장 경로")
    return parser.parse_args()


def main() -> None:
    """벤치마크 로드 → DB 병합 → 모델 학습/예측 → (선택) CSV 저장"""
    args = parse_args()
    setup_logging(args.log_level)

    # 벤치마크 지수 로드
    bench_path = Path(args.benchmark_csv)
    bench_df = pd.read_csv(bench_path, parse_dates=["date"])

    trainer = BuySignalTrainer(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # 학습 대상 티커 결정: CLI 지정 또는 DB 전체
    tickers = args.tickers or []
    if args.all_tickers:
        meta = trainer.db.load_ticker_metadata()
        tickers = list(meta.index) if not meta.empty else []
        logging.info("Loaded %d tickers from DB", len(tickers))
    if not tickers:
        logging.error("티커가 비어 있습니다. --tickers 또는 --all-tickers 중 하나를 지정하세요.")
        sys.exit(1)

    # 가격/지표/벤치마크 병합 데이터셋 생성
    dataset = trainer.load_dataset(tickers, bench_df)
    if dataset.empty:
        logging.error("No dataset built; DB나 벤치마크 CSV를 확인하세요.")
        sys.exit(1)

    # (선택) 데이터셋 저장
    if args.save_dataset:
        Path(args.save_dataset).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(args.save_dataset, index=False)
        logging.info("Saved dataset to %s", args.save_dataset)

    # 학습 및 간단한 CV 결과
    fit_info = trainer.fit(dataset)
    logging.info("Fit results: %s", fit_info)

    # 예측 및 (선택) 저장
    preds = trainer.predict(dataset)
    logging.info("Predictions generated: %d rows", len(preds))

    if args.save_preds:
        Path(args.save_preds).parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(args.save_preds, index=False)
        logging.info("Saved predictions to %s", args.save_preds)


if __name__ == "__main__":
    main()
