import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

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
    parser.add_argument("--all-tickers", action="store_true", help="DB에 있는 모든 티커 대상으로 학습/예측 수행")
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

    bench_df = pd.read_csv(Path(args.benchmark_csv), parse_dates=["date"])

    trainer = BuySignalTrainer(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    tickers = args.tickers or []
    if args.all_tickers:
        meta = trainer.db.load_ticker_metadata()
        tickers = list(meta.index) if not meta.empty else []
        logging.info("Loaded %d tickers from DB", len(tickers))
    if not tickers:
        logging.error("티커가 비어 있습니다. --tickers 또는 --all-tickers 중 하나를 지정하세요.")
        sys.exit(1)

    dataset = trainer.load_dataset(tickers, bench_df)
    if dataset.empty:
        logging.error("No dataset built; DB나 벤치마크 CSV를 확인하세요.")
        sys.exit(1)

    if args.save_dataset:
        Path(args.save_dataset).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(args.save_dataset, index=False)
        logging.info("Saved dataset to %s", args.save_dataset)

    fit_info = trainer.fit(dataset)
    logging.info("Fit results: %s", fit_info)

    preds = trainer.predict(dataset)
    logging.info("Predictions generated: %d rows", len(preds))

    thr = fit_info.get("threshold", getattr(trainer.model, "proba_threshold", 0.5))
    proba_stats = preds["buy_proba"].describe()
    n_signals = int((preds["buy_proba"] >= thr).sum())
    logging.info(
        "Summary -> samples=%d, proba(thr=%.3f) min/mean/max=%.4f/%.4f/%.4f, signals>=thr=%d",
        len(preds), thr, proba_stats["min"], proba_stats["mean"], proba_stats["max"], n_signals,
    )
    if fit_info.get("pr_auc") is not None:
        logging.info(
            "PR-AUC=%.4f, precision@thr=%.4f, recall@thr=%.4f",
            fit_info.get("pr_auc", float("nan")),
            fit_info.get("precision_at_threshold", float("nan")),
            fit_info.get("recall_at_threshold", float("nan")),
        )
    logging.info("CV scores: %s", fit_info.get("cv_scores"))

    if args.save_preds:
        Path(args.save_preds).parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(args.save_preds, index=False)
        logging.info("Saved predictions to %s", args.save_preds)

if __name__ == "__main__":
    main()

#PYTHONPATH=/app python scripts/train_buy_model.py \
  #--all-tickers \
  #--benchmark-csv data/kospi_benchmark.csv \
  #--start-date 2017-01-01 \
  #--end-date 2022-12-31 \
  #--db-path data/database/stocks.db \
  #--save-dataset data/output/buy_dataset.csv \
  #--save-preds data/output/buy_preds.csv
