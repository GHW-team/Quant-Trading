import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yaml

# 프로젝트 루트 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.all_ticker import TickerUniverse
from src.ml.backtest import DataLoader, BacktestRunner


def parse_args():
    parser = argparse.ArgumentParser(description="백테스트 실행 스크립트")
    parser.add_argument("--config", default="config/config.yaml", help="설정 파일 경로")
    parser.add_argument("--start", help="백테스트 시작일(YYYY-MM-DD)")
    parser.add_argument("--end", help="백테스트 종료일(YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", help="직접 지정할 티커 리스트")
    parser.add_argument("--equity-csv", default="plots/equity.csv", help="자산 곡선 저장 경로(csv)")
    parser.add_argument("--equity-plot", default="plots/equity.png", help="자산 곡선 시각화 경로(png)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

    db_path = config["data"]["database_path"]

    # 심볼 결정: CLI > config.exchanges(KOSPI 기본)
    if args.symbols:
        symbols = args.symbols
    else:
        exchanges = config["data"].get("exchanges") or ["KOSPI"]
        symbols = TickerUniverse().get(exchanges)

    # 기간: CLI > config (필수)
    start = args.start or config["data"].get("start_date")
    end = args.end or config["data"].get("end_date")
    if not start or not end:
        raise ValueError("백테스트 시작/종료일을 지정하세요 (--start/--end 또는 config)")

    loader = DataLoader(db_path)
    feeds = loader.load(symbols=symbols, start=start, end=end)

    runner = BacktestRunner(commission=0.00015)
    runner.add_feeds(feeds)
    runner.run()
    metrics = runner.metrics()

    print(f"Final equity: {runner.summary():.2f}")
    print(f"Sharpe ratio: {metrics.get('sharpe'):.4f}" if metrics.get("sharpe") is not None else "Sharpe ratio: N/A")
    print(f"Sortino ratio: {metrics.get('sortino'):.4f}" if metrics.get("sortino") is not None else "Sortino ratio: N/A")
    print(f"Max drawdown (%): {metrics.get('max_drawdown'):.2f}" if metrics.get("max_drawdown") is not None else "Max drawdown (%): N/A")
    print(f"Max drawdown duration (bars): {metrics.get('max_drawdown_duration')}" if metrics.get("max_drawdown_duration") is not None else "Max drawdown duration (bars): N/A")

    # 트레이드 요약
    trades = runner.trades()
    if trades and trades.get("total"):
        total = trades.get("total", {})
        closed = total.get("closed", 0)
        # Backtrader TradeAnalyzer 구조: 승/패 카운트는 top-level 'won'/'lost' 딕셔너리의 'total' 키에 있다.
        won = trades.get("won", {}).get("total", 0)
        lost = trades.get("lost", {}).get("total", 0)
        print(f"Trades closed: {closed}, won: {won}, lost: {lost}")

    # 자산 곡선 저장
    eq = runner.equity_curve()
    if eq is not None and args.equity_csv:
        eq_path = Path(args.equity_csv)
        eq_path.parent.mkdir(parents=True, exist_ok=True)
        eq.to_csv(eq_path, header=["equity"])
        print(f"Equity curve saved to {eq_path}")
        if args.equity_plot:
            eq_plot_path = Path(args.equity_plot)
            eq_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 4))
            eq.plot()
            plt.title("Equity Curve")
            plt.xlabel("Index")
            plt.ylabel("Equity")
            plt.tight_layout()
            plt.savefig(eq_plot_path, bbox_inches="tight")
            plt.close()
            print(f"Equity plot saved to {eq_plot_path}")

    # 전략 차트 저장은 제외 (자산 곡선만 저장)


if __name__ == "__main__":
    main()
