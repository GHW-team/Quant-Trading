# scripts/run_pipeline.py - 파이프라인 실행용 CLI (config.yaml + CLI override)
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.all_ticker import TickerUniverse
from src.data.pipeline import DataPipeline


# ============================================
# 로깅 설정
# ============================================
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


# ============================================
# 설정 로더
# ============================================
class ConfigLoader:
    """
    config.yaml을 기본으로 로드하고, CLI 인자로만 덮어씌운다.
    """

    def __init__(self, config_file: str = "config.yaml") -> None:
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        self.yaml_config = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        if self.config_file.exists():
            with self.config_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.logger.info("YAML config loaded: %s", self.config_file)
                return cfg or {}
        self.logger.warning("Config file not found: %s", self.config_file)
        return {}

    def get_config(self, cli_args: argparse.Namespace) -> Dict[str, Any]:
        config = self._build_config_from_yaml()
        config = self._merge_cli_args(config, cli_args)

        self.logger.info("\n%s", "=" * 70)
        self.logger.info("Final Configuration")
        self.logger.info("%s", "=" * 70)
        self.logger.info("Tickers: %s", config["tickers"] or "(auto via exchanges)")
        period_str = config.get("period") or f"{config.get('start_date')} ~ {config.get('end_date')}"
        self.logger.info("Period: %s", period_str)
        self.logger.info("Indicators: %s", config["indicators"])
        self.logger.info("Batch size: %s", config["batch_size"])
        self.logger.info("Database: %s", config["database_path"])
        self.logger.info("%s\n", "=" * 70)
        return config

    def _build_config_from_yaml(self) -> Dict[str, Any]:
        data_cfg = self.yaml_config.get("data", {}) or {}
        indicators_cfg = self.yaml_config.get("indicators", {}) or {}
        batch_cfg = self.yaml_config.get("batch", {}) or {}
        fetcher_cfg = self.yaml_config.get("fetcher", {}) or {}
        logging_cfg = self.yaml_config.get("logging", {}) or {}

        return {
            # 데이터 설정
            "tickers": data_cfg.get("tickers", ["005930.KS", "000660.KS"]),
            "exchanges": data_cfg.get("exchanges", []),
            "database_path": data_cfg.get("database_path", "data/database/stocks.db"),
            "period": data_cfg.get("period", "1y"),
            "start_date": data_cfg.get("start_date"),
            "end_date": data_cfg.get("end_date"),
            "update_if_exists": data_cfg.get("update_if_exists", True),
            "interval": data_cfg.get("interval", "1d"),

            # 지표 설정
            "indicators": indicators_cfg.get("list", ["ma_5", "ma_20", "ma_200", "macd"]),
            "indicator_version": indicators_cfg.get("version", "v1.0"),

            # 배치 설정
            "batch_size": batch_cfg.get("size", 100),

            # Fetcher 설정
            "max_workers": fetcher_cfg.get("max_workers", 5),
            "max_retries": fetcher_cfg.get("max_retries", 3),
            "log_level": logging_cfg.get("level", "INFO"),
        }

    def _merge_cli_args(self, config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
        # CLI에서 제공된 값만 덮어쓰기
        if cli_args.tickers:
            config["tickers"] = cli_args.tickers
            self.logger.debug("Override from CLI: tickers = %s", config["tickers"])
        if cli_args.exchanges:
            config["exchanges"] = cli_args.exchanges
            self.logger.debug("Override from CLI: exchanges = %s", config["exchanges"])

        if cli_args.period:
            config["period"] = cli_args.period
            config["start_date"] = None
            config["end_date"] = None
            self.logger.debug("Override from CLI: period = %s", config["period"])
        if cli_args.start_date:
            config["start_date"] = cli_args.start_date
            config["period"] = None
            self.logger.debug("Override from CLI: start_date = %s", config["start_date"])
        if cli_args.end_date:
            config["end_date"] = cli_args.end_date
            self.logger.debug("Override from CLI: end_date = %s", config["end_date"])
        if cli_args.indicators:
            config["indicators"] = cli_args.indicators
            self.logger.debug("Override from CLI: indicators = %s", config["indicators"])
        if cli_args.batch_size:
            config["batch_size"] = cli_args.batch_size
            self.logger.debug("Override from CLI: batch_size = %s", config["batch_size"])
        if cli_args.update_if_exists is not None:
            config["update_if_exists"] = cli_args.update_if_exists
            self.logger.debug("Override from CLI: update_if_exists = %s", config["update_if_exists"])

        return config


# ============================================
# 파이프라인 실행 함수
# ============================================
def _resolve_tickers(config: Dict[str, Any]) -> Any:
    """tickers와 exchanges 설정을 해석하고 상충 시 에러를 낸다."""
    tickers = config.get("tickers") or []
    exchanges = config.get("exchanges") or []

    if tickers and exchanges:
        raise ValueError("tickers와 exchanges를 동시에 지정할 수 없습니다. 하나만 선택하세요.")

    if tickers:
        return tickers
    if exchanges:
        return TickerUniverse().get(exchanges)
    raise ValueError("티커 리스트가 비어 있습니다. tickers 또는 exchanges를 지정하세요.")


def run_full_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("▶ Full Pipeline 실행 (Fetch → Save → Calculate → Save Indicators)")
    logger.info("=" * 70)

    with DataPipeline(
        db_path=config["database_path"],
        max_workers=config["max_workers"],
        max_retries=config["max_retries"],
        batch_size_default=config["batch_size"],
    ) as pipeline:
        results = pipeline.run_full_pipeline(
            ## 임시수정!!
            ticker_list=_resolve_tickers(config)[:50],
            start_date=config.get("start_date"),
            end_date=config.get("end_date"),
            period=config.get("period"),
            interval=config["interval"],
            update_if_exists=config["update_if_exists"],
            indicator_list=config["indicators"],
            version=config["indicator_version"],
            batch_size=config["batch_size"],
        )
        _print_results(results, logger)


def run_price_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("▶ Price Pipeline 실행 (Fetch → Save Price Data)")
    logger.info("=" * 70)

    with DataPipeline(
        db_path=config["database_path"],
        max_workers=config["max_workers"],
        max_retries=config["max_retries"],
        batch_size_default=config["batch_size"],
    ) as pipeline:
        results = pipeline.run_price_pipeline(
            ticker_list=_resolve_tickers(config),
            start_date=config.get("start_date"),
            end_date=config.get("end_date"),
            period=config.get("period"),
            interval=config["interval"],
            update_if_exists=config["update_if_exists"],
            batch_size=config["batch_size"],
        )
        _print_results(results, logger)


def run_indicator_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("▶ Indicator Pipeline 실행 (Load → Calculate → Save)")
    logger.info("=" * 70)

    with DataPipeline(
        db_path=config["database_path"],
        max_workers=config["max_workers"],
        max_retries=config["max_retries"],
        batch_size_default=config["batch_size"],
    ) as pipeline:
        results = pipeline.run_indicator_pipeline(
            ticker_list=_resolve_tickers(config),
            indicator_list=config["indicators"],
            start_date=config.get("start_date"),
            end_date=config.get("end_date"),
            version=config["indicator_version"],
            batch_size=config["batch_size"],
        )
        _print_results(results, logger)


def _print_results(results: Any, logger: logging.Logger) -> None:
    logger.info("Result summary: %s", results.get("summary"))


# ============================================
# CLI 파서
# ============================================
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="데이터 파이프라인 실행기 (config.yaml + CLI args)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    pipeline_group = parser.add_mutually_exclusive_group(required=True)
    pipeline_group.add_argument("--full", action="store_true", help="전체 파이프라인 실행")
    pipeline_group.add_argument("--price", action="store_true", help="가격 파이프라인만 실행")
    pipeline_group.add_argument("--indicators", action="store_true", help="지표 파이프라인만 실행")

    parser.add_argument("--tickers", nargs="+", help="종목 코드 (예: 005930.KS 000660.KS)")
    parser.add_argument("--exchanges", nargs="+", help="거래소 코드 리스트 (예: KOSPI KOSDAQ S&P500)")
    parser.add_argument("--period", help="기간 (예: 1y, 6m, 3m, 1m)")
    parser.add_argument("--start-date", help="시작 날짜 (YYYY-MM-DD, period와 함께 사용 불가)")
    parser.add_argument("--end-date", help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--indicators-list", dest="indicators", nargs="+", help="계산할 지표")
    parser.add_argument("--batch-size", type=int, help="배치 크기 (기본: 100)")
    parser.add_argument(
        "--update-if-exists",
        dest="update_if_exists",
        action="store_true",
        help="기존 데이터 업데이트",
    )
    parser.add_argument(
        "--no-update",
        dest="update_if_exists",
        action="store_false",
        help="기존 데이터 보존",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨 (기본: INFO)",
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='설정 파일 경로 (기본: config/config.yaml)'
    )

    return parser


# ============================================
# 메인 실행
# ============================================
def main():
    parser = create_parser()
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    try:
        config_loader = ConfigLoader(config_file=args.config)
        config = config_loader.get_config(args)

        if args.full:
            run_full_pipeline(config, logger)
        elif args.price:
            run_price_pipeline(config, logger)
        elif args.indicators:
            run_indicator_pipeline(config, logger)

        logger.info("\n✅ Pipeline execution completed!")

    except KeyboardInterrupt:
        logger.warning("\n⛔ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error("\n❌ Pipeline failed with error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
