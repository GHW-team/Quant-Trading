# scripts/run_pipeline.py - 데이터 파이프라인 실행 (config + env + CLI)

import sys
import os
import logging
import argparse
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pipeline import DataPipeline
from src.data.indicator_calculator import IndicatorCalculator

# ============================================
# 로깅 설정
# ============================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


# ============================================
# 설정 로더 (2-tier cascading)
# ============================================

class ConfigLoader:
    """
    설정을 다음 순서로 로드합니다:
    1. config.yaml (기본값)
    2. CLI 인자 (최우선)
    """

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.yaml_config = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.logger.info(f"✓ YAML config loaded: {self.config_file}")
                    return config or {}
            else:
                self.logger.warning(f"⚠ Config file not found: {self.config_file}")
                return {}
        except Exception as e:
            self.logger.error(f"✗ Failed to load config: {e}")
            return {}

    def get_config(self, cli_args: argparse.Namespace) -> Dict[str, Any]:
        """
        2-tier cascading으로 최종 설정 생성

        순서:
        1. YAML config (기본값)
        2. CLI 인자로 덮어쓰기 (최우선)
        """
        config = self._build_config_from_yaml()
        config = self._merge_cli_args(config, cli_args)

        self.logger.info(f"\n{'='*70}")
        self.logger.info("Final Configuration")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Tickers: {config['tickers']}")

        period_str = config.get('period') or f"{config.get('start_date')} ~ {config.get('end_date')}"
        self.logger.info(f"Period: {period_str}")

        self.logger.info(f"Indicators: {config['indicators']}")
        self.logger.info(f"Batch size: {config['batch_size']}")
        self.logger.info(f"Database: {config['database_path']}")
        self.logger.info(f"{'='*70}\n")

        return config

    def _build_config_from_yaml(self) -> Dict[str, Any]:
        """YAML 설정 파일을 읽어 설정 빌드"""
        config = {
            # 데이터 설정
            'tickers': self.yaml_config.get('data', {}).get('tickers', ['005930.KS', '000660.KS']),
            'database_path': self.yaml_config.get('data', {}).get('database_path', 'data/database/stocks.db'),
            'period': self.yaml_config.get('data', {}).get('period'),
            'start_date': self.yaml_config.get('data', {}).get('start_date'),
            'end_date': self.yaml_config.get('data', {}).get('end_date'),
            'update_if_exists': self.yaml_config.get('data', {}).get('update_if_exists', True),
            'interval': self.yaml_config.get('data', {}).get('interval', '1d'),

            # 지표 설정
            'indicators': self.yaml_config.get('indicators', {}).get('list', ['ma_5', 'ma_20', 'ma_200', 'macd']),
            'indicator_version': self.yaml_config.get('indicators', {}).get('version', 'v1.0'),
            'lookback_enabled': self.yaml_config.get('indicators', {}).get('lookback', {}).get('enabled', True),

            # 배치 설정
            'batch_size': self.yaml_config.get('batch', {}).get('size', 100),

            # Fetcher 설정
            'max_workers': self.yaml_config.get('fetcher', {}).get('max_workers', 5),
            'max_retries': self.yaml_config.get('fetcher', {}).get('max_retries', 3),

            # 로깅 설정
            'log_level': self.yaml_config.get('logging', {}).get('level', 'INFO'),
        }

        return config

    def _merge_cli_args(self, config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
        """CLI 인자로 config.yaml 설정 덮어쓰기"""
        # CLI에서 제공된 인자들만 덮어쓰기
        if cli_args.tickers:
            config['tickers'] = cli_args.tickers
            self.logger.debug(f"✓ Override from CLI: tickers = {config['tickers']}")

        if cli_args.period:
            config['period'] = cli_args.period
            config['start_date'] = None  # 충돌 방지
            config['end_date'] = None
            self.logger.debug(f"✓ Override from CLI: period = {config['period']}")

        if cli_args.start_date:
            config['start_date'] = cli_args.start_date
            config['period'] = None  # 충돌 방지
            self.logger.debug(f"✓ Override from CLI: start_date = {config['start_date']}")

        if cli_args.end_date:
            config['end_date'] = cli_args.end_date
            self.logger.debug(f"✓ Override from CLI: end_date = {config['end_date']}")

        if cli_args.indicators:
            config['indicators'] = cli_args.indicators
            self.logger.debug(f"✓ Override from CLI: indicators = {config['indicators']}")

        if cli_args.batch_size:
            config['batch_size'] = cli_args.batch_size
            self.logger.debug(f"✓ Override from CLI: batch_size = {config['batch_size']}")

        if cli_args.update_if_exists is not None:
            config['update_if_exists'] = cli_args.update_if_exists
            self.logger.debug(f"✓ Override from CLI: update_if_exists = {config['update_if_exists']}")

        return config


# ============================================
# 파이프라인 실행 함수들
# ============================================

def run_full_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    """전체 파이프라인 실행 (Step 1-4)"""
    logger.info("\n" + "="*70)
    logger.info("🚀 Full Pipeline 실행 (Fetch → Save → Calculate → Save Indicators)")
    logger.info("="*70)

    with DataPipeline(
        db_path=config['database_path'],
        max_workers=config['max_workers'],
        max_retries=config['max_retries']
    ) as pipeline:
        results = pipeline.run_full_pipeline(
            ticker_list=config['tickers'],
            start_date=config.get('start_date'),
            end_date=config.get('end_date'),
            period=config.get('period'),
            interval=config['interval'],
            update_if_exists=config['update_if_exists'],
            indicator_list=config['indicators'],
            version=config['indicator_version'],
            batch_size=config['batch_size'],
        )

        _print_results(results, logger)


def run_price_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    """가격 데이터 파이프라인만 (Step 1-2)"""
    logger.info("\n" + "="*70)
    logger.info("📊 Price Pipeline 실행 (Fetch → Save Price Data)")
    logger.info("="*70)

    with DataPipeline(
        db_path=config['database_path'],
        max_workers=config['max_workers'],
        max_retries=config['max_retries']
    ) as pipeline:
        results = pipeline.run_price_pipeline(
            ticker_list=config['tickers'],
            start_date=config.get('start_date'),
            end_date=config.get('end_date'),
            period=config.get('period'),
            interval=config['interval'],
            update_if_exists=config['update_if_exists'],
            batch_size=config['batch_size'],
        )

        _print_results(results, logger)


def run_indicator_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    """지표 파이프라인만 (Step 3-4)"""
    logger.info("\n" + "="*70)
    logger.info("📈 Indicator Pipeline 실행 (Load → Calculate → Save)")
    logger.info("="*70)

    with DataPipeline(
        db_path=config['database_path'],
        max_workers=config['max_workers'],
        max_retries=config['max_retries']
    ) as pipeline:
        results = pipeline.run_indicator_pipeline(
            ticker_list=config['tickers'],
            indicator_list=config['indicators'],
            start_date=config.get('start_date'),
            end_date=config.get('end_date'),
            version=config['indicator_version'],
            batch_size=config['batch_size'],
        )

        _print_results(results, logger)


def _print_results(results: Any, logger: logging.Logger) -> None:
    """결과 출력 (pipeline 내부에서 이미 상세 로깅됨)"""
    # pipeline.run_*() 함수들은 DataFrame dict를 반환
    # 파이프라인 내부에서 이미 모든 결과를 logger로 출력하므로
    # 여기서는 최종 완료 메시지만 출력
    logger.info("")


# ============================================
# CLI 인자 파서
# ============================================

def create_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description='데이터 파이프라인 실행 (config.yaml + CLI args)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # config.yaml 기본 설정으로 실행
  python scripts/run_pipeline.py --full

  # CLI로 종목 지정 (config.yaml 덮어쓰기)
  python scripts/run_pipeline.py --full --tickers 005930.KS 000660.KS

  # 기간 지정 (config.yaml 덮어쓰기)
  python scripts/run_pipeline.py --full --period 6m

  # 날짜 범위 지정
  python scripts/run_pipeline.py --full --start-date 2024-01-01 --end-date 2024-12-31

  # 가격 데이터만
  python scripts/run_pipeline.py --price

  # 지표만 (기존 DB 데이터 사용)
  python scripts/run_pipeline.py --indicators

  # 배치 크기 변경
  python scripts/run_pipeline.py --full --batch-size 50

  # 여러 옵션 조합
  python scripts/run_pipeline.py --full --tickers 005930.KS --period 1y --batch-size 50

설정 우선순위:
  config.yaml (기본값) → CLI args (우선)
        """
    )

    # 파이프라인 선택
    pipeline_group = parser.add_mutually_exclusive_group(required=True)
    pipeline_group.add_argument(
        '--full',
        action='store_true',
        help='전체 파이프라인 실행 (Fetch → Save Price → Calculate → Save Indicators)'
    )
    pipeline_group.add_argument(
        '--price',
        action='store_true',
        help='가격 데이터 파이프라인만 (Fetch → Save Price)'
    )
    pipeline_group.add_argument(
        '--indicators',
        action='store_true',
        help='지표 파이프라인만 (Load → Calculate → Save Indicators)'
    )

    # 데이터 설정
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='종목 코드 (예: 005930.KS 000660.KS)'
    )
    parser.add_argument(
        '--period',
        help='기간 (예: 1y, 6m, 3m, 1m)'
    )
    parser.add_argument(
        '--start-date',
        help='시작 날짜 (YYYY-MM-DD, period와 함께 사용 불가)'
    )
    parser.add_argument(
        '--end-date',
        help='종료 날짜 (YYYY-MM-DD)'
    )

    # 지표 설정
    parser.add_argument(
        '--indicators-list',
        dest='indicators',
        nargs='+',
        help='계산할 지표 (예: ma_5 ma_20 ma_200 macd)'
    )

    # 배치 설정
    parser.add_argument(
        '--batch-size',
        type=int,
        help='배치 크기 (기본: 100)'
    )

    # 기타
    parser.add_argument(
        '--update-if-exists',
        dest='update_if_exists',
        action='store_true',
        help='기존 데이터 업데이트'
    )
    parser.add_argument(
        '--no-update',
        dest='update_if_exists',
        action='store_false',
        help='기존 데이터 유지'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='로그 레벨 (기본: INFO)'
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
    """메인 함수"""
    parser = create_parser()
    args = parser.parse_args()

    # 로깅 설정
    logger = setup_logging(args.log_level)

    try:
        # 설정 로드
        config_loader = ConfigLoader(config_file=args.config)
        config = config_loader.get_config(args)

        # 파이프라인 선택 및 실행
        if args.full:
            run_full_pipeline(config, logger)
        elif args.price:
            run_price_pipeline(config, logger)
        elif args.indicators:
            run_indicator_pipeline(config, logger)

        logger.info("\n✅ Pipeline execution completed!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
