# src/data/pipeline.py - 4단계 데이터 파이프라인 통합

import os
import logging
from typing import List, Optional, Dict

from .data_fetcher import StockDataFetcher
from .db_manager import DatabaseManager
from .calculate_and_save import IndicatorPipeline

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    4단계 데이터 파이프라인 통합 클래스

    Step 1: yfinance로 OHLCV 데이터 수집
    Step 2: 수집한 데이터를 DB(daily_prices)에 저장
    Step 3: DB의 OHLCV 데이터로 기술적 지표 계산
    Step 4: 계산한 지표를 DB(technical_indicators)에 저장
    """

    def __init__(self, db_path: str = None):
        """
        초기화

        Args:
            db_path: 데이터베이스 경로 (None이면 .env에서 로드)
        """
        # 환경변수에서 기본값 로드
        self.db_path = db_path or os.getenv(
            'DATABASE_PATH', 'data/database/stocks.db'
        )

        # 각 모듈 초기화
        max_workers = int(os.getenv('FETCH_MAX_WORKERS', '5'))
        max_retries = int(os.getenv('FETCH_MAX_RETRIES', '3'))

        self.fetcher = StockDataFetcher(
            max_workers=max_workers,
            max_retries=max_retries
        )
        self.db_manager = DatabaseManager(self.db_path)
        self.indicator_pipeline = IndicatorPipeline(self.db_path)

        logger.info("DataPipeline initialized")

    def run_full_pipeline(
        self,
        ticker_list: Optional[List[str]] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        indicators: Optional[List[str]] = None,
        update_if_exists: Optional[bool] = None
    ) -> dict:
        """
        전체 4단계 자동 실행

        Args:
            ticker_list: 종목 코드 리스트 (None이면 .env에서 로드)
            period: 기간 (1y, 2y 등) (None이면 .env에서 로드)
            interval: 간격 (1d, 1h 등) (None이면 .env에서 로드)
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            indicators: 계산할 지표 리스트 (None이면 .env에서 로드)
            update_if_exists: 기존 데이터 업데이트 여부 (None이면 .env에서 로드)

        Returns:
            각 단계별 결과 딕셔너리
        """
        # 환경변수로부터 기본값 설정
        if ticker_list is None:
            tickers_str = os.getenv('DEFAULT_TICKERS', '')
            ticker_list = [t.strip() for t in tickers_str.split(',') if t.strip()]

        period = period or os.getenv('DEFAULT_PERIOD', '1y')
        interval = interval or os.getenv('DEFAULT_INTERVAL', '1d')

        if indicators is None:
            indicators_str = os.getenv(
                'DEFAULT_INDICATORS', 'ma_5,ma_20,ma_200,macd'
            )
            indicators = [ind.strip() for ind in indicators_str.split(',') if ind.strip()]

        if update_if_exists is None:
            update_if_exists = (
                os.getenv('UPDATE_IF_EXISTS', 'false').lower() == 'true'
            )

        logger.info(f"Starting full pipeline for {len(ticker_list)} tickers")
        logger.info(f"Configuration: period={period}, interval={interval}")
        logger.info(f"Indicators: {indicators}")

        results = {
            'step1_fetch': {},
            'step2_save': {},
            'step3_4_indicators': {},
            'summary': {}
        }

        try:
            # ========== Step 1: Fetch data from yfinance ==========
            logger.info("=" * 60)
            logger.info("Step 1: Fetching data from yfinance")
            logger.info("=" * 60)

            data_dict = self.fetcher.fetch_multiple_stocks(
                ticker_list,
                period=period,
                interval=interval
            )

            results['step1_fetch'] = {
                ticker: len(df) for ticker, df in data_dict.items()
            }
            logger.info(f"✓ Step 1 complete: {len(data_dict)} tickers fetched")

            # ========== Step 2: Save price data to DB ==========
            logger.info("=" * 60)
            logger.info("Step 2: Saving price data to database")
            logger.info("=" * 60)

            for ticker, df in data_dict.items():
                try:
                    saved_count = self.db_manager.save_price_data_bulk(
                        ticker,
                        df,
                        update_if_exists=update_if_exists
                    )
                    results['step2_save'][ticker] = saved_count
                    logger.info(f"✓ Saved {saved_count} records for {ticker}")
                except Exception as e:
                    logger.error(f"✗ Failed to save {ticker}: {e}")
                    results['step2_save'][ticker] = 0

            logger.info(f"✓ Step 2 complete")

            # ========== Step 3 & 4: Calculate and save indicators ==========
            logger.info("=" * 60)
            logger.info("Step 3-4: Calculating and saving indicators")
            logger.info("=" * 60)

            for ticker in ticker_list:
                try:
                    saved_indicators = self.indicator_pipeline.process_indicators(
                        ticker,
                        indicators=indicators,
                        start_date=start_date,
                        end_date=end_date
                    )
                    results['step3_4_indicators'][ticker] = saved_indicators
                    logger.info(f"✓ Saved {saved_indicators} indicator records for {ticker}")

                except Exception as e:
                    logger.error(f"✗ Failed to process indicators for {ticker}: {e}")
                    results['step3_4_indicators'][ticker] = 0

            logger.info(f"✓ Step 3-4 complete")

            # ========== Summary ==========
            logger.info("=" * 60)
            logger.info("Pipeline Summary")
            logger.info("=" * 60)

            total_fetched = sum(results['step1_fetch'].values())
            total_saved_prices = sum(results['step2_save'].values())
            total_saved_indicators = sum(results['step3_4_indicators'].values())

            results['summary'] = {
                'total_fetched_records': total_fetched,
                'total_saved_prices': total_saved_prices,
                'total_saved_indicators': total_saved_indicators,
                'tickers_processed': len(ticker_list),
                'status': 'success'
            }

            logger.info(f"Total fetched records: {total_fetched}")
            logger.info(f"Total saved price records: {total_saved_prices}")
            logger.info(f"Total saved indicator records: {total_saved_indicators}")
            logger.info(f"✓ Pipeline completed successfully")

            return results

        except Exception as e:
            logger.error(f"✗ Pipeline failed: {e}")
            results['summary'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
        finally:
            self.close()

    def run_step_by_step(
        self,
        ticker_list: Optional[List[str]] = None,
        step_config: Optional[Dict] = None
    ) -> dict:
        """
        단계별 선택적 실행

        Args:
            ticker_list: 종목 리스트
            step_config: {
                'step1_fetch': True/False,
                'step2_save': True/False,
                'step3_4_indicators': True/False,
                'period': '1y',
                'interval': '1d',
                'indicators': ['ma_5', 'ma_20'],
                'update_if_exists': False,
                ...
            }

        Returns:
            각 단계별 결과
        """
        if step_config is None:
            step_config = {}

        # 기본 설정
        run_step1 = step_config.get('step1_fetch', True)
        run_step2 = step_config.get('step2_save', True)
        run_step3_4 = step_config.get('step3_4_indicators', True)

        period = step_config.get('period') or os.getenv('DEFAULT_PERIOD', '1y')
        interval = step_config.get('interval') or os.getenv('DEFAULT_INTERVAL', '1d')
        indicators = step_config.get('indicators') or os.getenv(
            'DEFAULT_INDICATORS', 'ma_5,ma_20,ma_200,macd'
        ).split(',')
        update_if_exists = step_config.get('update_if_exists', False)

        if ticker_list is None:
            tickers_str = os.getenv('DEFAULT_TICKERS', '')
            ticker_list = [t.strip() for t in tickers_str.split(',') if t.strip()]

        results = {}
        data_dict = {}

        try:
            # Step 1
            if run_step1:
                logger.info("Running Step 1: Fetch")
                data_dict = self.fetcher.fetch_multiple_stocks(
                    ticker_list, period=period, interval=interval
                )
                results['step1_fetch'] = {
                    ticker: len(df) for ticker, df in data_dict.items()
                }
                logger.info(f"✓ Step 1 complete")
            else:
                logger.info("Skipping Step 1: Fetch")
                results['step1_fetch'] = None

            # Step 2
            if run_step2:
                logger.info("Running Step 2: Save prices")
                if not data_dict:
                    # data_dict가 없으면 DB에서 로드
                    logger.warning("No data from Step 1, skipping Step 2")
                    results['step2_save'] = None
                else:
                    results['step2_save'] = {}
                    for ticker, df in data_dict.items():
                        saved_count = self.db_manager.save_price_data_bulk(
                            ticker, df, update_if_exists=update_if_exists
                        )
                        results['step2_save'][ticker] = saved_count
                    logger.info(f"✓ Step 2 complete")
            else:
                logger.info("Skipping Step 2: Save prices")
                results['step2_save'] = None

            # Step 3 & 4
            if run_step3_4:
                logger.info("Running Step 3-4: Calculate and save indicators")
                results['step3_4_indicators'] = {}
                for ticker in ticker_list:
                    try:
                        saved = self.indicator_pipeline.process_indicators(
                            ticker, indicators=indicators
                        )
                        results['step3_4_indicators'][ticker] = saved
                    except Exception as e:
                        logger.error(f"Failed to process {ticker}: {e}")
                        results['step3_4_indicators'][ticker] = 0
                logger.info(f"✓ Step 3-4 complete")
            else:
                logger.info("Skipping Step 3-4: Calculate and save indicators")
                results['step3_4_indicators'] = None

            return results

        except Exception as e:
            logger.error(f"Step-by-step execution failed: {e}")
            raise
        finally:
            self.close()

    def close(self):
        """모든 연결 종료"""
        try:
            self.db_manager.close()
            logger.info("DataPipeline closed successfully")
        except Exception as e:
            logger.error(f"Error closing pipeline: {e}")


# ==================== Convenience Functions ====================

def run_pipeline(
    ticker_list: Optional[List[str]] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
    indicators: Optional[List[str]] = None,
    db_path: str = None
) -> dict:
    """
    간단한 파이프라인 실행 (DataPipeline 래퍼)

    Args:
        ticker_list: 종목 리스트
        period: 기간
        interval: 간격
        indicators: 지표 리스트
        db_path: DB 경로

    Returns:
        파이프라인 결과
    """
    pipeline = DataPipeline(db_path)
    try:
        return pipeline.run_full_pipeline(
            ticker_list=ticker_list,
            period=period,
            interval=interval,
            indicators=indicators
        )
    finally:
        pipeline.close()
