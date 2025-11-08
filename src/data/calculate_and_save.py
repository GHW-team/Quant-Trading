# src/data/calculate_and_save.py - 지표 계산 및 DB 저장 파이프라인

import pandas as pd
import logging
from typing import List, Optional
from pathlib import Path

from .indicator_calculator import IndicatorCalculator
from .db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class IndicatorPipeline:
    """지표 계산 및 저장 파이프라인"""

    def __init__(self, db_path: str = "data/database/stocks.db"):
        """
        초기화

        Args:
            db_path: 데이터베이스 경로
        """
        self.db_manager = DatabaseManager(db_path)
        self.calculator = IndicatorCalculator()
        self.db_path = db_path

    def process_indicators(
        self,
        ticker_code: str,
        indicators: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        version: str = "v1.0"
    ) -> int:
        """
        특정 종목의 지표 계산 및 DB 저장

        Args:
            ticker_code: 종목 코드 (예: '005930')
            indicators: 계산할 지표 목록 (None이면 모든 지표)
                       예: ['ma_5', 'ma_20', 'ma_200', 'macd']
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            version: 계산 버전

        Returns:
            저장된 레코드 수
        """
        logger.info(f"Starting indicator pipeline for {ticker_code}")
        logger.info(f"Indicators to calculate: {indicators or 'all'}")

        try:
            # 1. Daily prices 로드
            logger.info(f"Loading daily prices for {ticker_code}")
            price_df = self.db_manager.load_price_data(
                ticker_code=ticker_code,
                start_date=start_date,
                end_date=end_date
            )

            if price_df.empty:
                logger.warning(f"No price data found for {ticker_code}")
                return 0

            logger.info(f"Loaded {len(price_df)} records for {ticker_code}")

            # 2. 지표 계산
            logger.info(f"Calculating indicators for {ticker_code}")
            indicators_df = self.calculator.calculate_indicators(
                df=price_df,
                indicators=indicators
            )

            # 3. DB 저장
            logger.info(f"Saving indicators to database for {ticker_code}")
            saved_count = self.db_manager.save_indicators_bulk(
                ticker_code=ticker_code,
                df=indicators_df,
                version=version
            )

            logger.info(f"✓ Successfully processed {ticker_code}: {saved_count} records saved")
            return saved_count

        except Exception as e:
            logger.error(f"Error processing {ticker_code}: {str(e)}")
            raise

    def process_multiple_tickers(
        self,
        ticker_codes: List[str],
        indicators: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        version: str = "v1.0"
    ) -> dict:
        """
        여러 종목의 지표 계산 및 DB 저장

        Args:
            ticker_codes: 종목 코드 리스트
            indicators: 계산할 지표 목록
            start_date: 시작 날짜
            end_date: 종료 날짜
            version: 계산 버전

        Returns:
            종목별 저장 결과 딕셔너리
        """
        results = {}
        failed = []

        logger.info(f"Processing {len(ticker_codes)} tickers")

        for ticker_code in ticker_codes:
            try:
                saved_count = self.process_indicators(
                    ticker_code=ticker_code,
                    indicators=indicators,
                    start_date=start_date,
                    end_date=end_date,
                    version=version
                )
                results[ticker_code] = {
                    'status': 'success',
                    'saved_count': saved_count
                }
            except Exception as e:
                logger.error(f"Failed to process {ticker_code}: {str(e)}")
                results[ticker_code] = {
                    'status': 'failed',
                    'error': str(e)
                }
                failed.append(ticker_code)

        logger.info(f"✓ Processing complete: {len(ticker_codes) - len(failed)}/{len(ticker_codes)} succeeded")
        if failed:
            logger.warning(f"Failed tickers: {failed}")

        return results

    def get_available_indicators(self) -> List[str]:
        """
        사용 가능한 지표 목록

        Returns:
            지표 목록
        """
        return self.calculator.get_available_indicators()

    def close(self):
        """DB 연결 종료"""
        self.db_manager.close()


# ============== 커맨드라인 사용을 위한 함수 ==============

def calculate_and_save(
    ticker_code: str,
    indicators: Optional[List[str]] = None,
    db_path: str = "data/database/stocks.db"
) -> int:
    """
    단일 종목 지표 계산 및 저장 (간단한 인터페이스)

    Args:
        ticker_code: 종목 코드
        indicators: 계산할 지표 목록
        db_path: 데이터베이스 경로

    Returns:
        저장된 레코드 수
    """
    pipeline = IndicatorPipeline(db_path)
    try:
        return pipeline.process_indicators(
            ticker_code=ticker_code,
            indicators=indicators
        )
    finally:
        pipeline.close()


def calculate_batch(
    ticker_codes: List[str],
    indicators: Optional[List[str]] = None,
    db_path: str = "data/database/stocks.db"
) -> dict:
    """
    여러 종목 지표 계산 및 저장 (배치 처리)

    Args:
        ticker_codes: 종목 코드 리스트
        indicators: 계산할 지표 목록
        db_path: 데이터베이스 경로

    Returns:
        처리 결과
    """
    pipeline = IndicatorPipeline(db_path)
    try:
        return pipeline.process_multiple_tickers(
            ticker_codes=ticker_codes,
            indicators=indicators
        )
    finally:
        pipeline.close()


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 예시: 단일 종목 처리
    # result = calculate_and_save('005930', indicators=['ma_5', 'ma_20', 'ma_200', 'macd'])
    # print(f"Saved {result} records")

    # 예시: 여러 종목 처리
    # results = calculate_batch(['005930', '000660'], indicators=['ma_5', 'ma_20', 'macd'])
    # print(results)

    print("Use calculate_and_save() or calculate_batch() function")
