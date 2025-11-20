#class IndicatorPipeline 
#지표 계산 및 저장 파이프라인
import logging
from src.data.db_manager import DatabaseManager
from src.data.indicator_calculator import IndicatorCalculator
from typing import List,Optional,Dict

logger = logging.getLogger(__name__)

class IndicatorPipeline:
    """지표 계산 및 저장 파이프라인"""
    def __init__(self,db_path: str = "data/database/stocks.db"):
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path=db_path)
        self.calculator = IndicatorCalculator()

    def process_single_ticker(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        indicator_list: Optional[List[str]],
        version: Optional[str]='v1.0'
    ):
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
        logger.info(f"Starting indicator pipeline for {ticker}")
        logger.info(f"Indicators to calculate: {indicator_list or 'all'}")
        try:
            #load data from db
            logger.info(f"Loading daily prices for {ticker}")
            df = self.db_manager.load_price_data(
                ticker_code=ticker,
                start_date=start_date,
                end_date=end_date,
            ) 
            
            if df.empty:
                logger.warning(f"No price data about {ticker}")
                raise ValueError(f"No price data found for {ticker}")
            
            logger.info(f"Loaded {len(df)} records for {ticker}")
                
            #calculate indicator from data
            logger.info(f"Calculating indicators for {ticker}")
            calculated_df = self.calculator.calculate_indicators(
                df=df,
                indicator_list=indicator_list,
            )
            
            #save indicator to db 
            logger.info(f"Saving indicators to database for {ticker}")
            saved_count = self.db_manager.save_indicators(
                ticker_code=ticker,
                df = calculated_df,
                version= version,
            )
            logger.info(f"{ticker}: Saved indicator {saved_count} records")
            return saved_count

        except Exception as e: 
            logger.error(f"Error processing {ticker}: {e}")
            raise

    def process_multiple_tickers(
        self,
        ticker_list: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        indicator_list: Optional[List[str]],
        version: Optional[str]= 'v1.0'
    )-> Dict[str,dict]:
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
        logger.info(f"Processing {len(ticker_list)} tickers")

        for ticker in ticker_list:
            try:
                saved_count = self.process_single_ticker(
                    ticker = ticker,
                    start_date= start_date,
                    end_date=end_date,
                    indicator_list=indicator_list,
                    version=version,
                    )
                results[ticker] = {
                    'status' : 'success',
                    'saved_count' : saved_count
                }
            except Exception as e:
                logger.error(f'Failed to process {ticker} : {e}')
                results[ticker] = {
                    'status' : 'failed',
                    'error' : str(e)
                }
                failed.append(ticker)
        
        logger.info(f"Processing complete : {len(ticker_list)-len(failed)}/{len(ticker_list)} successed")
        if failed:
            logger.warning(f"failed tickers: {failed}")
        
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

#=========================================================
#================= 커맨드라인 사용을 위한 함수 ==================
#=========================================================


#??: 만약 데이터베이스에 날짜 중간중간 데이터가 비어있는경우, 이 파이프라인을 end,start date None으로 작동시킬경우 어떻게 되는지
#--> 결국 ta 함수가 어떤식으로 작동하느냐에 달림
def calculate_and_save(
    ticker: str,
    indicator_list: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = "data/database/stocks.db",
    version: Optional[str] = 'v1.0',
) -> int:
    """
    단일 종목 지표 계산 및 저장 (간단한 인터페이스)

    Args:
        ticker: 종목 코드
        indicator_list: 계산할 지표 목록
        start_date: 지표 계산 시작 날짜
        end_date: 지표 계산 종료 날짜
        db_path: 데이터베이스 경로
        version: 지표 계산 알고리즘 버전

    Returns:
        저장된 레코드 수
    """
    pipeline = IndicatorPipeline(db_path=db_path)
    try:
        return pipeline.process_single_ticker(
            ticker=ticker,
            indicator_list=indicator_list,
            version=version,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        pipeline.close()

def calculate_batch(
    ticker_list: List[str],
    indicator_list: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = 'data/database/stocks.db',
    version: Optional[str] = 'v1.0',
):
    """
    여러 종목 지표 계산 및 저장 (배치 처리)

    Args:
        ticker_list: 종목 코드 리스트
        indicator_list: 계산할 지표 목록
        start_date: 지표 계산 시작 날짜
        end_date: 지표 계산 종료 날짜
        db_path: 데이터베이스 경로
        version: 지표 계산 알고리즘 버전

    Returns:
        처리 결과
    """
    pipeline = IndicatorPipeline(db_path=db_path)
    try:
        return pipeline.process_multiple_tickers(
            ticker_list=ticker_list,
            indicator_list=indicator_list,
            version=version,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        pipeline.close()
    
        