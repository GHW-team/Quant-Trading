import os
import logging
from typing import List,Optional,Dict
from dotenv import load_dotenv

from src.data.data_fetcher import StockDataFetcher
from src.data.db_manager import DatabaseManager
from src.data.calculate_and_save import IndicatorPipeline

# .env 파일 로드 (아직 로드되지 않았을 경우)
load_dotenv()

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self,db_path: str = None):
        #Load variables from env
        self.db_path = db_path or os.getenv(
            'DATABASE_PATH', 'data/database/stocks.db'
        )

        max_workers = int(os.getenv('FETCH_MAX_WORKERS', '5'))
        max_retries = int(os.getenv('FETCH_MAX_RETRIES', '3'))

        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.fetcher = StockDataFetcher(
            max_retries=max_retries,
            max_workers=max_workers,
        )
        self.indicator_pipeline = IndicatorPipeline(db_path=self.db_path)

        logger.info("DataPipeline initialized")
    
    def run_full_pipeline(
        self,
        ticker_list: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        indicator_list: List[str] = None,
        update_if_exists: Optional[bool] = None,
        period: Optional[str] = "1y",
        interval: Optional[str] = "1d",
        version: Optional[str] = 'v1.0',
    )-> dict:
        #=========Load variables from env============
        if ticker_list is None:
            tickers_str = os.getenv('DEFAULT_TICKERS','')
            ticker_list = [
                t.strip() for t in tickers_str.split(',') if t.strip()
            ]
        
        start_date = start_date or os.getenv('START_DATE')
        end_date = end_date or os.getenv('END_DATE')

        if indicator_list is None:
            indicators_str = os.getenv('DEFAULT_INDICATORS','')
            indicator_list = [
                i.strip() for i in indicators_str.split(',') if i.strip()
            ]

        if update_if_exists is None:
            update_if_exists = (
                os.getenv('UPDATE_IF_EXISTS','false').lower() == 'true'
            )

        logger.info(f"Starting full pipeline for {len(ticker_list)} tickers")
        logger.info(f"Configuration: data from {start_date} to {end_date}")
        logger.info(f"Indicators : {indicator_list}")

        results = {
            'step1_fetch': {},
            'step2_save' : {},
            'step3_4_indicators': {},
            'summary': {},
        }
            
        try:
            #===========1.data fetch=============
            logger.info("="*60)
            logger.info("Step1: Fetching data from yfinance")
            logger.info("="*60)

            df_dict = self.fetcher.fetch_with_date_range(
                ticker_list=ticker_list,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )

            results['step1_fetch']={
                ticker : len(df) for ticker, df in df_dict.items()
            }
            logger.info(f"Step 1 complete: {len(df_dict)} tickers fetched")

            #===========2.data store=============
            logger.info("="*60)
            logger.info("Step2: Saving price data to database")
            logger.info("="*60)

            for ticker, df in df_dict.items():
                try:
                    save_count = self.db_manager.save_price_data(
                        ticker_code=ticker,
                        stock_df=df,
                        update_if_exists=update_if_exists,
                    )
                    results['step2_save'][ticker] = save_count
                except Exception as e:
                    logger.error(f"Failed to save{ticker}: {e}")
                    results['step2_save'][ticker] = 0
            
            logger.info(f"Step 2 completed")
                    
            #===========3.indicator calculate and save=============
            logger.info("="*60)
            logger.info("Step 3-4: Calculating and saving indicators")
            logger.info("="*60)

            for ticker in ticker_list:
                try:
                    saved_indicators = self.indicator_pipeline.process_single_ticker(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        indicator_list=indicator_list,
                        version=version,
                    )
                    results['step3_4_indicators'][ticker] = saved_indicators
                    logger.info(f"Saved {saved_indicators} indicator records for {ticker}")

                except Exception as e:
                    logger.error(f"Failed to process indicators for {ticker}: {e}")
                    results['step3_4_indicators'][ticker] = 0
                    
            logger.info(f"Step 3-4 complete")

            #============Summary=============
            logger.info("="*60)
            logger.info("Pipeline Summary")
            logger.info("="*60)

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
            logger.error(f"Plpeline failed: {e}")
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
            step_config: Optional[Dict] = None,
    ) -> dict:
        if step_config is None: 
            step_config = {}
        
        run_step1_fetch = step_config.get('step1_fetch', False)
        run_step2_save = step_config.get('step2_save', False)
        run_step3_4_indicators = step_config.get('step3_4_indicators', False)

        start_date = step_config.get('start_date') or os.getenv('START_DATE')
        end_date = step_config.get('end_date') or os.getenv('END_DATE')

        indicator_str = step_config.get('indicators') or os.getenv(
            'DEFAULT_INDICATORS', 'ma_5,ma_20,ma_200,macd'
        )
        if isinstance(indicator_str,str):
            indicator_list = [
                i.strip() for i in indicator_str.split(',') if i.strip()
            ]
        else:
            indicator_list = indicator_str

        update_if_exists = step_config.get('update_if_exists', False)

        if ticker_list is None:
            tickers_str = os.getenv('DEFAULT_TICKERS', '')
            ticker_list = [t.strip() for t in tickers_str.split(',') if t.strip()]

        results = {}
        data_dict = {}

        try: 
            # Step 1
            if run_step1_fetch:
                data_dict = self.fetcher.fetch_with_date_range(
                    ticker_list=ticker_list,
                    start_date=start_date,
                    end_date=end_date,
                )
                results['step1_fetch'] = {
                    ticker : len(df) for ticker, df in data_dict.items()
                }
                logger.info('Step 1 complete')
            else:
                logger.info("Skipping Step 1: Fetch")
                results['step1_fetch'] = None
            
            # Step 2
            if run_step2_save:
                logger.info("Running Step 2: Save prices")
                if not data_dict:
                    logger.warning("No data from step 1, skipping Step 2")
                    results['step2_save'] = None
                else:
                    results['step2_save'] = {}
                    for ticker, df in data_dict.items():
                        saved_count = self.db_manager.save_price_data(
                            ticker_code=ticker,
                            stock_df=df,
                            update_if_exists=update_if_exists,
                        )
                        results['step2_save'][ticker] = saved_count
                    logger.info(f"Step 2 Completed")
            else: 
                logger.info("Skipping Step 2 : Save")
                results['step2_save'] = None
            
            # Step 3,4
            if run_step3_4_indicators:
                logger.info("Running Step 3-4: Calculate and save indicators")
                results['step3_4_indicators'] = {}
                for ticker in ticker_list:
                    try:
                        saved = self.indicator_pipeline.process_single_ticker(
                            ticker=ticker,
                            start_date=start_date,
                            end_date=end_date,
                            indicator_list=indicator_list,
                        )
                    except Exception as e:
                        logger.error(f"Failed to process {ticker}: {e}")
                        results['step3_4_indicators'][ticker] = 0
                logger.info(f"Step 3-4 complete")
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
        try:
            self.db_manager.close()
            logger.info("DataPipeline closed successfully")
        except Exception as e:
            logger.error(f"Error closing pipeline: {e}")
            
            
            
# ==================== Convenience Functions ====================

def run_pipeline(
    ticker_list: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    indicator_list: Optional[List[str]] = None,
    db_path: str = None,
) -> dict:
    pipeline = DataPipeline(db_path=db_path)

    return pipeline.run_full_pipeline(
        ticker_list=ticker_list,
        start_date=start_date,
        end_date=end_date,
        indicator_list=indicator_list,
    )