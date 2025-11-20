#IndicatorCalculator
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import logging 
from src.data.data_fetcher import StockDataFetcher
from typing import List, Optional

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    @staticmethod
    def _calc_sma(df: pd.DataFrame, length: int)-> pd.Series:
        return ta.sma(df['adj_close'],length=length)

    @staticmethod
    def _calc_ema(df: pd.DataFrame, length: int)-> pd.Series:
        return ta.ema(df['adj_close'],length=length)

    @staticmethod
    def _calc_macd(df: pd.DataFrame, 
                   fast:int,
                   slow:int,
                   signal:int)-> pd.Series:
        return ta.macd(df['adj_close'],fast=fast, slow=slow, signal=signal)

    INDICATORS_FUNCTIONS = {
        'ma_5' : lambda df : IndicatorCalculator._calc_ema(df,5),
        'ma_20' : lambda df : IndicatorCalculator._calc_ema(df,20),
        'ma_200' : lambda df : IndicatorCalculator._calc_ema(df,200),
        'macd' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,0],
        'macd_signal' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,1],
        'macd_hist' : lambda df : IndicatorCalculator._calc_macd(df,fast=12,slow=26,signal=9).iloc[:,2],
    }

    def __init__(self):
        self.calculated_indicators = []

    def calculate_indicators(
        self,
        df: pd.DataFrame, 
        indicator_list: Optional[List[str]] = None
        )-> Optional[pd.DataFrame]:
        
        #df validation
        if df.empty:
            logger.error("Empty dataframe provided")
            raise ValueError("Cannot calculate indicators: empty dataframe")

                
        #default setting : All indicators
        if indicator_list == None:
            indicator_list = self.get_available_indicators()
            logger.warning(f"No input indicators!!. Calculate about all indicators: {indicator_list}")
        
        #calculation
        result_df = df.copy()

        errors = {}
        for indicator in indicator_list:
            try:
                calc_func = self.INDICATORS_FUNCTIONS[indicator]
                result_df[indicator] = calc_func(df)
                logger.debug(f"Calculated {indicator}")
            except Exception as e:
                logger.warning(f"Error Calculating {indicator}: {e}")
                errors[indicator] = str(e)
            
        if errors:
            logger.warning(f"Some indicators failed: {errors}")
        
        self.calculated_indicators = indicator_list
        logger.info(f"All requested indicators calculated successfully: {indicator_list}")
        
        return result_df    

    @staticmethod
    def get_available_indicators()-> List[str]:
        return list(IndicatorCalculator.INDICATORS_FUNCTIONS.keys())

    @staticmethod
    def validate_indicators(indicator_list)->None:
        invalid = []
        available = IndicatorCalculator.INDICATORS_FUNCTIONS.keys()

        invalid = [ind for ind in indicator_list if ind not in available]
        if invalid:
            logger.warning(f"Invalid indicator exists: {invalid}")
            raise ValueError(
                f"Invalid indicators: {invalid}\n"
                f"Available: {available}"
            )
    
    @staticmethod
    def add_indicators(
        indicator: str,
        calculation_func
    ):
        if indicator in IndicatorCalculator.INDICATORS_FUNCTIONS:
            logger.warning(f"Indicator {indicator} already exists. Overwritting..")
        
        IndicatorCalculator.INDICATORS_FUNCTIONS[indicator] = calculation_func
        logger.info(f"Added new indicator: {indicator}")

if __name__ == "__main__":
    fetcher = StockDataFetcher()
    stock_df = fetcher.fetch_single_stock('035420.KS')
    stock_df.columns = (stock_df.columns
                        .str.strip()
                        .str.lower()
                        .str.replace(' ','_'))
    print(f"stock columns: {stock_df.columns}")

    indicator_list = ['ma_5','ma_20','macd']
    calculator = IndicatorCalculator()
    calculated_df = calculator.calculate_indicators(stock_df, indicator_list=None)

    print("")
    print(calculated_df.info())
    print("="*50)
    print(calculated_df.head(50))
    print("="*50)