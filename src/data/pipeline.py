import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

from src.data.all_ticker import TickerUniverse
from src.data.data_fetcher import StockDataFetcher
from src.data.db_manager import DatabaseManager
from src.data.indicator_calculator import IndicatorCalculator


logger = logging.getLogger(__name__)


class DataPipeline:
    """
    데이터 수집 → DB 저장 → 지표 계산을 수행하는 파이프라인.

    설정은 반드시 config.yaml에서 읽고, 추가 인자로 덮어쓴다.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_workers: Optional[int] = None,
        max_retries: Optional[int] = None,
        config_path: str = "config.yaml",
    ) -> None:
        self.config = self._load_config(config_path)

        data_cfg = self.config.get("data", {})
        fetcher_cfg = self.config.get("fetcher", {})
        batch_cfg = self.config.get("batch", {})

        self.db_path = db_path or data_cfg.get("database_path", "data/database/stocks.db")

        max_workers = max_workers or fetcher_cfg.get("max_workers", 5)
        max_retries = max_retries or fetcher_cfg.get("max_retries", 3)
        per_request_delay_sec = fetcher_cfg.get("per_request_delay_sec", 1.5)

        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.fetcher = StockDataFetcher(
            max_workers=max_workers,
            max_retries=max_retries,
            per_request_delay_sec=per_request_delay_sec,
        )
        self.calculator = IndicatorCalculator()
        self.ticker_provider = TickerUniverse()
        self.batch_size_default = batch_cfg.get("size", 100)

        logger.info("DataPipeline initialized (config=%s)", config_path)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_config(path: str) -> dict:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_tickers(self, ticker_list: Optional[List[str]]) -> List[str]:
        if ticker_list:
            return ticker_list

        data_cfg = self.config.get("data", {})
        if data_cfg.get("tickers"):
            return list(data_cfg["tickers"])

        exchanges = data_cfg.get("exchanges")
        return self.ticker_provider.get(exchanges)

    def _resolve_indicator_list(self, indicator_list: Optional[List[str]]) -> List[str]:
        if indicator_list:
            return indicator_list
        ind_cfg = self.config.get("indicators", {})
        return list(ind_cfg.get("list", []))

    @staticmethod
    def _convert_period_to_dates(period: str) -> Tuple[str, str]:
        """
        Period 문자열(1y/6m/3m/1m/1w/30d 등)을 (start_date, end_date)로 변환한다.
        """
        today = datetime.now()

        if not period or len(period) < 2:
            raise ValueError(f"Invalid period format: {period}")

        amount = int(period[:-1])
        unit = period[-1].lower()

        if unit == "y":
            start_dt = today.replace(year=today.year - amount)
        elif unit == "m":
            month = today.month - amount
            year = today.year
            while month <= 0:
                month += 12
                year -= 1
            start_dt = today.replace(year=year, month=month)
        elif unit == "d":
            start_dt = today - timedelta(days=amount)
        elif unit == "w":
            start_dt = today - timedelta(weeks=amount)
        else:
            raise ValueError(f"Unknown period unit: {unit}. Use 'y', 'm', 'w', or 'd'")

        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        logger.debug("Converted period '%s' to %s ~ %s", period, start_date, end_date)
        return start_date, end_date

    def _calculate_extended_start_date(
        self, start_date: Optional[str], indicator_list: List[str]
    ) -> Optional[str]:
        """
        지표 계산용 룩백을 고려한 확장 start_date 계산.
        """
        if not start_date:
            return None

        lookback_days = IndicatorCalculator.get_lookback_days(indicator_list)
        safe_lookback = int(lookback_days * 1.6) + 10

        dt_start = pd.to_datetime(start_date)
        dt_extended_start = dt_start - timedelta(days=safe_lookback)

        extended_date_str = dt_extended_start.strftime("%Y-%m-%d")
        logger.info(
            "Date extended for lookback: %s -> %s (-%d days, base lookback: %d days)",
            start_date,
            extended_date_str,
            safe_lookback,
            lookback_days,
        )
        return extended_date_str

    def _validate_date_args(
        self, start_date: Optional[str], end_date: Optional[str], period: Optional[str]
    ) -> None:
        if (start_date or end_date) and period:
            raise ValueError("start/end와 period 중 하나만 지정하세요.")
        if (start_date is None) ^ (end_date is None):
            raise ValueError("start_date와 end_date는 함께 지정하거나 둘 다 생략해야 합니다.")

    # ------------------------------------------------------------------ #
    # main pipelines
    # ------------------------------------------------------------------ #
    def run_full_pipeline(
        self,
        ticker_list: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        actions: bool = False,
        update_if_exists: Optional[bool] = None,
        indicator_list: Optional[List[str]] = None,
        version: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Dict[str, int]]:
        data_cfg = self.config.get("data", {})
        ind_cfg = self.config.get("indicators", {})

        ticker_list = self._resolve_tickers(ticker_list)
        indicator_list = self._resolve_indicator_list(indicator_list)

        interval = interval or data_cfg.get("interval", "1d")
        update_if_exists = (
            data_cfg.get("update_if_exists", True)
            if update_if_exists is None
            else update_if_exists
        )
        version = version or ind_cfg.get("version", "v1.0")
        batch_size = batch_size or self.batch_size_default

        if period is None:
            period = data_cfg.get("period")
        start_date = start_date or data_cfg.get("start_date")
        end_date = end_date or data_cfg.get("end_date")

        self._validate_date_args(start_date, end_date, period)

        if period:
            start_date, end_date = self._convert_period_to_dates(period)

        original_start_date = start_date
        start_date = self._calculate_extended_start_date(start_date, indicator_list)

        results: Dict[str, Dict[str, int]] = {
            "step1_fetch": {},
            "step2_save_price": {},
            "step3_calculate": {},
            "step4_save_indicator": {},
            "summary": {},
        }

        total_tickers = len(ticker_list)
        num_batches = (total_tickers + batch_size - 1) // batch_size
        logger.info("Processing %d tickers in %d batches (size=%d)", total_tickers, num_batches, batch_size)

        for i in range(0, total_tickers, batch_size):
            batch_tickers = ticker_list[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info("Batch %d/%d: %d tickers", batch_num, num_batches, len(batch_tickers))

            # Step 1: fetch
            try:
                batch_df_dict = self.fetcher.fetch_multiple_by_date(
                    ticker_list=batch_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    actions=actions,
                )
                results["step1_fetch"].update({t: len(df) for t, df in batch_df_dict.items()})
            except Exception as exc:
                logger.error("Failed to fetch batch %d: %s", batch_num, exc)
                for t in batch_tickers:
                    results["step1_fetch"][t] = 0
                continue

            # Step 2: save prices (filter to original_start_date)
            filtered_price_dict: Dict[str, pd.DataFrame] = {}
            for t, df in batch_df_dict.items():
                if original_start_date:
                    filtered_df = df[df["date"] >= original_start_date].copy()
                else:
                    filtered_df = df.copy()
                if not filtered_df.empty:
                    filtered_price_dict[t] = filtered_df

            if filtered_price_dict:
                save_prices = self.db_manager.save_price_data(
                    df_dict=filtered_price_dict, update_if_exists=update_if_exists
                )
                results["step2_save_price"].update(save_prices)

            # Step 3: calculate indicators
            batch_calculated: Dict[str, pd.DataFrame] = {}
            for t, df in batch_df_dict.items():
                try:
                    calc_df = self.calculator.calculate_indicators(df=df, indicator_list=indicator_list)
                    batch_calculated[t] = calc_df
                    results["step3_calculate"][t] = len(calc_df)
                except Exception as exc:
                    logger.error("Failed to calculate indicators for %s: %s", t, exc)
                    results["step3_calculate"][t] = 0

            # Step 4: save indicators (filter to original_start_date)
            filtered_indicator_dict: Dict[str, pd.DataFrame] = {}
            for t, df in batch_calculated.items():
                if original_start_date:
                    filtered_df = df[df["date"] >= original_start_date].copy()
                else:
                    filtered_df = df.copy()
                if not filtered_df.empty:
                    filtered_indicator_dict[t] = filtered_df

            if filtered_indicator_dict:
                save_ind = self.db_manager.save_indicators(
                    indicator_data_dict=filtered_indicator_dict, version=version
                )
                results["step4_save_indicator"].update(save_ind)

        results["summary"] = {
            "total_fetched_records": sum(results["step1_fetch"].values()),
            "total_saved_prices": sum(results["step2_save_price"].values()),
            "total_calculated_indicators": sum(results["step3_calculate"].values()),
            "total_saved_indicators": sum(results["step4_save_indicator"].values()),
            "tickers_processed": len(ticker_list),
            "batches_processed": num_batches,
            "status": "success",
        }

        logger.info("Pipeline completed successfully: %s", results["summary"])
        return results

    def run_price_pipeline(
        self,
        ticker_list: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        actions: bool = False,
        update_if_exists: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Dict[str, int]]:
        data_cfg = self.config.get("data", {})

        ticker_list = self._resolve_tickers(ticker_list)
        interval = interval or data_cfg.get("interval", "1d")
        update_if_exists = (
            data_cfg.get("update_if_exists", True)
            if update_if_exists is None
            else update_if_exists
        )
        batch_size = batch_size or self.batch_size_default

        if period is None:
            period = data_cfg.get("period")
        start_date = start_date or data_cfg.get("start_date")
        end_date = end_date or data_cfg.get("end_date")

        self._validate_date_args(start_date, end_date, period)
        if period:
            start_date, end_date = self._convert_period_to_dates(period)

        results: Dict[str, Dict[str, int]] = {
            "step1_fetch": {},
            "step2_save_price": {},
            "summary": {},
        }

        total_tickers = len(ticker_list)
        num_batches = (total_tickers + batch_size - 1) // batch_size

        for i in range(0, total_tickers, batch_size):
            batch_tickers = ticker_list[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info("Price batch %d/%d: %d tickers", batch_num, num_batches, len(batch_tickers))

            try:
                batch_df_dict = self.fetcher.fetch_multiple_by_date(
                    ticker_list=batch_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    actions=actions,
                )
                results["step1_fetch"].update({t: len(df) for t, df in batch_df_dict.items()})
            except Exception as exc:
                logger.error("Failed to fetch batch %d: %s", batch_num, exc)
                for t in batch_tickers:
                    results["step1_fetch"][t] = 0
                continue

            try:
                save_results = self.db_manager.save_price_data(
                    df_dict=batch_df_dict, update_if_exists=update_if_exists
                )
                results["step2_save_price"].update(save_results)
            except Exception as exc:
                logger.error("Failed to save batch %d: %s", batch_num, exc)
                for t in batch_df_dict.keys():
                    results["step2_save_price"][t] = 0
                continue

        results["summary"] = {
            "total_fetched_records": sum(results["step1_fetch"].values()),
            "total_saved_prices": sum(results["step2_save_price"].values()),
            "tickers_processed": len(ticker_list),
            "batches_processed": num_batches,
            "status": "success",
        }
        logger.info("Price pipeline completed: %s", results["summary"])
        return results

    def run_indicator_pipeline(
        self,
        ticker_list: Optional[List[str]] = None,
        indicator_list: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        version: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Dict[str, int]]:
        data_cfg = self.config.get("data", {})
        ind_cfg = self.config.get("indicators", {})

        ticker_list = self._resolve_tickers(ticker_list)
        indicator_list = self._resolve_indicator_list(indicator_list)
        version = version or ind_cfg.get("version", "v1.0")
        batch_size = batch_size or self.batch_size_default

        start_date = start_date or data_cfg.get("start_date")
        end_date = end_date or data_cfg.get("end_date")
        self._validate_date_args(start_date, end_date, period=None)

        original_start_date = start_date
        start_date = self._calculate_extended_start_date(start_date, indicator_list)

        results: Dict[str, Dict[str, int]] = {
            "step1_load_data": {},
            "step2_calculate": {},
            "step3_save_indicator": {},
            "summary": {},
        }

        total_tickers = len(ticker_list)
        num_batches = (total_tickers + batch_size - 1) // batch_size

        for i in range(0, total_tickers, batch_size):
            batch_tickers = ticker_list[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info("Indicator batch %d/%d: %d tickers", batch_num, num_batches, len(batch_tickers))

            # Step 1: load price data
            try:
                batch_price_dict = self.db_manager.load_price_data(
                    ticker_codes=batch_tickers,
                    start_date=start_date,
                    end_date=end_date,
                )
                for t in batch_tickers:
                    df = batch_price_dict.get(t, pd.DataFrame())
                    results["step1_load_data"][t] = len(df)
            except Exception as exc:
                logger.error("Failed to load data for batch %d: %s", batch_num, exc)
                for t in batch_tickers:
                    results["step1_load_data"][t] = 0
                continue

            # Step 2: calculate indicators
            batch_calculated: Dict[str, pd.DataFrame] = {}
            for t, df in batch_price_dict.items():
                if df.empty:
                    continue
                try:
                    calc_df = self.calculator.calculate_indicators(df=df, indicator_list=indicator_list)
                    batch_calculated[t] = calc_df
                    results["step2_calculate"][t] = len(calc_df)
                except Exception as exc:
                    logger.error("Failed to calculate indicators for %s: %s", t, exc)
                    results["step2_calculate"][t] = 0

            # Step 3: save indicators (filter to original_start_date)
            filtered_indicator_dict: Dict[str, pd.DataFrame] = {}
            for t, df in batch_calculated.items():
                if original_start_date:
                    filtered_df = df[df["date"] >= original_start_date].copy()
                else:
                    filtered_df = df.copy()
                if not filtered_df.empty:
                    filtered_indicator_dict[t] = filtered_df

            if filtered_indicator_dict:
                save_results = self.db_manager.save_indicators(
                    indicator_data_dict=filtered_indicator_dict,
                    version=version,
                )
                results["step3_save_indicator"].update(save_results)

        results["summary"] = {
            "total_loaded_price_data": sum(results["step1_load_data"].values()),
            "total_calculated_indicators": sum(results["step2_calculate"].values()),
            "total_saved_indicators": sum(results["step3_save_indicator"].values()),
            "tickers_processed": len(ticker_list),
            "batches_processed": num_batches,
            "status": "success",
        }
        logger.info("Indicator pipeline completed: %s", results["summary"])
        return results

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self.db_manager.close()
            logger.info("DataPipeline closed successfully")
        except Exception as exc:
            logger.error("Error closing pipeline: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== Convenience Functions ==================== #
def run_pipeline(
    ticker_list: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    indicator_list: Optional[List[str]] = None,
    db_path: str = None,
) -> Dict[str, Dict[str, int]]:
    pipeline = DataPipeline(db_path=db_path)
    return pipeline.run_full_pipeline(
        ticker_list=ticker_list,
        start_date=start_date,
        end_date=end_date,
        indicator_list=indicator_list,
    )
