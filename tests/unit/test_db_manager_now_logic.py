# tests/unit/test_db_manager_new_logic.py

import pandas as pd
import pytest
from unittest.mock import patch
from sqlalchemy.exc import SQLAlchemyError

def _selected_keys(stmt):
    # SQLAlchemy Select의 selected column key를 뽑는 헬퍼
    return [c.key for c in stmt.selected_columns]

class TestLoadIndicatorsSelectiveColumns:
    """
    반환 결과 DF 컬럼이 최소화되는지
    → ["date", "rsi", "ma_5"]만 남아야 함 (ticker_code는 결과에서 drop)

    SQL SELECT도 최소 컬럼만 가져오도록 구성되는지
    → 실제로 read_sql에 넘긴 Select 문에서 selected_columns가
    ["ticker_code", "date", "rsi", "ma_5"]인지 확인
    """
    def test_load_indicators_selective_columns_query_and_result(self, db_manager):
        tickers = ["005930.KS"]

        # read_sql이 반환할 DF (ticker_code는 groupby/drop을 위해 필요)
        df_mock = pd.DataFrame({
            "ticker_code": ["005930.KS"],
            "date": [pd.Timestamp("2024-01-01")],
            "rsi": [55.0],
            "ma_5": [100.0],
        })

        with patch("pandas.read_sql") as mock_read_sql:
            mock_read_sql.return_value = df_mock

            result = db_manager.load_indicators(
                ticker_codes=tickers,
                indicators=["rsi", "ma_5"],   # ✅ 선택 로드
                chunk_size=500,
            )

            # 1) 반환 컬럼이 date + 선택 지표만인지
            assert "005930.KS" in result
            assert list(result["005930.KS"].columns) == ["date", "rsi", "ma_5"]

            # 2) SQL SELECT 컬럼도 최소화 됐는지 (ticker_code, date, rsi, ma_5만 선택)
            called_stmt = mock_read_sql.call_args[0][0]
            keys = _selected_keys(called_stmt)
            # label('ticker_code') 때문에 ticker_code 포함
            assert keys == ["ticker_code", "date", "rsi", "ma_5"]


class TestLoadIndicatorsChunkCalls:
    """
    load_indicators()가 티커가 많을 때 chunk_size로 나눠서 
    read_sql을 여러 번 호출하는지.
    """
    def test_load_indicators_calls_read_sql_per_chunk(self, db_manager):
        tickers = ["A", "B", "C", "D", "E"]

        with patch("pandas.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()  # empty => 계속 continue

            db_manager.load_indicators(tickers, indicators=["rsi"], chunk_size=2)

            assert mock_read_sql.call_count == 3  # 2,2,1


class TestLoadPriceDataBulkBehavior:
    """
    db_manager.load_price_data()가 DB에서 한 번에 읽어온 가격 DF를
    ticker_code 기준으로 종목별 dict로 분리해서 반환하는지
    각 종목 DF에서 ticker_code 컬럼을 drop 했는지
    종목별 값이 올바른지 (삼성 close=10, 영풍 close=20 같은 것)
    """
    def test_load_price_data_bulk_splits_by_ticker(self, db_manager):
        tickers = ["005930.KS", "003520.KS"]

        df_mock = pd.DataFrame({
            "ticker_code": ["005930.KS", "003520.KS"],
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [10, 20],
            "volume": [0, 0],
            "adj_close": [10, 20],
        })

        with patch("pandas.read_sql") as mock_read_sql:
            mock_read_sql.return_value = df_mock

            result = db_manager.load_price_data(tickers, chunk_size=500)

            assert "005930.KS" in result and "003520.KS" in result
            assert result["005930.KS"].iloc[0]["close"] == 10
            assert result["003520.KS"].iloc[0]["close"] == 20
            assert "ticker_code" not in result["005930.KS"].columns  # drop 되었는지


class TestLoadPriceDataChunkCalls:
    """
    load_price_data()도 indicator와 동일하게 chunk_size 단위로 read_sql을 나눠 호출하는지.
    tickers 5개, chunk_size=2 → 3번 호출 기대
    """
    def test_load_price_data_calls_read_sql_per_chunk(self, db_manager):
        tickers = ["A", "B", "C", "D", "E"]

        with patch("pandas.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()

            db_manager.load_price_data(tickers, chunk_size=2)

            assert mock_read_sql.call_count == 3


class TestLoadIndicatorsInvalidIndicators:
    """
    load_indicators()에 indicators=["rsi","NOT_EXIST"]처럼 유효하지 않은 지표명이 섞여 들어왔을 때:
    invalid는 무시하고 유효한 것만 로드하는지 → 결과 컬럼이 ["date","rsi"]만 남아야 함
    invalid 무시했다는 로그가 남는지 → caplog로 "Invalid indicators ignored" 메시지 포함 여부 체크
    """
    def test_load_indicators_ignores_invalid(self, db_manager, caplog):
        tickers = ["005930.KS"]

        df_mock = pd.DataFrame({
            "ticker_code": ["005930.KS"],
            "date": [pd.Timestamp("2024-01-01")],
            "rsi": [55.0],
        })

        with patch("pandas.read_sql") as mock_read_sql:
            mock_read_sql.return_value = df_mock

            result = db_manager.load_indicators(
                tickers,
                indicators=["rsi", "NOT_EXIST"],
                chunk_size=500,
            )

        assert list(result["005930.KS"].columns) == ["date", "rsi"]
        # 로그 정책이 있다면(경고 찍게 했다면) 확인
        assert any("Invalid indicators ignored" in rec.message for rec in caplog.records)


from unittest.mock import MagicMock

class TestSaveChunkedTransactions:
    def test_save_price_data_splits_transactions_by_chunk(self, db_manager):
        # 티커 5개, chunk_size=2면 begin()이 3번 호출돼야 함
        df_dict = {
            "A": pd.DataFrame({"date":[pd.Timestamp("2024-01-01")],"open":[1],"high":[1],"low":[1],"close":[1],"volume":[1],"adj_close":[1]}),
            "B": pd.DataFrame({"date":[pd.Timestamp("2024-01-01")],"open":[1],"high":[1],"low":[1],"close":[1],"volume":[1],"adj_close":[1]}),
            "C": pd.DataFrame({"date":[pd.Timestamp("2024-01-01")],"open":[1],"high":[1],"low":[1],"close":[1],"volume":[1],"adj_close":[1]}),
            "D": pd.DataFrame({"date":[pd.Timestamp("2024-01-01")],"open":[1],"high":[1],"low":[1],"close":[1],"volume":[1],"adj_close":[1]}),
            "E": pd.DataFrame({"date":[pd.Timestamp("2024-01-01")],"open":[1],"high":[1],"low":[1],"close":[1],"volume":[1],"adj_close":[1]}),
        }

        # engine.begin()을 context manager로 흉내내기
        cm = MagicMock()
        cm.__enter__.return_value = MagicMock()
        cm.__exit__.return_value = False

        db_manager.engine.begin = MagicMock(return_value=cm)

        db_manager.save_price_data(df_dict, update_if_exists=True, chunk_size=2)

        assert db_manager.engine.begin.call_count == 3


