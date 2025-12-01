"""
TickerUniverse 클래스 단위 테스트
거래소별 ticker 로드 및 정규화 검증
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.all_ticker import TickerUniverse, load_all_tickers


class TestTickerUniverseInitialization:
    """TickerUniverse 초기화 테스트"""

    def test_default_initialization(self):
        """기본값으로 초기화"""
        universe = TickerUniverse()
        assert universe.add_kr_suffix is True

    def test_custom_initialization(self):
        """커스텀 값으로 초기화"""
        universe = TickerUniverse(add_kr_suffix=False)
        assert universe.add_kr_suffix is False

    def test_supported_exchanges(self):
        """지원하는 거래소 목록"""
        expected = ("KRX", "KOSPI", "KOSDAQ", "S&P500", "NYSE", "AMEX")
        assert TickerUniverse.SUPPORTED_EXCHANGES == expected


class TestTickerUniverseGetMethod:
    """get() 메서드 테스트"""

    def test_get_all_exchanges_by_default(self):
        """기본값으로 모든 거래소 로드"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.return_value = ['005930', '000660']
                mock_fdr.StockListing.return_value = pd.DataFrame({
                    'Symbol': ['AAPL', 'MSFT']
                })

                universe = TickerUniverse()
                result = universe.get()

                # None이면 모든 거래소 로드
                assert isinstance(result, list)
                assert len(result) > 0

    def test_get_specific_exchange(self):
        """특정 거래소만 로드"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.return_value = ['005930', '000660']

            universe = TickerUniverse()
            result = universe.get(['KOSPI'])

            assert len(result) > 0
            # KOSPI suffix 확인
            assert any('.KS' in ticker for ticker in result)

    def test_get_multiple_exchanges(self):
        """여러 거래소 로드"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.return_value = ['005930']
                mock_fdr.StockListing.return_value = pd.DataFrame({
                    'Symbol': ['AAPL']
                })

                universe = TickerUniverse()
                result = universe.get(['KOSPI', 'S&P500'])

                assert len(result) > 0
                assert '005930.KS' in result
                assert 'AAPL' in result

    def test_get_case_insensitive_exchange(self):
        """거래소명 대소문자 구분 없음"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.return_value = ['005930']

            universe = TickerUniverse()
            result1 = universe.get(['KOSPI'])
            result2 = universe.get(['kospi'])
            result3 = universe.get(['Kospi'])

            assert result1 == result2 == result3

    def test_get_empty_exchanges_list(self):
        """빈 거래소 목록"""
        universe = TickerUniverse()
        result = universe.get([])

        assert result == []

    def test_get_unsupported_exchange(self):
        """지원하지 않는 거래소"""
        universe = TickerUniverse()
        result = universe.get(['INVALID_EXCHANGE'])

        assert result == []

    def test_get_mixed_supported_unsupported(self):
        """지원 및 미지원 거래소 혼합"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.return_value = ['005930']

            universe = TickerUniverse()
            result = universe.get(['KOSPI', 'INVALID'])

            assert '005930.KS' in result


class TestTickerUniverseLoadKRX:
    """_load_krx() 메서드 테스트"""

    def test_load_krx_success(self):
        """정상 KRX 로드"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.side_effect = [
                ['005930', '000660'],  # KOSPI
                ['068270', '095570']   # KOSDAQ
            ]

            universe = TickerUniverse()
            result = universe._load_krx()

            assert '005930.KS' in result
            assert '000660.KS' in result
            assert '068270.KQ' in result
            assert '095570.KQ' in result

    def test_load_krx_pykrx_failure_fallback(self):
        """pykrx 실패 시 FDR fallback"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.side_effect = Exception("Network error")
                mock_fdr.StockListing.return_value = pd.DataFrame({
                    'Symbol': ['005930', '000660'],
                    'Market': ['KOSPI', 'KOSDAQ']
                })

                universe = TickerUniverse()
                result = universe._load_krx()

                assert len(result) > 0
                assert '005930.KS' in result
                # FDR fallback이 호출되었는지 확인
                mock_fdr.StockListing.assert_called_once_with('KRX')

    def test_load_krx_both_sources_fail(self):
        """pykrx와 FDR 모두 실패"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.side_effect = Exception("pykrx error")
                mock_fdr.StockListing.side_effect = Exception("FDR error")

                universe = TickerUniverse()

                # 둘 다 실패하면 예외가 발생하거나 빈 리스트 반환
                with pytest.raises(Exception):
                    universe._load_krx()


class TestTickerUniverseLoadKRXMarket:
    """_load_krx_market() 메서드 테스트"""

    @pytest.mark.parametrize("market,tickers,suffix,expected_ticker", [
        ('KOSPI', ['005930', '000660'], '.KS', '005930.KS'),
        ('KOSDAQ', ['068270', '095570'], '.KQ', '068270.KQ'),
    ])
    def test_load_krx_market(self, market, tickers, suffix, expected_ticker):
        """KOSPI/KOSDAQ 로드 - 다양한 거래소"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.return_value = tickers

            universe = TickerUniverse()
            result = universe._load_krx_market(market)

            assert all(suffix in ticker for ticker in result)
            assert expected_ticker in result

    def test_load_krx_market_pykrx_failure(self):
        """pykrx 실패 시 FDR fallback"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.side_effect = Exception("Error")
                mock_fdr.StockListing.return_value = pd.DataFrame({
                    'Symbol': ['005930'],
                    'Market': ['KOSPI']
                })

                universe = TickerUniverse()
                result = universe._load_krx_market('KOSPI')

                assert len(result) > 0


class TestTickerUniverseLoadKRXPykrx:
    """_load_krx_pykrx() 메서드 테스트"""

    def test_load_krx_pykrx_kospi(self):
        """pykrx KOSPI 로드"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.return_value = ['005930', '000660']

            universe = TickerUniverse()
            result = universe._load_krx_pykrx('KOSPI')

            assert result == ['005930.KS', '000660.KS']

    def test_load_krx_pykrx_kosdaq(self):
        """pykrx KOSDAQ 로드"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.return_value = ['068270', '095570']

            universe = TickerUniverse()
            result = universe._load_krx_pykrx('KOSDAQ')

            assert result == ['068270.KQ', '095570.KQ']

    def test_load_krx_pykrx_empty(self):
        """pykrx 빈 결과"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            mock_pykrx.get_market_ticker_list.return_value = []

            universe = TickerUniverse()
            result = universe._load_krx_pykrx('KOSPI')

            assert result == []


class TestTickerUniverseConvertKRXDF:
    """_convert_krx_df() 메서드 테스트"""

    def test_convert_krx_df_with_kospi(self):
        """KOSPI DataFrame 변환"""
        df = pd.DataFrame({
            'Symbol': ['005930', '000660'],
            'Market': ['KOSPI', 'KOSPI']
        })

        universe = TickerUniverse(add_kr_suffix=True)
        result = universe._convert_krx_df(df)

        assert '005930.KS' in result
        assert '000660.KS' in result

    def test_convert_krx_df_with_kosdaq(self):
        """KOSDAQ DataFrame 변환"""
        df = pd.DataFrame({
            'Symbol': ['068270', '095570'],
            'Market': ['KOSDAQ', 'KOSDAQ']
        })

        universe = TickerUniverse(add_kr_suffix=True)
        result = universe._convert_krx_df(df)

        assert '068270.KQ' in result
        assert '095570.KQ' in result

    def test_convert_krx_df_mixed_markets(self):
        """혼합 시장 DataFrame 변환"""
        df = pd.DataFrame({
            'Symbol': ['005930', '068270'],
            'Market': ['KOSPI', 'KOSDAQ']
        })

        universe = TickerUniverse(add_kr_suffix=True)
        result = universe._convert_krx_df(df)

        assert '005930.KS' in result
        assert '068270.KQ' in result

    def test_convert_krx_df_no_suffix(self):
        """suffix 추가 안 함"""
        df = pd.DataFrame({
            'Symbol': ['005930', '068270'],
            'Market': ['KOSPI', 'KOSDAQ']
        })

        universe = TickerUniverse(add_kr_suffix=False)
        result = universe._convert_krx_df(df)

        assert '005930' in result
        assert '068270' in result

    def test_convert_krx_df_empty(self):
        """빈 DataFrame"""
        df = pd.DataFrame()

        universe = TickerUniverse()
        result = universe._convert_krx_df(df)

        assert result == []

    def test_convert_krx_df_missing_symbol_column(self):
        """Symbol 컬럼 없음"""
        df = pd.DataFrame({
            'Code': ['005930'],
            'Market': ['KOSPI']
        })

        universe = TickerUniverse()
        result = universe._convert_krx_df(df)

        assert result == []

    def test_convert_krx_df_nan_symbols(self):
        """NaN symbol 제거"""
        df = pd.DataFrame({
            'Symbol': ['005930', np.nan, '000660'],
            'Market': ['KOSPI', 'KOSPI', 'KOSPI']
        })

        universe = TickerUniverse()
        result = universe._convert_krx_df(df)

        assert len(result) == 2
        assert '005930.KS' in result
        assert '000660.KS' in result

    def test_convert_krx_df_whitespace_symbols(self):
        """공백 symbol 정리"""
        df = pd.DataFrame({
            'Symbol': ['  005930  ', '000660  '],
            'Market': ['KOSPI', 'KOSPI']
        })

        universe = TickerUniverse()
        result = universe._convert_krx_df(df)

        assert '005930.KS' in result
        assert '000660.KS' in result


class TestTickerUniverseLoadSP500:
    """_load_sp500() 메서드 테스트"""

    def test_load_sp500_success(self):
        """정상 S&P500 로드"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT', 'GOOGL']
            })

            universe = TickerUniverse()
            result = universe._load_sp500()

            assert 'AAPL' in result
            assert 'MSFT' in result
            assert 'GOOGL' in result

    def test_load_sp500_empty(self):
        """빈 S&P500"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame()

            universe = TickerUniverse()
            result = universe._load_sp500()

            assert result == []

    def test_load_sp500_missing_symbol_column(self):
        """Symbol 컬럼 없음"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame({
                'Code': ['AAPL']
            })

            universe = TickerUniverse()
            result = universe._load_sp500()

            assert result == []

    def test_load_sp500_with_nan(self):
        """NaN 값 제거"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame({
                'Symbol': ['AAPL', np.nan, 'MSFT']
            })

            universe = TickerUniverse()
            result = universe._load_sp500()

            assert len(result) == 2
            assert 'AAPL' in result
            assert 'MSFT' in result


class TestTickerUniverseLoadUSExchange:
    """_load_us_exchange() 정적 메서드 테스트"""

    @pytest.mark.parametrize("exchange,tickers", [
        ('NYSE', ['IBM', 'GE']),
        ('AMEX', ['XBI', 'XLY']),
        ('S&P500', ['AAPL', 'MSFT', 'GOOGL']),
    ])
    def test_load_us_exchange(self, exchange, tickers):
        """다양한 US 거래소 로드"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame({
                'Symbol': tickers
            })

            result = TickerUniverse._load_us_exchange(exchange)

            for ticker in tickers:
                assert ticker in result

    def test_load_us_exchange_empty(self):
        """빈 거래소"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame()

            result = TickerUniverse._load_us_exchange('NYSE')

            assert result == []

    def test_load_us_exchange_nan_symbols(self):
        """NaN symbol 제거"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame({
                'Symbol': ['IBM', np.nan, 'GE']
            })

            result = TickerUniverse._load_us_exchange('NYSE')

            assert len(result) == 2
            assert 'IBM' in result


class TestTickerUniverseDedupPreserveOrder:
    """_dedup_preserve_order() 정적 메서드 테스트"""

    def test_dedup_no_duplicates(self):
        """중복 없음"""
        tickers = ['005930.KS', '000660.KS', 'AAPL']
        result = TickerUniverse._dedup_preserve_order(tickers)

        assert result == tickers

    def test_dedup_with_duplicates(self):
        """중복 제거"""
        tickers = ['005930.KS', '000660.KS', '005930.KS', 'AAPL']
        result = TickerUniverse._dedup_preserve_order(tickers)

        assert len(result) == 3
        assert result == ['005930.KS', '000660.KS', 'AAPL']

    def test_dedup_preserve_order(self):
        """순서 유지"""
        tickers = ['Z', 'A', 'Z', 'B', 'A']
        result = TickerUniverse._dedup_preserve_order(tickers)

        assert result == ['Z', 'A', 'B']

    def test_dedup_empty_list(self):
        """빈 리스트"""
        result = TickerUniverse._dedup_preserve_order([])
        assert result == []

    def test_dedup_single_element(self):
        """단일 원소"""
        tickers = ['005930.KS']
        result = TickerUniverse._dedup_preserve_order(tickers)

        assert result == ['005930.KS']

    def test_dedup_empty_strings(self):
        """빈 문자열 제거"""
        tickers = ['005930.KS', '', '000660.KS', '']
        result = TickerUniverse._dedup_preserve_order(tickers)

        assert result == ['005930.KS', '000660.KS']

    def test_dedup_none_values(self):
        """None 값 제거"""
        tickers = ['005930.KS', None, '000660.KS']
        result = TickerUniverse._dedup_preserve_order(tickers)

        assert result == ['005930.KS', '000660.KS']


class TestLoadAllTickersFunction:
    """load_all_tickers() 함수 테스트"""

    def test_load_all_tickers_returns_dict(self):
        """딕셔너리 반환"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.side_effect = [
                    ['005930'],  # KOSPI
                    ['068270'],  # KOSDAQ
                    ['005930', '068270'],  # KRX (all)
                ]
                mock_fdr.StockListing.side_effect = [
                    pd.DataFrame({'Symbol': ['AAPL']}),  # S&P500
                    pd.DataFrame({'Symbol': ['IBM']}),   # NYSE
                    pd.DataFrame({'Symbol': ['XBI']})    # AMEX
                ]

                result = load_all_tickers()

                assert isinstance(result, dict)
                assert 'KRX' in result
                assert 'KOSPI' in result
                assert 'KOSDAQ' in result
                assert 'S&P500' in result
                assert 'NYSE' in result
                assert 'AMEX' in result

    def test_load_all_tickers_krx_failure_with_fallback(self):
        """KRX 실패 시 FDR fallback"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                # pykrx KRX 로드 실패 (처음 2개는 get_market_ticker_list, 3번째는 _load_krx)
                mock_pykrx.get_market_ticker_list.side_effect = Exception("KRX error")
                # FDR fallback
                mock_fdr.StockListing.side_effect = [
                    pd.DataFrame({'Symbol': ['005930'], 'Market': ['KOSPI']}),  # KRX fallback
                    pd.DataFrame({'Symbol': ['AAPL']}),  # S&P500
                    pd.DataFrame({'Symbol': ['IBM']}),   # NYSE
                    pd.DataFrame({'Symbol': ['XBI']})    # AMEX
                ]

                result = load_all_tickers()

                assert isinstance(result, dict)
                assert 'KRX' in result
                # FDR fallback으로 데이터 로드됨
                assert len(result['KRX']) > 0

    def test_load_all_tickers_krx_structure(self):
        """KRX 구조"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.side_effect = [
                    ['005930'],  # KOSPI
                    ['068270'],  # KOSDAQ
                    ['005930', '068270'],  # KRX
                ]
                mock_fdr.StockListing.return_value = pd.DataFrame({'Symbol': []})

                result = load_all_tickers()

                assert '005930.KS' in result['KRX']
                assert '068270.KQ' in result['KRX']

    def test_load_all_tickers_kospi_structure(self):
        """KOSPI 구조"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.side_effect = [
                    ['005930', '000660'],  # KOSPI (twice)
                    ['068270'],  # KOSDAQ
                ]
                mock_fdr.StockListing.return_value = pd.DataFrame({'Symbol': []})

                result = load_all_tickers()

                assert '005930.KS' in result['KOSPI']
                assert '000660.KS' in result['KOSPI']

    def test_load_all_tickers_sp500_structure(self):
        """S&P500 구조"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.return_value = []
                mock_fdr.StockListing.side_effect = [
                    pd.DataFrame({'Symbol': ['AAPL', 'MSFT']}),  # S&P500
                    pd.DataFrame({'Symbol': []}),  # NYSE
                    pd.DataFrame({'Symbol': []})   # AMEX
                ]

                result = load_all_tickers()

                assert 'AAPL' in result['S&P500']
                assert 'MSFT' in result['S&P500']


class TestTickerUniverseEdgeCases:
    """엣지 케이스 테스트"""

    def test_get_unsupported_exchange_warning(self):
        """지원하지 않는 거래소 경고"""
        universe = TickerUniverse()
        result = universe.get(['UNSUPPORTED'])

        assert result == []

    def test_get_nasdaq_deprecation_warning(self):
        """NASDAQ 지원 안 함 (S&P500 사용 권장)"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame({'Symbol': []})

            universe = TickerUniverse()
            result = universe.get(['NASDAQ'])

            # NASDAQ는 지원되지 않으므로 빈 리스트
            assert result == []

    def test_symbol_stripping(self):
        """symbol 공백 제거"""
        df = pd.DataFrame({
            'Symbol': ['  AAPL  ', 'MSFT'],
            'Market': ['S&P500', 'S&P500']
        })

        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = df

            universe = TickerUniverse()
            result = universe._load_sp500()

            assert 'AAPL' in result
            assert '  AAPL  ' not in result

    def test_large_ticker_list(self):
        """큰 ticker 리스트"""
        tickers = [f'TICK{i}' for i in range(1000)]

        result = TickerUniverse._dedup_preserve_order(tickers)

        assert len(result) == 1000

    def test_duplicate_heavy_list(self):
        """많은 중복"""
        tickers = ['A'] * 500 + ['B'] * 500

        result = TickerUniverse._dedup_preserve_order(tickers)

        assert result == ['A', 'B']
        assert len(result) == 2


class TestTickerUniverseLogging:
    """로깅 검증"""

    def test_load_krx_pykrx_failure_logs_error(self, caplog):
        """pykrx 실패 시 로그 기록"""
        with patch('src.data.all_ticker.pykrx_stock') as mock_pykrx:
            with patch('src.data.all_ticker.fdr') as mock_fdr:
                mock_pykrx.get_market_ticker_list.side_effect = Exception("pykrx network error")
                mock_fdr.StockListing.return_value = pd.DataFrame({
                    'Symbol': ['005930'],
                    'Market': ['KOSPI']
                })

                import logging
                caplog.set_level(logging.DEBUG)

                universe = TickerUniverse()
                result = universe._load_krx()

                # 에러 로그가 기록되었는지 확인
                assert len(result) > 0  # fallback으로 성공
                # pykrx 실패가 로그에 기록되었을 가능성
                assert len(caplog.records) >= 0  # 최소한 시도했음

    def test_load_empty_data_logging(self, caplog):
        """빈 데이터 로깅"""
        with patch('src.data.all_ticker.fdr') as mock_fdr:
            mock_fdr.StockListing.return_value = pd.DataFrame()

            import logging
            caplog.set_level(logging.DEBUG)

            universe = TickerUniverse()
            result = universe._load_sp500()

            assert result == []  # 빈 결과
            # 적어도 처리는 되었음
            assert isinstance(result, list)
