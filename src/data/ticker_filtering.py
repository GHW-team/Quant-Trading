
#필터링
#모멘텀 판단 지표 
#최근 20일 변동성 상위 30% & 최근 수익률 상위 20% & 최근 거래량 상위 50%
#간단: 고정값으로 하고 백테스트 수동으로 많이하기

import pandas as pd
import yfinance as yf
import numpy as np

#테스트 용 예시 티커
test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

