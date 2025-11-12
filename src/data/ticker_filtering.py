
#필터링
#모멘텀 판단 지표 
#최근 20일 변동성 상위 30% & 최근 수익률 상위 20% & 최근 거래량 상위 50%
#간단: 고정값으로 하고 백테스트 수동으로 많이하기

import pandas as pd
import yfinance as yf
import numpy as np

#테스트 용 예시 티커
#test_tickers = ['AAPL', 'META', 'AMZN', 'TSLA', 'SPY']
#crawling_ticker파일에서 코스피 : kospi, 나스닥=nasdaq100_tickers, S&P = sp500_tickers를 test_tickers 대신 대입할 것
df_tickers = pd.read_csv('nasdaq100.csv')
nasdaq100_tickers = df_tickers['Symbol'].tolist()

n=20 #최근 n일 설정

data = yf.download(nasdaq100_tickers, period=(f"{n+1}d"))
data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()


selected_tickers = []
for ticker in nasdaq100_tickers:
    df = data[data['Ticker'] == ticker].copy()
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    log_volatility = df['Return'].rolling(window=n).std(ddof=1) * np.sqrt(252) #ddof=1 : 표본 표준편차, log수익률을 쓴 이유는 연율화로 비교하기 위해
    log_volatility =  log_volatility.iloc[-1]
    print ({ticker},log_volatility) #변동성은 주가가 하락해도 양수로 나옴에 유의

    #최근 n일 수익률
    total_return = df['Close'].iloc[-1] / df['Close'].iloc[-n-1] - 1

    #최근 n일 평균 거래량
    avg_volume = df['Volume'].iloc[-n:].mean()

    selected_tickers.append({
        'Ticker': ticker,
        'Log_Volatility': log_volatility,
        'Total_Return': total_return,
        'Avg_Volume': avg_volume
    })

    print(selected_tickers)

filtered_df = pd.DataFrame(selected_tickers)
top_return = filtered_df['Total_Return'].quantile(0.8)
top_volume = filtered_df['Avg_Volume'].quantile(0.5)

final_selection = filtered_df[
    (filtered_df['Total_Return']>=top_return)&
    (filtered_df['Avg_Volume']>=top_volume)
]

final_tickers = final_selection['Ticker'].tolist()
print("✅ 최종 필터링된 티커:", final_tickers)
#방법1. ema_momentum = ema.iloc[-1] - ema.iloc[0]   
#문제점: 20일 안에 추세반전이 나오면 왜곡되지않음??


    
