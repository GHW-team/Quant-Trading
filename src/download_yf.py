import yfinance as yf

def download_yf(tickers, start, end, auto_adjust=True):
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
    )
    return df