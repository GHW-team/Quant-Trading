import pandas as pd

def add_ma(df: pd.DataFrame, windows=(5, 20)):
    for w in windows:
        df[f"ma_{w}"] = df["close"].rolling(window=w).mean()
    return df

def add_macd(df: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9):
    ema_short = df["close"].ewm(span=short, adjust=False).mean()
    ema_long  = df["close"].ewm(span=long, adjust=False).mean()
    df["macd"] = ema_short - ema_long
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df