import sqlite3

def upsert_price_daily(db_path, rows):
    # db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = ("INSERT INTO price_daily VALUES (?,?,?,?,?,?,?,?) "
           "ON CONFLICT(ticker, date) DO UPDATE SET "
           "open=excluded.open, high=excluded.high, low=excluded.low, "
           "close=excluded.close, adj_close=excluded.adj_close, volume=excluded.volume")
    with sqlite3.connect(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()

def upsert_features_daily(db_path, rows):
    sql = (
        "INSERT INTO features_daily "
        "(ticker, date, ma_5, ma_20, ma_200) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(ticker, date) DO UPDATE SET "
        "ma_5=excluded.ma_5, ma_20=excluded.ma_20, ma_200=excluded.ma_200"
    )
    with sqlite3.connect(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()