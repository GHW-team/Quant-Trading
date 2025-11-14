import sqlite3
import csv
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
db_path = root / "data" / "market.db"
csv_path = root / "data" / "price_daily_dump.csv"

def export_price_daily(ticker, start_date, end_date):
    if not db_path.exists():
        raise SystemExit("DB not found")

    where = []
    params = []
    if ticker:
        where.append("ticker = ?")
        params.append(ticker)
    if start_date:
        where.append("date >= ?")
        params.append(start_date)
    if end_date:
        where.append("date <= ?")
        params.append(end_date)

    if where:
        where_sql = " WHERE " + " AND ".join(where)
    else:
        where_sql = ""
    # print(where)
    # print(where_sql)
    # print(params)
    sql = "SELECT * FROM price_daily" + where_sql + " ORDER BY ticker, date;"
    # print(sql)
    # csv_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn, csv_path.open("w", encoding="utf-8", newline="") as f:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(cur)

TICKERS = "005930.KS"
START = "2025-10-20"
END = "2025-10-30"

export_price_daily(TICKERS, START, END)

