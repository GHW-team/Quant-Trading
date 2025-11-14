import sqlite3
import csv
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
db_path = root / "data" / "market.db"
csv_path = root / "data" / "features_daily_dump.csv"

def export_features_daily(ticker=None, start_date=None, end_date=None):
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

    sql = "SELECT * FROM features_daily" + where_sql + " ORDER BY ticker, date;"

    with sqlite3.connect(db_path) as conn, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(cur)


# 하드코딩 테스트용

TICKER = "005930.KS"
START = "2025-10-20"
END = "2025-10-30"

export_features_daily(TICKER, START, END)
