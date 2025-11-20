import sqlite3

def upsert_labels_daily(db_path, rows):
    sql = """
    INSERT INTO labels_daily
      (ticker, date, label, horizon)
    VALUES (?,?,?,?)
    ON CONFLICT(ticker, date, horizon) DO UPDATE SET
      label = excluded.label
    """
    with sqlite3.connect(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()
