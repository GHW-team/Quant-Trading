import sqlite3

def insert_serity(db_path, ticker, name, exchange):
    params = []
    params.append(ticker)
    if name:
        params.append(name)
    if exchange:
        params.append(exchange)

    params_sql = ", ".join(["?"] * len(params))
    # print(params_sql)
    sql = "INSERT INTO securities VALUES (" + params_sql + ");"
    
    with sqlite3.connect(db_path) as conn:
        conn.execute(sql, params)
        conn.commit()
        