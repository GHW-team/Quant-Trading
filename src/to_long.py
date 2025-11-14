import pandas as pd

def to_long(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        try:
            fields = {"Open","High","Low","Close","Adj Close","Volume"}
            level0 = set(map(str, df.columns.get_level_values(0)))
            # print(level0)
            level1 = set(map(str, df.columns.get_level_values(1)))
            # print(level1)
            if len(fields & level0) > 0:
                df = df.droplevel(1, axis=1)
            elif len(fields & level1) > 0:
                df = df.droplevel(0, axis=1)
            else:
                df = df.droplevel(-1, axis=1)
        except:
            df = df.copy()
        # print(df)

    out = (df.reset_index().assign(ticker=ticker).rename(columns={
              "Date":"date","Open":"open","High":"high","Low":"low",
              "Close":"close","Adj Close":"adj_close","Volume":"volume",}))
    # print(out)

    for c in ["open","high","low","close","adj_close","volume"]:
        if c not in out.columns:
            out[c] = None
        #else:
            #out[c] = out[c].where(out[c].notna(), None)
    # print(out)

    return out[["ticker","date","open","high","low","close","adj_close","volume"]]
