import pandas as pd

# ---------- 1. 구성 종목 크롤링 ----------
#미국 주식
# S&P 500 구성 종목 (Slickcharts 기준)
sp500_url = "https://www.slickcharts.com/sp500"
sp500_tables = pd.read_html(sp500_url)
sp500_df = sp500_tables[0]
sp500_df.to_csv("sp500.csv", index=False, encoding="utf-8-sig")
print("sp500.csv 저장 완료")

# NASDAQ 100 구성 종목
nasdaq100_url = "https://www.slickcharts.com/nasdaq100"
nasdaq100_tables = pd.read_html(nasdaq100_url)
nasdaq100_df = nasdaq100_tables[0]
nasdaq100_df.to_csv("nasdaq100.csv", index=False, encoding="utf-8-sig")
print("nasdaq100.csv 저장 완료")

# ---------- 2. Symbol(티커)만 추출 ----------
# 각 CSV 파일에서 Symbol 컬럼만 리스트로 변환
sp500_tickers = pd.read_csv("sp500.csv")["Symbol"].tolist()
nasdaq100_tickers = pd.read_csv("nasdaq100.csv")["Symbol"].tolist()

print("티커 리스트 추출 완료")
# ---------- 3. 결과 확인 ----------
print(f"\nS&P 500 티커 예시 ({len(sp500_tickers)}개):")
print(sp500_tickers[:10])

print(f"\nNASDAQ 100 티커 예시 ({len(nasdaq100_tickers)}개):")
print(nasdaq100_tickers[:10])

#미국 채권 금리
interest_rates = ["^IRX", "^TNX", "TYX" ]


#원자재
commodities_futures = ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]



#한국 주식
# 1. KOSPI 전체 종목 리스트 가져오기
# Excel 파일 읽기 (.xls)
file_path = r"C:\Users\najdorf\Downloads\korea_stock_list.xls" #파일 경로 수정할 것. , 다운로드 링크: https://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage

kospi_all = pd.read_html(file_path)

# 상장구분이 KOSPI인 종목만 선택
kospi_all = kospi_all[0]
kospi_all.rename(columns=lambda x: x.strip(), inplace=True)
# 1. '유가' 항목만 추출 (즉, KOSPI 종목), 코스닥을 선택하려면 '유가' 대신 '코스닥'
kospi = kospi_all[kospi_all['시장구분'] == '유가'].copy()

# 2. 종목코드를 문자열로 변환하고, 6자리 맞춘 뒤 ".KS" 붙이기
kospi['티커'] = kospi['종목코드'].astype(str).str.zfill(6) + '.KS'

# 3. 결과 일부 확인
#print(kospi[['회사명', '티커']])