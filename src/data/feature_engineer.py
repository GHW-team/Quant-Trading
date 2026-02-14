import pandas as pd
import numpy as np
import pandas_ta_classic as ta

class FeatureEngineer:
    """ML 모델 학습에 필요한 기술적 지표 피처를 생성하는 클래스"""
    
    def __init__(self):
        pass

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        레짐 분류용 피처를 생성합니다.
        입력 데이터프레임에는 'close', 'adj_close', 'volume', 'high', 'low' 컬럼이 있어야 합니다.
        """
        df = df.copy()
        
        # 1. 이동평균선 계산 (기본 지표)
        df['ma_50'] = ta.sma(df['adj_close'], length=50)
        df['ma_200'] = ta.sma(df['adj_close'], length=200)
        
        # 2. 이동평균 이격도 (MA Spread) - 중요 피처 (x1)
        # (50일 이평 - 200일 이평) / 200일 이평
        # 양수면 정배열(추세), 음수면 역배열
        df['ma_spread'] = (df['ma_50'] - df['ma_200']) / df['ma_200']
        
        # 3. 50일 이평선의 기울기 (MA50 Slope) (x2)
        # (현재 MA50 - 10일전 MA50) / 10일전 MA50
        df['ma_50_slope'] = df['ma_50'].pct_change(10)
        
        # 4. 가격 vs 200일 이평선 괴리율 (x3)
        # 종가 / 200일 이평 - 1
        df['price_vs_ma200'] = (df['adj_close'] / df['ma_200']) - 1
        
        # 5. 기간별 수익률 (Changes) (x4, x5)
        df['ret_20'] = df['adj_close'].pct_change(20) # 1개월
        df['ret_60'] = df['adj_close'].pct_change(60) # 3개월
        
        # 6. 실현 변동성 (Volatility) (x6)
        # 일별 수익률의 20일 표준편차
        df['daily_ret'] = df['adj_close'].pct_change()
        df['vol_20'] = df['daily_ret'].rolling(20).std()
        
        # 7. 볼린저 밴드 폭 (Band Width) (x7)
        # (상단 - 하단) / 중단
        # 밴드 폭이 좁으면 횡보, 넓으면 추세 가능성
        bb = ta.bbands(df['adj_close'], length=20, std=2)
        # 컬럼명 예시: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0 (라이브러리 버전에 따라 다를 수 있음)
        df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
        
        # 8. ADX (추세 강도) (x8)
        adx = ta.adx(df['high'], df['low'], df['adj_close'], length=14)
        df['adx'] = adx['ADX_14']
        
        # NaN(계산 불가 구간) 제거
        df.dropna(inplace=True)
        
        return df

    def get_feature_names(self):
        """학습에 사용할 피처 컬럼명 리스트 반환"""
        return [
            'ma_spread', 'ma_50_slope', 'price_vs_ma200', 
            'ret_20', 'ret_60', 'vol_20', 'bb_width', 'adx'
        ]
