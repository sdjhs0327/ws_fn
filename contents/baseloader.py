# 기본 데이터셋을 다운로드하는 클래스

# 기본 패키지
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

## 야후 파이낸스 패키지
import yfinance as yf
## 연준 패키지
from fredapi import Fred
fred = Fred(api_key='cfb4f49f5c1a9396f671b8049d992e56')

## 사전 정의한 함수 모음
import myfuncs as mf

## 필수 데이터셋 다운로드
class BaseLoader:
    def __init__(self):
        self.df_ref = None
        self.df = None
        
    def get_idx_df(self):
        ## 사전에 획득한 금, 다우존스배당지수
        gold = pd.read_csv(f"gold.csv", encoding='utf-8').set_index('Date')[['Close']].rename(columns={'Close':'Gold'})
        gold.index = pd.to_datetime(gold.index)
        div = pd.read_csv(f"div.csv", encoding='utf-8').set_index('Date')[['Close']].rename(columns={'Close':'Div'})
        div.index = pd.to_datetime(div.index)
        ## DFF: Federal Funds Rate, DTB3: 3개월 미국 국채 금리, DGS2: 2년 미국 국채 금리, DGS10: 10년 미국 국채 금리, DGS20: 20년 미국 국채 금리
        ds20 = fred.get_series('DGS20')
        ds10 = fred.get_series('DGS10')
        ds2 = fred.get_series('DGS2').fillna(method='pad')
        tb3 = fred.get_series('DTB3').fillna(method='pad')
        dff = fred.get_series('DFF').fillna(method='pad')
        ## rec = fred.get_series('USREC').fillna(method='pad') ## 월간 경기침체 데이터 ## 별도 수집 데이터로 대체

        ## 기본지수 데이터
        ## ^IXIC : 나스닥종합주가지수, ^GSPC: S&P500지수, ^DJI: 다우존스지수, GC=F : 금선물지수
        tickers = ['^IXIC', '^GSPC', '^DJI', 'GC=F']
        df_ref = yf.download(tickers, ignore_tz = True, auto_adjust=True)
        df_ref = df_ref['Close']

        ## 75년 이전에는 Gold 데이터가 없음
        df_ref = df_ref['1975':]
        df_ref = df_ref.fillna(method = 'pad')
        df_ref['Div'] = div['Div'].copy()
        df_ref['Gold'] = gold['Gold'].copy()
        df_ref['DGS10'] = ds10
        df_ref['DGS20'] = ds20
        df_ref['DGS2'] = ds2
        df_ref['DTB3'] = tb3
        df_ref['DFF'] = dff

        return df_ref
    
    def impute_idx_df(self, df_ref):
        ## Imputation
        df_ref_imp = mf.imputation(df_ref, '^GSPC', '^DJI') ## S&P500를 기준으로 다우존스 지수 보간
        df_ref_imp = mf.imputation(df_ref_imp, '^DJI', 'Div') ## 다우존스 지수를 기준으로 배당지수 보간
        df_ref_imp = mf.imputation(df_ref_imp, 'DGS10', 'DGS20', log_diff=False)
        df_ref_imp = mf.imputation(df_ref_imp, 'DGS20', 'DGS10', log_diff=False)
        df_ref_imp = mf.imputation(df_ref_imp, 'DGS10', 'DGS2', log_diff=False)
        df_ref_imp = mf.imputation(df_ref_imp, 'DGS2', 'DTB3', log_diff=False)
        df_ref_imp = mf.imputation(df_ref_imp, 'GC=F', 'Gold') ## 금 선물 가격을 기준으로 금 가격 보간
        ## column selection
        df_ref_imp = df_ref_imp[['^GSPC', '^DJI', '^IXIC', 'Div', 'Gold', 'DTB3', 'DGS2', 'DGS10', 'DGS20']]
        
        df_ref_scaled = df_ref_imp.copy()
        _cols = ['^GSPC', '^DJI', '^IXIC', 'Div', 'Gold']
        df_ref_scaled[_cols] = df_ref_scaled[_cols]/df_ref_scaled[_cols].iloc[0] * 100
        df_ref_scaled.columns = ['S&P500', 'DowJones', 'NASDAQ', 'Div', 'Gold', 'DTB3', 'DGS2', 'DGS10', 'DGS20']
        df_ref_scaled['Cash'] = (df_ref_scaled['DTB3']/100 * 1/252 + 1).shift(1).fillna(1).cumprod() * 100
        
        return df_ref_imp, df_ref_scaled
    
    def get_target_df(self):
        ## 프로젝트마다 필요한 데이터
        tickers = ['SCHD', 'SPY', 'QQQ', 'QLD', 'UPRO', 'TQQQ', 'TLT', 'IEF', 'SHY', 'IAU', 'SGOV']
        df_ori = yf.download(tickers, ignore_tz = True, auto_adjust=True)
        df_ori = df_ori['Close']
        
        return df_ori
    
    def impute_target_df(self, df_ori, df_ref_scaled):
        ## Imputation
        ## 보간 시계열 데이터
        tickers = ['SCHD', 'SPY', 'QQQ', 'QLD', 'UPRO', 'TQQQ', 'TLT', 'IEF', 'SHY', 'IAU', 'SGOV']
        df_imp = pd.concat([df_ref_scaled, df_ori], axis=1)
        df_imp = mf.imputation(df_imp, 'Div', 'SCHD')
        df_imp = mf.imputation(df_imp, 'S&P500', 'SPY')
        df_imp = mf.imputation(df_imp, 'NASDAQ', 'QQQ')
        df_imp = mf.imputation(df_imp, 'SPY', 'UPRO')
        df_imp = mf.imputation(df_imp, 'QQQ', 'QLD')
        df_imp = mf.imputation(df_imp, 'QQQ', 'TQQQ')
        df_imp = mf.imputation(df_imp, 'DGS20', 'TLT')
        df_imp = mf.imputation(df_imp, 'TLT', 'IEF')
        df_imp = mf.imputation(df_imp, 'IEF', 'SHY')
        df_imp = mf.imputation(df_imp, 'Gold', 'IAU')
        df_imp = mf.imputation(df_imp, 'Cash', 'SGOV')
        df_imp = df_imp[tickers]
        df_imp = df_imp/df_imp.iloc[0] * 100
        
        return df_imp
    
    def loader(self):
        self.df_ref = self.get_idx_df()
        self.df_ref_imp, self.df_ref_scaled = self.impute_idx_df(self.df_ref)
        self.df_ori = self.get_target_df()
        self.df_imp = self.impute_target_df(self.df_ori, self.df_ref_scaled)
        
    def save_df(self):
        self.loader()
        self.df_imp.to_csv('trend_data.csv', encoding='utf-8')
        self.df_ori.to_csv('origin_data.csv', encoding='utf-8')
        