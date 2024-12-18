import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def simulate_portfolio_assets(asset_rets_df, weights_list, initial_investment=100):
    """
    n개의 자산에 대한 투자 비중에 따른 포트폴리오 시뮬레이션.

    Parameters:
        asset_returns (list of pd.Series): 각 자산의 수익률 시계열 데이터프레임.
        weights_list (list of list): 자산별 비중 리스트. 각 리스트의 합은 1이어야 함.
        initial_investment (float): 초기 투자 금액 (기본값: 100).

    Returns:
        pd.DataFrame: 각 비중 조합에 대한 포트폴리오 가치 시계열 데이터프레임.
    """
    asset_rets_list = [asset_rets_df[col] for col in asset_rets_df.columns]
    portfolio_values = {}
    for weights in weights_list:
        _weights = [i/sum(weights) for i in weights]
        # 포트폴리오 수익률 계산 (각 자산의 비중 곱)
        portfolio_returns = sum(w * r for w, r in zip(_weights, asset_rets_list))
        # 초기 투자 금액 기준 포트폴리오 가치 시계열 계산
        portfolio_cum_value = initial_investment * (1 + portfolio_returns).cumprod()
        portfolio_values[str(weights)] = portfolio_cum_value

    # 데이터프레임으로 변환
    portfolio_df = pd.DataFrame(portfolio_values)
    return portfolio_df

def split_time_series(df, n_days):
    split_days = n_days
    dfs = []
    for i in range(len(df)-split_days+1):
        dfs.append(df[i:split_days+i])
    return dfs

## beta를 이용한 imputation
def imputation(df, basis_name, target_name):
    dataset = df.copy()
    ori_rets = np.log(dataset[[target_name]]).diff()
    rets = np.log(dataset[[target_name, basis_name]].dropna(subset=[basis_name])).diff()
    # 공분산과 분산 계산
    cov_matrix = np.cov(rets.dropna()[target_name], rets.dropna()[basis_name])
    cov_stock_market = cov_matrix[0, 1]  # 공분산
    var_market = cov_matrix[1, 1]        # market 분산
    # 베타 계산
    beta = cov_stock_market / var_market
    ori_rets[target_name][ori_rets[target_name].isna()] = rets[basis_name]*beta
    ori_rets = ori_rets.fillna(0)
    df_imputed = np.exp(ori_rets.cumsum())
    dataset[target_name] = df_imputed[target_name]
    print(beta)
    return dataset

## 주요 이평선 ## 피보나치
def get_signals(df, target):
    dataset = df[[target]].copy()
    dataset['M20'] = dataset[target].rolling(20).mean()
    dataset['M60'] = dataset[target].rolling(60).mean()
    dataset['M120'] = dataset[target].rolling(120).mean()
    dataset['M200'] = dataset[target].rolling(200).mean()
    dataset['M500'] = dataset[target].rolling(500).mean()

    ## dataset['D10'] = dataset[target].max() * (1-.1)
    ## dataset['D20'] = dataset[target].max() * (1-.2)
    ## dataset['D30'] = dataset[target].max() * (1-.3)
    dataset['F-4'] = dataset[target].max() * (1-0.6180339887498949)
    dataset['F-3'] = dataset[target].max() * (1-0.38196601125010515)
    dataset['F-2'] = dataset[target].max() * (1-0.2360679774997897)
    dataset['F-1'] = dataset[target].max() * (1-0.14589803375031546)
    
    return dataset


def calc_agr(df):
    ## AGR 계산
    cols = df.columns
    _yrr = []
    for col in cols:
        col_srs = df[col].copy()
        _years = col_srs.index.year.unique()
        _ls = []
        for _year in _years:
            _srs = col_srs[col_srs.index.year == _year]
            _val = _srs[-1]/_srs[0]-1
            _ls.append(_val)
        _yrr.append(_ls)
        
    agr_df = pd.DataFrame(_yrr, index=cols, columns=df.index.year.unique()).T
    return agr_df


## 그래프 x축 %로 표시
def percent_formatter(x, pos):
    return f"{int(x)}%"