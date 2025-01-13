# 기본 패키지
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np


def imputation(df, basis_name, target_name, log_diff=True):
    """
    기준 자산과 대상 자산의 수익률 시계열 데이터프레임을 이용하여 대상 자산의 수익률을 보간하는 함수.
    Parameters:
        df (pd.DataFrame): 기준 자산과 대상 자산의 수익률 시계열 데이터프레임.
        basis_name (str): 기준 자산의 컬럼 이름.
        target_name (str): 대상 자산의 컬럼 이름.
        log_diff (bool): 로그 수익률 여부.(기본값: True)
    Returns:
        pd.DataFrame: 보간된 대상 자산의 수익률이 추가된 데이터프레임.
    """
    dataset = df.copy()
    if log_diff:
        ori_rets = np.log(dataset[[target_name]]).diff()
        rets = np.log(dataset[[target_name, basis_name]].dropna(subset=[basis_name])).diff()
    else:
        ori_rets = dataset[[target_name]].diff()
        rets = dataset[[target_name, basis_name]].dropna(subset=[basis_name]).diff()
    # 공분산과 분산 계산
    cov_matrix = np.cov(rets.dropna()[target_name], rets.dropna()[basis_name])
    cov_stock_market = cov_matrix[0, 1]  # 공분산
    var_market = cov_matrix[1, 1]        # market 분산
    # 베타 계산
    beta = cov_stock_market / var_market
    ori_rets[target_name][ori_rets[target_name].isna()] = rets[basis_name]*beta
    ori_rets = ori_rets.fillna(0)
    if log_diff:
        df_imputed = np.exp(ori_rets.cumsum())
        dataset[target_name] = df_imputed[target_name]
    else:
        df_imputed = ori_rets.cumsum()
        df_imputed = df_imputed + dataset[target_name].mean() - df_imputed.mean()
        dataset[target_name][dataset[target_name].isna()] = df_imputed[target_name][dataset[target_name].isna()] 
    
    print(beta)
    return dataset

def cal_YRR(df, ticker, method ='a', unit = 'daily'):
    """
    연환산 수익률 계산
    parameter:
        df (pd.DataFrame): 시계열 데이터.
        ticker (str): Ticker.
        method (str): 수익률 계산 방법. 'a'는 절대 수익률, 'g'는 기하 수익률. (기본값: 'a')
        unit (str): 연환산 단위. 'daily'는 일 단위, 'monthly'는 월 단위. (기본값: 'daily')
    return:
        float: 연환산 수익률
    """
    if method == 'g':
        total_err = np.log(df[ticker][-1]/df[ticker][0])
    elif method == 'a':
        total_err = (df[ticker][-1]-df[ticker][0])/df[ticker][0]
    if unit == 'daily':
        yrr = (1+total_err)**(250/len(df))-1
    elif unit == 'monthly':
        yrr = (1+total_err)**(12/len(df))-1
    return yrr*1e2


def split_time_series(df, n_days):
    """
    시계열 데이터를 n_days 기간씩 나누는 함수.
    Parameters:
        df (pd.DataFrame): 시계열 데이터.
        n_days (int): 나눌 기간.
    Returns:
        list: n_days 기간씩 나눈 데이터프레임 리스트
    """
    split_days = n_days
    dfs = []
    for i in range(len(df)-split_days+1):
        dfs.append(df[i:split_days+i])
    return dfs


def get_rr_df(df, assets, years=5):
    """
    Rolling Returns 데이터프레임을 계산하는 함수.
    Parameters:
        df (pd.DataFrame): 시계열 데이터.
        assets (list): 자산 목록.
        years (int): 연환산 기간.(기본값: 5)
    Returns:
        pd.DataFrame: Rolling Returns 데이터프레임.
    """
    data = df[assets]
    data = data.resample('M').last()
    dfs = split_time_series(data, 12*years)
    ## calculate Rolling Returns
    _ls = []
    _idx = []
    for _df in dfs:
        _ls.append([cal_YRR(_df, col, method ='a', unit = 'monthly') for col in _df.columns])
        _idx.append(_df.index[0])  
    res = pd.DataFrame(_ls, columns = _df.columns, index=_idx)
    return res


def get_ttr_df(df, assets):
    """
    전고점 회복 시간(TTR) 데이터프레임을 계산하는 함수.
    Parameters:
        df (pd.DataFrame): 시계열 데이터.
        assets (list): 자산 목록.
    Returns:
        pd.DataFrame: TTR 데이터프레임.
    """
    def calculate_ttr_series(drawdown_series):
        ttr_data = []
        ttr_index = []
        in_drawdown = False
        start_date = None

        for date, value in drawdown_series.items():
            if value < 0 and not in_drawdown:
                # Start of a drawdown
                in_drawdown = True
                start_date = date
            elif value == 0 and in_drawdown:
                # Recovery point
                in_drawdown = False
                ttr_data.append((date - start_date).days)  # TTR in days
                ttr_index.append(date)

        return pd.Series(ttr_data, index=ttr_index)

    ttr_dict = {}

    for asset in assets:
        # Calculate drawdown
        drawdown = df[asset] / df[asset].cummax() - 1
        # Calculate TTR
        ttr_dict[asset] = calculate_ttr_series(drawdown)

    return pd.DataFrame(ttr_dict).interpolate(method='time')   