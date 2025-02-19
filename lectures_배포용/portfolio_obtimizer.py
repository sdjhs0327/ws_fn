import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import datetime as dt
from quant_functions import anal_funcs


class PortfolioAllocator(object):
    def __init__(self) -> None:
        print("cal_optimal => eg. process, obtimal, min_risk = cal_optimal(df)")
        print("eg. process, obtimal, min_risk = cal_optimal(df)")
        pass
    
    # target_sum을 만족하는 0 이상 k개 숫자의 모든 순열을 찾는 함수
    def permutations_k_sum(self, n, k, target_sum, prefix=[]):
        # 종료조건: k개의 숫자를 모두 선택했을 때, 합이 target_sum이면 prefix를 반환
        if k == 0:
            if target_sum == 0:
                return [prefix]  # prefix 리스트가 유효한 조합이므로 반환
            return []  # target_sum이 0이 아니라면 빈 리스트 반환
        if n < 0:
            return []  # 탐색 시작점이 음수인 경우 빈 리스트 반환
        if target_sum < 0:
            return []  # target_sum이 음수일 경우 빈 리스트 반환

        permutations = []  # 가능한 순열을 저장할 리스트

        # 숫자 i를 선택해 재귀적으로 다음 숫자들을 선택
        for i in range(n+1):
            # n부터 i를 선택하여 target_sum-i를 새로운 target으로 설정하고, prefix에 선택된 i를 추가
            permutations += self.permutations_k_sum(n, k-1, target_sum - i, prefix + [i])
        
        # 최종적으로 가능한 모든 순열을 반환
        return permutations
    
    ## 리벨런싱을 반영한 포트폴리오 구성 시뮬레이션 함수
    def cal_rebalancing(self, df, ratio=[0.5, 0.5], rebalancing=None, unit=None):
        # 주어진 데이터프레임(df)을 복사하여 작업을 수행
        new_df = df.copy()
        df_dict = []  # 리벨런싱 시점별 데이터프레임을 저장할 리스트
        years = sorted(new_df.index.year.unique())  # 유일한 연도 리스트
        months = sorted(new_df.index.month.unique())  # 유일한 월 리스트

        # 월 단위 리벨런싱을 수행할 때 일자 조정 (월간 집계로 인한 계산 오류 방지)
        if (rebalancing == 'm') & (unit == 'monthly'):
            temp_df = new_df.copy()
            temp_df.index = temp_df.index - dt.timedelta(days=1)  # 인덱스를 하루 전으로 조정
            new_df = pd.concat([new_df, temp_df])  # 수정된 데이터프레임을 기존 데이터와 결합
            new_df = new_df.sort_index()  # 인덱스를 기준으로 정렬
            new_df = new_df.shift(-1).dropna()  # 데이터 이동 및 결측값 제거

        # 리벨런싱 주기에 따라 데이터프레임을 분할
        if rebalancing == 'm':  # 월별 리벨런싱
            for year in years:
                for month in months:
                    temp = new_df[(new_df.index.year == year) & (new_df.index.month == month)]
                    if len(temp) > 0:
                        df_dict.append(temp)  # 각 월에 해당하는 데이터프레임 추가

        elif rebalancing == 'y':  # 연별 리벨런싱
            for year in years:
                temp = new_df[(new_df.index.year == year)]
                if len(temp) > 0:
                    df_dict.append(temp)  # 각 연도에 해당하는 데이터프레임 추가
        else:
            df_dict.append(new_df)  # 리벨런싱이 없으면 전체 데이터프레임을 추가

        # 리벨런싱을 적용하여 전략 가치 계산
        prev_val = 1  # 이전 리벨런싱의 마지막 가치를 저장하는 변수
        temp_series = pd.Series()  # 전략의 누적 가치를 저장할 시리즈
        for i in range(len(df_dict)):
            # 각 리벨런싱 구간의 첫날을 기준으로 비율을 조정하여 누적 수익률 계산
            temp = df_dict[i] / np.array(df_dict[i])[0]
            temp_AWP = (temp * ratio).sum(axis=1) * prev_val  # 비중을 적용한 가중 누적 수익률 계산
            prev_val = temp_AWP.iloc[-1]  # 다음 구간의 초기 가치를 현재 구간의 마지막 값으로 설정
            temp_series = pd.concat([temp_series, temp_AWP])  # 누적 시리즈에 추가

        # 전략의 누적 가치를 'strategy' 열에 추가
        new_df['strategy'] = temp_series

        # 월 단위로 계산한 데이터프레임을 원래 형태로 되돌림
        if (rebalancing == 'm') & (unit == 'monthly'):
            new_df['temp_y'] = new_df.index.year  # 임시 열로 연도 추가
            new_df['temp_m'] = new_df.index.month  # 임시 열로 월 추가
            new_df = new_df.drop_duplicates(['temp_y', 'temp_m'])  # 연도와 월 중복 제거
            new_df = new_df.drop(columns=['temp_y', 'temp_m'])  # 임시 열 제거
            new_df.index = df.index  # 인덱스를 원래 인덱스로 되돌림

        # 리벨런싱을 반영한 최종 데이터프레임 반환
        return new_df
    
    def is_efficient(self, df):
        # 효율적인 포트폴리오를 담을 리스트
        efficient = []
        
        for i, row in df.iterrows():
            dominated = False  # 현재 로우가 지배되는지 여부
            
            for j, other_row in df.iterrows():
                if i == j:
                    continue  # 자기 자신은 비교하지 않음
                
                # 다른 로우가 현재 로우를 지배하는지 판단
                if (
                    other_row['Risk'] <= row['Risk'] and
                    other_row['Return'] >= row['Return'] and
                    (other_row['Risk'] < row['Risk'] or other_row['Return'] > row['Return'])
                ):
                    dominated = True
                    break  # 하나라도 지배하는 포트폴리오가 있으면 중단
            if not dominated:
                efficient.append(i)  # 지배되지 않는 포트폴리오의 인덱스 추가
        return df.loc[efficient]

    
    ## 최적화 비율 탐색 함수: 그리디 알고리즘
    def cal_optimal(self, df, unit='monthly', rebalancing='m', d=10, min_edge=30):
        '''   
        * df: 자산가치 시계열 데이터프레임
        * unit: 데이터 집계 주기 - 'daily' 또는 'monthly'
        * rebalancing: 리벨런싱 주기 - None, 'm' (월별), 'y' (연별)
        * d: 비중 시뮬레이션 단계 크기, 예를 들어 10은 10% 단위의 변동을 의미
        * min_edge: 첫 번째 자산의 최소 비중 (예: 50)
        '''

        k = len(df.T)  # 자산의 개수
        combinations = self.permutations_k_sum(100//d, k, 100//d)  # 비중 조합 생성
        combinations = pd.DataFrame(combinations) * d  # 비중을 백분율로 변환
        combinations = combinations[combinations[0] >= min_edge]  # 첫 번째 자산의 최소 비중 적용

        # 시뮬레이션 결과를 저장할 리스트 초기화
        pyrr, pydd, pstn, pvol, psharp, pweight = [], [], [], [], [], []
        print(f"총 {len(combinations)}번 시행 예정")
        
        # 각 비중 조합에 대해 시뮬레이션 실행
        for p in range(len(combinations)):
            if p == 0:
                print(f"1번째 시행 중")
            if (p % 100 == 99):
                print(f"{p + 1}번째 시행 중")
                
            # 현재 비중 설정
            weights = np.array(combinations)[p] 
            weights = weights / np.sum(weights)  # 비중 합계를 1로 조정

            # 리벨런싱 적용하여 전략 데이터프레임 생성
            result_df = self.cal_rebalancing(df, weights, rebalancing=rebalancing, unit=unit)
            # 위험 지표 및 성과 지표 계산
            sortino = anal_funcs.get_Vol_report(result_df, 'a', unit=unit, rf=0)

            # 각 지표의 마지막 값(최종 성과)을 리스트에 추가
            pyrr.append(sortino['Return'][len(sortino)-1])  # 최종 수익률
            pydd.append(sortino['Volatility(Down)'][len(sortino)-1])  # 다운사이드 변동성
            pstn.append(sortino['Sortino_Ratio'][len(sortino)-1])  # Sortino 비율
            pvol.append(sortino['Volatility'][len(sortino)-1])  # 총 변동성
            psharp.append(sortino['Sharpe_Ratio'][len(sortino)-1])  # Sharpe 비율
            pweight.append(weights)  # 최종 비중 저장

        # 리스트를 배열로 변환하여 데이터 프레임에 저장
        pyrr = np.array(pyrr)
        pydd = np.array(pydd)
        pstn = np.array(pstn)
        pvol = np.array(pvol)
        psharp = np.array(psharp)
        pweight = np.array(pweight)

        # 결과 데이터를 데이터프레임으로 정리
        process = pd.DataFrame(pweight, columns=df.columns)
        process['Return'] = pyrr
        process['Volatility'] = pvol
        process['Volatility(Down)'] = pydd
        process['Sharpe Ratio'] = psharp
        process['Sortino Ratio'] = pstn
        
        # # 최적의 Sortino 비율을 갖는 포트폴리오 선택
        # obtimal = process[process['Sortino Ratio'] == process['Sortino Ratio'].max()]
        # obtimal = obtimal.reset_index(drop=True)
        
        # # 최저 변동성을 갖는 포트폴리오 선택
        # min_risk = process[process['Volatility'] == process['Volatility'].min()]
        # min_risk = min_risk.reset_index(drop=True)
        
        process['Point'] = None
        process['Point'][process[process['Sortino Ratio'] == process['Sortino Ratio'].max()]['Return'].idxmax()] = 'Obtimal'
        process['Point'][process[process['Volatility(Down)'] == process['Volatility(Down)'].min()]['Return'].idxmax()] = 'MinRisk'

        obtimal = process[process['Point'] == 'Obtimal'].reset_index(drop=True)
        min_risk = process[process['Point'] == 'MinRisk'].reset_index(drop=True)
        
        eff_idx = self.is_efficient(process[['Return', 'Volatility(Down)']].rename(columns={'Volatility(Down)':'Risk'}))
        process['Efficient'] = False
        process.loc[eff_idx.index, 'Efficient'] = True
        process['Efficient'][process['Return']<min_risk['Return'].values[0]] = False
        
        return process, obtimal, min_risk


