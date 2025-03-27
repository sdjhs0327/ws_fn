import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class VidyaAssetAllocation:
    def __init__(self, data, cash_asset='SGOV', short_period=2, long_period=200, alpha_factor=0.2, transaction_cost=0.001, vidya=False):
        """
        GTAA 전략 구현 클래스 (단기 SMA와 장기 VIDYA 크로스 기반)
        data: 시계열 데이터 (Date, 종가 포함)
        cash_asset: 매도 시 현금 보유 자산
        short_period: 단기 이동평균 기간
        long_period: 장기 VIDYA 기간
        alpha_factor: 평활 계수 스케일링 (기본 0.2)
        transaction_cost: 거래 비용 (예: 0.1% = 0.001)
        """
        self.data = data.copy()
        self.short_period = short_period
        self.long_period = long_period
        self.cash_asset = cash_asset
        self.initial_capital = 1000000000
        self.transaction_cost = transaction_cost
        self.alpha_factor = alpha_factor
        self.vidya = vidya

    def calculate_cmo(self, price_series, period=14):
        delta = price_series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta <= 0, -delta, 0)
        sum_gain = pd.Series(gain, index=price_series.index).rolling(window=period).sum()
        sum_loss = pd.Series(loss, index=price_series.index).rolling(window=period).sum()
        cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
        return cmo.fillna(0)

    def calculate_vidya(self, price_series, period=50, alpha_factor=0.2):
        cmo = self.calculate_cmo(price_series, period)
        normalized_cmo = np.abs(cmo) / 100
        alpha = alpha_factor * normalized_cmo
        alpha = alpha.clip(lower=0.01, upper=0.99)
        vidya = np.zeros_like(price_series)
        vidya[period] = price_series.iloc[period]
        for i in range(period + 1, len(price_series)):
            vidya[i] = vidya[i - 1] + alpha.iloc[i] * (price_series.iloc[i] - vidya[i - 1])
        return pd.Series(vidya, index=price_series.index)
    
    def calculate_sma(self, price_series, period=5):
        return price_series.rolling(window=period, min_periods=period).mean()

    def generate_signals(self):
        signal_df = pd.DataFrame(index=self.data.index)
        short_df = pd.DataFrame(index=self.data.index)
        long_df = pd.DataFrame(index=self.data.index)
        for col in self.data.columns:
            short = self.calculate_sma(self.data[col], period=self.short_period)
            if self.vidya:
                long = self.calculate_vidya(self.data[col], period=self.long_period, alpha_factor=self.alpha_factor)
            else:
                long = self.calculate_sma(self.data[col], period=self.long_period)
            signal_df[col] = (short > long).astype(int)
            short_df[col] = short
            long_df[col] = long
        return signal_df, short_df, long_df

    def backtest(self):
        cash_asset = self.cash_asset
        initial_capital = self.initial_capital
        transaction_cost = self.transaction_cost
        portfolio, short, long = self.generate_signals()
        portfolio = portfolio.shift(1).fillna(0)
        num_assets = portfolio.drop(columns=[cash_asset], errors='ignore').sum(axis=1).replace(0, np.nan)
        weights = portfolio.drop(columns=[cash_asset], errors='ignore').div(num_assets, axis=0).fillna(0)
        weights[cash_asset] = 1 - weights.sum(axis=1)
        returns = self.data.pct_change().fillna(0)
        weight_changes = weights.diff().fillna(0).abs()
        transaction_costs = (weight_changes * transaction_cost).sum(axis=1)
        portfolio_returns = (weights * returns).sum(axis=1) - transaction_costs
        portfolio_value = (1 + portfolio_returns).cumprod() * initial_capital
        portfolio = pd.DataFrame({'Value': portfolio_value}, index=portfolio_value.index)
        portfolio = portfolio / portfolio.iloc[0] * 100
        return portfolio, short, long
