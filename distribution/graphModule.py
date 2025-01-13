import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


figsize=(12, 8)
## 한글, 마이너스 깨짐 방지
plt.rcParams["figure.figsize"] = figsize
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

## shock case
with open('shockCase.json', encoding='utf-8') as f:
    shock_cases = json.load(f)
## color 설정
with open('colors.json') as f:
    mycolors = json.load(f)

highlight_periods = [(shock_cases['1차오일쇼크_t0'], shock_cases['1차오일쇼크_t1']),
                     (shock_cases['2차오일쇼크_t0'], shock_cases['2차오일쇼크_t1']),
                     (shock_cases['물가충격_t0'], shock_cases['물가충격_t1']),
                     (shock_cases['걸프전_t0'], shock_cases['걸프전_t1']),
                     (shock_cases['닷컴버블_t0'], shock_cases['닷컴버블_t1']),
                     (shock_cases['금융위기_t0'], shock_cases['금융위기_t1']),
                     (shock_cases['코로나_t0'], shock_cases['코로나_t1'])]

def trend_plot(df, assets, highlight_periods=highlight_periods, colors=None, title=True):
    """
    Plots a cumulative return graph for given assets.

    Parameters:
        df (DataFrame): A DataFrame containing date-indexed asset data.
        assets (list): List of asset column names to include in the plot.
        highlight_periods (list of tuples): List of (start_date, end_date) tuples to highlight on the graph.
        colors (list): List of colors for the lines, must match the number of assets.
    """
    data = df[assets].copy()
    _df = data.reset_index()
    _df = _df.melt(id_vars='Date', value_vars=_df.columns, var_name='Ticker', value_name='Value')

    # Use provided colors or default palette
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    lineplot = sns.lineplot(data=_df, x='Date', y='Value', hue='Ticker', palette=colors, linestyle='-', linewidth=2)
    if title:
        plt.title(f'Trends of {", ".join(assets)} ({data.index[0].year}~{data.index[-1].year})', fontsize=22, fontweight='bold')
    else:
        pass
    plt.ylabel(f"{data.index[0].year}Y=100", fontsize=14, labelpad=-50, loc="top", rotation=0, color=mycolors['color_around'])
    plt.xlabel("Date", fontsize=14, color=mycolors['color_around'])
    plt.xticks(fontsize=12, color=mycolors['color_around'])
    plt.yticks(fontsize=12, color=mycolors['color_around'])

    # Adjust tick params
    plt.gca().tick_params(axis="y", pad=1)

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Add grid
    plt.grid(color=mycolors["color_around2"], linestyle="--", linewidth=0.7, alpha=0.7)

    x_min, x_max = data.index.min(), data.index.max()

    # Highlight periods if provided
    if highlight_periods:
        for start, end in highlight_periods:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)

            # Adjust highlight periods to fit within the x-axis range
            if start_date > x_max or end_date < x_min:
                continue
            adjusted_start = max(start_date, x_min)
            adjusted_end = min(end_date, x_max)
            plt.axvspan(adjusted_start, adjusted_end, facecolor=mycolors['color_around'], alpha=0.30)

    # Convert x-axis to numeric format
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{round(y, 1)}'))

    # Show the plot
    plt.tight_layout()
    plt.show()


def asset_histogram_plot(df, assets, bins=50, colors=None, title=True):
    """
    Plots histograms for the returns of given assets, separated into individual horizontal subplots.

    Parameters:
        df (DataFrame): A DataFrame containing the asset return data.
        assets (list): List of asset column names to include in the plot.
        bins (int): Number of bins for the histogram.
        colors (list): List of colors for the histogram bars, must match the number of assets.
    """
    data = df[assets].copy()

    # Use provided colors or default palette
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    n_assets = len(assets)
    fig, axes = plt.subplots(1, n_assets, figsize=(8 * n_assets, 8), sharey=True)

    if n_assets == 1:
        axes = [axes]  # Ensure axes is iterable for a single asset

    for ax, asset, color in zip(axes, assets, colors):
        var_5_percent = np.percentile(data[asset].dropna(), 5)

        sns.histplot(data[asset], bins=bins, kde=True, color=color, label=asset, stat="density", ax=ax)
        if title:
            ax.set_title(f'Return Distribution of {asset}', fontsize=16, fontweight='bold')
        else:
            pass
        ax.set_xlabel("Return", fontsize=12, color=mycolors['color_around'])
        ax.set_ylabel("Density", fontsize=12, color=mycolors['color_around'])
        ax.legend(title="Asset", fontsize=10)
        ax.grid(color=mycolors["color_around2"], linestyle="--", linewidth=0.7, alpha=0.7)

        ax.axvline(var_5_percent, color=mycolors['color_around'], linestyle="--", linewidth=1)
        ax.annotate(f"VaR(5%) {var_5_percent:.2%}", xy=(var_5_percent, ax.get_ylim()[1] * 0.5),
                    xytext=(var_5_percent + (data[asset].max() * 0.01), ax.get_ylim()[1] * 0.95),
                    fontsize=12, color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()


def asset_histogram_merged_plot(df, assets, bins=50, colors=None, title=True):
    """
    Plots a histogram for the returns of given assets.

    Parameters:
        df (DataFrame): A DataFrame containing the asset return data.
        assets (list): List of asset column names to include in the plot.
        bins (int): Number of bins for the histogram.
        colors (list): List of colors for the histogram bars, must match the number of assets.

    Returns:
        None
    """
    data = df[assets].copy()

    # Use provided colors or default palette
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    plt.figure(figsize=figsize)
    for asset, color in zip(assets, colors):
        sns.histplot(data[asset], bins=bins, kde=True, color=color, label=asset, stat="density")

    if title:
        plt.title(f'Return Distributions of {", ".join(assets)} ({data.index[0].year}~{data.index[-1].year})', fontsize=22, fontweight='bold')
    else:
        pass
    plt.xlabel("Return", fontsize=14, color=mycolors['color_around'])
    plt.ylabel("Density", fontsize=14, rotation=0, labelpad=-50, loc="top", color=mycolors['color_around'])
    plt.legend(title="Assets", fontsize=12)
    plt.grid(color=mycolors["color_around2"], linestyle="--", linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()


def drawdown_plot(df, assets, highlight_periods=highlight_periods, colors=None, title=True):
    """
    Plots drawdown graphs for given assets.

    Parameters:
        df (DataFrame): A DataFrame containing the cumulative returns data.
        assets (list): List of asset column names to include in the plot.
        highlight_periods (list of tuples): List of (start_date, end_date) tuples to highlight on the graph.
        colors (list): List of colors for the lines, must match the number of assets.
    """
    def calculate_drawdown(data):
        peak = data.expanding(min_periods=1).max()
        drawdown = (data - peak) / peak
        return drawdown

    drawdowns = {asset: calculate_drawdown(df[asset]) for asset in assets}
    
    # Use provided colors or default palette
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    fig, axes = plt.subplots(1, len(assets), figsize=(8 * len(assets), 8), sharey=True)

    if len(assets) == 1:
        axes = [axes]  # Ensure axes is iterable for a single asset

    for i, (asset, color) in enumerate(zip(assets, colors)):
        ax = axes[i]
        drawdown_data = drawdowns[asset]
        drawdown_data.plot(ax=ax, color=color, linewidth=1, label=f"{asset} Drawdown")

        if title:
            ax.set_title(f'Drawdown of {asset}', fontsize=16, fontweight='bold')
        else:
            pass
        ax.set_xlabel("Date", fontsize=12, color=mycolors['color_around'])
        ax.set_ylabel("Drawdown", fontsize=12, color=mycolors['color_around'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(color=mycolors["color_around2"], linestyle="--", linewidth=0.7, alpha=0.7)

        x_min, x_max = df.index.min(), df.index.max()
        # Highlight periods if provided
        if highlight_periods:
            for start, end in highlight_periods:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                # Adjust highlight periods to fit within the x-axis range
                if start_date > x_max or end_date < x_min:
                    continue
                adjusted_start = max(start_date, x_min)
                adjusted_end = min(end_date, x_max)
                ax.axvspan(adjusted_start, adjusted_end, facecolor=mycolors['color_around'], alpha=0.30)

        # Mark and annotate MDD
        min_date = drawdown_data.idxmin()
        min_value = drawdown_data.min()
        ax.scatter(min_date, min_value, color=color, zorder=5)
        ax.annotate(f"{min_value:.2%}", xy=(min_date, min_value), xytext=(min_date, min_value - 0.03),
                    fontsize=10, color=color)

        ## ax.legend()

    plt.tight_layout()
    plt.show()
    

def ttr_plot(ttr_df, assets, highlight_periods=highlight_periods, colors=None, title=True):
    """
    Plots TTR (Time to Recovery) graphs for given assets in a similar style to the MDD plot.

    Parameters:
        ttr_df (DataFrame): A DataFrame containing TTR data (in days) for each asset.
                            Index should represent the dates when recovery was complete.
        assets (list): List of asset column names to include in the plot.
        highlight_periods (list of tuples): List of (start_date, end_date) tuples to highlight on the graph.
        colors (list): List of colors for the lines, must match the number of assets.
    """
    # 기본 색상 설정
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    # 서브플롯 생성 (자산별 1열)
    fig, axes = plt.subplots(1, len(assets), figsize=(8 * len(assets), 8), sharey=True)

    # 단일 자산일 경우 axes를 리스트 형태로 변경
    if len(assets) == 1:
        axes = [axes]

    # y축이 일수를 퍼센트처럼 표시되지 않도록 주의 (TTR은 퍼센트가 아닌 일수)
    # 별도 y formatter 없이 일 수로 표시
    
    for i, (asset, color) in enumerate(zip(assets, colors)):
        ax = axes[i]
        
        # 해당 자산의 TTR 시리즈
        ttr_data = ttr_df[asset].dropna()
        
        # TTR 데이터를 Plot
        # 데이터가 이벤트 발생 시점에 discrete하게 존재하므로 line plot을 사용하면
        # 연속적인 트렌드 파악에 유용
        ttr_data.plot(ax=ax, color=color, linewidth=2, label=f"{asset} TTR")

        # 제목 및 레이블
        if title:
            ax.set_title(f'Time to Recovery of {asset}', fontsize=16, fontweight='bold')
        else:
            pass
        ax.set_xlabel("Date", fontsize=12, color=mycolors['color_around'])
        ax.set_ylabel("TTR (Days)", fontsize=12, color=mycolors['color_around'])
        ax.grid(color=mycolors['color_around2'], linestyle="--", linewidth=0.7, alpha=0.7)
        
        # highlight_periods 처리 (MDD 플롯과 동일한 방식)
        x_min, x_max = ttr_df.index.min(), ttr_df.index.max()

        # Highlight periods if provided
        if highlight_periods:
            for start, end in highlight_periods:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)

                # Adjust highlight periods to fit within the x-axis range
                if start_date > x_max or end_date < x_min:
                    continue
                adjusted_start = max(start_date, x_min)
                adjusted_end = min(end_date, x_max)
                ax.axvspan(adjusted_start, adjusted_end, facecolor=mycolors['color_around'], alpha=0.30)
        
        # 최소 TTR 및 최대 TTR 지점 표시 (옵션)
        if not ttr_data.empty:
            # # 최소값 표시
            # min_date = ttr_data.idxmin()
            # min_value = ttr_data.min()
            # ax.scatter(min_date, min_value, color=color, zorder=5)
            # ax.annotate(f"{min_value:.0f} days", xy=(min_date, min_value),
            #             xytext=(min_date, min_value - (max_value * 0.05)), # 조금 아래에 표시
            #             fontsize=10, color=color)

            # 최대값 표시
            max_date = ttr_data.idxmax()
            max_value = ttr_data.max()
            ax.scatter(max_date, max_value, color=color, zorder=5)
            ax.annotate(f"{max_value:.0f} days", xy=(max_date, max_value),
                        xytext=(max_date, max_value + (max_value * 0.02)), # 조금 위에 표시
                        fontsize=10, color=color)
        
        ## ax.legend()

    plt.tight_layout()
    plt.show()


def rr_trend_plot(rr_df, assets, highlight_periods=highlight_periods, colors=None, title=True):
    """
    Plots trends of rolling returns for given assets.

    Parameters:
        rr_df (DataFrame): DataFrame containing rolling return data indexed by date.
        assets (list): List of asset column names to plot.
        highlight_periods (list of tuples, optional): List of (start_date, end_date) tuples to highlight on the graph.
                                                      If None, uses a default set of periods.
        colors (list, optional): List of colors for the lines. If None, uses default colors.
        figsize (tuple, optional): Figure size.
    """
    # 기본 colors 설정
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    _df = rr_df[assets].copy().reset_index()
    _df = _df.melt(id_vars='index', value_vars=_df.columns, var_name='Ticker', value_name='Value').rename(columns={'index':'Date'})

    plt.figure(figsize=figsize)
    lineplot = sns.lineplot(data=_df, x='Date', y='Value', hue='Ticker', palette=colors, linestyle='-', linewidth=2)
    if title:
        plt.title(f'Trends of 5Y Rolling Returns ({rr_df.index[0].year}~{rr_df.index[-1].year})', 
              fontsize=22, fontweight='bold', color=mycolors['color_basic'])
    else:
        pass
    plt.ylabel("Return", fontsize=14, labelpad=-40, color=mycolors['color_around'], loc="top", rotation=0)
    plt.xlabel("")
    plt.xticks(fontsize=12, color=mycolors['color_around'])
    plt.yticks(fontsize=12, color=mycolors['color_around'])

    # % 단위를 추가하는 포맷터 함수 정의
    def percent_formatter(x, pos):
        return f"{round(x, 1)}%"
    # Y축에 % 포맷터 적용
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(percent_formatter))

    # 눈금과 축 간격 줄이기
    plt.gca().tick_params(axis="y", pad=1)

    # 그리드 추가
    plt.grid(color=mycolors['color_around2'], linestyle="--", linewidth=0.7, alpha=0.7)

    if highlight_periods:
        # 강조 영역 추가
        for start, end in highlight_periods:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            if start_date > _df['Date'].max() or end_date < _df['Date'].min():
                continue
            adjusted_start = max(start_date, _df['Date'].min())
            adjusted_end = min(end_date, _df['Date'].max())
            plt.axvspan(adjusted_start, adjusted_end, facecolor=mycolors['color_around'], alpha=0.30)

    # 그래프 표시
    plt.tight_layout()
    plt.show()
    
    
def rr_box_plot(rr_df, assets, colors=None, title=True):
    """
    Plots a box plot of rolling returns for given assets.

    Parameters:
        rr_df (DataFrame): DataFrame containing rolling return data indexed by date.
        assets (list): List of asset column names to plot.
        colors (list, optional): List of colors for the box plot. If None, uses default colors.
        figsize (tuple, optional): Figure size. Default is (12, 8).
    """
    # 기본 색상 설정
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    plot_df = rr_df[assets].copy()

    plt.figure(figsize=figsize)
    boxplot = sns.boxplot(data=plot_df, palette=colors, width=0.5, linewidth=1, 
                          boxprops=dict(edgecolor=mycolors['color_basic'], linewidth=0))

    # 제목, 라벨, 티크 스타일 적용 (mycolors는 외부에서 정의되어 있다고 가정)
    if title:
        plt.title(f'Distribution of 5Y Rolling Returns ({rr_df.index[0].year}~{rr_df.index[-1].year})',
              fontsize=22, fontweight='bold', color=mycolors['color_basic'])
    else:
        pass
    plt.ylabel("Return", fontsize=14, labelpad=-40, color=mycolors['color_around'], loc="top", rotation=0)
    plt.xlabel("")
    plt.xticks(fontsize=12, color=mycolors['color_around'])
    plt.yticks(fontsize=12, color=mycolors['color_around'])

    # 중앙값 표시
    medians = plot_df.median()
    for i, median in enumerate(medians):
        plt.text(i, median + 0.25, f'{median:.2f}%', ha='center', va="center", fontsize=11,
                 color="white", fontweight="bold")

    # % 단위 포매터 정의
    def percent_formatter(x, pos):
        return f"{round(x, 1)}%"

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(percent_formatter))
    plt.gca().tick_params(axis="y", pad=1)

    # 그리드 및 기준선 추가
    plt.grid(color=mycolors['color_around2'], linestyle="--", linewidth=0.7, alpha=0.7)
    plt.axhline(y=0, color=mycolors['color_around'], linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()
    
    
def corr_plot(corr_df, cmap=None, title=True):
    """
    Plots a heatmap of the correlation matrix.
    Parameters:
        corr_df (DataFrame): DataFrame containing the correlation matrix.
        cmap (str, optional): Colormap to use for the heatmap. If None, uses a custom colormap.
        figsize (tuple, optional): Figure size. Default is (12, 8).
    """
    # cmap이 None이면 기본 컬러맵 설정
    if cmap is None:
        custom_colors = ["#F7FBFF", "#6BAED6", "#08306B"]
        cmap = LinearSegmentedColormap.from_list("custom", custom_colors)

    # 히트맵 시각화
    plt.figure(figsize=figsize)
    sns.heatmap(corr_df, annot=True, cmap=cmap, fmt='.2f',
                linewidths=0.5, cbar_kws={"shrink": .8}, cbar=False, annot_kws={"size": 12})
    if title:
        plt.title('Asset Correlations', fontsize=22, fontweight="bold", color=mycolors['color_basic'])
    else:
        pass
    plt.ylabel("Assets", fontsize=14, labelpad=-50, color=mycolors['color_around'], loc="top", rotation=0)
    plt.xlabel("")
    plt.yticks(rotation=0, fontsize=12, color=mycolors['color_around'])
    plt.xticks(rotation=0, fontsize=12, color=mycolors['color_around'])
    plt.tight_layout()
    plt.show()