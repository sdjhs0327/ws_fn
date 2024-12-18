import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

figsize=(12, 8)
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

def trend_plot(df, assets, highlight_periods=highlight_periods, colors=None):
    """
    Plots a cumulative return graph for given assets.

    Parameters:
        df (DataFrame): A DataFrame containing date-indexed asset data.
        assets (list): List of asset column names to include in the plot.
        highlight_periods (list of tuples): List of (start_date, end_date) tuples to highlight on the graph.
        colors (list): List of colors for the lines, must match the number of assets.

    Returns:
        None
    """
    data = df[assets].copy()
    _df = data.reset_index()
    _df = _df.melt(id_vars='Date', value_vars=_df.columns, var_name='Ticker', value_name='Value')

    # Use provided colors or default palette
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    lineplot = sns.lineplot(data=_df, x='Date', y='Value', hue='Ticker', palette=colors, linestyle='-', linewidth=1)
    
    plt.title(f'Trends of {", ".join(assets)} ({data.index[0].year}~{data.index[-1].year})', fontsize=22, fontweight='bold')
    plt.ylabel(f"{data.index[0].year}Y=100", fontsize=14, labelpad=-100, loc="top", rotation=0)
    plt.xlabel("Date", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

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
            plt.axvspan(adjusted_start, adjusted_end, facecolor="gray", alpha=0.45)

    # Convert x-axis to numeric format
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

    # Show the plot
    plt.tight_layout()
    plt.show()


def asset_histogram_plot(df, assets, bins=50, colors=None):
    """
    Plots histograms for the returns of given assets, separated into individual horizontal subplots.

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

    n_assets = len(assets)
    fig, axes = plt.subplots(1, n_assets, figsize=(8 * n_assets, 8), sharey=True)

    if n_assets == 1:
        axes = [axes]  # Ensure axes is iterable for a single asset

    for ax, asset, color in zip(axes, assets, colors):
        var_5_percent = np.percentile(data[asset].dropna(), 5)

        sns.histplot(data[asset], bins=bins, kde=True, color=color, label=asset, stat="density", ax=ax)
        ax.set_title(f'Return Distribution of {asset}', fontsize=16, fontweight='bold')
        ax.set_xlabel("Return", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(title="Asset", fontsize=10)
        ax.grid(color=mycolors["color_around2"], linestyle="--", linewidth=0.7, alpha=0.7)

        ax.axvline(var_5_percent, color=mycolors['color_around'], linestyle="--", linewidth=1)
        ax.annotate(f"VaR(5%) {var_5_percent:.2%}", xy=(var_5_percent, ax.get_ylim()[1] * 0.5),
                    xytext=(var_5_percent + (data[asset].max() * 0.01), ax.get_ylim()[1] * 0.95),
                    fontsize=12, color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()


def asset_histogram_merged_plot(df, assets, bins=50, colors=None):
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

    plt.title(f'Return Distributions of {", ".join(assets)} ({data.index[0].year}~{data.index[-1].year})', fontsize=22, fontweight='bold')
    plt.xlabel("Return", fontsize=14)
    plt.ylabel("Density", fontsize=14, rotation=0, labelpad=-50, loc="top")
    plt.legend(title="Assets", fontsize=12)
    plt.grid(color=mycolors["color_around2"], linestyle="--", linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()


def drawdown_plot(df, assets, highlight_periods=None, colors=None):
    """
    Plots drawdown graphs for given assets.

    Parameters:
        df (DataFrame): A DataFrame containing the cumulative returns data.
        assets (list): List of asset column names to include in the plot.
        highlight_periods (list of tuples): List of (start_date, end_date) tuples to highlight on the graph.
        colors (list): List of colors for the lines, must match the number of assets.

    Returns:
        None
    """
    def calculate_drawdown(data):
        peak = data.expanding(min_periods=1).max()
        drawdown = (data - peak) / peak
        return drawdown

    drawdowns = {asset: calculate_drawdown(df[asset]) for asset in assets}
    
    # Use provided colors or default palette
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    fig, axes = plt.subplots(1, len(assets), figsize=(6 * len(assets), 6), sharey=True)

    if len(assets) == 1:
        axes = [axes]  # Ensure axes is iterable for a single asset

    for i, (asset, color) in enumerate(zip(assets, colors)):
        ax = axes[i]
        drawdown_data = drawdowns[asset]
        drawdown_data.plot(ax=ax, color=color, linewidth=1, label=f"{asset} Drawdown")

        ax.set_title(f'Drawdown of {asset}', fontsize=16, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown", fontsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(color=mycolors["color_around2"], linestyle="--", linewidth=0.7, alpha=0.7)

        # Highlight periods if provided
        if highlight_periods:
            for start, end in highlight_periods:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                ax.axvspan(start_date, end_date, color="gray", alpha=0.3)

        # Mark and annotate MDD
        min_date = drawdown_data.idxmin()
        min_value = drawdown_data.min()
        ax.scatter(min_date, min_value, color=color, zorder=5)
        ax.annotate(f"{min_value:.2%}", xy=(min_date, min_value), xytext=(min_date, min_value - 0.03),
                    fontsize=10, color=color)

        ax.legend()

    plt.tight_layout()
    plt.show()


def return_risk_profile_plot(df, assets, target_col='Return', risk_col='Volatility(Down)', colors=None):
    """
    Enhanced Return vs Downside Risk plot with improved design and annotations.
    """
    data = df[[target_col, risk_col]].copy()
    data.index = assets  # Set index as asset names

    # Calculate Sortino Ratios
    data['Sortino Ratio'] = data[target_col] / data[risk_col]

    # Use provided colors or default palette
    if colors is None:
        colors = sns.color_palette('tab10', len(assets))

    # Graph settings
    plt.figure(figsize=figsize)
    sizes = 500

    x = data[risk_col]
    y = data[target_col]
    labels = data.index

    scatter = plt.scatter(x, y, c=colors, s=sizes, edgecolors="white", linewidth=2, alpha=0.9)

    # Add labels
    for i, label in enumerate(labels):
        plt.text(
            x[i], y[i] - abs(y.max() - y.min()) * 0.11, label, fontsize=12, ha="center", va="center", 
            color="white", fontweight="bold", bbox=dict(facecolor=colors[i], edgecolor='none', alpha=0.8, boxstyle="round,pad=0.3")
        )

    # Axis formatters
    def percent_formatter(x, pos):
        return f"{round(x, 1)}%"

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(percent_formatter))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(percent_formatter))

    # Axis settings
    plt.title(f"Return-Risk Profile", fontsize=22, fontweight="bold")
    plt.xlabel("Downside Risk", fontsize=14)
    plt.ylabel("Return", fontsize=14, rotation=0, labelpad=-50, loc="top")
    plt.grid(color=mycolors["color_around2"], linestyle="--", linewidth=0.7, alpha=0.7)
    plt.xlim(0, x.max() * 1.1)
    plt.ylim(0, y.max() * 1.1)

    # Layout adjustments
    plt.tight_layout()
    plt.show()
