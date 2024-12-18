import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

## shock case
with open('shockCase.json', encoding='utf-8') as f:
    shock_cases = json.load(f)

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
    plt.ylabel(f"{data.index[0].year}Y=100", fontsize=14, labelpad=-100, fontweight="bold", loc="top", rotation=0)
    plt.xlabel("Date", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Adjust tick params
    plt.gca().tick_params(axis="y", pad=1)

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Add grid
    plt.grid(color="gray", linestyle="--", linewidth=0.7, alpha=0.7)

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
