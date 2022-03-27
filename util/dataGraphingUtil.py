from datetime import timedelta

import pandas as pd

# Graphing Settings
import talib
from matplotlib import pyplot as plt
from pandas import to_datetime
import matplotlib as mpl

from util.langUtil import timedeltatoyahootimestr

BAR_WIDTH = 5
BAR_WIDTH_DICT = {
    '1M': 0.2,
    '2M': 0.2,
    '5M': 0.2,
    '15M': 0.3,
    '30M': 0.4,
    '60M': 0.5,
    '1h': 0.6,
    '90M': 0.7,
    '1d': 0.8,
    '5d': 1.5,
    '1wk': 2.5,
    '1m': 5,
    '3m': 10,
}
DATE_FORMAT_DICT = {  # Use parser for parsing slow, use dict for axis format
    '1m': '%Y-%m-%d %H:%M:%S',
    '2m': '%Y-%m-%d %H:%M',
    '5m': '%Y-%m-%d %H:%M',
    '15m': '%Y-%m-%d %H:%M',
    '30m': '%Y-%m-%d %H:%M',
    '60m': '%Y-%m-%d %H:%M',
    '1h': '%Y-%m-%d %H:%M',
    '90m': '%Y-%m-%d %H:%M',
    '1d': '%Y-%m-%d',
    '5d': '%Y-%m-%d',
    '1wk': '%Y-%m-%d',
    '1m': '%Y-%m-%d',
    '3m': '%Y-%m-%d',
}
FIGSIZE = (24, 12)
PLOTSTYLE = "seaborn"
PLOTENGINE = "TkAgg"


# Base Plot functions

def init_plot():
    mpl.use(PLOTENGINE)


def plot_single():
    # mpl.use(PLOTENGINE)
    plt.style.use(PLOTSTYLE)
    return plt.subplots(1, 1, figsize=FIGSIZE, sharex=True)


def plot(nrows, ncols, height_ratios, width_ratios):
    # mpl.use(PLOTENGINE)
    plt.style.use(PLOTSTYLE)
    return plt.subplots(nrows, ncols, figsize=FIGSIZE,
                        gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios}, sharex=True)


# Plotting functions

def candlestick(ax, df: pd.DataFrame):
    """Plots candlestick data based on df on ax"""

    bar_width = get_barwidth_from_interval(get_interval(df))

    # Create candlestick chart
    (up, down) = (df[df.Close >= df.Open], df[df.Close < df.Open])
    (col1, col2) = ('g', 'r')
    (w1, w2) = (bar_width, bar_width / 6)
    # Plot up
    ax.bar(up.index, up.Close - up.Open, w1, bottom=up.Open, color=col1)
    ax.bar(up.index, up.High - up.Close, w2, bottom=up.Close, color=col1)
    ax.bar(up.index, up.Low - up.Open, w2, bottom=up.Open, color=col1)
    # Plot down
    ax.bar(down.index, down.Close - down.Open, w1, bottom=down.Open, color=col2)
    ax.bar(down.index, down.High - down.Open, w2, bottom=down.Open, color=col2)
    ax.bar(down.index, down.Low - down.Close, w2, bottom=down.Close, color=col2)
    ax.axis(xmin=df.index[0], xmax=df.index[-1])

    plt.show()


# Indicators


def sma(ax, df, period):
    pass


def ema(ax, df, period, exp):
    pass


def macd(ax, df: pd.DataFrame):
    macd, macdsignal, macdhist = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)


def macd_bar(ax, macdhist: pd.DataFrame):
    pass


def profit_graph(ax, df, ls):
    pass


def load_signals(ax, sdf):
    """Loads buy signatures onto time series - date_index, buy_or_sell, short_or_long, indicator_warnings"""
    for datum in sdf:
        id = datum['id']
        datum_close = next(d for d in sdf if d['id'] == id and d['date'] != datum['date'])
        # Remove datums
        if not datum_close:
            sdf.drop([id])
        else:
            sdf.drop[id, datum_close['id']]

        pass
    # Number of "error" keys

    # Fetch successful shorts

    # Successful longs

    # Unsuccessful longs and shorts

    pass


def plot_optimisations(ax, ivar_list, profit_list, primary_axis = [0, 1], **kwd):
    pass


def mac_diagram(ax, macd_df):
    pass

# Signals


def plot_signals(signal_d, df: pd.DataFrame, canvas, see_fail=False):
    axes = canvas.axes


# Util

def get_interval(df: pd.DataFrame) -> str:
    """Gets closest interval to standard set of intervals"""
    if "Datetime" in df.columns:
        return to_datetime(df.loc[:, "Datetime"][1]) - to_datetime(df.loc[:, "Datetime"][0])
    elif "Date" in df.columns:
        return to_datetime(df.loc[:, "Date"][1]) - to_datetime(df.loc[:, "Date"][0])
    return "1M"


def get_barwidth_from_interval(interval: timedelta):
    interval = timedeltatoyahootimestr(interval)
    return BAR_WIDTH_DICT[interval]
