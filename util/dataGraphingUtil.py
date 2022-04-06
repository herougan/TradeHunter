import math
from datetime import timedelta

import matplotlib.axes
import pandas as pd

# Graphing Settings
import talib
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Patch
from pandas import to_datetime
import matplotlib as mpl

from settings import PLOTTING_SETTINGS
from util.langUtil import timedeltatoyahootimestr, strtodatetime, is_datetime, strtotimedelta

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
    '1mo': '%Y-%m-%d',
    '3mo': '%Y-%m-%d',
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

def candlestick_plot(ax, df: pd.DataFrame, xlim=None):
    """Plots candlestick data based on df on ax.
    df index must* be index form, not date form."""

    # If index, w = 1: width is as thick as the distance between 2 adjacent units, e.g. x=1 to x=2
    # if date, x=day if >week, x=1 week
    bar_width = PLOTTING_SETTINGS['candle_fixed_width'] / 10
    if is_datetime(df.index[0]):
        bar_width = get_barwidth_from_interval(get_interval(df))

    if not xlim:
        xlim = [df.index[0], df.index[-1]]
        
    # Do not plot things outside the scope
    df = df[df.index > xlim[0]]

    # Create candlestick chart
    (up, down) = (df[df.Close >= df.Open], df[df.Close < df.Open])
    (col1, col2) = ('g', 'r')
    (w1, w2) = (bar_width, bar_width / 6)
    # Plot up
    ax.bar(up.index, up.Close - up.Open, w1, bottom=up.Open, color=col1)
    ax.bar(up.index, up.High - up.Low, w2, bottom=up.Low, color=col1)
    # ax.bar(up.index, up.High - up.Close, w2, bottom=up.Close, color=col1)
    # ax.bar(up.index, up.Low - up.Open, w2, bottom=up.Open, color=col1)
    # Plot down
    ax.bar(down.index, down.Close - down.Open, w1, bottom=down.Open, color=col2)
    ax.bar(down.index, down.High - down.Low, w2, bottom=down.Low, color=col2)
    # ax.bar(down.index, down.High - down.Open, w2, bottom=down.Open, color=col2)
    # ax.bar(down.index, down.Low - down.Close, w2, bottom=down.Close, color=col2)
    # ax.axis(xmin=df.index[0], xmax=df.index[-1])
    ax.set_xlim(left=xlim[0], right=xlim[1])


def macd_histogram_plot(ax, df: pd.DataFrame, xlim=None):
    (up, down) = (df[df >= 0], df[df < 0])
    (up1, up2) = (up[up.lt(up.shift(periods=-1))], up[up.ge(up.shift(periods=-1))])
    # todo, off-shifted by 1. sometimes values can drop at interxions
    (down1, down2) = (down[down.lt(down.shift(periods=-1))], down[down.ge(down.shift(periods=-1))])
    (col1, col2, col3, col4) = ('g', 'r', 'lightgreen', 'lightsalmon')

    # Check if there is data to be plotted
    has_data = len([d for d in df if not math.isnan(d)]) > 1

    if has_data:
        if not xlim:
            xlim = [df.index[0], df.index[-1]]

        bar_w = PLOTTING_SETTINGS['candle_fixed_width'] / 10
        if is_datetime(df.index[0]):
            bar_w = get_barwidth_from_interval(get_interval(df))

        ax.bar(up1.index, up1.values, bar_w, color=col1)
        ax.bar(up2.index, up2.values, bar_w, color=col3)
        ax.bar(down1.index, down1.values, bar_w, color=col2)
        ax.bar(down2.index, down2.values, bar_w, color=col4)
        # ax.bar(up.index, up.values, bar_w, color='blue', align='center')
        # ax.bar(down.index, down.values, bar_w, color='orange', align='center')
        # ax.axis(xmin=df.index[0], xmax=df.index[-1])
        ax.set_xlim(left=xlim[0], right=xlim[1])


def line_plot(ax, df: pd.DataFrame, style={}, xlim=None):
    _style = generic_style()
    _style.update(style)

    has_data = len([d for d in df if not math.isnan(d)]) > 1

    if has_data:
        ax.plot(df, alpha=_style['alpha'], color=_style['colour'],
                linewidth=_style['linewidth'], marker=_style['marker'])
        # style = default_style.update(style)
        #
        # x = [open_value, close_value]
        # y = [start_date, end_date]
        # ax.plot(x, y, color=colour, marker=marker)
        if not xlim:
            xlim = [df.index[0], df.index[-1]]
        ax.set_xlim(left=xlim[0], right=xlim[-1])


def generic_style():
    return {
        'alpha': 0.9,
        'colour': 'r',
        'linewidth': 0.6,
        'marker': '',
    }


# Dataframe


def plot_timeseries(ax, df: pd.DataFrame, style={}):
    # Style options
    default_style = {
        'transparency': 0.85,
        'colour': 'b',
    }
    default_style.update(style)
    style = default_style

    ax.plot(df, color=style['colour'], alpha=style['transparency'])


def plot_line(ax, x, y, style={}, xlim=[]):
    # Style options
    default_style = {
        'transparency': 0.85,
        'colour': 'b',
    }
    default_style.update(style)
    style = default_style
    if not xlim:
        xlim = [x[0], x[-1]]

    ax.plot(x, y, color=style['colour'], alpha=style['transparency'])
    ax.set_xlim(left=xlim[0], right=xlim[1])

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


def plot_optimisations(ax, ivar_list, profit_list, primary_axis=[0, 1], **kwd):
    pass


def mac_diagram(ax, macd_df):
    pass


def plot_robot_instructions(axes, instructions, xlim):
    for instruction in instructions:
        plot_robot_instruction(axes, instruction, xlim)


def plot_robot_instruction(axes, instruction, xlim):
    index = instruction['index']
    data = instruction['data']
    type = instruction['type']
    colour = instruction['colour']

    ax = axes[index][0]  # row, column - in this case, assume only 0

    # Plot based on the different types...
    if type == "null":
        pass
    elif type.lower() == "macd_hist":
        macd_histogram_plot(ax, data, xlim)
    elif type.lower() == "line":
        line_plot(ax, data, {'colour': colour}, xlim)
    elif type.lower() == "hist":
        pass
    elif type.lower() == "area":
        pass
    elif type.lower() == "between":
        pass
    pass


# Signals


def plot_signals(ax, signals, xlim):
    style = {
        'transparency': 0.9
    }
    for signal in signals:
        # Start date will be different if there is calculated from earlier peak/troughs
        if 'baseline' not in signal:
            signal['baseline'] = signal['start']
        # plot_stop_take_box(ax, signal, style)
        # plot_open_close_pos(ax, signal, style, xlim)
    plot_stop_take(ax, signals, style, xlim)
    plot_open_close_all_pos(ax, signals, style, xlim)


def plot_open_signals(ax, signals, xlim):
    # Style
    style = {
        'transparency': 0.6
    }
    for signal in signals:
        if 'baseline' not in signal:
            signal['baseline'] = signal['start']
        if 'end' not in signal or not signal['end']:
            signal['end'] = signal['start']
        # plot_stop_take_box(ax, signal, style)
        # plot_open_close_pos(ax, signal, style)  # no close pos
    plot_stop_take(ax, signals, style, xlim)


def plot_stop_take(ax, signals, style={}, xlim=None):
    """Stop_loss, take_profit rectangles drawn using bar plot"""

    # index: Where the bar starts, width: distance to bar end,
    # height: How tall is the bar, base: Where the bar starts (y)
    profit_index = []
    profit_height = []
    profit_width = []
    profit_base = []

    loss_index = []
    loss_height = []
    loss_width = []
    loss_base = []

    left_lim = math.inf
    right_lim = 0
    for signal in signals:
        # if signal['stop_loss'] < signal['take_profit']:
        if signal['vol'] >= 0:  # long
            profit_index.append(signal['baseline'])
            profit_height.append(signal['take_profit'] - signal['open_price'])
            profit_width.append(signal['end'] - signal['baseline'])
            profit_base.append(signal['open_price'])

            loss_index.append(signal['baseline'])
            loss_height.append(signal['open_price'] - signal['stop_loss'])
            loss_width.append(signal['end'] - signal['baseline'])
            loss_base.append(signal['stop_loss'])

            if not xlim and left_lim > signal['baseline']:
                left_lim = signal['baseline']
            if not xlim and right_lim < signal['end']:
                right_lim = signal['end']
        elif signal['vol'] < 0:  # short
        # elif signal['stop_loss'] >= signal['take_profit']:
            profit_index.append(signal['baseline'])
            profit_height.append(signal['open_price'] - signal['take_profit'])
            profit_width.append(signal['end'] - signal['baseline'])
            profit_base.append(signal['take_profit'])

            loss_index.append(signal['baseline'])
            loss_height.append(signal['stop_loss'] - signal['open_price'])
            loss_width.append(signal['end'] - signal['baseline'])
            loss_base.append(signal['open_price'])

            if not xlim and left_lim > signal['baseline']:
                left_lim = signal['baseline']
            if not xlim and right_lim < signal['end']:
                right_lim = signal['end']

    # Style options
    if 'transparency' not in style:
        style.update({'transparency': 0.8})
    transparency = style['transparency']
    profit_col = 'g'
    loss_col = 'r'

    if not xlim:
        xlim = [left_lim - 1, right_lim + 1]
    ax.set_xlim(xlim[0], xlim[1])

    # ax.bar(profit_index, profit_height, profit_width, color=profit_col, alpha=transparency, bottom=profit_base)
    # ax.bar(loss_index, loss_height, loss_width, color=loss_col, alpha=transparency, bottom=loss_base)

#
# def plot_stop_take_box(ax, signal, style={}, **kwd):
#     """Stop_loss, take_profit rectangle. Mainly only accepts index.
#     WARNING: TEMPORARILY UNUSED - CANNOT GET MATPLOTLIB's RECTANGLE TO WORK"""
#     # Variables
#     stop_loss = signal['stop_loss']
#     take_profit = signal['take_profit']
#     pos_value = signal['open_price']
#
#     # Check if index based or date based
#     if signal['baseline']._typ == 'int64index':
#         baseline_date = signal['baseline']
#         end_date = signal['end']
#         start_date = signal['start']
#         period = baseline_date - end_date
#         # Calculate width
#         width = period
#     else:
#         baseline_date = strtodatetime(signal['baseline'])
#         end_date = strtodatetime(signal['end'])
#         start_date = strtodatetime(signal['start'])
#         period = baseline_date - end_date
#         if 'interval' in signal:
#             interval = strtotimedelta(signal['interval'])  # need interval to calculate rect length
#             width = period / interval
#         elif 'interval' in kwd:
#             interval = strtotimedelta(kwd['interval'])  # need interval to calculate rect length
#             width = period / interval
#         else:
#             width = 5
#     if width < 1:
#         width = 1
#
#     # Style options
#     if 'transparency' not in style:
#         style.update({'transparency': 0.8})
#     transparency = style['transparency']
#
#     # Draw stop loss rectangle
#     profit_rects = []
#     loss_rects = []
#     if take_profit > stop_loss:
#         loss_rects.append(Rectangle((stop_loss, baseline_date), width, pos_value - stop_loss))
#         profit_rects.append(Rectangle((pos_value, baseline_date), width, take_profit - pos_value))
#     else:
#         loss_rects.append(Rectangle((pos_value, baseline_date), width, stop_loss - pos_value))
#         profit_rects.append(Rectangle((take_profit, baseline_date), width, pos_value - take_profit))
#
#     # Add rectangles
#     l_pc = PatchCollection(loss_rects)
#     p_pc = PatchCollection(profit_rects)
#     ax.add_collection(p_pc, facecolor='g', alpha=transparency)
#     ax.add_collection(l_pc, facecolor='r', alpha=transparency)


def plot_open_close_pos(ax: matplotlib.axes.Axes, signal, style={}, xlim=None):
    """Start Price and 'Open' for 'Open Position' price are equivalent.
    End date == 'Close' date etc. They do not represent close prices but closing
    positions! Dates must be in datetime or int index format. Strings will not work!"""

    # Variables
    open_value = signal['open_price']
    close_value = signal['close_price']
    end_date = signal['end']
    start_date = signal['start']
    # if signal['baseline']._typ == 'int64index':
    #     end_date = signal['end']
    #     start_date = signal['start']
    # else:
    #     end_date = strtodatetime(signal['end'])
    #     start_date = strtodatetime(signal['start'])
    net = signal['net']

    # Style options
    default_style = {
        'transparency': 0.8,
        'loss_colour': 'r',
        'profit_colour': 'g',
        'marker': 'x',
    }

    default_style.update(style)
    style = default_style
    transparency = style['transparency']
    colour = style['loss_colour']
    if net > 0:
        colour = style['profit_colour']
    marker = style['marker']

    # Draw circle on open and close positions
    x = [open_value, close_value]
    y = [start_date, end_date]
    ax.plot(x, y, color=colour, marker=marker)

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])


def plot_open_close_all_pos(ax: matplotlib.axes.Axes, signals, style={}, xlim=None):

    # Style options
    default_style = {
        'transparency': 0.8,
        'loss_colour': 'r',
        'profit_colour': 'g',
        'marker': 'x',
    }

    default_style.update(style)
    style = default_style
    transparency = style['transparency']
    colour = style['loss_colour']
    marker = style['marker']

    for signal in signals:

        # Variables
        open_value = signal['open_price']
        close_value = signal['close_price']
        end_date = signal['end']
        start_date = signal['start']
        # if signal['baseline']._typ == 'int64index':
        #     end_date = signal['end']
        #     start_date = signal['start']
        # else:
        #     end_date = strtodatetime(signal['end'])
        #     start_date = strtodatetime(signal['start'])
        net = signal['net']
        if net > 0:
            colour = style['profit_colour']
        else:
            colour = style['loss_colour']

        # Draw circle on open and close positions
        x = [open_value, close_value]
        y = [start_date, end_date]
        # ax.plot(x, y, color=colour, marker=marker)

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])


# Util

def get_interval(df: pd.DataFrame) -> str:
    """Gets interval from dataframe. Must have adjacent data!"""
    for i in range(1, len(df)):
        if not math.isnan(df[-i]) and not math.isnan(df[-i-1]):
            return strtodatetime(df.index[-i]) - strtodatetime(df.index[-i-1])
    return timedelta(minutes=1)  # default


def get_yahoo_interval_str(df: pd.DataFrame) -> str:
    return timedeltatoyahootimestr(get_interval(df))


def get_barwidth_from_interval(interval: timedelta):
    interval = timedeltatoyahootimestr(interval)
    return BAR_WIDTH_DICT[interval]
