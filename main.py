# Custom packages
import sys
from datetime import date, timedelta

# UI
from PyQt5.QtWidgets import QApplication
from pandas import to_datetime

import UI.tradehunter

# Custom Util
sys.path.append('util')
from util.dataRetrievalUtil import retrieve, load_df, load_df_list, init_common, force_overwrite_common
from util.langUtil import strtoyahootimestr

# Settings
import settings

# Packages
import yfinance as yf
from matplotlib import pyplot as plt
import matplotlib as mpl
import talib
import pandas as pd


def test_3():
    init_common()
    # force_overwrite_common()

    th = UI.tradehunter.TradeHunterApp()


def test_2():
    mpl.use('TkAgg')
    plt.style.use("seaborn")
    # Options
    figsize = (26, 16)
    length_of_time = 666  # In days
    interval = '60m'
    risk_ratio = 1.5
    stop_loss = 10
    take_profit = 15
    pip_value = 0.001
    stop_loss = stop_loss * pip_value
    take_profit = take_profit * pip_value
    max_length = 50
    bar_width_dict = {
        '60m': 0.6,
        '90m': 0.8,
        '1d': 1,
        '1wk': 2,
    }
    date_format_dict = {
        '60m': '%Y-%m-%d %H:%M',
        '90m': '%Y-%m-%d %H:%M',
        '1d': '%Y-%m-%d',
        '1wk': '%Y-%m-%d',
    }
    df = yf.download("AAPL",
                     start=date.today() - timedelta(days=length_of_time),
                     end=date.today(),
                     interval=interval)
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1, 1]}, sharex=True)

    # MACD
    data = df["Close"]
    macd, macdsignal, macdhist = talib.MACD(data, fastperiod=12, slowperiod=26, signalperiod=9)
    sma5 = talib.SMA(data, timeperiod=2)
    sma200 = talib.SMA(data, timeperiod=200)
    macd_df = pd.DataFrame(index=df.index,
                           data={"macd": macd,
                                 "macd_signal": macdsignal,
                                 "macdhist": macdhist, })
    profit = {
        'main': [0] * len(macd),
        'fail': {
            'sma': [0] * len(macd),
            'sma_macd': [0] * len(macd),
            'macd': [0] * len(macd),
        }
    }

    # Plot everything plus long/short

    # Base evaluation functions

    ###################
    # Start algorithm
    ###################
    if len(macd) <= 200: return
    over = False
    long = False
    short = False

    # setup; index based
    pos = {
        'long': {
            'main': [],
            'fail': {
                'sma': [],
                'macd': [],
                'sma_macd': [],
            }
        },
        'short': {
            'main': [],
            'fail': {
                'sma': [],
                'macd': [],
                'sma_macd': [],
            }
        },
    }
    closed = {
        'long': {
            'main': [],
            'main_close': [],
            'fail': {
                'sma': [],
                'macd': [],
                'sma_macd': [],
                'sma_close': [],
                'macd_close': [],
                'sma_macd_close': [],
            }
        },
        'short': {
            'main': [],
            'main_close': [],
            'fail': {
                'sma': [],
                'macd': [],
                'sma_macd': [],
                'sma_close': [],
                'macd_close': [],
                'sma_macd_close': [],
            }
        },
    }

    def deal(_open, _close):
        return (_open - _close) > take_profit

    def escape(_open, _close):
        return (_close - _open) > stop_loss

    for i in range(200, len(macd)):

        datum = data[i]
        _over = macd_df.macd[i] < macd_df.macd_signal[i]
        if over != _over:
            if over:
                long = True
            if ~over:
                short = True
            over = _over

        profit['main'][i] = profit['main'][i - 1]
        # Try to close positions
        for l in pos['long']['main']:
            if deal(data[l], datum):
                print("Closing a long signal")
                profit['main'][i] += data[l] - datum
                pos['long']['main'].remove(l)
            elif escape(data[l], datum):
                print("Escaping a long signal")
                profit['main'][i] += data[l] - datum
                pos['long']['main'].remove(l)

        for s in pos['short']['main']:
            if deal(datum, data[s]):
                print("Closing a short signal")
                profit['main'][i] += datum - data[l]
                pos['short']['main'].remove(s)
            elif escape(datum, data[s]):
                print("Escaping a short signal")
                profit['main'][i] += datum - data[l]
                pos['short']['main'].remove(s)

        if long:
            error_str = ""
            if macd_df.macdhist[i] > 0:
                error_str = "Failed to place a long signal due to MACD_hist"
                pos['long']['fail']['macd'].append(i)
            if df.Close[i] < sma200[i]:
                if error_str:
                    error_str += " and SMA"
                    pos['long']['fail']['sma_macd'].append(i)
                else:
                    error_str = "Failed to place a long signal due to SMA"
                    pos['long']['fail']['sma'].append(i)
            if not error_str:
                print("Placing a long signal")
                pos['long']['main'].append(i)
            else:
                print(error_str)

        if short:
            error_str = ""
            if macd_df.macdhist[i] < 0:
                error_str += "Failed to place a short signal due to MACD_hist"
                pos['short']['fail']['macd'].append(i)
            if df.Close[i] > sma200[i]:
                if error_str:
                    error_str += " and SMA"
                    pos['short']['fail']['sma_macd'].append(i)
                else:
                    error_str = "Failed to place a short signal due to SMA"
                    pos['short']['fail']['sma'].append(i)
            if not error_str:
                print("Placing a short signal")
                pos['short']['main'].append(i)
            else:
                print(error_str)

        for l in pos['long']['fail']['sma']:
            pass
        for l in pos['long']['fail']['macd']:
            pass
        for l in pos['long']['fail']['sma_macd']:
            pass

        for s in pos['short']['fail']['sma']:
            pass
        for s in pos['short']['fail']['macd']:
            pass
        for s in pos['short']['fail']['sma_macd']:
            pass

    print('max', max(profit['main']))
    print('min', min(profit['main']))
    print('final', profit['main'][-1])

    # Remove weekends
    # dates = df.index
    # df.index = range(len(df.index))
    # x_tick_labels = []
    # for _date in dates:
    #     x_tick_labels.append(_date.strftime(date_format_dict[interval]))


def test_1():
    # Data Retrieval
    # r = retrieve('AAPL', '2019-01-01', '2021-06-12', False)
    mpl.use('TkAgg')
    plt.style.use("seaborn")
    # Options
    figsize = (24, 12)
    candlestick_no = 100
    display_trade_info = True
    length_of_time = 100  # In days
    interval = '60m'
    bar_width_dict = {
        '60m': 0.6,
        '90m': 0.8,
        '1d': 1,
        '1wk': 2,
    }
    date_format_dict = {
        '60m': '%Y-%m-%d %H:%M',
        '90m': '%Y-%m-%d %H:%M',
        '1d': '%Y-%m-%d',
        '1wk': '%Y-%m-%d',
    }

    # Tunables
    x = 0

    # Download Data
    df = yf.download("EURUSD=X",
                     start=date.today() - timedelta(days=length_of_time),
                     end=date.today(),
                     interval=interval)
    # Remove weekends
    dates = df.index
    df.index = range(len(df.index))

    # MACD
    macd, macdsignal, macdhist = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    sma5 = talib.SMA(df["Close"], timeperiod=2)
    sma200 = talib.SMA(df["Close"], timeperiod=200)
    macd_talib = pd.DataFrame(index=df.index,
                              data={"macd": macd,
                                    "macd_signal": macdsignal})
    macd_hist = pd.DataFrame(index=df.index,
                             data={"macdhist": macdhist})
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    # Create df sma
    ax1.plot(sma5, alpha=0.05, color="b", linewidth=0.3)
    ax1.plot(df.Close, alpha=0.6, color="b", linewidth=0.3)
    ax1.plot(sma200, alpha=1, color="b", linewidth=0.6)
    # Create candlestick chart
    (up, down) = (df[df.Close >= df.Open], df[df.Close < df.Open])
    (col1, col2) = ('g', 'r')
    (w1, w2) = (bar_width_dict[interval], bar_width_dict[interval] / 6)
    # Plot up
    ax1.bar(up.index, up.Close - up.Open, w1, bottom=up.Open, color=col1)
    ax1.bar(up.index, up.High - up.Close, w2, bottom=up.Close, color=col1)
    ax1.bar(up.index, up.Low - up.Open, w2, bottom=up.Open, color=col1)
    # Plot down
    ax1.bar(down.index, down.Close - down.Open, w1, bottom=down.Open, color=col2)
    ax1.bar(down.index, down.High - down.Open, w2, bottom=down.Open, color=col2)
    ax1.bar(down.index, down.Low - down.Close, w2, bottom=down.Close, color=col2)
    ax1.axis(xmin=df.index[0], xmax=df.index[-1])

    # Create macd time series
    ax2.plot(macd_talib)
    # Create macd hist chart
    (up, down) = (macd_hist[macd_hist.macdhist >= 0], macd_hist[macd_hist.macdhist < 0])
    (up1, up2) = (up[up.lt(up.shift(periods=-1))], up[up.ge(up.shift(periods=-1))])
    (down1, down2) = (down[down.lt(down.shift(periods=-1))], down[down.ge(down.shift(periods=-1))])
    (col1, col2, col3, col4) = ('g', 'r', 'lightgreen', 'lightsalmon')
    w = bar_width_dict[interval]
    ax2.bar(up1.index, up1.macdhist, w, color=col1)
    ax2.bar(up2.index, up2.macdhist, w, color=col3)
    ax2.bar(down1.index, down1.macdhist, w, color=col2)
    ax2.bar(down2.index, down2.macdhist, w, color=col4)
    ax2.axis(xmin=df.index[0], xmax=df.index[-1])

    # Readjust x-labels after removing weekends
    x_tick_labels = []
    for _date in dates:
        x_tick_labels.append(_date.strftime(date_format_dict[interval]))
    ax2.set(xticklabels=x_tick_labels)

    plt.show()


if __name__ == '__main__':
    test_3()