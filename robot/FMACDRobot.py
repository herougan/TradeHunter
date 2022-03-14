"""FMACD Robot"""
from typing import List

import pandas as pd
import talib

from robot.abstract.robot import robot
from datetime import datetime, timedelta


# Pre-Retrieve done by DataTester

# Definitions:
#
# Check - Indicator result: long, short or none
# Signal - Dictionary, trade deal = { type, vol (amount), start, end, }


class FMACDRobot(robot):
    IVAR_STEP = 0.05
    N_ARGS = 2
    ARGS_STR = ['stop_loss', 'take_profit', 'fast_period', 'slow_period', 'signal_period']
    ARGS_DEFAULT = [1, 1.5, 12, 26, 9]
    ARGS_RANGE = [[0.01, 10], [0.01, 10]]

    # Signal Definition:
    #  {}: 'net', 'start', 'end', 'type (short/long)', 'vol', 'type', 'equity',
    #  + 'macd_fail', 'sma_fail', 'ema_fail',

    def __init__(self, ivar=ARGS_DEFAULT, xvar={}):
        if len(ivar) == 2:
            self.ivar = ivar
        else:
            self.ivar = FMACDRobot.ARGS_DEFAULT
        self.ivar_range = FMACDRobot.ARGS_RANGE
        self.steps = 0

        # External attributes
        self.xvar = xvar
        self.symbol = ""
        self.period = timedelta()
        self.interval = timedelta()

        # Preparation attributes
        self.prepare_period = 200

        # Indicator Data
        self.indicators = {
            'SMA200': [],
            'SMA5': [],
            'EMA': [],
            'MACD': [],
            'MACD_SIGNAL': [],
            'MACD_HIST': [],
            'MACD_DF': [],
        }

        # Statistical Data
        self.profit_data = []
        self.asset_data = []
        # Operational variables
        self.signals = []
        self.signal_headers = []
        self.completed_signals = []

        # Data
        self.df = {}

        # Upon completion
        self.profit_df = []
        self.equity_df = []
        self.signal_df = []

        # Status
        self.market_active = False
        self.started = False
        self.last_date = datetime.now()

    def reset(self):
        self.steps = 0

        self.profit_df = []
        self.equity_df = []
        self.signal_df = []

    # ======= Start =======

    def start(self, symbol: str, interval: timedelta, period: timedelta):
        """Begin by understanding the incoming data. Setup data will be sent
        E.g. If SMA-200 is needed, at the minimum, the past 400 data points should be known.
        old_data = retrieve()
        From then on, the robot receives data realtime - simulated by feeding point by point. (candlestick)
        data_meta: Symbol, Period, Interval, xvar variables, ...
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval

        self.reset()
        self.interval = interval
        self.retrieve_prepare()

        self.build_indicators()

        # Set up Indicators

        self.started = True

    def retrieve_prepare(self, df):

        self.df = df

    def next(self, candlesticks: pd.DataFrame):
        """Same as above, but only when updates are missed so the whole backlog would
         included. Only candlestick_list[-1] will be measured for signals. The time has set
         for the rest but they are needed to calculate the indicators."""

        self.df.append(candlesticks)

        profit = 0
        liquid_assets = 0
        self.profit_data.append(profit)
        self.asset_data.append(liquid_assets)

        # update indicators
        self.build_indicators()

        # close signals
        for signal in [_signal for _signal in self.signals if not _signal['end']]:
            pass

        # check signals
        check_dict = self.check_indicators()  # indicator check dictionary
        f_check = check_dict.values()[0]
        for key in check_dict.keys():
            if not f_check:
                continue
            if f_check != check_dict[key]:
                f_check = 0
        # make signals
        self.create_signal(f_check, check_dict, candlesticks[-1])
        # end -

    def calc_macd(self):
        self.indicators['MACD'], self.indicators['MACD_SIGNAL'], self.indicators['MACD_HIST'] = \
            talib.MACD(self.df, fastperiod=12, slowperiod=26, signalperiod=9)
        self.indicators['MACD_DF'] = pd.DataFrame(index=self.df.index,
                                                  data={"macd": self.indicators['MACD'],
                                                        "macd_signal": self.indicators['MACD_SIGNAL'],
                                                        "macdhist": self.indicators['MACD_HIST'], })
        self.indicators['SMA5'] = talib.SMA(self.df, timeperiod=2)
        self.indicators['SMA200'] = talib.SMA(self.df, timeperiod=200)

    # ======= End =======

    def finish(self):
        pass

    def on_complete(self):
        pass
        # self.profit_df = create_profit_df_from_list(self.profit_data, self.asset_data)
        # self.signal_df = create_signal_df_from_list(self.completed_signals, self.signals)
        # self.summary_df = create_summary_df_from_list(self.profit_data, self.asset_data, self.completed_signals)

    # Retrieve results

    def get_data(self):
        return self.df

    def get_time_data(self):
        return self.df['datetime']

    def get_profit(self):
        return self.profit_d, self.equity_d

    def get_signals(self):
        return self.signals

    def get_curr_data_time(self):
        return self.df['datetime'].tolist()[-1]

    # Indicator

    def check_indicators(self):
        """Returns a list of integer-bools according to the signal-generating indicators.
        0: False, 1: Long, 2: Short"""
        return {
            'MACD': self.check_macd(),
            'MACD_HIST': self.check_macd_hist(),
            'SMA': self.check_sma(),
        }

    def check_macd(self):
        if len(self.indicators['MACD']) > 2 and len(self.indicators['MACD_SIGNAL']) > 2:
            if self.indicators['MACD'][-2] > self.indicators['MACD_SIGNAL'][-2]:
                if self.indicators['MACD'][-1] < self.indicators['MACD_SIGNAL'][-1]:
                    return 1
            else:
                if self.indicators['MACD'][-1] > self.indicators['MACD_SIGNAL'][-1]:
                    return 2
        return 0

    def check_macd_hist(self):
        if self.indicators['MACD_HIST'][-1] > 0:
            if self.indicators['MACD_HIST'][-1] > self.indicators['MACD_HIST'][-2]:
                return 1
        else:
            if self.indicators['MACD_HIST'][-1] < self.indicators['MACD_HIST'][-2]:
                return 2
        return 0

    def check_sma(self):
        if self.indicators['SMA200'][-1] > self.df['close'][-1]:
            return 1
        else:
            return 2

    # Signal

    def create_signal(self, check, check_dict, df):

        start = self.get_curr_data_time()
        vol = self.

        if check:
            signal = {
                'type': check,
                'vol': 0,
                'start': start,
                'end': 0,
            }
        self.signals.append(signal)

    def close_signal(self, check, check_dict, df):



        pass

    # Optimisation

    def step_ivar(self, idx, up=True):
        if len(self.ivar_range) >= idx:
            i = -1
            if up:
                i = 1
            self.ivar[idx] += (self.ivar_range[idx][1] - self.ivar[idx]) * robot.IVAR_STEP * i

    # Utility

    def if_go(self):
        # Check if [-1] and [-2] sma exists for the df
        pass

    def sell(self, ind):
        # self.signals = [signal for signal in self.signals if signal['type'] == 'Buy']
        signal = self.signals[ind]
        del self.signals[ind]
        completed_signal = {

        }
        self.completed_signals.append(completed_signal)

    def add_signal(self):
        # if signals were = { 'date': [], 'vol': [] }
        # checking for a signal would be troublesome!
        # more convenient if for signal in signals!
        # therefore signals = [{ 'date': date, 'vol': vol, ...}...]
        pass

    def check_stop_loss(self):
        pass

    def check_profit_loss(self):
        pass

    def close_trade(self):

        # Calculate profit
        pass
