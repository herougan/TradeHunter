'''FMACD Robot'''
from typing import List

import pandas as pd
import talib

from robot.abstract.robot import robot
from datetime import datetime, timedelta

# IDEA TODO
# THEN, AS A GREAT PROJECT - WE MOVE ALL CODE LOGIC TO DATETESTINGUTIL - SO THAT FMACD ROBOT JUST DOES EVERYTHING
# IN QUIET. (OUTSIDE USES While Not self.Done(): self.Next() or something like that)
# DATATESTER loops through itself and produces stuff. FMACDRobot just needs to feed data into datatester.
# Makes sense!
#
# Pre-Retrieve done by DataTester


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

        # External attributes
        self.xvar = xvar

        # Indicator Data
        self.indicators = {
            'SMA200': 0,
            'EMA': 0,
            'MACD1': 0,
            'MACD2': 0,
            'MACDHIST': 0,
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

        # Meta does not matter, but just in case
        self.symbol = ""

        # Upon completion
        self.profit_df = None
        self.signal_df = None
        self.summary_df = None

        # Status
        self.market_active = False
        self.started = False
        self.last_date = datetime.now()

    def reset(self):
        self.steps = 0

        self.profit_df = None
        self.signal_df = None
        self.summary_df = None

    # ======= Start =======

    def start(self, data_meta: List[str], interval: timedelta):
        """Begin by understanding the incoming data.
        Setup data will be sent
        E.g. If SMA-200 is needed, at the minimum, the past 400 data points should be known.

        old_data = retrieve()

        From then on, the robot receives data realtime - simulated by feeding point by point. (candlestick)
        """
        self.reset()
        self.interval = interval
        self.retrieve_prepare()

        self.build_indicators()

        # Set up Indicators

        self.started = True

    def retrieve_prepare(self, df):

        # self.df = retrieve("symbol", datetime.now(), datetime.now() - self.interval * self.prepare_period,
        #                    self.interval,
        #                    False, False)

        self.df = df  # Past data todo

        # If no such file, SMA will be empty.
        self.calc_macd()

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
        self.calc_macd()

        # check signals
        for signal in [_signal for _signal in self.signals if not _signal['end']]:
            pass

        # make signals
        if self.if_go():
            pass
        # end-

    def calc_macd(self):
        macd, macdsignal, macdhist = talib.MACD(self.df, fastperiod=12, slowperiod=26, signalperiod=9)
        sma5 = talib.SMA(self.df, timeperiod=2)
        sma200 = talib.SMA(self.df, timeperiod=200)
        macd_df = pd.DataFrame(index=self.df.index,
                               data={"macd": macd,
                                     "macd_signal": macdsignal,
                                     "macdhist": macdhist, })

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

    def get_result(self):
        pass

    def get_profit(self):
        profit_d = []
        equity_d = []
        return profit_d, equity_d

    def get_signals(self):
        pass

    # Indicators

    def build_indicators(self):
        pass

    def indicator_next(self, candlestick: pd.DataFrame):
        pass

    def get_indicator_df(self):
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
        return False

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
