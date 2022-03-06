import datetime
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from util.dataRetrievalUtil import retrieve


class robot:
    IVAR_STEP = 0.05

    # we are not programming it here!

    def __init__(self, ivar, xvar):
        # Internal Arguments defined here:
        # e.g. 0: arg A, 1: arg B etc...
        self.ivar = []
        # Range of the variable that can be explored
        self.ivar_range = []  # Self set in this case*, No need to be set externally

        self.prepare_period = 400  # Number of data points before current starting point (of the same interval)

        self.xvar = []

        self.df = pd.DataFrame()  # Stock dataframe

        # Indicator
        self.indicators = []  # sma200, ema, macd

        # Universal Constants
        self.IVAR_STEP = 0.05

        self.last = None  # Last timestamp of tick/candlestick data

        self.started = False

    # (External) IVar Management

    def get_ivar_len(self) -> int:
        return len(self.ivar)

    def ivar(self, *ivar):
        self.ivar = ivar

    def mutate_ivar(self, idx, _ivar):
        self.ivar[idx] = _ivar

    def step_ivar(self, idx, up=True):
        if len(self.ivar_range) >= idx:
            i = -1
            if up:
                i = 1
            self.ivar[idx] += (self.ivar_range[idx][1] - self.ivar[idx]) * robot.IVAR_STEP * i

    # START

    def start_test(self, data_meta: List[str], interval: timedelta):
        """Begin by understanding the incoming data.
        Setup data will be sent
        E.g. If SMA-200 is needed, at the minimum, the past 400 data points should be known.

        old_data = retrieve()

        From then on, the robot receives data realtime - simulated by feeding point by point. (candlestick)
        """
        self.interval = interval
        p_df = self.retrieve_prepare()

        self.build_indicators()

        # Set up Indicators

        self.started = True

    def retrieve_prepare(self):

        df = retrieve("symbol", datetime.now(), datetime.now() - self.interval * self.prepare_period, self.interval, False, False)


        # If no such file, SMA will be empty.

        df = {}

        return df

    def next(self, candlestick: pd.DataFrame):

        # update indicators

        # check signals

        # make signals

        # end-
        pass

    def next(self, candlestick_list: pd.DataFrame):
        """Same as above, but only when updates are missed so the whole backlog would
         included. Only candlestick_list[-1] will be measured for signals. The time has set
         for the rest but they are needed to calculate the indicators."""
        pass

    def finish(self):
        pass

    # De-init

    def get_data(self):
        return self.df

    # Indicators

    def build_indicators(self):
        pass

    def indicator_next(self, candlestick: pd.DataFrame):
        pass

    def get_indicator_df(self):
        pass

    # Signals

    def get_signal_df(self):
        pass

    def get_pl_df(self):
        pass

    def record_profits(self):
        pass

    def record_signals(self):
        pass

    # Self-Optimisation Util
