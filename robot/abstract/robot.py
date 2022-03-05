from typing import List

import pandas as pd


class robot:

    def __init__(self):
        # Internal Arguments defined here:
        # e.g. 0: arg A, 1: arg B etc...
        self.ivar = []
        # Range of the variable that can be explored
        self.ivar_range = []  # Self set in this case*, No need to be set externally

        # Universal Constants
        self.IVAR_STEP = 0.05

        self.last = None  # Last timestamp of tick/candlestick data

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
            self.ivar[idx] += (self.ivar_range[idx][1] - self.ivar[idx]) * IVAR_STEP * i

    def start_test(self, data_meta: List[str]):
        """Begin by understanding the incoming data.
        Setup data will be sent (according to the robot's needs - todo)
        E.g. If SMA-200 is needed, at the minimum, the past 400 data points should be known.

        old_data = retrieve()

        From then on, the robot receives data realtime - simulated by feeding point by point. (candlestick)
        """
        pass

    # ...

    def get_indicator_df(self):
        pass

    def get_signal_df(self):
        pass

    def get_pl_df(self):
        pass

    def record_profits(self):
        pass

    def record_signals(self):
        pass

    # Self-Optimisation Util
