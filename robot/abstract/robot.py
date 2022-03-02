import pandas as pd

IVAR_STEP = 0.05


class robot:

    def __init__(self):
        # Internal Arguments defined here:
        # e.g. 0: arg A, 1: arg B etc...
        self.ivar = []
        # Range of the variable that can be explored
        self.ivar_range = []  # Self set in this case*, No need to be set externally

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

    def start_test(self, time_series: pd.DataFrame):
        pass

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
