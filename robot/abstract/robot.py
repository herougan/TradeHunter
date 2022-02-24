import pandas as pd


class robot:

    def __init__(self):

        # Internal Arguments defined here:
        # e.g. 0: arg A, 1: arg B etc...
        self.ivar = []
        # Range of the variable that can be explored
        self.ivar_range = []

    def ivar(self):
        pass

    def start_eval(self, time_series: pd.DataFrame):
        pass

    def get_indicator_df(self):
        pass

    def get_signal_df(self):
        pass

    def get_pl_df(self):
        pass