from datetime import timedelta
from typing import List
import pandas as pd


class robot:
    IVAR_STEP = 0.05

    # N_ARGS = 2
    # ARGS_STR = ['stop_loss', 'take_profit']
    ARGS_DEFAULT = [1, 1.5]
    # ARGS_RANGE = [[0.01, 10], [0.01, 10]]
    ARGS_DICT = {}

    def __init__(self, ivar=ARGS_DEFAULT, xvar={}):
        pass

    def reset(self):
        pass

    # ======= Start =======

    def start(self, data_meta: List[str], interval: timedelta):
        pass

    def retrieve_prepare(self):
        pass

    def next(self, candlestick_list: pd.DataFrame):
        pass

    # ======= End =======

    def finish(self):
        pass

    def on_complete(self):
        pass

    # Retrieve results

    def get_data(self):
        pass

    def get_result(self):
        pass

    def get_profit(self):
        pass

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
        pass

    # Utility

    def sell(self, ind):
        pass

    def add_signal(self):
        pass

    def check_stop_loss(self):
        pass

    def check_profit_loss(self):
        pass

    def close_trade(self):
        pass
