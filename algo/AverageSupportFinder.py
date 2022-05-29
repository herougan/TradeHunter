from datetime import datetime

import pandas as pd

from settings import IVarType


class AverageSupportFinder:

    ARGS_DICT = {
        'smoothing_period': {
            'default': 10,
            'range': [3, 50],
            'step': 1,
            'comment': 'Factor distinguishing between different bundles. The greater the number,'
                       'the more supports are bundled together. Adjacent distance for bundling '
                       'is directly equal to d_c * stddev (variability) e.g. stddev(200) (+ flat base of 1 pip)'
                       'Acts as a multiplier to stddev. If stddev type is flat, distinguishing amount is'
                       'd_c * 100 pips'
                       'UPDATE: d_c * pips only.',
            'type': IVarType.DISCRETE,
        },
        'min_base': {
            'default': 3,
            'range': [1, 5],
            'step': 1,
            'type': IVarType.DISCRETE,
        },
    }
    PREPARE_PERIOD = 10

    def __init__(self, ivar=None):

        # == Main Args ==
        if ivar is None:
            ivar = self.ARGS_DICT
        if ivar:
            self.ivar = ivar
        else:
            self.ivar = self.ARGS_DICT

        # ARGS_DICT
        self.smoothing_period = ivar['smoothing_period']['default']
        self.min_base = ivar['min_base']['default']
        self.min_left, self.min_right = self.min_base // 2, self.min_base // 2  # No difference at the moment

        # Variable arrays
        self.bundles = []
        self.last_idx, idx = 0, 0
        self.df = pd.DataFrame()
        self.smooth_df, self.delta_df = pd.DataFrame(), pd.DataFrame()

        # Collecting data across time
        self.n_bundles = []
        self.time_start, self.time_stop = None, None
        self.idx = 0

        # == Testing ==
        self.test_mode = None

    def reset(self, ivar):
        self.__init__(ivar)

    def start(self, meta, pre_data: pd.DataFrame, test_mode=False):
        """"""
        self.reset(self.ivar)
        self.test_mode = test_mode

        # == Preparation ==
        self.df = pd.DataFrame()
        self.idx = len(pre_data) - 1

        # == Statistical Data ==
        self.n_supports = []

        # == Status ==
        self.started = True
        self.time_start = datetime.now()

        # == Pre-setup ==
        for i in range(len(pre_data)):
            # self.pre_next(pre_data[i:i+1])
            self.n_supports.append(0)

    # External command

    def support_find(self, data):
        """One shot find"""
        pass

    def set_pip_value(self, pip):
        pass

    # ==== Algo ====

    def next(self, candlestick):

        self.df = pd.concat([self.df, candlestick])
        self.idx += 1

        # ===== (1) ======
        # Search through data
        increasing, decreasing = 0, 0
        condition = 0  # -2 (Trough), -1, 0, 1, 2 (Peak)
        for i in range(self.last_idx, len(self.smooth_df)):
            if self.smooth_df.Delta[i] > 0:
                increasing += 1
                decreasing = 0
            else:
                increasing = 0
                decreasing += 1
            # If has been increasing for some time:
            if increasing >= self.min_left:
                if condition < 0:  # Check if trough
                    condition -= 1
                else:  # Update as 'has increased'
                    condition = 1
            if decreasing >= self.min_left:
                if condition > 0:  # Check if peak
                    condition += 1
                else:  # Update as 'has decreased'
                    condition = -1
            # If condition hit, go from next
            if condition <= -2:
                # Add trough
                self.bundles.append()
                self.last_idx = i + 1
            elif condition >= 2:
                # Add peak
                self.bundles.append()
                self.last_idx = i + 1

        self.last_idx = 1  # todo

        # ===== (2) ======
        # Record peaks and troughs

    def build_indicators(self):
        """Builds smooth_df according to data. Does not stop at $self.idx"""
        if len(self.df) < self.smoothing_period:
            smooth_values, delta_values = [], []
            # Average values up to smoothing period
            for i in range(len(self.smooth_df), len(self.df)):
                # Start at len - period, len = speriod
                start = i - self.smoothing_period
                value, l = 0, self.smoothing_period
                if start < 0:  # If len is too short, start is negative, so:
                    start, l = 0, i  # Set start to 0 and $stop at i (len = i also)
                for u in range(start, i):
                    value += self.df.Close[u]
                smooth_values.append(value/l)
                delta_values.append(smooth_values[-1] > smooth_values[-2])  # todo [-1] does not exist until 2nd loop
            return pd.DataFrame()  # -------------------------------- consider concat-ing every loop
        # Continue/Start building self.smooth_df
        else:
            smooth_data, delta_data, index = [], [], []
            for i in range(len(self.smooth_df), len(self.df)):
                # Assuming there is at least $smoothing_period length of data # todo
                smooth_data.append(self.smooth_df.Close[-1] +
                                   (self.df[i] - self.df[i-self.smoothing_period]) / self.smoothing_period)
                index.append(self.df.index[-1])
                delta_data.append(self.smooth_data[-1] > self.smooth_data[-2])  # todo
            # Append new data
            data = pd.DataFrame(index=index, data={
                'Close': smooth_data,
                'Delta': delta_data
            })
            pd.concat([self.smooth_data, data])


    # ===== Comments ======
    # Differences with 'ClassicSupportFinder'
    # Delta-ism
    # Late
    # Strength

    def add_into_bundle(self, bundle, support):
        bundle['supports'].append(bundle)

    def create_bundle(self, support):
        self.bundles.append()
        pass

    def create_support(self, data):
        support = data
        self.supports.append(support)
        # Try to bundle
        if True:
            self.add_into_bundle(None, support)
        else:
            self.create_bundle(support)
        pass
