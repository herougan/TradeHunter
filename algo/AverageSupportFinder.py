import math
from datetime import datetime

import pandas as pd

from settings import IVarType
from util.langUtil import try_divide


class AverageSupportFinder:

    ARGS_DICT = {
        'smoothing_period': {
            'default': 8,
            'range': [3, 20],
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
    OTHER_ARGS_DICT = {
        'distinguishing_constant': {
            'default': 3,
        },
        'lookback': {
            'default': 150,
        }
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
        self.bundles, self.supports = [], []  # supports make up bundles
        self.last_idx, idx = 0, 0
        self.df = pd.DataFrame()
        self.smooth_df, self.delta_df = pd.DataFrame(), pd.DataFrame()

        # Constants
        self.pip = 0.0001
        self.distinguishing_constant = self.OTHER_ARGS_DICT['distinguishing_constant']['default']
        self.distinguishing_value = self.distinguishing_constant * self.pip
        self.lookback = self.OTHER_ARGS_DICT['lookback']['default']

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
        self.df = pre_data
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
            # Indicators will be built 'one-shot' in first 'next' loop

    # External command

    def support_find(self, data):
        """One shot find"""
        pass

    def set_pip_value(self, pip):
        pass

    # ==== Algo ====

    def next(self, candlestick):

        self.df = pd.concat([self.df, candlestick])
        self.build_indicators()
        self.idx += 1

        # ===== (1) ======
        # Search through data
        increasing, decreasing = 0, 0
        past_increasing, past_decreasing = 0, 0
        condition = 0  # -2 (Trough), -1, 0, 1, 2 (Peak)
        for i in range(self.last_idx, len(self.smooth_df)):
            if self.smooth_df.Delta[i]:
                increasing += 1
                past_decreasing, decreasing = decreasing, 0
            else:
                past_increasing, increasing = increasing, 0
                decreasing += 1
            # If has been increasing for some time:
            if increasing >= self.min_left:
                if condition < 0:  # Check if trough (if 'has' decreased)
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
                self.create_support_at(i - increasing, increasing, past_decreasing)  # *Is increasing, was decreasing
                self.last_idx = i + 1
            elif condition >= 2:
                # Add peak
                self.create_support_at(i - decreasing, past_increasing, decreasing)
                self.last_idx = i + 1

        # ===== (2) =====
        # Record peaks and troughs

        # ===== (3) =====
        # Clean
        self.delete_decayed()

    def build_indicators(self):
        """Builds smooth_df according to data. Does not stop at $self.idx"""

        for i in range(len(self.smooth_df), len(self.df)):
            if i < self.smoothing_period:  # At i = self.sm_p, recalculate at len = sm_p; up to i, avg up to #i
                smooth = sum(self.df.Close[max(0, i-self.smoothing_period):i+1])\
                         / min(len(self.smooth_df) + 1, self.smoothing_period)
                if i == 0:
                    past_smooth = self.df.Close[i]
                else:
                    past_smooth = self.smooth_df.Smooth[-1]
            else:  # Calculate from previous value
                smooth = self.smooth_df.Smooth.iloc[-1] + (self.df.Close[i] - self.df.Close[i - self.smoothing_period])\
                         / self.smoothing_period
                past_smooth = self.smooth_df.Smooth.iloc[-1]
            data = pd.DataFrame(
                index=[self.df.index[i]],
                data={
                    'Smooth': [smooth],
                    'Delta': [smooth > past_smooth],
                }
            )
            # Concat to sdf every loop
            self.smooth_df = pd.concat([self.smooth_df, data])

    def get_instructions(self):
        # Lines should get lighter the weaker they are
        # Data should be pd.DataFrame format with index and 'height'/value
        data = pd.DataFrame(index=[self.get_idx_date(bundle['pos']) for bundle in self.bundles], data={
            'strength': [bundle['strength'] for bundle in self.bundles],
            'height': [bundle['height'] for bundle in self.bundles],
            'pos': [bundle['pos'] for bundle in self.bundles],
        })
        # data = pd.DataFrame(index=[[self.df.index.get_loc(bundle['peak']) for bundle in self.bundles]], data={
        #     'strength': [[bundle['strength'] for bundle in self.bundles]],
        #     'height': [[bundle['height'] for bundle in self.bundles]],
        # })
        return [{
            'index': 0,
            'data': data,
            'type': 'support',
            'colour': 'black',
        },{
            'index': 0,
            'data': pd.DataFrame(index=self.smooth_df.index, data=self.smooth_df.Smooth.values),
            'type': 'line',
            'colour': 'orange',
        },{
            'index': 1,
            'data': pd.DataFrame(index=self.smooth_df.index, data=self.smooth_df.Delta.values),
            'type': 'line',
            'colour': 'orange',
        }]

    # Util functions

    def add_into_bundle(self, bundle, support):
        bundle['supports'].append(support)

    def create_bundle(self, support):
        bundle = {
            'strength': support['strength'],
            'pos': support['pos'],
            'height': support['height'],
            'supports': [support]
        }
        self.bundles.append(bundle)

    def create_support(self, data):
        support = data
        self.supports.append(support)
        # Try to bundle
        added = False
        for bundle in self.bundles:
            if self.within_bundle(bundle, support):
                bundle['supports'].append(support)
                added = True

        # Make new bundle
        if not added:
            self.create_bundle(support)
            # self.calculate_bundle(bundle)  # calculated on construct

        self.supports.append(support)
        return support

    def create_support_at(self, idx, left, right):
        self.create_support({
            'strength': left + right,
            'pos': idx,
            # 'height': self.df.Close[idx]
            'height': self.smooth_df.Smooth[idx]
        })

    def calculate_bundle(self, bundle):
        strength = sum([support['strength'] for support in bundle['supports']])
        if strength < 1 or len(bundle['supports']) < 1:
            self.delete_bundle(bundle)
            return
        height = sum([support['height'] for support in bundle['supports']])
        bundle['strength'] = try_divide(strength, math.sqrt(len(bundle['supports'])))
        bundle['pos'] = bundle['supports'][-1]['pos']
        bundle['height'] = try_divide(height, len(bundle['supports']))

    def delete_bundle(self, bundle):
        self.bundles.remove(bundle)

    def get_bundle(self, support):
        pass

    def get_idx_date(self, idx):
        if idx < 0 or idx > self.idx:
            idx = 0
        return self.df.index[idx]

    def within_bundle(self, bundle, support):
        if abs(bundle['height'] - support['height']) < self.distinguishing_value:
            return True
        return False

    def delete_support(self, bundle, support):
        bundle['supports'].remove(support)

    def delete_decayed(self):
        for bundle in self.bundles:
            for support in bundle['supports']:
                if support['pos'] < max(0, self.idx-self.lookback):
                    self.delete_support(bundle, support)
            self.calculate_bundle(bundle)
        for bundle in self.bundles:
            if bundle['strength'] < 2:
                self.delete_bundle(bundle)

    def decay_all(self):
        """Start decay past lookback/2"""
        pass

    # ===== Comments ======
    # Differences with 'ClassicSupportFinder'
    # Delta-ism
    # Late
    # Strength

    # Idea: Decay then delete when strength too low