import math
from datetime import datetime

import pandas as pd


class ClassicSupportFinder:
    ARGS_DICT = {
        'distinguishing_constant': {
            'default': 0,
            'range': [],
            'step': 0,
        },
        'decay_constant': {
            'default': 0,
            'range': [],
            'step': 0,
            'step_type': 'continuous'
        },
        'lookback_period': {
            'default': 0,
            'range': [],
            'step': 0,
            'step_type': 'discrete'
        },
        'variability_period': {
            'default': 0,
            'range': [],
            'step': 0
        },
        'strength_cutoff': {
            'default': 0,
            'range': [],
            'step': 0
        },
        'critical_symmetry': {
            'default': 0,
            'range': [],
            'step': 0
        },
        'max_base': {
            'default': 0,
            'range': [],
            'step': 0
        },
        'min_base': {
            'default': 0,
            'range': [],
            'step': 0
        },
        'delta_constant': {
            'default': 0,
            'range': [],
            'step': 0
        },
        'width_decay': {
            'default': 0,
            'range': [],
            'step': 0
        },
        'bundling_constant': {
            'default': 0,
            'range': [],
            'step': 0
        },
    }
    # Constants
    PEAK, TROUGH = 1, -1
    # Other args
    PREPARE_PERIOD = 5

    def __init__(self, ivar=ARGS_DICT):

        # == Main Args ==
        self.ivar = ivar

        # ARGS_DICT
        self.distinguishing_constant = ivar['distinguishing_constant']['default']
        self.decay_constant = ivar['decay_constant']['default']
        self.lookback_period = ivar['lookback_period']['default']
        self.variability_period = ivar['variability_period']['default']
        self.strength_cutoff = ivar['strength_cutoff']['default']
        self.critical_symmetry = ivar['critical_symmetry']['default']
        self.max_base = ivar['max_base']['default']
        self.min_base = ivar['min_base']['default']
        self.delta_constant = ivar['delta_constant']['default']
        self.width_decay = ivar['width_decay']['default']
        self.bundling_constant = ivar['bundling_constant']['default']
        self.ivar_check()

        # == Variables ==
        # Variable arrays
        self.decay = math.pow(math.exp(), - self.decay_constant)
        self.bundles = []
        self.delta_data = []

        # Stats
        self.n_supports = []
        self.avg_strength = []

        # Continuing arrays
        past_df = pd.DataFrame()

        # Constants
        last_peak, last_trough, new_peak, new_trough = 0, 0, 0, 0
        last_lookback, last_support, last_delta, delta_flipped = 0, None, 0, False
        idx = 0

        # == Testing ==
        self.test_mode = None

    def ivar_check(self):
        """Ensures the IVar variables are 1) within range and 2) in the correct format."""
        for key in self.ivar.keys():
            arg = self.ivar[key]

    def reset(self, ivar):
        self.__init__(ivar)

    def start(self, meta_or_param, pre_data: pd.DataFrame, test_mode=False):
        """"""
        self.reset()
        self.test_mode = test_mode

        # == Data Meta ==
        pass

        # == Preparation ==
        self.df = pre_data
        self.build_indicators()

        # == Statistical Data ==
        self.n_supports = []
        self.avg_strength = []
        for i in range(len(pre_data)):
            self.n_supports.append(0)
            self.avg_strength.append(0)

        # == Status ==
        self.started = True
        self.time = datetime.now()

    def support_find(self, data):
        """Find supports in data w.r.t current (latest) index"""
        pass

    # ==== Algo ====

    def next(self, candlestick):
        self.past_data.append(candlestick)

    def pre_next(self, candlestick):
        # Go through pre-dataframe and construct supports, ignore data that is too old (no decay)
        pass

    # ==============

    def get_instructions(self):
        return []

    # Util functions

    def calc_strength(self, support, idx):
        start = support['start']
        end = support['end']
        peak = support['peak']
        dist = idx - peak
        length = end - start
        return math.pow(self.ARGS_DICT['decay'], dist) * math.log(length, 2)

    def decay(self, strength):
        return strength * self.ARGS_DICT['decay']

    def decay_by(self, strength, length):
        return strength * math.pow(self.ARGS_DICT['decay'], length)

    def within_bundle(self, bundle, support):
        if abs(bundle['height'] - support['height']) < self.ARGS_DICT['bundling_constant']:
            return True
        return False

    def bundle_add(self, bundle, support):
        pass

    def bundle_decay(self, bundle):
        for support in bundle.supports:
            support['strength'] = self.decay(support['strength'])
        self.calculate_bundle_strength(bundle)

    def create_bundle(self, support):
        bundle = {
            'supports': [support]
        }
        self.bundles.append(bundle)
        return bundle

    def create_support(self, peak, start, end, height, type):
        support = {
            'peak': peak,
            'start': start,
            'end': end,
            'height': height,
            'type': type,
            'open': True,
        }

        # Add into some bundle
        added = False
        for bundle in self.bundles:
            if self.within_bundle(bundle, support):
                self.bundle_add(bundle, support)
                added = True
                break

        if not added:
            self.create_bundle(support)

        last_support = support
        return support

    def get_bundle(self, support):
        for bundle in self.bundles:
            if support in bundle['supports']:
                return bundle
        return None

    def calculate_bundle_strength(self, bundle):
        strength = 0
        for support in bundle['supports']:
            # strength += self.calc_strength(support, idx)
            strength += support['strength']
        bundle['strength'] = strength
        return strength

    def try_extend_peak(self, support, idx):
        """Extend length of peak. This affects its strength. Upon extension, recalculate
        decay effects."""
        if support['end'] - support['start']:  # base too long, reset
            support['open'] = False
            return False
        elif support['type'] == self.PEAK and support['height'] < self.past_data.High[idx]:  # base too high
            support['open'] = False
            return False
        elif support['type'] == self.TROUGH and support['height'] > self.past_data.Low[idx]:  # base too low
            support['open'] = False
            return False
        # Calculate decay effect
        # todo
        self.calculate_bundle_strength(self.get_bundle(support))
        return True
