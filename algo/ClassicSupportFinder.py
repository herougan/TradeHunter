import math

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
    PEAK, TROUGH = 1, -1
    OTHER_ARGS_DICT = {

    }

    def __init__(self):
        self.decay = math.pow(math.exp(), -self.ARGS_DICT['decay_constant'])
        self.bundles = []
        self.delta_data = []

        past_df = pd.DataFrame()

        last_peak, last_trough, new_peak, new_trough = 0, 0, 0, 0
        last_lookback, last_support, last_delta, delta_flipped = 0, None, 0, False
        idx = 0

    def reset(self):
        self.__init__()

    def support_find(self):
        pass

    # ==== Algo ====

    def next(self, candlestick):
        self.past_data.append(candlestick)

    # ==============

    def get_plotting_instructions(self):
        pass

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
        for support in self.supports:
            support['strength'] = self.decay(support['strength'])

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

    def calculate_bundle_strength(self, bundle, idx):
        strength = 0
        for support in bundle['supports']:
            strength += self.calc_strength(support, idx)
        bundle['strength'] = strength
        return strength

    def try_extend_peak(self, support, idx):
        if support['end'] - support['start']:  # base too long, reset
            support['open'] = False
            return False
        elif support['type'] == self.PEAK and support['height'] < self.past_data.High[idx]:  # base too high
            support['open'] = False
            return False
        elif support['type'] == self.TROUGH and support['height'] > self.past_data.Low[idx]:  # base too low
            support['open'] = False
            return False
        self.calculate_bundle_strength(self.get_bundle(support))
        return True
