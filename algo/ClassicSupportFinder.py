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
    PREPARE_PERIOD = 0

    def __init__(self, ivar=ARGS_DICT):

        # == Main Args ==
        self.time = None
        self.started = None
        self.df = None
        if ivar:
            self.ivar = ivar
        else:
            self.ivar = self.ARGS_DICT

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

        # Stats
        self.n_supports = []
        self.avg_strength = []

        # Df
        self.past_df = pd.DataFrame()

        # Variable arrays
        self.decay = math.pow(math.e, - self.decay_constant)
        self.bundles = []
        self.delta_data = []  # -1: delta-descending, 0: within delta, 1: delta-ascending

        # Tracking variables
        self.last_peak, self.last_trough, self.new_peak, self.new_trough = 0, 0, 0, 0
        self.last_lookback, self.last_support, self.last_delta, self.delta_flipped = 0, None, 0, False
        self.idx = 0

        # == Testing ==
        self.test_mode = None

    def ivar_check(self):
        """Ensures the IVar variables are 1) within range and 2) in the correct format."""
        for key in self.ivar.keys():
            arg = self.ivar[key]

    def reset(self, ivar):
        self.__init__(ivar)

    def start(self, meta_or_param, pre_data: pd.DataFrame, test_mode=False):
        """?Start"""
        self.reset(self.ivar)
        self.test_mode = test_mode

        # == Data Meta ==
        pass

        # == Preparation ==
        self.df = pre_data
        self.idx += len(pre_data) - 1

        # == Statistical Data ==
        self.n_supports = []
        self.avg_strength = []
        for i in range(len(pre_data)):
            self.n_supports.append(0)
            self.avg_strength.append(0)

        # == Status ==
        self.started = True
        self.time = datetime.now()

        # Setup consequences of pre_data
        for datum in pre_data:
            self.pre_next(datum)  # df.Close, Open, High, Low

    def support_find(self, data):
        """Find supports in data w.r.t current (latest) index"""
        pass

    # ==== Algo ====

    def next(self, candlestick):

        # Next
        self.df.append(candlestick)
        self.idx += 1

        # Note: This algorithm is index agnostic
        new_supports = []
        _max, _min = 0, math.inf
        if len(self.past_df) < 2:
            return

        # ===== Algorithm ======
        # 1) Compare old[-1] and new candle
        diff = self.past_df.Close[-2] - self.past_df.Close[-1]
        delta_flipped = False
        if abs(diff) < self.delta_constant:
            self.delta_data.append(0)
        else:
            if diff > 0:  # Past candle is higher than latest candle
                delta_val = -1
            else:
                delta_val = 1
            self.delta_data.append(delta_val)
            # 1 to -1 or -1 to 1. 0s break the chain
            delta_flipped = (self.last_delta != delta_val)
            self.last_delta = delta_val

        # 2) Get next peak/trough, 3 modes: Find next trough, find next peak, find next any
        if self.last_peak > self.last_trough:  # look for next trough
            if delta_flipped:  # found
                height = min(self.past_df.Close[self.supports[-1]['start'] + 1:self.idx])  # Value
                peak = self.past_df[self.past_df.Close[self.supports[-1]['end']:self.idx] == height][-1] or 0  # Where
                start = self.last_support[-1]['end']
                end = self.idx
                self.create_support(peak, start, end, height, self.TROUGH)
            else:  # try extend last peak
                if self.try_extend_peak(self.supports[-1], self.idx):
                    pass
                else:  # failed to extend, reset to no last_support status
                    self.last_peak, self.last_trough = self.idx, self.idx
                # supports[-1]['end'] += 1
                # # if base too long, reset
                # if supports[-1]['end'] - supports[-1]['start']:
                #     last_peak, last_trough = idx, idx
        elif self.last_peak < self.last_trough:  # look for next peak
            if delta_flipped:  # found
                height = max(self.past_df.Close[self.supports[-1]['start'] + 1:self.idx])
                peak = self.past_df[self.past_df.Close[self.supports[-1]['end']:self.idx] == height][-1] or 0
                start = self.last_support[-1]['end']
                end = self.idx
                self.create_support(peak, start, end, height, self.PEAK)
            else:  # try extend last trough
                if self.try_extend_peak(self.supports[-1], self.idx):
                    pass
                else:  # failed to extend, reset to no last_support status
                    self.last_peak, self.last_trough = self.idx, self.idx
        else:  # last_peak = last_trough (only possible if just started or reset)
            if self.delta_data[-1] == -1:  # found potential trough
                self.last_trough = self.idx
            elif self.delta_data[-1] == 1:  # potential peak
                self.last_peak = self.idx

            # ===== Bundling =====

            # Already done in algorithm part

            # ===== Return function =====

            # None in this case
            print(self.bundles)

    def pre_next(self, candlestick):
        # Pre_data supports will be ignored! If that is not desired, do not include pre_data
        pass

    # ==============

    def get_supports(self):
        return self.bundles

    def get_resistances(self):
        """Only get support ceilings"""
        last = self.past_df.Close[-1]
        _bundles = []
        for bundle in self.bundles:
            if bundle['height'] > last:
                _bundles.append(bundle)
        return _bundles

    def get_resistance_supports(self):
        """Only get support floors"""
        last = self.past_df.Close[-1]
        _bundles = []
        for bundle in self.bundles:
            if bundle['height'] < last:
                _bundles.append(bundle)
        return _bundles

    def get_instructions(self):
        # Lines should get lighter the weaker they are
        # Data should be pd.DataFrame format with index and 'height'/value
        data = pd.DataFrame(index=[[self.get_idx_date(bundle['peak']) for bundle in self.bundles]], data={
            'strength': [[bundle['strength'] for bundle in self.bundles]],
            'height': [[bundle['height'] for bundle in self.bundles]],
        })
        print(self.bundles)  # todo
        return [{
            'index': 0,
            'data': data,
            'type': 'line',
            'colour': 'black',
        }]

    # Util functions

    def get_idx_date(self, idx):
        return self.df.index[idx]

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
