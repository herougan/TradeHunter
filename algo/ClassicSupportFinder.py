import math
from datetime import datetime

import pandas as pd

from settings import IVarType


class ClassicSupportFinder:
    # todo do decay first
    # decaying all supports in bundle.
    # however, when extending support, add 1 * decay^(idx-peak)
    # todo apply constants now
    ARGS_DICT = {
        'distinguishing_constant': {
            'default': 0.2,
            'range': [0.1, 1],
            'step': 0.05,
            'comment': 'Factor distinguishing between different bundles. The greater the number,'
                       'the more supports are bundled together. Adjacent distance for bundling '
                       'is directly equal to d_c * stddev (500) (+ flat base of 1 pip)',
            'type': IVarType.CONTINUOUS,
        },
        'decay_constant': {  # Done
            'default': 0,
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
        'width_strength_coefficient': {  # todo unused how much with contributes to strength
            'default': 1,  # a.k.a width decay
            'range': [0.5, 2],
            'step': 0.01,
            'comment': 'Default w=1: base strength contribution = base width * (w=1)^(base) = base. Otherwise, '
                       'strength = base * w^(base)',
            'type': IVarType.CONTINUOUS,
        },
        'clumping_strength_coefficient': {  # Done
            'default': 1,
            'range': [0.2, 2],
            'step': 0.01,
            'comment': 'Affects strength addition: (X+Y)/c. c=1 is default addition. The greater'
                       'the number, the less strength emphasis is on having multiple supports.'
                       'Smaller numbers (<1) enhance the importance of having multiple supports.'
                       'Sum(X_n)=(X_1+...X_N)/c^(N-1). Sum(X)=X/c^0=X, as expected.',
            'type': IVarType.CONTINUOUS,
        },
        'variability_period': {  # todo unused
            'default': 0,
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
        'strength_cutoff': {  # todo unused min strength, otherwise deleted
            'default': 0,  # strength = log(base)
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
        'critical_symmetry': {  # todo unused max(min(l, r) * c, max(l, r))
            'default': 0,
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
        'max_base': {
            'default': 0,
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
        'min_base': {
            'default': 0,
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
        'delta_constant': {
            'default': 0,
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
        'value_type': {
            'default': 'close',  # index 0
            'idx ': 0,  # todo change to default
            'range': ['close', 'open', 'high_low', 'average'],
            'type': IVarType.ENUM
        }
    }
    OTHER_ARGS_DICT = {
        'lookback_period': {
            'default': 20,
        },
    }
    # Constants
    PEAK, TROUGH = 1, -1
    # Other args
    PREPARE_PERIOD = 0
    GREATEST_AGE = 50

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
        self.variability_period = ivar['variability_period']['default']
        self.strength_cutoff = ivar['strength_cutoff']['default']
        self.critical_symmetry = ivar['critical_symmetry']['default']
        self.max_base = ivar['max_base']['default']
        self.min_base = ivar['min_base']['default']
        self.delta_constant = ivar['delta_constant']['default']
        self.width_decay = ivar['width_decay']['default']
        self.bundling_constant = ivar['bundling_constant']['default']
        self.value_type = ivar['value_type']['default']
        self.ivar_check()
        # OTHER ARGS
        self.lookback_period = self.OTHER_ARGS_DICT['lookback_period']['default']

        # == Variables ==

        # Stats
        self.n_supports = []
        self.avg_strength = []

        # Variable arrays
        self.decay = math.pow(math.e, - self.decay_constant)
        self.bundles = []  # supports build up into bundles
        self.supports = []  # handles to the supports themselves
        self.delta_data = []  # -1: delta-descending, 0: within delta, 1: delta-ascending
        self.accum_df = pd.DataFrame()
        self.delta_df = pd.DataFrame()

        # Tracking variables
        self.last_peak, self.last_trough, self.new_peak, self.new_trough = 0, 0, 0, 0
        self.peak, self.trough, self.has_new = 0, 0, False
        self.last_lookback, self.last_support, self.last_delta, self.delta_flipped = 0, None, 0, False
        self.idx = 0

        # Collecting data across time
        self.avg_strength = []
        self.n_bundles = []

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
        self.reset(self.ivar)  # External codes should reset it instead.
        self.test_mode = test_mode

        # == Data Meta ==
        pass

        # == Preparation ==
        self.df = pd.DataFrame()
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
        for i in range(max(0, len(pre_data)-self.lookback_period), len(pre_data)):
            self.pre_next(pre_data[i:i+1])  # df.Close, Open, High, Low
        self.delta_df.index_name = 'date'

    def support_find(self, data):
        """Find supports in data w.r.t current (latest) index"""
        for i in range(len(data)-self.GREATEST_AGE, len(data)):
            pass

    # ==== Algo ====

    def next(self, candlestick):

        # Next
        self.df = self.df.append(candlestick)
        self.idx += 1

        # Note: This algorithm is index agnostic
        # self.supports = []  # temporary
        _max, _min = 0, math.inf
        if len(self.df) < 2:
            return

        # ===== Algorithm ======
        # (1) Compare old[-1] and new candle
        diff = self.df.Close[-2] - self.df.Close[-1]
        self.delta_flipped = False
        if abs(diff) < self.delta_constant:
            self.delta_data.append(0)
        else:
            if diff > 0:  # Past candle is higher than latest candle
                delta_val = -1
            else:
                delta_val = 1
            self.delta_data.append(delta_val)
            # 1 to -1 or -1 to 1. 0s break the chain
            if self.last_delta != 0:
                self.delta_flipped = (self.last_delta != delta_val)
            self.last_delta = delta_val
        # Update delta df
        self.delta_df = self.delta_df.append(pd.DataFrame({
            'delta': self.delta_data[-1]
        }, index=[self.df.index[-1]]))
        if len(self.accum_df > 0):
            self.accum_df = self.accum_df.append(pd.DataFrame({
                'delta': self.delta_data[-1] + self.accum_df.delta[-1]
            }, index=[self.df.index[-1]]))
        else:
            self.accum_df = self.accum_df.append(pd.DataFrame({
                'delta': self.delta_data[-1]
            }, index=[self.df.index[-1]]))

        # (2) Get next peak/trough:
        if self.trough == self.peak:  # Find any peak/trough
            if self.last_delta == 0:
                pass  # ignore and continue
            # Do not create support, but create left base first
            elif self.last_delta == 1:
                self.trough = self.idx - 1
            elif self.last_delta == -1:
                self.peak = self.idx - 1

        elif self.trough > self.peak:  # Find new peak
            if self.delta_flipped:  # Found!
                # 'default' peak properties
                self.peak = self.idx - 1
                left_base = self.peak - self.trough
                start = self.trough
                end = self.idx
                height = self.df.Close[self.peak]
                # Check if supports (previous and current) have min_base
                if left_base < self.min_base//2:  # new left base = old right base
                    # Destroy left support
                    self.delete_support(self.supports[-1])
                    # Do not create new support, past support cannot be extended also
                    self.has_new = False
                else:  # left base > min_base // 2, OK
                    # Try to find true peak (a.k.a delta=0 peak)
                    # todo: 1) check if sorting works 2) check if df.index.get_loc works
                    peaks = self.df[self.trough:self.peak+1][self.df.Close >= height].sort_values(by=['Close'],
                                                                                                  ascending=False)
                    # If no alt. peaks, loop will terminate at df.Close == height
                    for i, peak in peaks.iterrows():
                        # Check if alt. left_base is of minimum length,
                        _peak = self.df.index.get_loc(peak['index'])
                        _left_base = _peak - self.trough
                        if _left_base >= self.min_base//2:  # Add as new peak
                            # Adjust previous support's base
                            self.update_support(self.supports[-1], 'end', _peak)
                            # Register peak
                            height = peak['Close']
                            self.peak = _peak
                            self.create_support(self.peak, start, end, height, self.PEAK)
                            self.has_new = True
                            break
                        else:  # otherwise continue
                            continue
            else:
                if self.has_new:
                    if self.try_extend(self.supports[-1]):
                        pass  # if extension (to the right) successful, do nothing
                    else:
                        # Reset status to 'neutral'
                        self.has_new = False
                        # self.trough = self.peak = self.idx  # no need to reset completely
                else:  # No older support to extend
                    pass  # a.k.a do nothing, just continue

        elif self.peak > self.trough:  # Find new trough
            if self.delta_flipped:
                self.trough = self.idx - 1
                left_base = self.trough - self.trough
                start = self.trough
                end = self.idx
                depth = self.dc.Close[self.trough]
                # Check if supports have min_base
                if left_base < self.min_base // 2:
                    # Destroy left support
                    self.delete_support(self.supports[-1])
                    # Past support cannot be extended
                    self.has_new = False
                else:
                    # Try to find true trough
                    troughs = self.df[self.peak:self.trough+1][self.df.Close <= depth].sort_values(by=['Close'],
                                                                                                   ascending=True)
                    for i, trough in troughs.iterrows():
                        # Check if alt. trough has min_base
                        _trough = self.df.index.get_loc(trough['index'])
                        _left_base = _trough - self.peak
                        if _left_base >= self.min_base // 2:
                            # Adjust previous support's base
                            self.update_support(self.supports[-1], 'end', _trough)
                            # Register trough
                            depth = trough['Close']
                            self.trough = _trough
                            self.create_support(self.trough, start, end, depth, self.TROUGH)
                            self.has_new = True
                            break
                        else:
                            continue
            else:
                if self.has_new:
                    if self.try_extend(self.supports[-1]):
                        pass
                    else:
                        self.has_new = False
                else:
                    pass

        # ===== Bundling =====

        # Bundling is automatic when creating supports

        # Decay bundles
        self.decay_all()

        # ===== Return function =====

        # None in this case
        print(self.bundles)

    def pre_next(self, candlestick):
        self.df = self.df.append(candlestick)
        # Pre_data supports will be ignored! If that is not desired, do not include pre_data
        self.delta_data.append(0)
        self.delta_df = self.delta_df.append(pd.DataFrame({
            'delta': 0
        }, index=[self.df.index[-1]]))
        self.accum_df = self.accum_df.append(pd.DataFrame({
            'delta': 0
        }, index=[self.df.index[-1]]))

    # ==============

    def get_supports(self):
        return self.bundles

    def get_value(self, idx, peak_type=TROUGH):
        if self.value_type == 'close':
            return self.df.Close[idx]
        if self.value_type == 'high_low':
            if peak_type == self.TROUGH:
                return self.df.Low[idx]
            elif peak_type == self.PEAK:
                return self.df.High[idx]
        if self.value_type == 'open':
            return self.df.Open[idx]
        if self.value_type == 'average':
            return self.df.Close[idx]  #?
        return None

    def get_sort_height(self, idx, peak_type=TROUGH):
        """Sorts based on value type"""
        if self.value_type == 'close':
            return self.df.Close[idx]
        if self.value_type == 'high_low':
            if peak_type == self.TROUGH:
                return self.df.Low[idx]
            elif peak_type == self.PEAK:
                return self.df.High[idx]
        if self.value_type == 'open':
            return self.df.Open[idx]
        if self.value_type == 'average':
            return self.df.Close[idx]  #?
        pass

    def get_resistances(self):
        """Only get support ceilings"""
        last = self.df.Close[-1]
        _bundles = []
        for bundle in self.bundles:
            if bundle['height'] > last:
                _bundles.append(bundle)
        return _bundles

    def get_resistance_supports(self):
        """Only get support floors"""
        last = self.df.Close[-1]
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
        return [{
            'index': 0,
            'data': data,
            'type': 'support',
            'colour': 'black',
        }, {
            'index': 1,
            'data': self.delta_df.copy(),
            'type': 'line',
            'colour': 'black',
        }, {
            'index': 2,
            'data': self.accum_df.copy(),
            'type': 'line',
            'colour': 'black',
        }]

    # Util functions

    def bundle_add(self, bundle, support):
        pass

    def bundle_decay(self, bundle):
        for support in bundle['supports']:
            self.decay_support(support)
        self.calculate_bundle_strength(bundle)

    def calc_strength(self, support, idx):
        start = support['start']
        end = support['end']
        peak = support['peak']
        dist = idx - peak
        length = end - start
        return math.pow(self.ARGS_DICT['decay'], dist) * math.log(length, 2)

    def calculate_bundle_strength(self, bundle):
        strength = 0
        for support in bundle['supports']:
            # strength += self.calc_strength(support, idx)
            strength += support['strength']
        bundle['strength'] = strength / math.pow(self.clumping_strength, len(bundle['supports'])-1)
        return strength

    def create_bundle(self, support):
        """Create new bundle around support."""
        bundle = {
            'supports': [support]
        }
        self.bundles.append(bundle)
        return bundle

    def combine_bundles(self):
        """Use closeness/2 metric. Combine from top to bottom."""
        pass

    def create_support(self, peak, start, end, height, type):
        """Create support within bounds end and start, at peak, with value of height.
        Types are TROUGH or PEAK. 'open' is whether the support is available for
        base extension (to increase its strength). Then, add it into closest bundle if possible.
        Otherwise, it becomes a bundle of its own."""
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

        self.supports.append(support)
        return support

    def decay(self, strength):
        return strength * self.ARGS_DICT['decay']

    def decay_all(self):
        for bundle in self.bundles:
            self.bundle_decay(bundle)

    def decay_by(self, strength, length):
        return strength * math.pow(self.ARGS_DICT['decay'], length)

    def decay_support(self, support):
        support['strength'] = self.decay(support['strength'])

    def delete_support(self, _support):
        for bundle in self.bundles:
            for support in bundle.supports:
                if support == _support:
                    bundle.supports.remove(support)
        self.supports.remove(support)

    def get_bundle(self, support):
        for bundle in self.bundles:
            if support in bundle['supports']:
                return bundle
        return None

    def get_idx_date(self, idx):
        return self.df.index[idx]

    def try_extend(self, support):
        """Extend length of peak. This affects its strength. Upon extension, recalculate
        decay effects."""
        if support['end'] - support['start']:  # base too long, reset
            support['open'] = False
            return False
        elif support['type'] == self.PEAK and support['height'] < self.past_data.High[self.idx]:  # base too high
            support['open'] = False
            return False
        elif support['type'] == self.TROUGH and support['height'] > self.past_data.Low[self.idx]:  # base too low
            support['open'] = False
            return False

        # Calculate new strength
        support['end'] += 1
        support['strength'] += 1 * math.pow(self.decay, self.idx - support['peak'])
        self.calculate_bundle_strength(self.get_bundle(support))
        return True

    def update_support(self, support, arg, val):
        support[arg] = val
        self.calculate_bundle_strength(self.get_bundle(support))

    def within_bundle(self, bundle, support):
        if abs(bundle['height'] - support['height']) < self.distinguishing_constant:
            return True
        return False