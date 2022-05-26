import math
from datetime import datetime

import pandas as pd

from settings import IVarType
from util.langUtil import try_divide


class ClassicSupportFinder:
    ARGS_DICT = {
        'distinguishing_constant': {
            'default': 5,
            'range': [1, 25],
            'step': 0.05,
            'comment': 'Factor distinguishing between different bundles. The greater the number,'
                       'the more supports are bundled together. Adjacent distance for bundling '
                       'is directly equal to d_c * stddev (variability) e.g. stddev(200) (+ flat base of 1 pip)'
                       'Acts as a multiplier to stddev. If stddev type is flat, distinguishing amount is'
                       'd_c * 100 pips'
                       'UPDATE: d_c * pips only.',
            'type': IVarType.CONTINUOUS,
        },
        'decay_constant': {
            'default': 0.9,
            'range': [0.1, 1],
            'step': 0.05,
            'type': IVarType.CONTINUOUS,
        },
        'width_coefficient': {
            'default': 1,  # a.k.a width decay
            'range': [0, 2],
            'step': 0.01,
            'comment': 'Strength = width_coefficient * base + 1. At 0, Strength = 1 at all times. At greater numbers,'
                       'base width greatly increases strength.',
            'type': IVarType.CONTINUOUS,
        },
        'clumping_coefficient': {  # Done
            'default': 1,
            'range': [0.2, 2],
            'step': 0.01,
            'comment': 'Affects strength addition: (X+Y)/c. c=1 is default addition. The greater'
                       'the number, the less strength emphasis is on having multiple supports.'
                       'Smaller numbers (<1) enhance the importance of having multiple supports.'
                       'Sum(X_n)=(X_1+...X_N)/c^(N-1). Sum(X)=X/c^0=X, as expected.',
            'type': IVarType.CONTINUOUS,
        },
        'variability_period': {
            'default': 500,
            'range': [200, 700],
            'step': 50,
            'comment': 'Used in determining bundling with distinguishing_constant. +-Stddev lines formed'
                       'from stddev(variability_period) calculation.',
            'type': IVarType.CONTINUOUS,
        },
        'symmetry_coefficient': {
            'default': 0,
            'range': [0, 1],
            'step': 0.1,
            'type': IVarType.CONTINUOUS,
            'comment': 'This greater this coefficient, the more it demands the left and right bases'
                       'of a support to be symmetrical. At 1, both sides can only be as wide'
                       'as their shortest side. At 0, both sides are as wide as their longest'
                       'side. Formula: min(min(l, r) * 1/c, max(l, r)) + max(l, r). Where the left'
                       'term represents the shorter side, compensated, and the right side is the'
                       'longer side. If c = 0, min(min, max) will be assumed to be max(l, r).',
        },
        'max_base': {
            'default': 12,
            'range': [5, 50],
            'step': 1,
            'type': IVarType.DISCRETE,
        },
        'min_base': {
            'default': 3,
            'range': [1, 5],
            'step': 1,
            'type': IVarType.DISCRETE,
        },
        'delta_constant': {
            'default': 3,
            'range': [1, 10],
            'step': 1,
            'type': IVarType.CONTINUOUS,
            'comment': 'This is the main part of the algorithm that sets it apart from'
                       'the strictly inc. dec. peak algorithm. It allows for a \'give\' of '
                       'delta before considering it an increase or decrease. Peaks are defined not by'
                       'strictly decreasing numbers on both sides but instead, a looser requirement of'
                       'delta-decreasing numbers on both sides. If the values on the side increase but'
                       'within delta, it does not count as breaking the peak.'
                       'delta_constant is in units of pips.'
        },
        'smoothing_period': {
            'default': 6,
            'range': [3, 20],
            'step': 1,
            'type': IVarType.DISCRETE,
            'comment': 'Conducting the same algorithm on a smoothed surface may generate supports missed'
                       'when operating on the candlestick data. These supports, from here on forwards called'
                       'smooth supports will corroborate the supports. Only when these supports cannot be'
                       'bundled with any existing bundles will they form their own bundle. Note. Peaks and troughs'
                       'are detected with delta=0, strict peaks/troughs. min_base will be of the same size as'
                       'the normal min_base.'
        },
        # Unused
        # 'value_type': {  # Not used at the moment
        #     'default': 'close',  # index 0
        #     'idx ': 0,
        #     'range': ['close', 'open', 'high_low', 'average'],
        #     'type': IVarType.ENUM
        # },
        # 'variability_type': {
        #     'default': 'flat',
        #     'idx': 1,
        #     'range': ['stddev', 'flat'],
        #     'type': IVarType.ENUM
        # }
    }
    OTHER_ARGS_DICT = {
        'lookback_period': {
            'default': 20,
        },
        'strength_cutoff': {
            'default': 0,  # strength = log(base)
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
        'date_cutoff': {
            'default': 25,  # strength = log(base)
            'range': [],
            'step': 0,
            'type': IVarType.CONTINUOUS,
        },
    }
    # Constants
    PEAK, TROUGH = 1, -1
    # Other args
    PREPARE_PERIOD = 0

    def __init__(self, ivar=None):

        # == Main Args ==
        if ivar is None:
            ivar = self.ARGS_DICT
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
        self.symmetry_coefficient = ivar['symmetry_coefficient']['default']
        self.max_base = ivar['max_base']['default']
        self.min_base = ivar['min_base']['default']
        self.delta_constant = ivar['delta_constant']['default']
        self.delta_value = self.delta_constant * 0.0001
        self.width_coefficient = ivar['width_coefficient']['default']
        self.clumping_strength = ivar['clumping_coefficient']['default']
        # self.value_type = ivar['value_type']['default']
        # self.variability_type = ivar['variability_type']['default']
        self.ivar_check()
        # OTHER ARGS
        self.lookback_period = self.OTHER_ARGS_DICT['lookback_period']['default']
        self.strength_cutoff = self.OTHER_ARGS_DICT['strength_cutoff']['default']
        self.date_cutoff = self.OTHER_ARGS_DICT['date_cutoff']['default']

        # Constants
        self.pip = 0.0001
        self.min_left = self.min_base //2

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

        # Indicators
        self.stdev = []

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
        for i in range(len(data)-self.date_cutoff, len(data)):
            pass

    def set_pip_value(self, pip):
        self.pip = pip

    # ==== Algo ====

    def next(self, candlestick):

        # Next
        # self.df = self.df.append(candlestick)
        self.df = pd.concat([self.df, candlestick])
        self.idx += 1

        # Note: This algorithm is index agnostic
        # self.supports = []  # temporary
        _max, _min = 0, math.inf
        if len(self.df) < 2:
            return
        self.build_indicators()

        # ===== Algorithm ======
        # (1) Compare old[-1] and new candle
        diff = self.df.Close[-2] - self.df.Close[-1]
        self.delta_flipped = False
        if abs(diff) < self.delta_value:
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
        self.delta_df = pd.concat([self.delta_df, pd.DataFrame({
            'delta': self.delta_data[-1]
        }, index=[self.df.index[-1]])])
        if len(self.accum_df > 0):
            self.accum_df = pd.concat([self.accum_df, pd.DataFrame({
                'delta': self.delta_data[-1] + self.accum_df.delta[-1]
            }, index=[self.df.index[-1]])])
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
                    if self.has_new:
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
                        _peak = self.df.index.get_loc(i)
                        _left_base = _peak - self.trough
                        if _left_base >= self.min_base//2:  # Add as new peak
                            # Adjust previous support's base
                            # self.update_support(self.supports[-1], 'end', _peak)  # no need to. auto extended!
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
                else:  # No older support to extend. Old trough and peak cannot be further than min_base/2 away
                    # Last support was trough. Only reset trough.
                    self.trough = max(self.trough, self.idx-self.min_left)  # reset

        elif self.peak > self.trough:  # Find new trough
            if self.delta_flipped:
                self.trough = self.idx - 1
                left_base = self.trough - self.peak
                start = self.peak
                end = self.idx
                depth = self.df.Close[self.trough]
                # Check if supports have min_base
                if left_base < self.min_base // 2:
                    # Destroy left support
                    if self.has_new:
                        self.delete_support(self.supports[-1])
                    # Past support cannot be extended
                    self.has_new = False
                else:
                    # Try to find true trough
                    troughs = self.df[self.peak:self.trough+1][self.df.Close <= depth].sort_values(by=['Close'],
                                                                                                   ascending=True)
                    for i, trough in troughs.iterrows():
                        # Check if alt. trough has min_base
                        _trough = self.df.index.get_loc(i)
                        _left_base = _trough - self.peak
                        if _left_base >= self.min_base // 2:
                            # Adjust previous support's base
                            # self.update_support(self.supports[-1], 'end', _trough)
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
                else:  # Reset peak only (Searching for trough)
                    self.peak = max(self.peak, self.idx-self.min_left)

        # ===== Bundling =====

        # Bundling is automatic when creating supports

        # Decay bundles
        self.decay_all()

        # ===== Return function =====

        # None in this case
        # print(self.bundles)

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
        data = pd.DataFrame(index=[self.get_idx_date(bundle['peak']) for bundle in self.bundles], data={
            'strength': [bundle['strength'] for bundle in self.bundles],
            'height': [bundle['height'] for bundle in self.bundles],
            'peak': [bundle['peak'] for bundle in self.bundles],
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
        },
        #     {
        #     'index': 0,
        #     'data': smooth_data,
        #     'type': 'support',
        #     'colour': 'red',
        # },
            {
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

    def build_indicators(self):
        # self.stdev = talib.STDDEV(self.df, self.variability_period)
        pass

    # Util functions

    def bundle_add(self, _bundle, support):
        for bundle in self.bundles:
            if bundle == _bundle:
                bundle['supports'].append(support)

    def calc_strength(self, support):
        """Takes width and time decay into account. Recalculates the current strength value of a support."""
        strength = self.calc_raw_strength(support)
        dist = self.idx - support['peak']
        support['strength'] = math.pow(self.decay_constant, dist) * strength
        return support['strength']

    def calc_raw_strength(self, support):
        left = support['end'] - support['peak']
        right = support['peak'] - support['start']
        # Symmetry considerations
        base = min(min(left, right) * try_divide(1, self.symmetry_coefficient), max(left, right))\
               + max(left, right)
        if math.isnan(base):  # Occurs only on 0 * inf
            base = min(left, right)
        # Max base consideration
        base = min(base, self.max_base)
        # Width contribution consideration
        return base * self.width_coefficient + 1

    def calculate_bundle(self, bundle):
        strength = 0
        peak = 0  # (position)
        height = 0
        for support in bundle['supports']:
            # strength += self.calc_strength(support, idx)
            strength += support['strength']
            # peak += support['strength'] * support['peak']
            height += support['strength'] * support['height']
            peak = support['peak']

        bundle['strength'] = try_divide(strength, math.pow(self.clumping_strength, len(bundle['supports'])-1))
        # bundle['peak'] = try_divide(bundle['peak'], len(bundle['supports']) * strength)
        bundle['peak'] = peak  # Last added peak
        bundle['height'] = try_divide(height, len(bundle['supports']) * strength)

        return strength

    def combine_bundles(self):
        """Use closeness/2 metric. Combine from top to bottom."""
        pass

    def create_bundle(self, support):
        """Create new bundle around support."""
        bundle = {
            'strength': 0,
            'peak': 0,
            'height': 0,
            'supports': [support]
        }
        self.bundles.append(bundle)
        # print(F'Creating {bundle}')
        return bundle

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
        support.update({
            'strength': self.calc_raw_strength(support)
        })

        # Add into some bundle
        added = False
        for bundle in self.bundles:  # todo created with NaN strength
            if self.within_bundle(bundle, support):
                self.bundle_add(bundle, support)
                # print(F'Creating {support} in {bundle}')
                self.calculate_bundle(bundle)
                added = True
                break
        if not added:
            bundle = self.create_bundle(support)
            # print(F'Creating {support} in new {bundle}')
            self.calculate_bundle(bundle)

        self.supports.append(support)
        return support

    def decay_all(self):
        for bundle in self.bundles:
            self.decay_bundle(bundle)
        self.delete_decayed()

    def decay_bundle(self, bundle):
        for support in bundle['supports']:
            self.decay_support(support)
        self.calculate_bundle(bundle)

    def decay_by(self, strength, length):
        return strength * math.pow(self.ARGS_DICT['decay'], length)

    def decay_support(self, support):
        support['strength'] = support['strength'] * self.decay_constant

    def delete_support(self, _support):
        for bundle in self.bundles:
            for support in bundle['supports']:
                if support == _support:
                    # print(F'Deleting {_support} from {bundle}')
                    bundle['supports'].remove(support)
                    # If bundle has no supports, remove it
                    if len(bundle['supports']) == 0:
                        self.bundles.remove(bundle)
                    self.supports.remove(support)

    def delete_decayed(self):
        for bundle in self.bundles:
            if bundle['strength'] < self.strength_cutoff:
                self.bundles.remove(bundle)

    def get_bundle(self, support):
        for bundle in self.bundles:
            if support in bundle['supports']:
                return bundle
        return None

    def get_idx_date(self, idx):
        if idx < 0 or idx > self.idx:
            idx = 0
        return self.df.index[idx]

    def try_extend(self, support):
        """Extend length of peak. This affects its strength. Upon extension, recalculate
        decay effects."""
        if support['end'] - support['start'] > self.max_base:  # base too long, reset
            return False
        elif support['type'] == self.PEAK and support['height'] < self.df.High[self.idx]:  # new base too high
            return False
        elif support['type'] == self.TROUGH and support['height'] > self.df.Low[self.idx]:  # new base too low
            return False

        # Calculate new strength
        support['end'] += 1
        # support['strength'] += 1 * math.pow(self.decay, self.idx - support['peak']) * self.width_coefficient
        support['strength'] = self.calc_strength(support)
        # Recalculate bundle strength
        self.calculate_bundle(self.get_bundle(support))
        return True

    def update_support(self, support, arg, val):
        support[arg] = val
        self.calculate_bundle(self.get_bundle(support))

    def within_bundle(self, bundle, support):
        if abs(bundle['height'] - support['height']) < self.distinguishing_constant * self.pip:
            return True
        return False