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
        for i in range(0, len(pre_data)):
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
            if self.last_delta != 0:
                delta_flipped = (self.last_delta != delta_val)
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

        # (2) Get next peak/trough, 3 modes: Find next trough, find next peak, find next any
        if self.last_peak > self.last_trough:  # look for next trough
            if delta_flipped:  # found
                height = min(self.df.Close[self.supports[-1]['start'] + 1:self.idx])  # Value
                peak = self.df[self.df.Close[self.supports[-1]['end']:self.idx] == height][-1] or 0  # Where
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
                height = max(self.df.Close[self.supports[-1]['start'] + 1:self.idx])
                peak = self.df[self.df.Close[self.supports[-1]['end']:self.idx] == height][-1] or 0
                start = self.last_support[-1]['end']
                end = self.idx
                self.create_support(peak, start, end, height, self.PEAK)
            else:  # try extend last trough
                if self.try_extend_peak(self.supports[-1], self.idx):
                    pass
                else:  # failed to extend, reset to no last_support status
                    self.last_peak, self.last_trough = self.idx, self.idx
        elif self.last_peak == self.last_trough:  # last_peak = last_trough (only possible if just started or reset)
            if self.delta_data[-1] == -1:  # found potential trough
                self.last_trough = self.idx  # todo not true!
                # plus not support created
            elif self.delta_data[-1] == 1:  # potential peak
                self.last_peak = self.idx
        # rewrite thoughts here...
            # if 1 or -1, peak or trough is the index behind it!
            # additional condition: if last non-zero delta has value or not
                # if starting from 0: nope,
                    # trough = peak = 0
                # if peak extended too long, (descending): delta will be -1
                    # peak confirmed, (older) trough < peak -> seeking 1 for new trough
                # if peak was too short (now is new_trough): delta will be the 1
                    # peak deleted, peak marker stays. (newer) trough > peak -> seeking new peak -1
                # peak is within limits
                    # peak confirmed
            # try_extend -> pass within -1
            #            -> fail within 0, extend once, but return false so that no more extend
            # When discovering a support & confirming,
                # base might include alternative max/min point
                # left_base can start at most at where last_trough/peak are:
                    # in the case of neutral status, peak=trough=where peak/trough stops
                # when moving through each alternative max/min point, check if 'base' condition
                        # Objective height vs delta height
                    # is fufilled. otherwise, move to next point!
                    # NOTE: if original min/max point does not fulfill min_base (left) conditions
                    # then other points will not work anyway (they will be more left)
                    # If peak confirm created -> look for other valid min/max points
                        # guaranteed to terminate
            # if base destroyed (-1), ok for future (1) since (0) current base was created
            # if base too extended (-1), there hasnt yet been a (0) current. so either
                # part of the previous base is included in the new delta calculation
                # the latest point which made the base extend too far (depends on if last point inc/not inc)
                # or only from next point onwards:
                # find some delta. if none non-zero, set as 0. delta_flipped cannot be anything. until 1, -1 or -1, 1
                # if 1 or -1, define the last 1 or -1 as (cannot be 1 and -1) the new peak/trough depending
                    # do not create a support for this.
                    # do not try to extend. (if new_support:... support[-1]...)
        # this may be too intensive if done on a per-tick basis

        # (2) Get next peak/trough:
        if self.trough == self.peak:  # Find any peak/trough
            if self.last_delta == 0:
                pass  # ignore and continue
            # Do not create support, but create left base first
            elif self.last_delta == 1:
                self.trough = self.idx - 1
            elif self.last_delta == -1:
                self.peak = self.idx - 1
        if self.trough > self.peak:  # Find new peak
            if self.delta_flipped:  # Found!
                # Peak properties
                self.peak = self.idx - 1
                left_base = self.peak - self.trough
                # Handle left support
                if self.has_new:  # If there is a support on the left
                    if left_base < self.min_base//2:  # new left_base = prev old_base
                        # Destroy support (self.supports, self.bundles)
                        pass
                    else:  # OK
                        pass
                # Register peak
                self.create_support()  # todo
                new_support = True
            else:
                if self.has_new:
                    if self.try_extend(self.supports[-1]):
                        pass
                    else:
                        # Reset status to 'neutral'
                        self.has_new = False
                        self.trough = self.peak = self.idx
                else:  # No older support to extend
                    pass
        if self.peak > self.trough:  # Find new trough
            self.has_new = True
        # last_trough > last_peak: # find peak
            # if delta_flipped: peak found
            #   check old trough, if new_support:
                #   if too short (right):
                #       destroy support in supports AND bundles
                #   elif OK:
                #       do nothing
            #   register new peak regardless (last_trough=old_trough):
            #       peak = idx - 1
            #       left_base = peak - trough
            #       right_base = 1 (definitely, since we found a -1) - delta_flipped
            #       add into some bundle (or it becomes its own bundle)
            #       new_support = True
            # else: no peak
            #   if new_support:
            #       try_extend(support[-1]):
            #           if too long (left):
            #               new_support = False
            #               trough = peak = now. # todo something is wrong here! how to get new_support = True
        # last_trough < last_peak: # find trough
            #   same as above
        # clean up bundles - if below some threshold, delete support


        # last_peak last_trough new_peak new_trough:
        #       found new peak!, cementing new trough ->
            #           last_trough, last_peak, new_trough, new_peak
            #                   (newest extreme point is always not a support yet)
            #                   (consider making a subtle support -> delete if not,
            #                   extend otherwise)
            #           definitely solidify trough if not already made:
            #               new_peak - last_peak = width of new_trough
            #               (consider max constraints!)
            #   latest_bundle -> most recent editted bundle.
            #       latest_bundle.delete/try_extend(support) if support == _support...
            #       or delete.
        # last_peak=last_trough=new_peak=new_trough:
            #
            #
            #
            #
        #
            #
            #
            #
            #
            #
        #   Note about decay. it gets abit confusing (ONLY WITH EXTENSION + DECAY)
            #   so one possibility is keeping decay_count in support array.
            #   and recalculating on every bundle calculate run
            # ===== Bundling =====

            # Already done in algorithm part

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

        # Calculate new strength
        self.calculate_bundle_strength(self.get_bundle(support))  # todo wrong method!
        return True

    def delete_support(self, _support):
        for bundle in self.bundles:
            for support in bundle.supports:
                if support == _support:
                    bundle.supports.remove(support)
        self.supports.remove(support)