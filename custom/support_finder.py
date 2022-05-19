# Super Attributes
import math
import random

import pandas as pd

from util.dataRetrievalUtil import load_df_list, load_df

# Download data
df_list = load_df_list()
r = random.randint(0, len(df_list))
df = load_df(df_list[r])
print(df_list[r])
# df = pd.DataFrame(index=df.index, data={
#     'data': df.Close
# })

# ##### Method 1 #####

# ===== Hyper parameters =====
distinguishing_constant = 0.1  # 0 to 1
decay_constant = 0.01  # The greater the number, the greater the decay of strength a.k.a soft cutoff (0.01 to 1)
lookback_period = 200  # Hard cutoff
variability_period = 100  # 50 to 150
strength_cutoff = 10  # When strength drops below cutoff, support is destroyed
critical_symmetry = 0.5  # 0 to 1, how equal must the length of the left and right bases be
max_base = 20  # Max length of left or right base
min_base = 10
delta_constant = 100  # flat pips
width_decay = 0  # Logarithmic
bundling_constant = 100  # pips, to bundle nearby supports together to create a stronger support
# ===== Process Hyper parameters =====
decay = math.pow(math.exp(), -decay_constant)
# ===== Constants =====
PEAK, TROUGH = 1, -1

# it would count as a peak... what do you think of this?

# ======================================
#
#   Look back {lookback_period} candlesticks for peaks and troughs, no matter the size.
#       Each peak and trough is marked by its relative strength =
#                                       (length of left base, length of right base)
#                                              Base defined by length to next peak
#                                       (length of strictly decreasing left base, str. dec. right base)
#                                              Length to next increasing candlestick
#                                       (age, aging with decaying strength)
#                                               m = (1 - decay) ^ age
#                                       dict = {left, right, str_left, str_right, age}
#       Re-measure last peak/trough (in some bundle):
#               Base could have expanded by 1 (unless marked as closed)
#               Check if within 'delta' of previous data. If true, increase support's length by 1.
#                   Search for support method:
#                           Check all bundles, for each support in bundle, check if start = support start
#                                   If found, increase length by 1. Then re-calculate bundle strength
#               If it can't, mark it as closed.
#
#       Search for new peak and troughs since last peak/trough {{last_lookback}}
#               (This algorithm does not use moving average and gentle peaks are also considered peaks)
#               Check if delta-increasing or delta-decreasing (can be both true)
#                       Mark
#               If delta-x flips sign (neutral ignored), this signifies either a peak or trough
#                       so -1 -1 -1 0 0 0 1 1 0 1 is considered a flip.
#                       !!!This does not work as there may be peaks where delta-x remains true!!!
#                       !!!One signal is enough +1 for up, -1 for down 0 for both. They can never be both 0!!!
#                       !!!After receiving a new peak/trough, compare to past peak/trough. If too close but diminutive,
#                           destroy last peak - false peak (If within min_length * 2 of each other)!!!
#                   if x = increasing, x' = decreasing vice versa
#                   - Record left_base = delta-x True in a row before this moment.
#                   When the delta-x' flips sign again, this marks the end of the right base.
#                           OR when the base height exceeds the absolute height of peak
#                           OR if base length reaches max length
#                           OR if end of data
#                           [Alt: When base height exceeds X% of the average-to-peak (SMA - 10) height of peak]
#                           [Peaks should be 2 deviations (Standard Deviation - 20, 90% Quartile) away]
#                   - Peak = max(data[left_base:right_base+1])
#                   - Record right_base = end - peak
#               dict = {
#                   left_base=min(max_base, left_base), right_base=min(max_base, right_base),
#                   base=min(min(l_b, r_b)*symmetry, max(l_b, r_b))
#               }
#               Set {{last_lookback}} as this peak.
#               Measure base of peak/trough
#       Mark each peak and trough with a support line
#               Support line strength = strength of peak/trough
#       Bundles drop support lines past the {lookback_period} or those with less than {strength_cutoff} strength
#       {{last_lookback}} is the latest peak or trough.
#
#   Refresh bundles
#       For each bundle, for each support
#           Decay supports by 1
#       If support strength weaker than cutoff,
#           Remove
#       If no more supports,
#           Destroy bundle
#
#   Bundling
#       Calculate {{market_variability}} based on {variability_period}
#               Find standard deviation within variability_period (Should be > lookback)
#       Based on {distinguishing_constant} and {{market_variability}}, progressively bundle supports together
#               Using bundle average, search for supports within {distinguishing_constant} * standard deviation distance
#                   Remaining supports become bundles
#                   Collect supports for new bundle
#                   Repeat
#               Calculate bundle strength
#
#   Submit bundle averages and bundle strength
#               dict = {
#                   supports:   [{start, price, strength},...]
#               }
#           Calculate average price and strength
#               price = avg([support['price'] for support in bundle['supports']]),
#               strength = sum([support['strength'] for support in bundle['supports']])
#           bundles = [{price, strength},...]
#
#
# ======================================

# ===== Handles =====
bundles = []  # Contains supports
delta_data = []  # Combined delta+ and delta- (x and x'): 1 = d-inc; 0 = both; -1 = d-dec;

past_df = pd.DataFrame()

# Search for delta-critical points
last_peak, last_trough, new_peak, new_trough = 0, 0, 0, 0
last_lookback, last_support, last_delta, delta_flipped = 0, None, 0, False
idx = 0


# ===== Util Functions =====


def calc_strength(support, idx):
    start = support['start']
    end = support['end']
    peak = support['peak']
    dist = idx - peak
    length = end - start
    return math.pow(decay, dist) * math.log(length, 2)


def decay(strength):
    return strength * decay


def decay_by(strength, length):
    return strength * math.pow(decay, length)


def within_bundle(bundle, support):
    if abs(bundle['height'] - support['height']) < bundling_constant:
        return True
    return False


def bundle_add(bundle, support):
    pass


def bundle_decay(bundle):
    for support in supports:
        support['strength'] = decay(support['strength'])


def create_bundle(support):
    bundle = {
        'supports': [support]
    }
    bundles.append(bundle)
    return bundle


def create_support(peak, start, end, height, type):
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
    for bundle in bundles:
        if within_bundle(bundle, support):
            bundle_add(bundle, support)
            added = True
            break

    if not added:
        create_bundle(support)

    last_support = support
    return support


def get_bundle(support):
    for bundle in bundles:
        if support in bundle['supports']:
            return bundle
    return None


def calculate_bundle_strength(bundle, idx):
    strength = 0
    for support in bundle['supports']:
        strength += calc_strength(support, idx)
    bundle['strength'] = strength
    return strength


def try_extend_peak(support, idx):
    if support['end'] - support['start']:  # base too long, reset
        support['open'] = False
        return False
    elif support['type'] == PEAK and support['height'] < df.High[idx]:  # base too high
        support['open'] = False
        return False
    elif support['type'] == TROUGH and support['height'] > df.Low[idx]:  # base too low
        support['open'] = False
        return False
    calculate_bundle_strength(get_bundle(support))
    return True


##### START #####

for candle in df:  # Simulation of going through df

    # ===== Temporary Variables =====
    # Next
    past_df.append(candle)
    idx += 1

    # Supports (to be bundled)
    supports = []  # contains dict = {type_peak (T/F); peak; start; end; height;}
    # --
    _max = 0
    _min = math.inf
    # --

    # Need at least 2 data points to continue:
    if len(past_df) < 2:
        delta_data.append(0)
        continue

    # ===== Algorithm =====
    # Compare last candle with new candle:
    diff = past_df.Close[-2] - past_df.Close[-1]
    delta_flipped = False
    if abs(diff) < delta_constant:  # No delta-movement
        delta_data.append(0)
    else:
        if diff > 0:  # New candle is lower, therefore (only) delta-decreasing
            val = -1  # Ignores 0 case, only captures last delta-movement
        else:  # Otherwise delta-increasing
            val = 1
        delta_data.append(val)
        delta_flipped = last_delta != val
        last_delta = val  # Ignores 0 case, only captures last delta-movement

    # Get next peak/trough, 3 modes:
    # 1) Find next trough, 2) find next peak, 3) find next any
    if last_peak > last_trough:  # look for next trough
        if delta_flipped:  # found
            height = min(past_df.Close[supports[-1]['start'] + 1:idx])  # Value
            peak = past_df[past_df.Close[supports[-1]['end']:idx] == height][-1] or 0  # Where
            start = last_support[-1]['end']
            end = idx
            create_support(peak, start, end, height, TROUGH)
        else:  # try extend last peak
            if try_extend_peak(supports[-1], idx):
                pass
            else:
                last_peak, last_trough = idx, idx
            # supports[-1]['end'] += 1
            # # if base too long, reset
            # if supports[-1]['end'] - supports[-1]['start']:
            #     last_peak, last_trough = idx, idx
    elif last_peak < last_trough:  # look for next peak
        if delta_flipped:  # found
            height = max(past_df.Close[supports[-1]['start'] + 1:idx])
            peak = past_df[past_df.Close[supports[-1]['end']:idx] == height][-1] or 0
            start = last_support[-1]['end']
            end = idx
            create_support(peak, start, end, height, PEAK)
        else:  # try extend last trough
            if try_extend_peak(supports[-1], idx):
                pass
            else:
                last_peak, last_trough = idx, idx
    else:  # last_peak = last_trough (only possible if just started or reset)
        if delta_data[-1] == -1:  # found potential trough
            last_trough = idx
        elif delta_data[-1] == 1:  # potential peak
            last_peak = idx

        # ===== Bundling =====

        # Already done in algorithm part

        # ===== Return function =====

        # None in this case
        print(bundles)

# ##### Method 2 #####

# Naiive method: delta=0 of method 1, no bundling.
pass_data = []

for candlestick in df:
    pass_data.append(candlestick)
    pass

# ##### Method 3 #####

# Method by Deviation

pass_data = []

for candlestick in df:
    pass_data.append(candlestick)
    pass
