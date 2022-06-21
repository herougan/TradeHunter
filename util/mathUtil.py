import math
from math import floor
from statistics import stdev

import talib


def quartile_out(quartile, data):
    """Takes out extremities"""
    pass


def moving_average(period, data):
    avg = []
    if len(data) < period:
        return avg
    for i in range(period - 1, len(data)):
        avg.append(try_mean(data[i - period + 1:i]))
    return avg


def moving_stddev(period, data):
    avg = []
    if len(data) < period:
        return avg
    for i in range(period - 1, len(data)):
        avg.append(try_stdev(data[i - period + 1:i]))
    return avg


def adjusted_dev(period, data, order=1):
    # Does not work!
    above, below = data, data
    stdev_data = talib.STDDEV(data, period)
    for i, row in above.iterrows():
        above.iloc[i].data += stdev_data.iloc[i].data * order
    for u, row in below.iterrows():
        above.iloc[i].data -= stdev_data.iloc[i].data * order
    return above, below


def index_arr_to_date(date_index, index):
    """Given an index, return date from date_index."""
    if index < 0 or index > len(date_index):
        return 0
    return date_index.iloc[index]


def date_to_index_arr(index, dates_index, dates):
    """Given an index that corresponds to a date_array, find the relative index of input date."""
    try:
        _dates = []
        for date in dates:
            _dates.append(index[list(dates_index).index(date)])
        return _dates
    except:
        print('Error! Date cannot be found. Continuing with 0.')
        return [0 for date in dates]


# Try


def try_int(s: str) -> int:
    try:
        if s is None:
            return 0
        return int(s)
    except ValueError:
        return 0


def try_float(s: str) -> float:
    try:
        if s is None:
            return 0
        return float(s)
    except ValueError:
        return 0


def try_key(dict: {}, key: str):
    if key in dict:
        return dict['key']
    else:
        return "-"


def try_divide(n1, n2):
    if n2 == 0:
        return math.inf
    return n1 / n2


def try_normalise(list):
    i, t = 0, 0
    for l in list:
        i += 1
        t += math.pow(l, 2)
    return math.pow(t, 0.5)


def try_limit(n1, n):
    if n1 > n[1]:
        return n[1]
    elif n1 < n[0]:
        return n[0]
    return n1


def try_max(list):
    if len(list) < 1:
        return 0
    return max(list)


def try_min(list):
    if len(list) < 1:
        return 0
    return min(list)


def try_mean(list):
    if len(list) < 1:
        return 0
    t, l = 0, len(list)
    for i in list:
        if i is None:
            t += 0
            l -= 0
            continue
        t += i
    return try_divide(t, l)


def try_width(list):
    """1D only"""
    if len([l for l in list if l is not None]) < 1:
        return 0
    max, min = 0, math.inf
    # Get max and min
    for x in list:
        if x is None:
            continue
        if x > max:
            max = x
        if x < min:
            min = x
    if min == math.inf:
        return math.inf
    return max - min


def try_stdev(list):
    if len(list) < 2:
        return 0
    return stdev(list)


def try_radius(list):
    """1D only"""
    pass


def try_diameter(list):
    """1D only"""
    pass


def try_centre(list):
    """1D only"""
    pass


def try_avg_eccentricity(list):
    """1D only"""
    pass


def try_packing_dist(list):
    """1D only. Returns minimum distance such that all vertices..."""
    pass


def try_dominating_dist(list):
    """1D only"""
    pass


def try_sgn(n1):
    n1 = try_float(n1)
    if n1:
        sgn = math.copysign(1, n1)
        return sgn
    return 0


def get_dist(n1, n2):
    _t = 0
    for i in range(len(n1)):
        _d = n1[i] - n2[i]
        _t += _d * _d
    return math.sqrt(_t)


# If/In

def in_range(n1, n2=[0, 1]):
    n1 = try_float(n1)
    if n1 and len(n2) >= 2:
        if n2[0] < n1 < n2[-1]:
            return True
    return False


def on_range(n1, n2=[0, 1]):
    n1 = try_float(n1)
    if n1 and len(n2) >= 2:
        if n2[0] == n1 or n1 == n2[-1]:
            return True
    return False


def in_std_range(n1, avg, stdev, order=1):
    return in_range(n1, [avg - order * stdev, avg + order * stdev])


# Is


def is_integer(x):
    y = try_int(x)
    if not y or y - x != 0:
        return False
    return True


def get_scale_colour(col1, col2, val):
    """Takes in two colours and the val (between 1 and 0) to decide
    the colour value in the continuum from col1 to col2.
    col1 and col2 must be named colours."""
    pass


def to_candlestick(ticker_data, interval: str, inc=False):
    pass


def get_scale_grey(val):
    hexa = 15 * 16 + 15 * val
    first_digit = hexa // 16
    second_digit = hexa - first_digit * 16
    hexa = F'{to_single_hex(first_digit)}{to_single_hex(second_digit)}'

    return F'#{hexa}{hexa}{hexa}'


def get_inverse_single_hex(val):
    val = try_int(val)
    _val = val % 16
    _val = 16 - _val
    if _val < 10:
        return str(_val)
    elif 10 <= _val < 11:
        return 'A'
    elif 11 <= _val < 12:
        return 'B'
    elif 12 <= _val < 13:
        return 'C'
    elif 13 <= _val < 14:
        return 'D'
    elif 14 <= _val < 15:
        return 'E'
    elif 15 <= _val < 16:
        return 'F'
    return None


def to_single_hex(val):
    val = try_int(val)
    _val = val % 16
    if _val < 10:
        return str(_val)
    elif 10 <= _val < 11:
        return 'A'
    elif 11 <= _val < 12:
        return 'B'
    elif 12 <= _val < 13:
        return 'C'
    elif 13 <= _val < 14:
        return 'D'
    elif 14 <= _val < 15:
        return 'E'
    elif 15 <= _val < 16:
        return 'F'
    return None


# Array

def sorted_insert(array: list, value):
    for i in range(len(array)):
        if value > array[i]:
            array.insert(i+1, value)
            return
    # Else
    array.append(value)


def sorted_dict_insert(array, value, value_key):
    for i in range(len(array)):
        if value > array[i][value_key]:
            array.insert(i+1, {value_key: value})
            return
    # Else
    array.append({value_key: value})