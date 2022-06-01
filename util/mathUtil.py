from math import floor

import talib

from util.dataRetrievalUtil import try_stdev
from util.langUtil import try_mean, try_int


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

    hexa = 15*16+15 * val
    first_digit = hexa//16
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