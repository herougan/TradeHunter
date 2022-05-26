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
