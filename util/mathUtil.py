from util.dataRetrievalUtil import try_stdev
from util.langUtil import try_mean


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


def adjusted_dev(period, data):
    pass


def index_arr_to_date(date_index, index):
    pass


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
