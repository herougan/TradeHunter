from util.dataRetrievalUtil import try_stdev
from util.langUtil import try_mean


def quartile_out(quartile, data):
    """Takes out extremities"""
    pass


def moving_average(period, data):
    avg = []
    if len(data) < period:
        return avg
    for i in range(period-1, len(data)):
        avg.append(try_mean(data[i-period+1:i]))
    return avg


def moving_stddev(period, data):
    avg = []
    if len(data) < period:
        return avg
    for i in range(period-1, len(data)):
        avg.append(try_stdev(data[i-period+1:i]))
    return avg


def adjusted_dev(period, data):
    pass