import math
from datetime import timedelta, datetime
import unicodedata
from typing import List
from dateutil import parser


def strtotime(s: str):
    """XM X minutes, XH X hours, Xd X days, Xw X weeks, Xm X months, all separated by a space"""
    t = timedelta()
    s_array = s.split()
    for _s in s_array:
        (d, a) = drsplit(_s)
        if d is None:
            pass
        elif a == "M" or a.casefold() in "minutes".casefold():
            t += timedelta(minutes=d)
        elif a == "H" or a.casefold() in "hours".casefold():
            t += timedelta(hours=d)
        elif a == "d" or a.casefold() in "days".casefold():
            t += timedelta(days=d)
        elif a == "w" or a.casefold() in "weeks".casefold():
            t += timedelta(weeks=d)
        elif a == "mo" or a.casefold() in "months".casefold():
            t += timedelta(weeks=d * 4)
        elif a == "y" or a.casefold() in "years".casefold():
            t += timedelta(weeks=d * 48)
        elif a == "s" or a.casefold() in "seconds".casefold():
            t += timedelta(seconds=d)
        elif s == "max":
            return s
    return t


def strtoyahootimestr(s: str):
    """XM X minutes, XH X hours, Xd X days, Xw X weeks, Xm X months, all separated by a space
    Interval closest to '1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo' will be chosen."""
    interval = ['1M', '2M', '5M', '15M', '30M', '60M', '1h', '90M', '1d', '5d', '1wk', '1mo', '3mo']
    idx, prev_idx = len(interval) // 2, 0
    left, right = 0, len(interval)
    chosen_interval = strtotime(s)

    while not (right - left) < 2:
        prev_idx = idx
        diff = strtotime(interval[idx]) - chosen_interval
        # Check if chosen interval is smaller or greater than measured interval, move boundaries accordingly
        if diff > timedelta(0):
            right = idx
            idx = (idx + left) // 2
        elif diff == timedelta(0):
            return interval[idx]
        else:
            left = idx
            idx = (idx + right) // 2

    # Compare which is better
    diff1 = strtotime(interval[idx]) - chosen_interval
    diff2 = strtotime(interval[prev_idx]) - chosen_interval
    if diff1 > timedelta(0):
        # interval_1 is larger than chosen interval, so interval_2 is smaller (diff2 is negative)
        if diff1 + diff2 > timedelta(0):
            # diff1 is larger than diff2, and so chosen interval is closer to interval_2
            idx = prev_idx
    else:
        # interval_1 is smaller than chosen interval, so interval_2 is bigger (diff2 is positive, diff1 is negative)
        if diff1 + diff2 < timedelta(0):
            # diff2 (positive) is not large enough to compensate for diff1 and so chosen interval is closer to interval_2
            idx = prev_idx

    return interval[idx]


def timedeltatoyahootimestr(_interval: timedelta):
    """XM X minutes, XH X hours, Xd X days, Xw X weeks, Xm X months, all separated by a space
    Interval closest to '1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo' will be chosen."""
    interval = ['1M', '2M', '5M', '15M', '30M', '60M', '1h', '90M', '1d', '5d', '1wk', '1m', '3m']
    idx, prev_idx = len(interval) // 2, 0
    left, right = 0, len(interval)
    chosen_interval = _interval

    while not (right - left) < 2:
        prev_idx = idx
        diff = strtotime(interval[idx]) - chosen_interval
        # Check if chosen interval is smaller or greater than measured interval, move boundaries accordingly
        if diff > timedelta(0):
            right = idx
            idx = (idx + left) // 2
        elif diff == timedelta(0):
            return interval[idx]
        else:
            left = idx
            idx = (idx + right) // 2

    # Compare which is better
    diff1 = strtotime(interval[idx]) - chosen_interval
    diff2 = strtotime(interval[prev_idx]) - chosen_interval
    if diff1 > timedelta(0):
        # interval_1 is larger than chosen interval, so interval_2 is smaller (diff2 is negative)
        if diff1 + diff2 > timedelta(0):
            # diff1 is larger than diff2, and so chosen interval is closer to interval_2
            idx = prev_idx
    else:
        # interval_1 is smaller than chosen interval, so interval_2 is bigger (diff2 is positive, diff1 is negative)
        if diff1 + diff2 < timedelta(0):
            # diff2 (positive) is not large enough to compensate for diff1 and so chosen interval is closer to interval_2
            idx = prev_idx

    return interval[idx]


def yahoolimitperiod(period: timedelta, interval: str):
    """Divides period into smaller chunks depending on the interval. Outputs new_period, n_loop"""
    n_loop = 1
    loop_period = period

    min_dict = {
        '1M': '7d',
        '2M': '7d',
        '5M': '7d',
        '15M': '60d',
        '30M': '60d',
        '60M': '60d',
        '90M': '60d',
        '1h': '60d',
    }

    eff_interval = '1M'
    diff = strtotime(eff_interval) - strtotime(interval)
    for key in min_dict.keys():
        _diff = strtotime(key) - strtotime(interval)
        if timedelta() > _diff > diff:
            diff = _diff
            eff_interval = key

    max_period = strtotime(min_dict[eff_interval])

    if period > max_period:
        n_loop = math.ceil(period / max_period)
        eff_period = period / n_loop
        return eff_period, n_loop
    return period, 1


def yahoolimitperiod_leftover(period: timedelta, interval: str):
    """Divides period into smaller defined chunks, depending on the interval.
    Outputs new_period, n_loop and period_leftover"""
    n_loop = 1
    loop_period = period

    min_dict = {
        '1M': '7d',
        '2M': '7d',
        '5M': '7d',
        '15M': '60d',
        '30M': '60d',
        '60M': '60d',
        '90M': '60d',
        '1h': '60d',
    }

    eff_interval = '1M'
    diff = strtotime(eff_interval) - strtotime(interval)
    for key in min_dict.keys():
        _diff = strtotime(key) - strtotime(interval)
        if timedelta() > _diff > diff:
            diff = _diff
            eff_interval = key

    max_period = strtotime(min_dict[eff_interval])

    if period > max_period:
        n_loop = math.floor(period / max_period)
        leftover = period - max_period * n_loop
        if leftover < strtotime(interval):
            leftover = strtotime(interval)
        return max_period, n_loop, leftover
    return period, 1, timedelta(0)


def strtodatetime(s: str) -> datetime:
    # 2022 - 02 - 23
    # OR
    # 2022 - 02 - 23
    # 09: 30:00 - 05: 00
    # hh: mm:ss tzd
    return parser.parse(s)


def drsplit(s: str):
    alpha = s.lstrip('0123456789')
    digit = s[:len(s) - len(alpha)]
    if not digit:
        digit = 0
    return int(digit), alpha


def is_lnumber(s: str):
    return not s == s.lstrip('0123456789')


def is_not_rnumber(s: str):
    return s == s.rstrip('0123456789')


def timedeltatosigstr(s: timedelta):
    """Takes in datetime and returns string containing only one significant time denomination without spaces"""
    if s.days > 0:
        return F'{s.days}d'
    elif s.seconds >= 60 * 60:
        return F'{s.seconds // (60 * 60)}h'
    elif s.seconds > 60:
        return F'{s.seconds // 60}M'
    else:
        return F"{s.seconds}s"


def to_dataname(s, interval, period):
    return F'{s}-{interval}-{timedeltatosigstr(period)}'


def from_dataname(s: str):
    arr = s.split('-')
    if len(arr) < 3:
        return ('Str_Error', '', '')
    return (arr[0], arr[1], arr[2])


def normify_name(s: str):
    return s.replace(' ', '')


def snake_to_proper_case(s: str):
    """to_proper_case -> To Proper Case"""
    s_arr = s.split('_')
    for i in range(len(s_arr)):
        s_arr[i] = s_arr[i].upper()
    return ' '.join(s_arr)


def remove_special_char(s: str):
    return s.replace('_', '')


def try_int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 0


def craft_instrument_filename(sym: str, interval: str, period: str):
    return F'{sym}__{interval}__{period}.csv'


def craft_test_filename(ta_name: str, ivar_name: str, ds_names: List[str]):
    pass


def try_key(dict: {}, key: str):
    if key in dict:
        return dict['key']
    else:
        return "-"


def pip_conversion(currency_pair: str):
    if 'USD' in currency_pair and 'JPY' in currency_pair:
        return 1 / 100
    else:
        return 1 / 10000


def get_size_bytes(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def check_if_valid_timestr(s: str):
    return is_lnumber(s) and is_not_rnumber(s)

# data = yf.download(  # or pdr.get_data_yahoo(...
#         # tickers list or string as well
#         tickers = "SPY AAPL MSFT",
#
#         # use "period" instead of start/end
#         # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#         # (optional, default is '1mo')
#         period = "ytd",
#
#         # fetch data by interval (including intraday if period < 60 days)
#         # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#         # (optional, default is '1d')
#         interval = "1m",
#
#         # group by ticker (to access via data['SPY'])
#         # (optional, default is 'column')
#         group_by = 'ticker',
#
#         # adjust all OHLC automatically
#         # (optional, default is False)
#         auto_adjust = True,
#
#         # download pre/post regular market hours data
#         # (optional, default is False)
#         prepost = True,
#
#         # use threads for mass downloading? (True/False/Integer)
#         # (optional, default is True)
#         threads = True,
#
#         # proxy URL scheme use use when downloading?
#         # (optional, default is None)
#         proxy = None
#     )
