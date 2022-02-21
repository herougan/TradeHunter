from datetime import timedelta, datetime
import re
import unicodedata


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
            t += timedelta(hours=d)
        elif a == "w" or a.casefold() in "weeks".casefold():
            t += timedelta(weeks=d)
        elif a == "m" or a.casefold() in "months".casefold():
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
    interval = ['1M', '2M', '5M', '15M', '30M', '60M', '1h', '90M', '1d', '5d', '1wk', '1m', '3m']
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


def drsplit(s: str):
    alpha = s.lstrip('0123456789')
    digit = s[:len(s) - len(alpha)]
    if not digit:
        digit = 0
    return int(digit), alpha


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


def to_dataname():
    return F'{s}-{interval}-{timedeltatosigstr(period)}-{start.year}-{end.year}'


def from_dataname():
    return (s, interval, period, start, end)


def normify_name(s: str):
    return s.replace(' ', '_')