# Stats Imports
import pandas as pd

# Data Brokers
import yfinance as yf
from yahoofinancials import YahooFinancials

# Utils
from datetime import timedelta, datetime
import datetime
from typing import List
import os
# Custom Utils
from statMathUtil import date_to_string as datestring
from langUtil import strtotime


# Retrieval from yfinance

def retrieve(s: str, start: str, end: str, progress: bool = False):
    df = yf.download(s,
                     start=start,
                     end=end,
                     progress=progress,
                     )
    df.head()

    return df


def retrieve(s: str, period: str, interval: str, write: bool = False, progress: bool = False):
    return retrieve(s, datetime.now() - strtotime(period), datetime.now(), interval, write, progress)


def retrieve(s: str, start: str, end: str, interval: str, write: bool = False, progress: bool = False):
    period = datetime(start) - datetime(end)
    name = F'{s}-{interval}-{period}-{datetime(start).year}-{datetime(end).year}'
    # Similar retrievals will overwrite each other unless they start or end in different years.

    # Loop through smaller time periods if period is too big for given interval (denied by yfinance)
    n_loop = 1
    loop_period = period
    min_dict = {
        '1m': '7d',
        '2m': '7d',
        '5m': '7d',
        '15m': '60d',
        '30m': '60d',
        '60m': '60d',
        '90m': '60d',
        '1h': '60d',
    }
    min_interval = strtotime(min_dict[interval])
    if interval in min_dict:
        if period > min_interval:
            n_loop = min_interval // interval + 1
            loop_period = period / n_loop

    df_array = []
    # Retrieve data slowly...
    for i in range(n_loop):
        df = yf.download(s,
                         start=start,
                         end=start + loop_period,
                         interval=interval,
                         progress=progress)
        df_array.append(df)
        # Next starting date
        start += loop_period

    if write:
        write_df(df_array, name)

    return pd.concat(df_array)


# Read/Write from local
def write_df(df_array, name: str):
    for df in df_array:
        folder = F'../static/data'
        os.makedirs(folder, exist_ok=True)
        df.to_csv(F'{folder}{name}.csv')
    return True


def load_df():
    pass
