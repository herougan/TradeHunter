# Stats Imports
import pandas as pd

# Data Brokers
import yfinance as yf
from yahoofinancials import YahooFinancials

# Utils
from datetime import timedelta, datetime
from typing import List
import os
from os import listdir
from os.path import isfile, join
import glob
# Custom Utils
from statMathUtil import date_to_string as datestring
from langUtil import strtotime, timedeltatosigstr


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


def retrieve(s: str, start: datetime, end: datetime, interval: str, write: bool = False, progress: bool = False):
    period = end - start
    name = F'{s}-{interval}-{timedeltatosigstr(period)}-{start.year}-{end.year}'
    print(F'Retrieving {name}')
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
    max_period = strtotime(min_dict[interval])
    if interval in min_dict:
        if period > max_period:
            n_loop = max_period // period + 1
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

    final = pd.concat(df_array)

    if write:
        write_df(final, name)

    return final


# Read/Write from local
def write_df(df, name: str):
    folder = F'../data'
    os.makedirs(folder, exist_ok=True)
    df.to_csv(F'{folder}/{name}.csv')
    print(F'Creating {folder}/{name}.csv')


def load_df(name: str):
    folder = F'../data'
    if not name.endswith('.csv'):
        name += '.csv'
    df = pd.read_csv(F'{folder}/{name}')
    print(F'Reading {folder}/{name}')
    return df


def load_df_list():
    path = F'../data/'
    # Get list of files that end with .csv
    df_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    return df_list
