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
from util.statMathUtil import date_to_string as datestring
from util.langUtil import strtotime, timedeltatosigstr, normify_name


#   DataFrame

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
    folder = F'/static/data'
    os.makedirs(folder, exist_ok=True)
    df.to_csv(F'{folder}/{name}.csv')
    print(F'Creating {folder}/{name}.csv')


def load_df(name: str):
    folder = F'/static/data'
    if not name.endswith('.csv'):
        name += '.csv'
    df = pd.read_csv(F'{folder}/{name}')
    print(F'Reading {folder}/{name}')
    return df


def load_df_list():
    path = F'/static/data/'
    # Get list of files that end with .csv
    df_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    return df_list


def load_df_list(ds_name: str):
    folder = F'/static/datasetdef'
    if not ds_name.endswith('.csv'):
        ds_name += '.csv'
    # load df_list from list of paths
    ds = pd.read_csv(F'{folder}/{ds_name}.csv')
    return ds


#   DataSet

def load_dataset(ds_name: str):
    folder = F'/static/datasetdef'
    if not ds_name.endswith('.csv'):
        ds_name += '.csv'
    dsf = pd.read_csv(F'{folder}/{ds_name}')
    print(F'Reading {folder}/{ds_name}')
    return dsf


def load_dataset_list():
    path = F'static/datasetdef/'
    # Get list of files that end with .csv
    df_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    return df_list


def save_dataset(ds_name, dsf):
    folder = F'static/datasetdef'
    if not ds_name.endswith('.csv'):
        ds_name += '.csv'
    # save new dsf into ds_name
    dsf.to_csv(F'{folder}/{ds_name}')
    print(F'Saving into {folder}/{ds_name}')


def write_dataset(ds_name, dsf):
    folder = F'static/datasetdef'
    os.makedirs(folder, exist_ok=True)
    dsf.to_csv(F'{folder}/{ds_name}.csv')
    print(F'Creating dataset {folder}/{ds_name}.csv')


def write_new_dataset(ds_name, dsf):
    """This function writes the dataset but also notes it as a change"""
    write_dataset(ds_name, dsf)
    add_as_dataset_change(ds_name)
    print(F'Overwriting dataset {ds_name}.csv')


def write_new_empty_dataset(ds_name):
    """Same as above but dataset is empty"""

    ds_name = normify_name(ds_name)
    data = {
        'symbol': [],
        'interval': [],
        'period': [],
    }
    dsf = pd.DataFrame(data)
    write_dataset(ds_name, dsf)
    add_as_dataset_change(ds_name)

    print(F'Creating new dataset {ds_name}.csv')


# Dataset-Changes

def get_dataset_changes() -> pd.DataFrame:
    path = F'/static/common/datasetchanges.txt'
    dsc = pd.read_csv(path)
    return dsc


def update_all_dataset_changes():  # Downloading
    dsc = get_dataset_changes()
    print(F'Updating all datasets...')
    pass


def update_specific_dataset_change(ds_name):  # Downloading
    print(F'Updating dataset {ds_name}')

    pass


def add_as_dataset_change(ds_name: str):
    '''Changes to any instrument signature contained within the dataset or addition/subtraction of instruments
    count as a dataset change.'''
    path = F'/static/common/datasetchanges.txt'
    dsc = pd.read_csv(F'{os.getcwd()}{path}')

    _new = pd.DataFrame({
        'name': [ds_name]
    })
    dsc.append(_new)
    print(F'Overwriting dataset {ds_name}')
    return dsc


def remove_dataset_change(ds_name: str):
    dsc = get_dataset_changes()
    dsc.drop(name=ds_name)
    set_dataset_changes(dsc)


def set_dataset_changes(dsc: pd.DataFrame):
    path = F'/static/datasetdef/datasetchanges.txt'
    dsc.to_csv(path)


# List of instrument

def load_interval_suggestions():
    pass


def write_interval_suggestions(isdf: pd.DataFrame):
    pass


def add_interval_suggestion():
    isdf = load_interval_suggestions()
    write_interval_suggestions(isdf)


def load_period_suggestions():
    pass


def write_period_suggestions(isdf: pd.DataFrame):
    pass


def add_period_suggestion():
    isdf = load_interval_suggestions()
    write_interval_suggestions(isdf)


def load_symbol_suggestions():
    pass


def write_symbol_suggestions(isdf: pd.DataFrame):
    pass


def add_symbol_suggestion():
    isdf = load_interval_suggestions()
    write_interval_suggestions(isdf)


# Check
def is_valid_dataset():
    return True


def is_valid_df():
    return True


# Get bots


def load_trade_advisor(ta_name: str):
    pass


def load_trade_advisor_list():
    folder = ''
    pass


# Data Transformation Util

def table_to_dataframe(data):
    return pd.DataFrame(data)


def dataframe_to_table(df):
    table = []

    return table


# Init basic files


def init_common():

    datasetchanges = F'{os.getcwd()}/static/common/datasetchanges.txt'
    common_intervals = F'{os.getcwd()}/static/common/common_intervals.txt'
    common_periods = F'{os.getcwd()}/static/common/common_periods.txt'
    common_symbols = F'{os.getcwd()}/static/common/common_symbols.txt'

    if not file_exists(datasetchanges) or file_is_empty(datasetchanges):
        data = {
            'name': [],
            'date': []
        }
        df = pd.DataFrame(data)
        df.to_csv(datasetchanges)
    if not file_exists(common_intervals) or file_is_empty(common_intervals):
        data = {
            'interval': ['1M', '2M', '5M', '15M', '30M', '60M', '1h', '90M', '1d', '5d', '1wk', '2wk', '1m', '3m'],
        }
        df = pd.DataFrame(data)
        df.to_csv(common_intervals)
    if not file_exists(common_periods) or file_is_empty(common_periods):
        data = {
            'period': ['1wk', '1m', '3m', '6m', '1y', '2y', '5y', 'max'],
        }
        df = pd.DataFrame(data)
        df.to_csv(common_periods)
    if not file_exists(common_symbols) or file_is_empty(common_symbols):
        data = {
            'symbol': []
        }
        df = pd.DataFrame(data)
        df.to_csv(common_symbols)


def file_exists(path) -> bool:
    return os.path.exists(F'{path}')


def file_is_empty(path) -> bool:
    return os.path.getsize(path) == 0


def file_is_df_csv(path) -> bool:
    try:
        df = pd.read_csv(path)
        return True
    except:
        return False
