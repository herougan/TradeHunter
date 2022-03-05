# Stats Imports
import math

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
import settings
from robot.abstract.robot import robot
from util.dataTestingUtil import SUMMARY_STATS
from util.statMathUtil import date_to_string as datestring
from util.langUtil import strtotime, timedeltatosigstr, normify_name, yahoolimitperiod, yahoolimitperiod_leftover

# Run Robot imports
from robot import FMACDRobot, FilterRobot


#   DataFrame

def retrieve_str(s: str,
                 interval: str,
                 period: str,
                 write: bool = False,
                 progress: bool = False):
    return retrieve(s, datetime.now() - strtotime(period), datetime.now(), interval, write, progress)


def retrieve(
        s: str,
        start: datetime,
        end: datetime,
        interval: str,
        write: bool = False,
        progress: bool = False):
    period = end - start
    name = F'{s}-{interval}-{timedeltatosigstr(period)}'
    print(F'Retrieving {name}, write: {write}')
    # todo are other intervals such as '3M' allowed in yfinance?

    # Loop through smaller time periods if period is too big for given interval (denied by yfinance)
    loop_period, n_loop, leftover = yahoolimitperiod_leftover(period, interval)

    df_array = []
    # Retrieve data slowly...
    for i in range(n_loop + 1):
        if i == n_loop:
            _loop_period = leftover
        else:
            _loop_period = loop_period

        if _loop_period > timedelta(minutes=1):
            df = yf.download(s,
                             start=start,
                             end=start + _loop_period,
                             interval=interval,
                             progress=progress)
            df_array.append(df)
            # Next starting date
            start += _loop_period

    final = pd.concat(df_array)

    success = True
    if len(final.index) <= 1:
        success = False

    if write and success:
        write_df(final, name)

    return final, success


def retrieve_ds(ds_name: str, write: bool = False, progress: bool = False):
    df = load_dataset(ds_name)
    for index, row in df.iterrows():
        df, suc = retrieve_str(row['symbol'], row['interval'], row['period'], write, progress)
        if not suc:
            remove_from_dataset(ds_name, row['symbol'], row['interval'], row['period'])


# Read/Write from local
def write_df(df, name: str):
    folder = F'static/data'
    os.makedirs(folder, exist_ok=True)
    df.to_csv(F'{folder}/{name}.csv')
    print(F'Creating {folder}/{name}.csv')


def load_df(name: str):
    folder = F'static/data'
    if not name.endswith('.csv'):
        name += '.csv'
    df = pd.read_csv(F'{folder}/{name}')
    print(F'Reading {folder}/{name}')
    return df


def load_df_list():
    path = F'static/data/'
    # Get list of files that end with .csv
    df_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    return df_list


#   DataSet

def load_dataset(ds_name: str) -> pd.DataFrame:
    folder = F'static/datasetdef'
    if not ds_name.endswith('.csv'):
        ds_name += '.csv'
    dsf = pd.read_csv(F'{folder}/{ds_name}', index_col=0)
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


def remove_from_dataset(ds_name, symbol, interval, period):
    dsf = load_dataset(ds_name)
    dsf.drop(dsf[dsf.symbol == symbol and dsf.interval == interval and dsf.period == period].index)
    write_dataset(ds_name, dsf)


# Dataset Files

def load_dataset_data_list(ds_name: str) -> List[str]:
    folder = 'static/data/'

    ds_df = load_df(ds_name)
    d_list = []
    for index, row in ds_df.iterrows():
        # Form d_name (SYM_INT_PER)
        d_name = F'{row["symbol"]}_{row["interval"]}_{row["period"]}'
        # todo
        d_list.append(d_name)

    return d_list


def load_dataset_data(d_list: List[str]) -> List[pd.DataFrame]:
    all_data = []
    for d_name in d_list:
        all_data.append(load_dataset(d_name))

    return all_data


# Dataset-Changes

def get_dataset_changes() -> pd.DataFrame:
    path = F'static/common/datasetchanges.txt'
    dsc = pd.read_csv(path, index_col=0)
    return dsc


def update_all_dataset_changes():  # Downloading
    dsc = get_dataset_changes()
    print(F'Updating all datasets...')
    for index, row in dsc.iterrows(dsc):
        retrieve_ds(row['name'])
    clear_dataset_changes()


def update_specific_dataset_change(ds_name):  # Downloading
    print(F'Updating dataset {ds_name}')

    # Removing specific dataset change flag
    dsc = get_dataset_changes()
    for index, row in iter(dsc):
        if row['name'] == ds_name:
            dsc.drop([ds_name])
            # Download data
            retrieve_ds(ds_name)
            remove_dataset_change(ds_name)


def add_as_dataset_change(ds_name: str):
    '''Changes to any instrument signature contained within the dataset or addition/subtraction of instruments
    count as a dataset change.'''
    path = F'/static/common/datasetchanges.txt'
    dsc = pd.read_csv(F'{os.getcwd()}{path}', index_col=0)
    print("---------------")
    if ds_name in dsc['name']:
        print(F'Overwriting dataset {ds_name} - Already most updated')
    else:
        _new = pd.DataFrame([[ds_name]], columns=['name'], index=[len(dsc.index)])
        dsc = dsc.append(_new)
        print(F'Overwriting dataset {ds_name}')
        write_dataset_change(dsc)


def write_dataset_change(dsc_df: pd.DataFrame):
    path = F'static/common/datasetchanges.txt'

    print(F'Noting changes in datasetchanges.txt')
    dsc_df.to_csv(path)


def remove_dataset_change(ds_name: str):
    dsc = get_dataset_changes()
    dsc.drop(dsc[dsc.name == ds_name].index)
    set_dataset_changes(dsc)


def clear_dataset_changes():
    path = F'static/common/datasetchanges.txt'

    df = pd.DataFrame(columns=['name'])
    df.to_csv(path)


def set_dataset_changes(dsc: pd.DataFrame):
    path = F'/static/datasetdef/datasetchanges.txt'
    dsc.to_csv(path)


# List of instrument

def load_symbol_suggestions() -> pd.DataFrame:
    common_symbols = F'static/common/common_symbols.txt'
    ss_df = pd.read_csv(common_symbols, index_col=0)
    return ss_df['symbol']


def write_symbol_suggestions(ss_df: pd.DataFrame):
    common_symbols = F'static/common/common_symbols.txt'
    ss_df.to_csv(common_symbols)


def add_symbol_suggestion(ss_add):
    ss_df = load_symbol_suggestions()
    ss_df.append(ss_add)
    write_symbol_suggestions(ss_df)


def load_interval_suggestions():
    common_intervals = F'static/common/common_intervals.txt'
    is_df = pd.read_csv(common_intervals, index_col=0)
    return is_df['interval']


def write_interval_suggestions(is_df: pd.DataFrame):
    common_intervals = F'static/common/common_intervals.txt'
    is_df.to_csv(common_intervals)


def add_interval_suggestion():
    is_df = load_interval_suggestions()
    write_interval_suggestions(is_df)


def load_period_suggestions():
    common_periods = F'static/common/common_periods.txt'
    ps_df = pd.read_csv(common_periods, index_col=0)
    return ps_df['period']


def write_period_suggestions(ps_df: pd.DataFrame):
    common_periods = F'static/common/common_periods.txt'
    ps_df.to_csv(common_periods)


def add_period_suggestion():
    ps_df = load_interval_suggestions()
    write_period_suggestions(ps_df)


# Check
def is_valid_dataset():
    return True


def is_valid_df():
    return True


# Data Transformation Util

def table_to_dataframe(data):
    return pd.DataFrame(data)


def dataframe_to_table(df):
    table = []

    return table


# Trade Advisors/Robots


# Get bots


def load_trade_advisor(ta_name: str):
    path = F'robot'
    pass


def load_trade_advisor_list():
    path = F'robot'
    # Get list of files that end with .py
    robot_list = [os.path.splitext(f)[0] for f in listdir(path) if isfile(join(path, f)) and
                  f.endswith('.py') and '__init__' not in f]
    return robot_list


def init_robot(ta_name: str, ivar: pd.DataFrame):
    args_str = eval(F'{ta_name}.ARGS_STR')
    r = eval(F'{ta_name}.{ta_name}({ivar})')
    return r


# iVar


def ivar_to_list(idf: pd.DataFrame):
    ivar = []
    for col in idf:
        if not col == 'ivar_name':
            ivar.append(idf.loc[:, col])
    return ivar


def load_ivar(ta_name: str, ivar_name: str):
    idf = load_ivar_df(ta_name)
    return idf[idf['ivar_name'] == ivar_name]


def load_ivar_df(ta_name: str):
    folder = F'robot/ivar'
    ivar_file = F'{ta_name}_ivar'
    path = F'{folder}/{ivar_file}.csv'

    if not file_exists(path):
        generate_ivar(ta_name)

    idf = pd.read_csv(path, index_col=0)
    return idf


def load_ivar_list(ta_name: str):
    """Returns IVar names only"""
    folder = F'robot/ivar'
    ivar_file = F'{ta_name}_ivar'
    path = F'{folder}/{ivar_file}.csv'

    if not file_exists(path):
        generate_ivar(ta_name)

    idf = pd.read_csv(path, index_col=0)
    return idf.loc[:, 'name']


def generate_ivar(ta_name: str):
    folder = F'robot/ivar'
    ivar_file = F'{ta_name}_ivar'
    path = F'{folder}/{ivar_file}.csv'
    args_str = eval(F'{ta_name}.ARGS_STR')
    # args_str = eval(F'{ta_name}.ARGS_STR', {"__builtins__": {}})
    args = eval(F'{ta_name}.ARGS_DEFAULT')
    data = {
        'name': ['*Default'],
    }
    for i in range(len(args_str)):
        data.update({args_str[i]: args[i]})

    df = pd.DataFrame(data)
    df.to_csv(path)


def ivar_to_arr(idf: pd.DataFrame):
    pass


def insert_ivar(ta_name: str, ivar):
    path = get_ivar_path(ta_name)

    idf = load_ivar_df(ta_name)
    head = idf.head()
    data = {}
    for i in range(len(head)):
        data.update({
            head[i]: ivar[i]
        })
    n_idf = pd.DataFrame(data)
    idf.append(n_idf)
    idf.to_csv(path)


def get_ivar_path(ta_name):
    folder = F'robot/ivar'
    ivar_file = F'{ta_name}_ivar'
    path = F'{folder}/{ivar_file}.csv'
    return path


# Test Results


def get_test_steps(ds_name: str):
    """Get number of dataset(s)"""
    dsf = load_dataset(ds_name)
    return len(dsf)


def generic_result_df():
    data = {
        'dataset': []  # name
    }

    for stat in SUMMARY_STATS:
        data.update({
            stat: []
        })

    return pd.DataFrame(data)


def create_test_result(ta_name: str, ivar_name: str, rdf_list: List[pd.DataFrame]):
    name = F'{ta_name}__{ivar_name}__test.csv'
    folder = F'static/results/evaluation'
    path = F'{folder}/{name}.csv'

    final_df = generic_result_df()

    for rdf in rdf_list:
        final_df.append(rdf)
    final_df.append(create_dataset_result(rdf_list))

    final_df.to_csv(path, index=False)

    return final_df


def create_dataset_result(rdf_list):
    """Aggregates the results above from list of result dataframes.
    Result dataframes (from result_dict_to_dataset or otherwise) contain only 1 row."""

    data = {
        'dataset': ['Overall '],
    }

    for col in generic_result_df():
        data.update({
            col: [0]
        })
        for rdf in rdf_list:
            data[col][0] += rdf[col][0]
        data[col][0] /= len(rdf_list)

    rdf = pd.DataFrame(data, index=False)
    return rdf


def result_dict_to_dataset(result_dict):
    data = {}
    for key in result_dict:
        data.update({
            key: [result_dict[key]]
        })
    return pd.DataFrame(data, index=False)


def write_data_result(pl_df):
    # Meant for "simple plotting" - prints results for test on a SINGLE data
    pass


def load_test_result(test_name: str):
    pass


def load_test_meta(test_name: str):
    pass


# Robot -> Test (ivar) -> Dataset(s) -> Dataset -> Data -> P/L, Indicator, Buy/Sell Indicator
#                       -> Overall P/L

# Optimisation Results

def create_optimisation_result():
    ds_name = "test"
    ivar_name = "test"
    name = F'{ds_name}__{ivar_name}__optimisation.csv'

    data = [

    ]


# Robot -> Optimise (ivar) -> Dataset(s) -> Overall P/L optimisation (or progressive mode)
#                           -> Multiple IVars + Ivar/FinalProfit plot
#                           -> Final set IVars
#                          f:   Test potential IVars
#                               Save IVar

# Future Robot -> Execute (simple, ivar)


# Because of it's difficulty, I would rather just only test/optimise based on one dataset at a time.
# In terms of test/optimisation results its easy since it aggregates whatever data used for it
# But in terms of optimising... how would it work? Just optimise like normal but across bigger set?
# What about progressive optimisation -> Optimise against dataset(s)1, dataset(s)2 ....

# Note! It's not IVAR+DATASET/DATASETS per ROBOT for result but
# IVAR+DATASET(S) -> RESULT -> ROBOT. just search for results.
# Result takes in 1) IVAR variables 2) Dataset(s) names
# dataset names are written in a column form, other spaces left blank, 1st row are the ivar variables and robot name
# result should be named after robot.
# It should also contain all stats etc. 3) Stats
# Overall Profit Loss graph 4) Aggregate graphs

# NOTE! Will not save IVAR until given a name and manually saved. ALL IVAR will be deleted afterwards.
# Tests DO NOT create IVARs. Default IVAR must be available (as starting point). Optimisation tests thousands of IVARs.
# Can play with the last wave of IVARs ONLY. Save the few you want somehow.
# 2 kinds of IVAR: Pliable-Variables Stiff-Variables

# However, what about buy signals? Many many separate files would have to be generated - but this is a MUST.
# What about, for tests and optimisations, the first (or only) data of the first dataset will be chosen for
# the full plotting treatment (and saving of signal data)

# OR: Each instrument choice in dataset has a "signal" and profit dataframe. even indicator dataframes.
# So each test, whether multiple dataset(s) or not, the first dataset will have to print to multiple files
# crypto.csv -> robot_crypto_crypto1.csv, robot_crypto_crypto2.csv

# Lastly, clear all function if it gets too messy


# Init basic files


def init_common():
    datasetchanges = F'{os.getcwd()}/static/common/datasetchanges.txt'
    common_intervals = F'{os.getcwd()}/static/common/common_intervals.txt'
    common_periods = F'{os.getcwd()}/static/common/common_periods.txt'
    common_symbols = F'{os.getcwd()}/static/common/common_symbols.txt'

    if not file_exists(datasetchanges) or file_is_empty(datasetchanges):
        df = pd.DataFrame(columns=['name'])
        df.to_csv(datasetchanges)
    if not file_exists(common_intervals) or file_is_empty(common_intervals):
        data = {
            'interval': ['1M', '2M', '5M', '15M', '30M', '60M', '1h', '90M', '1d', '5d', '1wk', '2wk', '1m', '3m'],
        }
        df = pd.DataFrame(data)
        df.to_csv(common_intervals)
    if not file_exists(common_periods) or file_is_empty(common_periods):
        data = {
            'period': settings.SUGGESTIONS['periods'],
        }
        df = pd.DataFrame(data)
        df.to_csv(common_periods)
    if not file_exists(common_symbols) or file_is_empty(common_symbols):
        data = {
            'symbol': [
                'AAPL', 'TLSA', 'GOOGL', 'INTC', 'MSFT',
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(common_symbols)


def force_overwrite_common():
    common_symbols = F'{os.getcwd()}/static/common/common_symbols.txt'
    data = {
        'symbol': settings.SUGGESTIONS['symbols']
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


def dataframe_ok(df: pd.DataFrame) -> bool:
    if len(df.index):
        return True
    return False

# Robot methods
