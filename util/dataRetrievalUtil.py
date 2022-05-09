# Stats Imports
import platform
import random
from statistics import stdev

import GPUtil

import pandas as pd

# Data Brokers
import psutil as psutil
import yfinance as yf

# Utils
from datetime import timedelta, datetime
from typing import List
import os
from os import listdir
from os.path import isfile, join
# Custom Utils
import settings
from util.langUtil import strtotimedelta, timedeltatosigstr, normify_name, yahoolimitperiod_leftover, \
    get_size_bytes, try_int, craft_instrument_filename, strtoyahootimestr, get_yahoo_intervals


#   DataFrame

def retrieve_str(s: str,
                 interval: str,
                 period: str,
                 write: bool = False,
                 progress: bool = False, name=""):
    return retrieve(s, datetime.now() - strtotimedelta(period), datetime.now(), interval, write, progress, name)


def retrieve(
        s: str,
        start: datetime,
        end: datetime,
        interval: str,
        write: bool = False,
        progress: bool = False, name=""):
    period = end - start
    if not name:
        name = craft_instrument_filename(s, interval,
                                         timedeltatosigstr(period))  # F'{s}-{interval}-{timedeltatosigstr(period)}'
    print(F'Retrieving {name}, write: {write}')

    if interval not in get_yahoo_intervals():
        print(F'Note, interval of {interval} not accepted in yfinance api. Closest alternative will be used.')
    interval = strtoyahootimestr(interval)

    # Loop through smaller time periods if period is too big for given interval (denied by yfinance)
    loop_period, n_loop, leftover = yahoolimitperiod_leftover(period, interval)

    if strtotimedelta(interval) < timedelta(days=1) and period > timedelta(days=730):
        print('Note: Data past 730 days will likely fail to download.')

    df_array = []
    # Retrieve data slowly... 'Failed download' will be printed by yfinance upon failure
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
    if df is None:
        print("Failed to open dataset. Cancelling retrieval.")
        return False
    for index, row in df.iterrows():
        name = craft_instrument_filename(row['symbol'], row['interval'], row['period'])
        df, suc = retrieve_str(row['symbol'], row['interval'], row['period'], write, progress, name)
        if not suc:
            remove_from_dataset(ds_name, row['symbol'], row['interval'], row['period'])
    return True


# Read/Write from local
def write_df(df, name: str):
    folder = F'static/data'
    if not name.endswith('.csv'):
        name += '.csv'
    os.makedirs(folder, exist_ok=True)
    df.to_csv(F'{folder}/{name}')
    print(F'Creating {folder}/{name}')


def load_df(name: str):
    folder = F'static/data'
    if not name.endswith('.csv'):
        name += '.csv'
    path = F'{folder}/{name}'
    if not file_exists(path):
        print(F'{path} datafile not found')
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    print(F'Reading {path}')
    return df


def load_df_sig(sym: str, inv: str, per: str):
    path = F'{settings.DATA_FOLDER}/'
    name = craft_instrument_filename(sym, inv, per)
    # Get list of files that end with .csv
    df_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    for df in df_list:
        if df == name:
            return df
    return None


def load_df_list():
    path = F'{settings.DATA_FOLDER}/'
    # Get list of files that end with .csv
    df_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    return df_list


def load_ds_df_list(ds_name: str):
    dsf = load_dataset(ds_name)

    df_list = []
    # Get list of dsf's dfs
    for i, row in dsf.iterrows():
        # Find file
        df = load_df_sig(row['symbol'], row['interval'], row['period'])
        if df:
            df_list.append(df)
        pass
    return df_list


def remove_all_df():
    path = F'static/data/'
    df_list = load_df_list()
    for df in df_list:
        os.remove(F'{path}{df}')


def remove_df(df):
    path = settings.DATA_FOLDER
    df_list = load_df_list()
    for _df in df_list:
        if df == _df:
            os.remove(F'{path}/{df}')
            break


def remove_ds_df(ds_name: str):
    path = settings.DATA_FOLDER
    df_list = load_ds_df_list(ds_name)
    for _df in df_list:
        os.remove(F'{path}/{_df}')
        break


def get_random_df(ds_name: str):
    folder = settings.DATASETDEF_FOLDER
    if not ds_name.endswith('.csv'):
        ds_name += '.csv'

    # Get random row from dataset
    dsf = pd.read_csv(F'{folder}/{ds_name}', index_col=0)
    r = random.randint(0, len(dsf))
    row = dsf.iloc[r]
    d_name = F'{craft_instrument_filename(row["symbol"], row["interval"], row["period"])}'

    return d_name


#   DataSet

def load_dataset(ds_name: str) -> pd.DataFrame:
    folder = settings.DATASETDEF_FOLDER
    if not ds_name.endswith('.csv'):
        ds_name += '.csv'
    full_path = F'{folder}/{ds_name}'

    # If file doesn't exist
    if not file_exists(full_path):
        print(F'Failed to find file at {full_path}!')
        return None

    dsf = pd.read_csv(full_path, index_col=0)
    print(F'Reading {full_path}')
    return dsf


def load_dataset_list():
    """Load list of dataset files in datasetdef."""
    path = settings.DATASETDEF_FOLDER
    # Get list of files that end with .csv
    df_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    df_list.sort()
    return df_list


def save_dataset(ds_name, dsf):
    folder = settings.DATASETDEF_FOLDER
    if not ds_name.endswith('.csv'):
        ds_name += '.csv'
    # save new dsf into ds_name
    dsf.to_csv(F'{folder}/{ds_name}')
    print(F'Saving into {folder}/{ds_name}')


def write_dataset(ds_name, dsf):
    folder = settings.DATASETDEF_FOLDER
    os.makedirs(folder, exist_ok=True)
    if not ds_name.endswith('.csv'):
        ds_name = F'{ds_name}.csv'
    dsf.to_csv(F'{folder}/{ds_name}')
    print(F'Creating dataset {folder}/{ds_name}')


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


def remove_from_dataset(ds_name: str, symbol: str, interval: str, period: str):
    dsf = load_dataset(ds_name)
    dsf = dsf.drop(dsf[(dsf.symbol == symbol) & (dsf.interval == interval) & (dsf.period == period)].index)
    print(F"Removing {symbol}-{interval}-{period} from {ds_name}")
    dsf = dsf.reset_index(drop=True)
    write_dataset(ds_name, dsf)


def remove_dataset(ds_name: str):
    folder = settings.DATASETDEF_FOLDER
    print(F'Removing {ds_name} completely.')
    if not ds_name.endswith('.csv'):
        ds_name += '.csv'
    full_path = F'{folder}/{ds_name}'
    if not os.path.exists(full_path):
        return False
    if 'windows' in full_path.lower():
        return False
    os.remove(full_path)
    return True


def number_of_datafiles(ds_name_list):
    total_len = 0
    for ds_name in ds_name_list:
        dsf = load_dataset(ds_name)
        total_len += len(dsf.index)
    return total_len


# Dataset Files

def load_dataset_data_list(ds_name: str) -> List[str]:
    folder = 'static/data/'

    ds_df = load_df(ds_name)
    d_list = []
    for index, row in ds_df.iterrows():
        # Form d_name (SYM_INT_PER)
        d_name = F'{row["symbol"]}__{row["interval"]}__{row["period"]}'
        d_list.append(d_name)

    return d_list


def load_dataset_data(d_list: List[str]) -> List[pd.DataFrame]:
    all_data = []
    for d_name in d_list:
        all_data.append(load_dataset(d_name))

    return all_data


# Dataset-Changes

def get_dataset_changes() -> pd.DataFrame:
    path = settings.DATASET_CHANGES_PATH
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
    path = settings.DATASET_CHANGES_PATH
    dsc = pd.read_csv(F'{os.getcwd()}/{path}', index_col=0)
    if not ds_name.endswith('.csv'):
        ds_name = F'{ds_name}.csv'
    print("---------------")
    if ds_name in list(dsc['name']):
        print(F'Overwriting dataset {ds_name} - Abort, already most updated')
    else:
        _new = pd.DataFrame([[ds_name]], columns=['name'], index=[len(dsc.index)])
        dsc = dsc.append(_new)
        print(F'Overwriting dataset {ds_name}')
        write_dataset_change(dsc)


def write_dataset_change(dsc_df: pd.DataFrame):
    path = settings.DATASET_CHANGES_PATH
    print(F'Noting changes in datasetchanges.txt')
    dsc_df.to_csv(path)


def remove_dataset_change(ds_name: str):
    dsc = get_dataset_changes()
    dsc.drop(dsc[dsc.name == ds_name].index)
    set_dataset_changes(dsc)


def clear_dataset_changes():
    path = settings.DATASET_CHANGES_PATH

    df = pd.DataFrame(columns=['name'])
    df.to_csv(path)


def set_dataset_changes(dsc: pd.DataFrame):
    path = settings.DATASET_CHANGES_PATH
    dsc.to_csv(path)


# List of instruments

def load_speed_suggestions():
    return settings.SUGGESTIONS['simulation']['speed']


def load_contract_size_suggestions():
    return settings.SUGGESTIONS['contract_size']


def load_flat_commission_suggestions():
    return settings.SUGGESTIONS['flat_commission']


def load_capital_suggestions():
    return settings.SUGGESTIONS['capital']


def load_lag_suggestions():
    return settings.SUGGESTIONS['lag']


def load_leverage_suggestions():
    return settings.SUGGESTIONS['leverage']


def load_optim_depth_suggestions():
    return settings.SUGGESTIONS['optim_depth']


def load_setting(name: str):
    return settings.SUGGESTIONS[name]


def load_instrument_type_suggestions():
    return settings.SUGGESTIONS['instrument_type']


def load_symbol_suggestions() -> pd.DataFrame:
    common_symbols = F'static/common/common_symbols.txt'
    ss_df = pd.read_csv(common_symbols, index_col=0)
    return ss_df['symbol']


def write_symbol_suggestions(ss_df: pd.DataFrame):
    common_symbols = F'static/common/common_symbols.txt'
    ss_df.to_csv(common_symbols)


def add_symbol_suggestion(ss_add):
    ss_df = load_symbol_suggestions()
    ss_df = ss_df.append(ss_add)
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


def add_period_suggestions(ps2: pd.DataFrame):
    ps_df = load_interval_suggestions()
    ps_df = ps_df.append(ps2)
    write_period_suggestions(ps_df)


# Simulation

def load_sim_speed_suggestions():
    return settings.SUGGESTIONS['sim_speed']


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


def load_trade_advisor_list():
    path = F'robot'
    # Get list of files that end with .py
    robot_list = [os.path.splitext(f)[0] for f in listdir(path) if isfile(join(path, f)) and
                  f.endswith('.py') and '__init__' not in f]
    return robot_list


# Algos

def load_algo_list():
    path = F'robot'
    # Get list of files that end with .py
    robot_list = [os.path.splitext(f)[0] for f in listdir(path) if isfile(join(path, f)) and
                  f.endswith('.py') and '__init__' not in f]
    return robot_list


# IVar


def ivar_to_list(idf: pd.DataFrame):
    ivar = []
    for col in idf:
        if not col == 'ivar_name':
            ivar.append(idf.loc[:, col])
    return ivar


def load_ivar(ta_name: str, ivar_name: str):
    idf = load_ivar_df(ta_name)
    return idf[idf.index == ivar_name]


def load_ivar_vars(ta_name: str, ivar_name: str):
    keys = load_ivar(ta_name, ivar_name).keys()
    for key in keys:
        if key in ['name', 'fitness', 'type']:
            keys.remove(key)
    return load_ivar(ta_name, ivar_name).keys()


def load_ivar_as_dict(ta_name: str, ivar_name: str):
    idf = load_ivar(ta_name, ivar_name)
    ivar_dict = {}
    for col in idf.columns:
        ivar_dict.update({
            col: {
                'default': idf[col][0],
            }
        })
    ivar_dict.update({
        'name': {
            'default': idf.index[-1]
        }
    })
    return ivar_dict


def load_ivar_df(ta_name: str, meta=False) -> pd.DataFrame:
    """Load iVar entry point"""
    folder = F'{settings.ROBOT_FOLDER}{settings.IVAR_SUB_FOLDER}'
    ivar_file = F'{ta_name}_ivar'
    path = F'{folder}/{ivar_file}.csv'

    if not file_exists(path):
        generate_ivar(ta_name)

    idf = pd.read_csv(path, index_col='name')
    # Strip meta attributes
    if not meta:
        if 'type' in idf.columns:
            idf.drop('type', inplace=True, axis=1)
        if 'fitness' in idf.columns:
            idf.drop('fitness', inplace=True, axis=1)

    return idf


def load_ivar_as_list(ta_name: str, ivar_name: str):
    """Robot takes in input as a List."""
    idf = load_ivar(ta_name, ivar_name)
    ivars = []
    for col in idf.columns:
        ivars.append(idf[col][0])
    return ivars


def load_ivar_list(ta_name: str):
    """Returns IVar names only"""
    idf = load_ivar_df(ta_name)
    # folder = F'robot/ivar'
    # ivar_file = F'{ta_name}_ivar'
    # path = F'{folder}/{ivar_file}.csv'
    #
    # if not file_exists(path):
    #     generate_ivar(ta_name)
    #
    # idf = pd.read_csv(path, index_col=0)
    return list(idf.index)


def load_ivar_file_list():
    path = F'robot/ivar'
    # Get list of files that end with .csv
    ivar_file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    return ivar_file_list


def generate_ivar(ta_name: str):
    folder = F'robot/ivar'
    ivar_file = F'{ta_name}_ivar'
    path = F'{folder}/{ivar_file}.csv'
    # args_str = eval(F'{ta_name}.{ta_name}.ARGS_STR')
    # args = eval(F'{ta_name}.{ta_name}.ARGS_DEFAULT')
    args_dict = eval(F'{ta_name}.{ta_name}.ARGS_DICT')
    data = {
        # meta
        'name': ['*Default'],
        'fitness': [-1],
        'type': ['default'],
    }
    # for i in range(len(args_str)):
    #     data.update({args_str[i]: args[i]})
    for key in args_dict.keys():
        data.update({
            key: [args_dict[key]['default']]
        })

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path, index=False)


def ivar_to_arr(idf: pd.DataFrame):
    columns = idf.columns
    arr = []

    for i in range(len(idf.index)):  # rows
        _arr = []
        for u in range(len(columns)):
            _arr.append(idf[columns[u + 1]][i])
        arr.append(_arr)
    return arr


def insert_ivar(ta_name: str, ivar):
    ivar_list = []
    if 'ivar' not in ivar:
        # If not in dict form
        ivar_list.append({
            'ivar': ivar
        })
    else:
        ivar_list.append(ivar)
    insert_ivars(ta_name, ivar_list)


def insert_ivars(ta_name: str, ivar_list):
    path = get_ivar_path(ta_name)

    # Load stored ivars
    idf = load_ivar_df(ta_name)
    cols = list(idf.columns)
    if 'name' not in cols:  # Name is used as index
        cols.append('name')

    # Sanity check
    if len(ivar_list[-1]['ivar'].keys()) != len(cols) - 1:
        # len of ivar keys != len of ivar meta - name (fitness and type excluded automatically)
        print(F'Insert ivars not compatible. Forcing insert: {ivar_list}')
    one_true = False
    for col in cols:
        for key in ivar_list[-1]['ivar'].keys():
            if col == key:
                one_true = True
                break
        if one_true:
            break
    if not one_true:
        print(F'Not a single ivar column is similar. Cancelling process.')
        return

    # Add ivars
    for ivar_dict in ivar_list:
        data = {}
        # Flatten ivar dict structure
        ivar = ivar_dict['ivar']
        for key in ivar:
            if key not in cols:
                continue  # Ignore if not inside.
            if key == 'name':  # Add name as index
                continue
            data.update({
                key: [ivar[key]['default']],
            })

        # Create indexer
        name = ["none"]
        if 'name' in ivar_dict['ivar']:
            name = [ivar['name']['default']]

        # Fill in meta attributes
        data.update({
            'type': ['unknown'],
            'fitness': [0],
        })
        if 'fitness' in ivar_dict:
            data.update({
                'fitness': [ivar_dict['fitness']],
            })
        if 'type' in ivar_dict:
            data.update({
                'type': [ivar_dict['type']],
            })
        n_idf = pd.DataFrame(data, index=name)
        idf = idf.append(n_idf)
    idf.index.name = 'name'
    idf.to_csv(path)


def delete_ivar(ta_name: str, ivar_name: str):
    path = get_ivar_path(ta_name)

    # Load stored ivars
    idf = load_ivar_df(ta_name)

    # Drop row
    idf.drop(labels=ivar_name, axis=0)

    idf.index.name = 'name'
    idf.to_csv(path)


def get_ivar_path(ta_name):
    folder = F'robot/ivar'
    ivar_file = F'{ta_name}_ivar'
    path = F'{folder}/{ivar_file}.csv'
    return path


def get_test_steps(ds_name: str):
    """Get number of dataset(s)"""
    dsf = load_dataset(ds_name)
    return len(dsf)


def result_dict_to_dataset(result_dict):
    data = {}
    for key in result_dict:
        data.update({
            key: [result_dict[key]]
        })
    return pd.DataFrame(data, index=False)


# Algo's IVar

def load_algo_ivar_list():
    path = F'{settings.ALGO_FOLDER}{settings.IVAR_SUB_FOLDER}/'
    # Get list of files that end with .csv
    ivar_list = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
    return ivar_list


def load_algo_ivar_df(algo_name: str, meta=False):
    """May return empty dictionary. If there is no ivar file, no ivar dictionary assumed."""
    folder = F'{settings.ALGO_FOLDER}{settings.IVAR_SUB_FOLDER}'
    ivar_file = F'{algo_name}_ivar'
    path = F'{folder}/{ivar_file}.csv'

    if not file_exists(path):
        return pd.DataFrame()

    aidf = pd.read_csv(path, index_col='name')

    # Strip meta attributes
    if not meta:
        if 'type' in aidf.columns:
            aidf.drop('type', inplace=True, axis=1)
        if 'fitness' in aidf.columns:
            aidf.drop('fitness', inplace=True, axis=1)

    return aidf


def load_algo_ivar(algo_name: str, ivar_name: str):
    aidf = load_algo_ivar_df(algo_name)
    # for i, row in aidf.iterrows():
    #     if ivar_name == row['name']:
    #         return row
    # return None
    return aidf[aidf.index == ivar_name]


def load_algo_ivar_as_dict(algo_name: str, ivar_name: str):
    aidf = load_algo_ivar(algo_name, ivar_name)
    ivar_dict = {}
    for col in aidf.columns:
        ivar_dict.update({
            col: {
                'default': aidf[col][0],
            }
        })
    ivar_dict.update({
        'name': {
            'default': aidf.index[-1]
        }
    })
    return ivar_dict


# XVar

def build_generic_xvar():
    return {
        'lag': '',
        'type': '',
        'instrument_type': '',
        'capital': '',
        'leverage': '',
        'commision': '',
    }


def translate_xvar_dict(xvar):
    # '10 ms' -> 10
    if not 'lag' in xvar.keys():
        xvar['lag'] = load_lag_suggestions()[0]
    xvar['lag'] = try_int(xvar['lag'].split(' ')[0])

    # 10000 or '10000' -> 10000
    if not 'capital' in xvar.keys():
        xvar['capital'] = load_capital_suggestions()[0]
    xvar['capital'] = try_int(xvar['capital'])

    # '1:100' -> 100, '10:1' -> 0.1
    if not 'leverage' in xvar.keys():
        xvar['leverage'] = load_leverage_suggestions()[0]
    # leverage_to_float
    xvar['leverage'] = (lambda x: try_int(x[1]) / try_int(x[0]))(xvar['leverage'].split(':'))

    # Currency_Type # Do nothing

    # 10000 or '10000' -> 10000
    if not 'commission' in xvar.keys():
        xvar['commission'] = load_flat_commission_suggestions()[0]
    xvar['commission'] = try_int(xvar['commission'])

    if not 'contract_size' in xvar.keys():
        xvar['contract_size'] = load_contract_size_suggestions()[0]
    xvar['contract_size'] = try_int(xvar['contract_size'])

    return xvar


# Data modification

def columnify_datetime(df: pd.DataFrame):
    """Yahoo timeseries data has inconsistent column naming, datetime for intra-day and date for inter-day.
    This converts all date-like columns to 'datetime' column. The value itself will be left unchanged."""
    if 'date' in df.columns:
        return df.rename(columns={'date': 'datetime'})
    return df


# Init basic files


def init_common():
    datasetchanges = F'{os.getcwd()}/static/common/datasetchanges.txt'
    common_intervals = F'{os.getcwd()}/static/common/common_intervals.txt'
    common_periods = F'{os.getcwd()}/static/common/common_periods.txt'
    common_symbols = F'{os.getcwd()}/static/common/common_symbols.txt'

    if not file_exists(datasetchanges) or file_is_empty(datasetchanges):
        df = pd.DataFrame({
            'name': load_dataset_list()
        })
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
            'symbol': settings.SUGGESTIONS['symbols'],
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


# Hardware

def get_computer_specs():
    uname = platform.uname()
    svmem = psutil.virtual_memory()
    specs = {
        'system': uname.system,
        'n_cores': psutil.cpu_count(logical=True),
        'total_memory': get_size_bytes(svmem.total),
        'free_memory': get_size_bytes(svmem.available),
    }
    i = 0
    for partition in psutil.disk_partitions():
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        specs.update({
            F'device_{i}': partition.device,
            F'free_space_{i}': get_size_bytes(partition_usage.free),
        })
        i += 1

    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        id = gpu.id
        specs.update({
            F'name_{id}': gpu.name,
            F'gpu_load_{id}': F'{gpu.load * 100}%',
            F'gpu_free_mem_{id}': F'{gpu.memoryFree * 100} mb',
            F'gpu_total_mem_{id}': F'{gpu.memoryTotal * 100} mb',
        })

    return specs


# Stats

def try_stdev(list):
    if len(list) < 2:
        return 0
    return stdev(list)

# https://www.thepythoncode.com/article/get-hardware-system-information-python
# ======================================== System Information ========================================
# System: Linux
# Node Name: rockikz
# Release: 4.17.0-kali1-amd64
# Version: #1 SMP Debian 4.17.8-1kali1 (2018-07-24)
# Machine: x86_64
# Processor:
# ======================================== Boot Time ========================================
# Boot Time: 2019/8/21 9:37:26
# ======================================== CPU Info ========================================
# Physical cores: 4
# Total cores: 4
# Max Frequency: 3500.00Mhz
# Min Frequency: 1600.00Mhz
# Current Frequency: 1661.76Mhz
# CPU Usage Per Core:
# Core 0: 0.0%
# Core 1: 0.0%
# Core 2: 11.1%
# Core 3: 0.0%
# Total CPU Usage: 3.0%
# ======================================== Memory Information ========================================
# Total: 3.82GB
# Available: 2.98GB
# Used: 564.29MB
# Percentage: 21.9%
# ==================== SWAP ====================
# Total: 0.00B
# Free: 0.00B
# Used: 0.00B
# Percentage: 0%
# ======================================== Disk Information ========================================
# Partitions and Usage:
# === Device: /dev/sda1 ===
#   Mountpoint: /
#   File system type: ext4
#   Total Size: 451.57GB
#   Used: 384.29GB
#   Free: 44.28GB
#   Percentage: 89.7%
# Total read: 2.38GB
# Total write: 2.45GB
# ======================================== Network Information ========================================
# === Interface: lo ===
#   IP Address: 127.0.0.1
#   Netmask: 255.0.0.0
#   Broadcast IP: None
# === Interface: lo ===
# === Interface: lo ===
#   MAC Address: 00:00:00:00:00:00
#   Netmask: None
#   Broadcast MAC: None
# === Interface: wlan0 ===
#   IP Address: 192.168.1.101
#   Netmask: 255.255.255.0
#   Broadcast IP: 192.168.1.255
# === Interface: wlan0 ===
# === Interface: wlan0 ===
#   MAC Address: 64:70:02:07:40:50
#   Netmask: None
#   Broadcast MAC: ff:ff:ff:ff:ff:ff
# === Interface: eth0 ===
#   MAC Address: d0:27:88:c6:06:47
#   Netmask: None
#   Broadcast MAC: ff:ff:ff:ff:ff:ff
# Total Bytes Sent: 123.68MB
# Total Bytes Received: 577.94MB


# Example:

# tech_stocks = ['AAPL', 'MSFT', 'INTC']
# bank_stocks = ['WFC', 'BAC', 'C']
# commodity_futures = ['GC=F', 'SI=F', 'CL=F']
# cryptocurrencies = ['BTC-USD', 'ETH-USD', 'XRP-USD']
# currencies = ['EURUSD=X', 'JPY=X', 'GBPUSD=X']
# mutual_funds = ['PRLAX', 'QASGX', 'HISFX']
# us_treasuries = ['^TNX', '^IRX', '^TYX']

# USD/CAD	EUR/JPY
# EUR/USD	EUR/CHF
# USD/CHF	EUR/GBP
# GBP/USD	AUD/CAD
# NZD/USD	GBP/CHF
# AUD/USD	GBP/JPY
# USD/JPY	CHF/JPY
# EUR/CAD	AUD/JPY
# EUR/AUD	AUD/NZD