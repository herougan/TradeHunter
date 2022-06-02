import copy
import datetime
import importlib
import os
import random
from datetime import datetime, timedelta
import math
from os import listdir
from os.path import isfile, join
from typing import List

import PyQt5
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressBar, QPlainTextEdit
from matplotlib import pyplot as plt

from robot.abstract.robot import robot
from settings import EVALUATION_FOLDER, OPTIMISATION_FOLDER, PLOTTING_SETTINGS, TESTING_SETTINGS, OPTIMISATION_SETTINGS, \
    ALGO_ANALYSIS_FOLDER
from util.dataGraphingUtil import plot_robot_instructions, plot_signals, plot_open_signals, candlestick_plot, \
    get_interval, DATE_FORMAT_DICT, plot_line, plot_optimisations
from util.dataRetrievalUtil import load_dataset, load_df, get_computer_specs, number_of_datafiles, try_stdev, \
    insert_ivars, file_exists, try_delete_file
from util.langUtil import craft_instrument_filename, strtodatetime, try_key, remove_special_char, try_divide, try_max, \
    try_mean, get_test_name, get_file_name, get_instrument_from_filename, \
    try_min, try_sgn, in_std_range, is_datetimestring

#  Robot
# from robot import FMACDRobot, TwinSMA
from util.mathUtil import date_to_index_arr


def step_test_robot(r: robot, step: int):
    """..."""
    r.step(step)  # this is incorrect! step test should run through the current data file(s) in the dataset (idx=step)


def create_profit_df_from_list(profit, assets):
    # ,profits, assets ->
    # Profit_d: index, total_profit_value, total_assets_value
    data = {
        'total_profit_value': profit,
        'total_assets_value': assets,
    }
    return pd.DataFrame(data, index=0)


def create_signal_df_from_list(completed, leftover, headers):
    """Completed, Leftover: [{ 'date': date, 'vol': vol, ...}...],
    -> Signal Df: columns = ['', completed's cols]"""
    sdata = {
    }
    for header in headers:
        sdata.update({
            header: []
        })

    for _completed in completed:
        for header in headers:
            sdata[header].append(_completed[header])
    for _leftover in leftover:
        for header in headers:
            sdata[header].append(_leftover[header])
    sdf = pd.DataFrame(sdata, index=False)

    return sdf


def base_summary_dict():
    summary_dict = {
        'period': timedelta(0),
        'n_bars': 0,
        'ticks': 0,
        #
        'total_profit': 0,
        'gross_profit': 0,
        'gross_loss': 0,
        'virtual_profit': 0,
        #
        'profit_factor': 0,
        'recovery_factor': 0,
        'growth_factor': 0,
        'AHPR': 0,
        'GHPR': 0,
        #
        'total_trades': 0,
        'total_deals': 0,
        #
        'balance_drawdown_abs': 0,
        'balance_drawdown_max': 0,
        'balance_drawdown_rel': 0,
        'balance_drawdown_avg': 0,
        'balance_drawdown_len_avg': 0,
        'balance_drawdown_len_max': 0,
        'equity_drawdown_abs': 0,
        'equity_drawdown_max': 0,
        'equity_drawdown_rel': 0,
        'equity_drawdown_avg': 0,
        'equity_drawdown_len_avg': 0,
        'equity_drawdown_len_max': 0,
        #
        'expected_payoff': 0,
        'sharpe_ratio': 0,
        'standard_deviation': 0,
        'LR_correlation': 0,
        'LR_standard_error': 0,
        #
        'total_short_trades': 0,
        'total_long_trades': 0,
        'short_trades_won': 0,
        'long_trades_won': 0,
        'trades_won': 0,
        'trades_lost': 0,
        # 'short_trades_won_p': 0, # Calculable statistic
        # 'long_trades_won_p': 0,
        # 'trades_won_p': 0,
        # 'trades_lost_p': 0,
        #
        'largest_profit_trade': 0,
        'average_profit_trade': 0,
        'largest_loss_trade': 0,
        'average_loss_trade': 0,
        #
        'longest_trade_length': 0,
        'shortest_trade_length': 0,
        'average_trade_length': 0,
        'average_profit_length': 0,
        'average_loss_length': 0,
        'period_to_profit': 0,
        'period_to_gross': 0,
        #
        'max_consecutive_wins': 0,
        'max_consecutive_profit': 0,
        'avg_consecutive_wins': 0,
        'avg_consecutive_profit': 0,
        'max_consecutive_losses': 0,
        'max_consecutive_loss': 0,
        'avg_consecutive_losses': 0,
        'avg_consecutive_loss': 0,
        #
        'n_symbols': 0,
        'margin_level': 0,
        'z_score': 0,
        #
    }

    return summary_dict


# == Testing ==


def create_summary_df_from_list(stats_dict, signals_dict, df, additional={}):
    """Takes in the profit-loss dataframe, buy-sell signals,
    produces the data-1 summary stat dictionary"""

    # (profit_d, equity_d, signals, df, additional={}
    profit_d = stats_dict['profit']
    equity_d = stats_dict['equity']
    signals = signals_dict['signals']

    summary_dict = base_summary_dict().copy()
    summary_dict.update(additional)

    complete_signals, incomplete_signals, profit_signals, loss_signals = [], [], [], []
    long_signals, short_signals = [], []
    for signal in signals:
        if signal['end']:
            complete_signals.append(signal)
            if signal['net'] > 0:
                profit_signals.append(signal)
            else:
                loss_signals.append(signal)
        else:
            incomplete_signals.append(signal)
        if signal['type'] == 1:  # 1 is long, 2 is short
            long_signals.append(signal)
        else:
            short_signals.append(signal)

    l = len(complete_signals)
    l2 = len(signals_dict['open_signals'])
    l3 = len(profit_signals)
    l4 = len(loss_signals)
    l5 = len(long_signals)
    l6 = len(short_signals)

    summary_dict['period'] = stats_dict['datetime'][-1] - stats_dict['datetime'][0]
    summary_dict['n_bars'] = len(df.index)
    summary_dict['ticks'] = len(df.index)  # not applicable here
    #
    summary_dict['total_profit'] = profit_d[-1]
    summary_dict['gross_profit'] = stats_dict['gross_profit'][-1]
    summary_dict['gross_loss'] = stats_dict['gross_loss'][-1]
    summary_dict['virtual_profit'] = stats_dict['virtual_profit'][-1]

    # deepest loss in drawdown - stats
    mean = 0
    gmean = 0
    for signal in signals:
        mean += signal['net']
        gmean *= signal['net']
    summary_dict['AHPR'] = try_divide(mean, l)
    summary_dict['GHPR'] = math.pow(gmean, try_divide(1, l))

    summary_dict['profit_factor'] = try_divide(summary_dict['total_profit'], summary_dict['gross_profit'])
    summary_dict['growth_factor'] = try_divide(summary_dict['total_profit'], stats_dict['capital'])
    # summary_dict['recovery_factor'] = summary_dict['total_profit'] / summary_dict['max_drawdown']  # Implemented below

    summary_dict['total_trades'] = l
    summary_dict['total_deals'] = l  # + unclosed deals

    prev_d = 0
    drawdown, drawdowns = False, []
    drawdown_l, lowest_d = 1, 0
    for d in profit_d:  # Detect drawdowns
        if drawdown:
            if d < prev_d:
                drawdown_l += 1
                if d < lowest_d:
                    lowest_d = d
            else:
                drawdowns.append({
                    'length': drawdown_l,
                    'depth': prev_d - lowest_d,
                    'ledge': prev_d,
                })
                # reset
                prev_d = d
                drawdown = False
                drawdown_l, lowest_d = 1, 0
        else:
            if d < prev_d:
                drawdown = True
                lowest_d = d
            prev_d = d
    if drawdown:  # last 'unsettled' drawdown
        drawdowns.append({
            'length': drawdown_l + 1,
            'depth': prev_d - lowest_d,
            'ledge': prev_d,
        })
    greatest_depth, avg_depth = 0, 0  # depths are positive
    greatest_len, avg_len = 0, 0
    for drawdown in drawdowns:
        avg_depth += drawdown['depth']
        avg_len += drawdown['length']
        if greatest_len < drawdown['length']:
            greatest_len = drawdown['length']
    p_min = min(profit_d)
    summary_dict['balance_drawdown_abs'] = profit_d[0] - p_min
    summary_dict['balance_drawdown_max'] = greatest_depth
    min_drawdown = [dd for dd in drawdowns if dd['depth'] == p_min]
    if min_drawdown:
        summary_dict['balance_drawdown_rel'] = min_drawdown[0]['ledge'] - p_min
    else:
        summary_dict['balance_drawdown_rel'] = profit_d[0] - p_min
    summary_dict['balance_drawdown_avg'] = 0
    summary_dict['balance_drawdown_len_avg'] = 0
    if len(drawdowns) != 0:
        summary_dict['balance_drawdown_avg'] = try_divide(avg_depth, len(drawdowns))
        summary_dict['balance_drawdown_len_avg'] = try_divide(avg_len, len(drawdowns))
    summary_dict['balance_drawdown_len_max'] = greatest_len

    prev_d = 0  # Same as above but for equity
    drawdown, drawdowns = False, []
    drawdown_l, lowest_d = 0, 0
    for d in equity_d:  # Detect drawdowns
        if drawdown:
            if d < prev_d:
                drawdown_l += 1
                if d < lowest_d:
                    lowest_d = d
            else:
                drawdowns.append({
                    'length': drawdown_l,
                    'depth': prev_d - lowest_d,
                    'ledge': prev_d,
                })
                # reset
                prev_d = d
                drawdown = False
                drawdown_l, lowest_d = 0, 0
        else:
            if d < prev_d:
                drawdown = True
                lowest_d = d
            prev_d = d
    if drawdown:  # last 'unsettled' drawdown
        drawdowns.append({
            'length': drawdown_l + 1,
            'depth': 0,
            'ledge': prev_d,
        })
    greatest_depth, avg_depth = 0, 0  # depths are positive
    greatest_len, avg_len = 0, 0
    for drawdown in drawdowns:
        avg_depth += drawdown['depth']
        avg_len += drawdown['length']
        if greatest_len < drawdown['length']:
            greatest_len = drawdown['length']
    p_min = min(profit_d)
    summary_dict['equity_drawdown_abs'] = profit_d[0] - p_min
    summary_dict['equity_drawdown_max'] = greatest_depth
    min_drawdown = [dd for dd in drawdowns if dd['depth'] == p_min]
    if min_drawdown:
        summary_dict['equity_drawdown_rel'] = min_drawdown[0]['ledge'] - p_min
    else:
        summary_dict['equity_drawdown_rel'] = profit_d[0] - p_min
    summary_dict['equity_drawdown_avg'] = try_divide(avg_depth, len(drawdowns))
    summary_dict['equity_drawdown_len_avg'] = try_divide(avg_len, len(drawdowns))
    summary_dict['equity_drawdown_len_max'] = greatest_len

    expected_payoff = 0
    for signal in complete_signals:
        expected_payoff += signal['net']
    summary_dict['expected_payoff'] = try_divide(summary_dict['total_profit'], l)
    summary_dict['sharpe_ratio'] = 0  # No risk-free proxy
    # todo https: // www.investopedia.com / terms / s / sharperatio.asp  US treasury rate
    summary_dict['standard_deviation'] = 0
    summary_dict['LR_correlation'] = 0  # no line correlation yet
    summary_dict['LR_standard_error'] = 0
    if summary_dict['equity_drawdown_max'] == 0:
        summary_dict['recovery_factor'] = 0
    else:
        summary_dict['recovery_factor'] = summary_dict['total_profit'] / summary_dict['equity_drawdown_max']

    summary_dict['total_short_trades'] = l6
    summary_dict['total_long_trades'] = l5
    summary_dict['short_trades_won'] = len([signal for signal in short_signals if signal['net'] > 0])
    summary_dict['long_trades_won'] = len([signal for signal in long_signals if signal['net'] > 0])
    summary_dict['trades_won'] = l3
    summary_dict['trades_lost'] = l4

    largest_profit, average_profit = 0, 0
    largest_loss, average_loss = 0, 0
    for signal in profit_signals:
        average_profit += signal['net']
        if largest_profit > signal['net']:
            largest_profit = signal['net']
    for signal in loss_signals:
        average_loss += signal['net']
        if largest_loss < signal['net']:
            largest_loss = signal['net']
    summary_dict['largest_profit_trade'] = largest_profit
    summary_dict['average_profit_trade'] = try_divide(average_profit, l3)
    summary_dict['largest_loss_trade'] = largest_loss
    summary_dict['average_loss_trade'] = try_divide(average_loss, l3)

    # summary_dict['longest_trade'] = 0
    # summary_dict['longest_profit_trade'] = 0
    # summary_dict['longest_loss_trade'] = 0
    # summary_dict['average_trade_period'] = 0

    interval = get_interval(df).total_seconds()
    # if 'Datetime' in df.columns:  # -1 - -2 doesn't work
    #     interval = (strtodatetime(df.iloc[1].Datetime) - strtodatetime(df.iloc[0].Datetime)).total_seconds()
    # elif 'Date' in df.columns:
    #     interval = (strtodatetime(df.iloc[1].Date) - strtodatetime(df.iloc[0].Date)).total_seconds()
    # else:
    #     col = df.columns[0]
    #     interval = (strtodatetime(df.iloc[1][col]) - strtodatetime(df.iloc[0][col])).total_seconds()

    period_to_net, period_to_gross, avg_period = 0, 0, 0
    avg_profit_period, avg_loss_period = 0, 0
    longest_length, shortest_length = 0, len(df)
    winning, losing = False, False
    win_length, lose_length = 0, 0
    win_amount, lose_amount = 0, 0
    win_streak, lose_streak = [], []
    for signal in signals:
        net = abs(signal['net'])
        period = (strtodatetime(signal['end']) - strtodatetime(signal['start'])).total_seconds() / interval
        avg_period += period
        if signal['net'] > 0:
            avg_profit_period += period
        else:
            avg_loss_period += period
        if longest_length < period:
            longest_length = period
        if shortest_length > period:
            shortest_length = period
        period_to_net += try_divide(net, period)
        if signal['net'] > 0:
            period_to_gross += try_divide(net, period)
        if signal['net'] > 0:
            if winning:
                win_length += 1
                win_amount += signal['net']
            else:
                winning = True
                win_length = 1
                win_amount = signal['net']

            if losing:
                losing = False
                lose_streak.append({
                    'length': lose_length,
                    'amount': lose_amount,  # negative number
                })
                lose_length, lose_amount = 0, 0
        else:
            if losing:
                lose_length += 1
                lose_amount += signal['net']
            else:
                losing = True
                lose_length = 1
                lose_amount = signal['net']

            if winning:
                win_streak.append({
                    'length': win_length,
                    'amount': win_amount,
                })
                winning = False
                win_length, win_amount = 0, 0

    summary_dict['period_to_profit'] = try_divide(period_to_net, l)
    summary_dict['period_to_gross'] = try_divide(period_to_gross, l3)

    summary_dict['longest_trade_length'] = longest_length
    summary_dict['shortest_trade_length'] = shortest_length
    summary_dict['average_trade_length'] = try_divide(avg_period, l)
    summary_dict['average_profit_length'] = try_divide(avg_profit_period, l3)
    summary_dict['average_loss_length'] = try_divide(avg_loss_period, l4)

    mean = lambda list: sum(list) / len(list)
    summary_dict['max_consecutive_wins'] = try_max([s['length'] for s in win_streak])
    summary_dict['max_consecutive_profit'] = try_max([s['amount'] for s in win_streak])
    summary_dict['avg_consecutive_wins'] = try_mean([s['length'] for s in win_streak])
    summary_dict['avg_consecutive_profit'] = try_mean([s['amount'] for s in win_streak])

    summary_dict['max_consecutive_losses'] = try_max([s['length'] for s in lose_streak])
    summary_dict['max_consecutive_loss'] = try_max([s['amount'] for s in lose_streak])
    summary_dict['avg_consecutive_losses'] = try_mean([s['length'] for s in lose_streak])
    summary_dict['avg_consecutive_loss'] = try_mean([s['amount'] for s in lose_streak])

    summary_dict['n_symbols'] = 1
    summary_dict['margin_level'] = 0
    summary_dict['z_score'] = 0

    # summary_dict['market_fee']

    return summary_dict


def aggregate_summary_df_in_dataset(ds_name: str, summary_dict_list: List):
    """Aggregates the results above from list of result dataframes.
    Result dataframes (from result_dict_to_dataset or otherwise) contain only 1 row."""

    profits = [d['total_profit'] for d in summary_dict_list]
    # gross_profits = [d['gross_profit'] for d in summary_dict_list]
    # gross_loss = [d['gross_loss'] for d in summary_dict_list]

    summary_dict = base_summary_dict().copy()
    for i in range(len(summary_dict_list)):
        for key in summary_dict_list[i]:
            if not key in summary_dict:
                summary_dict[key] = summary_dict_list[i][key]
            else:
                summary_dict[key] += summary_dict_list[i][key]
    for key in summary_dict:
        if len(summary_dict_list) > 0:
            summary_dict[key] /= len(summary_dict_list)
        else:
            if key == 'period':
                summary_dict[key] = timedelta(0)
            else:
                summary_dict[key] = 0

    final_summary_dict = {
        'n_instruments': len(summary_dict_list),
        'standard_deviation_profit': try_stdev(profits),
        #
        'dataset': ds_name,
    }

    summary_dict.update(final_summary_dict)
    final_summary_dict = summary_dict
    return final_summary_dict


def aggregate_summary_df_in_datasets(summary_dict_list: List):
    """Aggregate summary and dataset summaries"""

    summary_dict_list = [summary_dict for summary_dict in summary_dict_list if len(summary_dict.keys()) > 0]
    if len(summary_dict_list) < 1:
        return {}

    final_summary_dict = summary_dict_list[0].copy()
    # total_instruments = final_summary_dict['n_instruments']
    total_instruments = sum([s['n_instruments'] for s in summary_dict_list])
    for i in range(len(summary_dict_list)):
        if i == 0:
            continue
        for key in summary_dict_list[i]:
            if not key.lower() == 'dataset':
                final_summary_dict[key] += summary_dict_list[i][key] * summary_dict_list[i]['n_instruments'] \
                                           / total_instruments
        # total_instruments += summary_dict_list[i]['n_instruments']
    del final_summary_dict['dataset']

    # for key in final_summary_dict:
    #     if not key.lower() == 'dataset':
    #         final_summary_dict[key] = try_divide(final_summary_dict[key], total_instruments)

    final_summary_dict.update({
        'n_datasets': len(summary_dict_list),
        'name': 'Total',
        #
        'datasets': [s['dataset'] + ", " for s in summary_dict_list],
    })

    return final_summary_dict


def create_test_meta(test_name, ivar, xvar, other, get_specs=True):
    """Test meta file contains test name, ivar (dict version) used, start and end date etc
    most importantly, xvar (dict version) attributes"""
    meta = {
        # Robot meta output
        'test_name': test_name,
        'test_meta': other,
        'robot_meta': 0,
    }
    # IVar Dict
    meta.update(ivar)
    # XVar Dict
    meta.update(xvar)
    # Computer specs
    if get_specs:
        meta.update(get_computer_specs())

    return meta


def create_test_result(test_name: str, summary_dict_list, robot_name: str):
    # folder = F'{EVALUATION_FOLDER}/{robot_name}'
    # result_path = F'{folder}/{test_name}.csv'
    # meta_path = F'{folder}/{test_name}__meta.csv'

    summary_data = {}
    final_summary_dict = summary_dict_list[-1]  # Setup base keys
    for key in final_summary_dict:
        summary_data.update({
            key: [final_summary_dict[key]]
        })
    for i in range(summary_dict_list - 1):  # Add to dicta's lists
        for key in final_summary_dict:
            summary_data[key].append(try_key(summary_dict_list, key))

    summary_df = pd.DataFrame(summary_data, index=0)
    # summary_df.to_csv(result_path)
    # meta_df.to_csv(meta_path)
    return summary_df


def create_optim_meta(optim_name, ivar, xvar, other, get_specs=True):
    meta = {
        # Robot meta output
        'optim_name': optim_name,
        'optim_meta': other,
        'robot_meta': 0,
    }
    # IVar Dict
    meta.update(ivar)
    # XVar Dict
    meta.update(xvar)
    # Computer specs
    if get_specs:
        meta.update(get_computer_specs())

    return meta


def create_optim_result(optim_name, result_dict, robot_name: str):
    result_df = pd.DataFrame([result_dict])
    return result_df


def create_optim_series(optim_name, fitness_collection, ivar_collection):
    # todo create 'index'-series of ivar scores - create optim_series file
    # Fetch keys
    if len(ivar_collection) < 1:
        return pd.DataFrame()
    eg_args_dict = ivar_collection[0]['ivar']

    # Build data
    optim_series_data = {
        'fitness': fitness_collection,
    }
    for key in eg_args_dict.keys():  # For each key, collect their values through
        optim_series_data.update({
            key: [ivar[key]['default'] for ivar in ivar_collection]
        })
    optim_series = pd.DataFrame(index=[list(range(len(fitness_collection)))], data=optim_series_data)
    return optim_series


# Forex type

def get_optimised_robot_list():
    folder = F'{OPTIMISATION_FOLDER}'
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(F'{folder}')
    return files


def get_tested_robot_list():
    folder = F'{EVALUATION_FOLDER}'
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(F'{folder}')
    return files


def get_analysed_algo_list():
    folder = F'{ALGO_ANALYSIS_FOLDER}'
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(F'{folder}')
    return files


def get_tests_list(robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(F'{folder}')
    _files = []
    for file in files:
        if not get_file_name(file).endswith('__meta'):
            _files.append(get_test_name(file))
    return _files


def get_optimisations_list(robot_name: str):
    folder = F'{OPTIMISATION_FOLDER}/{robot_name}'
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(F'{folder}')
    _files = []
    for file in files:
        if not get_file_name(file).endswith('__meta'):
            _files.append(get_test_name(file))
    return _files
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(F'{folder}')
    _files = []
    for file in files:
        if not get_file_name(file).endswith('__meta'):
            _files.append(get_test_name(file))
    return _files


def get_algo_results_list(algo_name: str):
    folder = F'{ALGO_ANALYSIS_FOLDER}/{algo_name}'
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(F'{folder}')
    _files = []
    for file in files:
        if not get_file_name(file).endswith('__meta'):
            _files.append(get_test_name(file))
    return _files


def write_test_result(test_name: str, summary_dicts: List, robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    path = F'{folder}/{test_name}.csv'

    if not os.path.exists(folder):
        os.makedirs(folder)

    sdfs = []
    for summary_dict in summary_dicts:
        sdfs.append(summary_dict)
    sdfs = pd.DataFrame(sdfs)
    sdfs.to_csv(path)

    print(F'Writing test result at {path}')
    return sdfs


def write_test_meta(test_name: str, meta_dict, robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    meta_path = F'{folder}/{test_name}__meta.csv'

    if not os.path.exists(folder):
        os.makedirs(folder)

    meta = pd.DataFrame([meta_dict])
    meta.to_csv(meta_path)
    print(F'Writing test meta at {meta_path}')
    return meta


def write_optim_result(optim_name: str, result_df, robot_name: str):
    folder = F'{OPTIMISATION_FOLDER}/{robot_name}'
    result_path = F'{folder}/{optim_name}.csv'

    if not os.path.exists(folder):
        os.makedirs(folder)

    print(F'Writing optimisation result at {result_path}')
    result_df.to_csv(result_path)
    return result_df


def write_optim_meta(optim_name: str, meta_dict, robot_name: str):
    folder = F'{OPTIMISATION_FOLDER}/{robot_name}'
    meta_path = F'{folder}/{optim_name}__meta.csv'

    if not os.path.exists(folder):
        os.makedirs(folder)

    meta = pd.DataFrame([meta_dict])
    meta.to_csv(meta_path)
    print(F'Writing test meta at {meta_path}')
    return meta


def write_optim_series(optim_name: str, optim_sdf, robot_name: str):
    folder = F'{OPTIMISATION_FOLDER}/{robot_name}'
    meta_path = F'{folder}/{optim_name}__series.csv'
    pass


def load_test_result(test_name: str, robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    if not test_name.endswith('.csv'):
        test_name += '.csv'
    result_path = F'{folder}/{test_name}'

    if file_exists(result_path):
        trdf = pd.read_csv(result_path, index_col=0)
        return trdf
    else:
        print(F'Error! {result_path} cannot be found!')
    return pd.DataFrame()


def load_test_meta(meta_name: str, robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    if not meta_name.endswith('.csv'):
        meta_name += '.csv'
    meta_path = F'{folder}/{meta_name}'

    if file_exists(meta_path):
        tmdf = pd.read_csv(meta_path, index_col=0)
        return tmdf
    else:
        print(F'Error! {meta_path} cannot be found!')
    return pd.DataFrame()


def load_optimisation_result(optim_name: str, robot_name: str):
    folder = F'{OPTIMISATION_FOLDER}/{robot_name}'
    if not optim_name.endswith('.csv'):
        optim_name += '.csv'
    result_path = F'{folder}/{optim_name}'

    if file_exists(result_path):
        trdf = pd.read_csv(result_path, index_col=0)
        return trdf
    else:
        print(F'Error! {result_path} cannot be found!')
    return pd.DataFrame()


def load_optimisation_meta(meta_name: str, robot_name: str):
    folder = F'{OPTIMISATION_FOLDER}/{robot_name}'
    if not meta_name.endswith('.csv'):
        meta_name += '.csv'
    meta_path = F'{folder}/{meta_name}'

    if file_exists(meta_path):
        omdf = pd.read_csv(meta_path, index_col=0)
        return omdf
    else:
        print(F'Error! {meta_path} cannot be found!')
    return pd.DataFrame()


def load_algo_result(result_name: str, algo_name: str):
    folder = F'{ALGO_ANALYSIS_FOLDER}/{algo_name}'
    if not result_name.endswith('.csv'):
        result_name += '.csv'
    result_path = F'{folder}/{result_name}'

    if file_exists(result_path):
        trdf = pd.read_csv(result_path, index_col=0)
        return trdf
    else:
        print(F'Error! {result_path} cannot be found!')
    return pd.DataFrame()


def load_algo_meta(meta_name: str, algo_name: str):
    folder = F'{ALGO_ANALYSIS_FOLDER}/{algo_name}'
    if not meta_name.endswith('.csv'):
        meta_name += '.csv'
    meta_path = F'{folder}/{meta_name}'

    if file_exists(meta_path):
        omdf = pd.read_csv(meta_path, index_col=0)
        return omdf
    else:
        print(F'Error! {meta_path} cannot be found!')
    return pd.DataFrame()


def delete_test(test_name: str, robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    path = F'{folder}/{test_name}.csv'
    meta_path = F'{folder}/{test_name}__meta.csv'
    try_delete_file(path)
    try_delete_file(meta_path)


def delete_optimisation(optim_name: str, robot_name: str):
    folder = F'{OPTIMISATION_FOLDER}/{robot_name}'
    path = F'{folder}/{optim_name}.csv'
    meta_path = F'{folder}/{optim_name}__meta.csv'
    try_delete_file(path)
    try_delete_file(meta_path)


def delete_algo_result(result_name: str, algo_name: str):
    folder = F'{ALGO_ANALYSIS_FOLDER}/{algo_name}'
    path = F'{folder}/{result_name}.csv'
    meta_path = F'{folder}/{result_name}__meta.csv'
    try_delete_file(path)
    try_delete_file(meta_path)


# Robot/Algos eval


def get_ivar_vars(robot: str):
    args_dict = eval(F'{robot}.{robot}.ARGS_DICT')
    return args_dict


def get_algo_ivar_vars(algo: str):
    args_dict = eval(F'{algo}.{algo}.ARGS_DICT')
    return args_dict


# Stock type


class DataTester:

    def __init__(self, xvar):

        self.xvar = xvar
        self.start_time = None
        self.end_time = None

        # Progress
        self.p_bar_2 = None
        self.p_bar = None

        # Robot/Algo
        self.algo = None
        self.algo_ivar = None
        self.robot = None
        self.robot_ivar = None
        self.robot_fixed_ivar = None

    def bind_progress_bar(self, p_bar: QProgressBar):
        self.p_bar = p_bar
        # self.p_window = p_window

    def bind_progress_bar_2(self, p_bar_2: QProgressBar):
        self.p_bar_2 = p_bar_2

    # == Full run ==
    def live_simulate(self, ta_name: str, ivar: dict, dfs: List[str], test_name: str):
        """Live simulate"""
        # Pre-download data
        # robot.start()
        time_index = []
        for df in dfs:
            time_index.append(
                {
                    'last_index': 0, # some time
                    'last_time': 0,
                }
            )

        # Bundle all data since last time_index
        pass

    # == Test ==
    def test(self, ta_name: str, ivar: dict, ds_names: List[str], test_name: str, store=True, meta_store=True):
        """Runs a test against a particular set of datasets. (Sets of sets of data) Depending on the robot, test() is
        optimised with techniques such as running indicators only once."""

        self.start_time = datetime.now()
        ta_name = remove_special_char(ta_name)

        print('Going to test: ' + F'{ta_name}.{ta_name}({ivar}) with i:{ivar}, x:{self.xvar}')
        self.robot = eval(F'{ta_name}.{ta_name}({ivar}, {self.xvar})')

        if self.p_bar:
            self.p_bar.setMaximum(number_of_datafiles(ds_names) + 1)
            self.p_bar.setValue(0)
            # self.p_window.show()

        # Testing util functions
        def test_dataset(ds_name: str):
            dsdf = load_dataset(ds_name)
            summary_dict_list = []

            for index, row in dsdf.iterrows():

                if self.p_bar:
                    self.p_bar.setValue(self.p_bar.value() + 1)

                # Load dataframe
                d_name = F'{craft_instrument_filename(row["symbol"], row["interval"], row["period"])}'
                df = load_df(d_name)
                # If dataframe cannot be found:
                if len(df) < 1:
                    print(F'Testing: {d_name} cannot be found...')
                    continue

                # Testing dataframe here
                # self.p_window.setWindowTitle(F'Testing against {d_name}')
                stats_dict, signals_dict, robot_dict = test_data(df, row['symbol'], row['interval'], row['period'])
                summary_dict_list.append(create_summary_df_from_list(stats_dict, signals_dict, df))

            return aggregate_summary_df_in_dataset(ds_name, summary_dict_list)

        def test_data(df, sym, interval, period):

            # Set robot to testing mode
            print(F"Robot {self.robot} starting test {sym}-{interval}-{period}...")
            if "JYP-X" in sym:
                self.robot.set_lot_size(1000)
            self.robot.start(sym, interval, period, df, True)
            # Robot
            # df = retrieve("symbol", datetime.now(), datetime.now()
            #               - strtotimedelta(interval) * robot.PREPARE_PERIOD,
            #               interval,
            #               False, False)
            # self.robot.retrieve_prepare(df)

            # Test
            self.robot.test()
            # for index, row in df.iterrows():
            #     print(index, row)
            #     robot.next(row)

            return self.robot.finish()

        # ===== Testing starts here! =====
        full_summary_dict_list = []
        for ds_name in ds_names:
            full_summary_dict_list.append(test_dataset(ds_name))
        full_summary_dict_list.append(aggregate_summary_df_in_datasets(full_summary_dict_list))
        # ======= Testing ended! =========

        # if self.p_bar:
        #     self.p_window.setWindowTitle('Writing results...')

        self.end_time = datetime.now()

        # Create Meta and Result
        meta = {
            'start': self.start_time,
            'end': self.end_time,
            'time_taken': self.start_time - self.end_time,
        }
        if meta_store:
            meta.update(self.robot.get_robot_meta())
        _ivar = {}
        for key in ivar.keys():
            if key == 'name':
                continue
            _ivar.update({
                key: ivar[key]['default']
            })

        # Create test meta and result
        test_meta = create_test_meta(test_name, _ivar, self.xvar, meta, meta_store)  # Returns dict
        test_result = full_summary_dict_list

        if len(test_name) < 1 or not store:
            pass
        else:
            # Write Meta and Result if storing
            write_test_meta(test_name, test_meta, meta['name'])
            write_test_result(test_name, test_result, meta['name'])

        if self.p_bar:
            self.p_bar.setValue(self.p_bar.maximum())
            # self.p_window.setWindowTitle('Done! Close this window.')

        return test_result, test_meta

    # == Visual ==
    def simulate_single(self, ta_name, ivar, svar, df_name, canvas):
        """Simulate a robot on some single dataframe and draw
        results on input canvas.
        This method only works for robot and not algo!"""

        ta_name = remove_special_char(ta_name)
        print('Starting visual simulation: ' + F'{ta_name}({ivar}) with i:{ivar}, x:{self.xvar}, s:{svar}')
        pre_path = 'robot'

        # Import robot py
        module = importlib.import_module(F'{pre_path}.{ta_name}')
        globals().update(
            {n: getattr(module, n) for n in module.__all__} if hasattr(module, '__all__')
            else
            {k: v for (k, v) in module.__dict__.items() if not k.startswith('_')
             })

        # Start robot
        self.robot = eval(F'{ta_name}({ivar}, {self.xvar})')
        sym, interval_str, period = get_instrument_from_filename(df_name)
        df = load_df(df_name)
        if len(df) < self.robot.PREPARE_PERIOD:  # Otherwise the robot will do nothing
            print(F"Not enough data! Minimum {self.robot.PREPARE_PERIOD} for {ta_name}")
            return False, F"Not enough data! Minimum {self.robot.PREPARE_PERIOD} for {ta_name}"

        # Variables
        sleep_time = 1 / svar['speed']  # seconds
        start = 1

        # Plotting handles
        signals = []
        instructions = []
        axes = canvas.axes
        main_ax = axes[0][0]  # row 0, col 0
        last_ax = axes[-1][-1]

        # Simulate and plot
        self.robot.start(sym, interval_str, period, df[0: start])
        starting_balance = self.xvar['capital']
        scope = svar['scope']

        # Drawing constants
        margin = PLOTTING_SETTINGS['plot_margin'][1]

        # todo use pd.concat(_df, df[:i+1])
        # self.__df = 0

        for i in range(start, len(df)):

            # Clear figure # todo dont replot whole figure. although open_signals - signals ... need to find cheatcode!
            # e.g. like baseline_date to start_date stop take, then start_date to end_date close deal!
            for ax_row in axes:
                for ax in ax_row:
                    ax.clear()

            # Fetch data, === Prepare Data ===
            self.robot.sim_next(df[i:i + 1])
            # indicator = self.robot.get_indicator_df()
            _df = df[:i + 1]
            _signals, _open_signals = self.robot.get_signals()
            signals = copy.deepcopy(_signals)
            open_signals = copy.deepcopy(_open_signals)
            instructions = self.robot.get_instructions()
            profit, equity, balance, margin = self.robot.get_profit()

            # Convert x to indices, remove weekends
            _dates = _df.index
            _df.index = list(range(len(_df.index)))
            # Instructions
            for instruction in instructions:
                # todo convert data.index to indices without assumption that indexes match up exactly
                # date_to_index_arr(instruction['data'].index)  # assign this instead in future
                if 'data' not in instruction:
                    continue
                instruction['data'].index = _df.index
            for signal in signals:
                signal['start'] = _df.index[_dates == signal['start']][0]
                signal['end'] = _df.index[_dates == signal['end']][0]
                if 'baseline' in signal:
                    signal['baseline'] = _df.index[_dates == signal['baseline']][0]
            for signal in open_signals:
                signal['start'] = _df.index[_dates == signal['start']][0]
                signal['end'] = signal['start'] + 1
                if 'baseline' in signal:
                    signal['baseline'] = _df.index[_dates == signal['baseline']][0]

            # Get xlim
            if len(_df.index) > scope:
                xlim = [_df.index[-scope - 1] - 1, _df.index[-1] + 1]
            else:
                # xlim = [_df.index[0] - 1, _df.index[-1] + 1]
                xlim = [0, scope]
            # Adjust profit data in case of length mismatch
            # adjusted = []
            # for data in [profit, equity, balance, margin]:
            #     if len(data) > len(_df):
            #         data = data[len(data) - len(_df):]
            #         adjusted.append(data)
            #     else:
            #         o = [data[0] for i in range(len(_df) - len(data))]
            #         o.extend(data)
            #         data = o
            #         adjusted.append(data)
            # if len(profit) > len(_df):
            #     profit = profit[len(profit) - len(_df):]
            # elif len(profit) < len(_df):
            #     o = [profit[0] for i in range(len(df) - len(profit))]
            #     o.extend(profit)
            #     profit = o
            # if len(equity) > len(_df):
            #     equity = equity[len(equity) - len(_df):]
            # elif len(equity) < len(_df):
            #     o = [equity[0] for i in range(len(df) - len(equity))]
            #     o.extend(equity)
            #     equity = o
            # if len(balance) > len(_df):
            #     balance = balance[len(balance) - len(_df):]
            # elif len(balance) < len(_df):
            #     o = [balance[0] for i in range(len(df) - len(balance))]
            #     o.extend(balance)
            #     balance = o

            # === Plot data ===

            # Plot
            plot_robot_instructions(axes, instructions, xlim)
            if len(signals) > 0:
                plot_signals(main_ax, signals, xlim)
            if len(signals) > 0:
                plot_open_signals(main_ax, open_signals, xlim)
            candlestick_plot(main_ax, _df, xlim)
            plot_line(last_ax, _df.index, profit, {'colour': 'g'}, xlim)
            plot_line(last_ax, _df.index, equity, {'colour': '#b35300'}, xlim)
            plot_line(last_ax, _df.index, balance, {'colour': 'b'}, xlim)
            plot_line(last_ax, _df.index, margin, {'colour': '#000000'}, xlim)
            # plot_line(last_ax, _df.index, adjusted[0], {'colour': 'g'}, xlim)
            # plot_line(last_ax, _df.index, adjusted[1], {'colour': '#b35300'}, xlim)
            # plot_line(last_ax, _df.index, adjusted[2], {'colour': 'b'}, xlim)
            # plot_line(last_ax, _df.index, adjusted[3], {'colour': '#000000'}, xlim)

            # Switch back to dates
            x_tick_labels = []
            for _date in _dates:
                x_tick_labels.append(strtodatetime(_date).strftime(DATE_FORMAT_DICT[interval_str]))
            for ax_row in axes:
                for ax in ax_row:
                    ax.set(xticklabels=x_tick_labels)
                    for label in ax.get_xticklabels():
                        label.set_ha("right")
                        label.set_rotation(45)

            # Wait before each step
            # time.sleep(sleep_time)

            # Adjust y-lim, expand with margin
            # ylim = main_ax.get_ylim()
            # yheight = ylim[1] - ylim[0]
            # main_ax.set_ylim(bottom=ylim[0] - yheight * margin,
            #                  top=ylim[1] + yheight * margin,)
            # for ax_row in axes:
            #     for ax in ax_row:
            #         ylim = ax.get_ylim()
            #         yheight = ylim[1] - ylim[0]
            #         ax.set_ylim(bottom=ylim[0] - yheight * margin,
            #                     top=ylim[1] + yheight * margin, )
            #         if ax == last_ax:
            #             last_ax.set_ylim(bottom=0)

            # line1.set_xdata(x)
            canvas.draw()
            PyQt5.QtWidgets.QApplication.processEvents()

        plt.show()
        return True, 'No Error'

    def simulate_algo_single(self, algo_name, ivar, svar, df_name, canvas):
        """...
        returns Success: bool, Error Message: str"""

        algo_name = remove_special_char(algo_name)
        print('Starting visual simulation: ' + F'{algo_name}({ivar}) with i:{ivar}, s:{svar}')
        pre_path = 'algo'

        # Import robot py
        module = importlib.import_module(F'{pre_path}.{algo_name}')
        globals().update(
            {n: getattr(module, n) for n in module.__all__} if hasattr(module, '__all__')
            else
            {k: v for (k, v) in module.__dict__.items() if not k.startswith('_')
             })

        # Start algo
        if ivar:
            self.algo = eval(F'{algo_name}({ivar})')
        else:
            self.algo = eval(F'{algo_name}()')
        sym, interval_str, period = get_instrument_from_filename(df_name)
        df = load_df(df_name)
        if len(df) < self.algo.PREPARE_PERIOD:  # Otherwise the robot will do nothing
            print(F"Not enough data! Minimum {self.algo.PREPARE_PERIOD} for {algo_name}")
            return False, F"Not enough data! Minimum {self.algo.PREPARE_PERIOD} for {algo_name}"

        # Variables
        sleep_time = 1 / svar['speed']  # seconds
        start = self.algo.PREPARE_PERIOD or 1  # index of dataframe to start at
        scope = svar['scope']

        # Plotting handles
        signals = []
        instructions = []
        axes = canvas.axes
        main_ax = axes[0][0]  # row 0, col 0
        last_ax = axes[-1][-1]

        self.algo.start({
            'sym': sym,
            'interval_str': interval_str,
            'period': period,
        }, df[0: start])

        # Drawing constants
        margin = PLOTTING_SETTINGS['plot_margin'][1]
        plot_stuffs = []

        # Slowly feed data
        for i in range(start, len(df)):

            # Clear all  # todo do not clear the data. only the extra stuff
            for ax_row in axes:
                for ax in ax_row:
                    ax.clear()

            # Next
            self.algo.next(df[i:i + 1])
            _df = df[:i+1]
            instructions = self.algo.get_instructions()

            # Convert dates to indices
            _dates = _df.index
            _df.index = list(range(len(_df.index)))
            # instructions
            for instruction in instructions:
                # todo check. if already in date form...
                if len(instruction['data']) and is_datetimestring(instruction['data'].index[-1]):
                    pass
                instruction['data'].index = date_to_index_arr(_df.index, _dates, instruction['data'].index)

            # Get xlim
            if len(_df.index) > scope:
                xlim = [_df.index[-scope - 1] - 1, _df.index[-1] + 1]
            else:
                xlim = [0, scope]

            # Plot
            plot_stuffs.append(plot_robot_instructions(axes, instructions, xlim))
            candlestick_plot(main_ax, _df, xlim)

            # Switch back to dates
            x_tick_labels = []
            for _date in _dates:
                x_tick_labels.append(strtodatetime(_date).strftime(DATE_FORMAT_DICT[interval_str]))
            for ax_row in axes:
                for ax in ax_row:
                    ax.set(xticklabels=x_tick_labels)
                    for label in ax.get_xticklabels():
                        label.set_ha("right")
                        label.set_rotation(45)

            canvas.draw()
            PyQt5.QtWidgets.QApplication.processEvents()

        return True, 'No Error'

    # == Optimise ==
    def optimise(self, ta_name: str, init_ivar: List[float], ds_names: List[str], optim_name: str, store=True,
                 meta_store=True, canvas=None):
        # todo makeover ivar['ivar']['name'] to ivar['name']
        # todo but yet, when storing, store with name
        """...
        Note this function uses progress bar #2."""
        ta_name = remove_special_char(ta_name)
        print('Starting optimisation: ' + F'{ta_name}.{ta_name}({init_ivar}) with i:{init_ivar}, x:{self.xvar}')

        self.robot = eval(F'{ta_name}.{ta_name}({init_ivar}, {self.xvar})')
        ivar_dict = {
            'ivar': init_ivar,
            'name': init_ivar['name'],
        }
        for meta in ['name', 'type', 'fitness']:  # Move meta attributes to a higher level
            if meta in init_ivar:
                ivar_dict.update({
                    meta: init_ivar[meta]
                })
                del ivar_dict['ivar'][meta]

        args_dict = self.robot.ARGS_DICT  # Basis for optimisation
        # # Manage types of vars
        for key in args_dict.keys():
            # currently not used
            if 'type' in args_dict[key]:
                if args_dict[key]['type'] == 'discrete':  # if discrete, cannot decrease min_step
                    pass
            args_dict[key]['variability'] = 0  # dec/inc min_step to reach variability average
            args_dict[key]['smoothness'] = 0  # min_step < noise width, inc min_step to ignore noise

        # Number of ivar to capture
        top_n = 3  # Maximum fitness
        # Base options
        runs = OPTIMISATION_SETTINGS['optimisation_width']
        max_depth = OPTIMISATION_SETTINGS['optimisation_depth']
        if 'optim_depth' in self.xvar:
            runs = self.xvar['optim_depth']
        if 'optim_width' in self.xvar:
            max_depth = self.xvar['optim_width']
        step_size = OPTIMISATION_SETTINGS['arg_step_size']  # alpha
        approach_size = OPTIMISATION_SETTINGS['approach_step_size']

        # Vector field setup
        optim_field = []  # [{
        # val_1,...val_i...,val_n
        # arg_1,...arg_i...,arg_n
        # },...{}]
        fitness_collection = []
        ivar_collection = []
        final_ivar_results = []  # {ivar, fitness}

        # Get ax
        if canvas:
            axes = canvas.axes
            ax = axes[-1][-1]

        # Setup progress bar
        if self.p_bar_2:
            self.p_bar_2.setMaximum(1)
            self.p_bar_2.setValue(0)

        # ===========================
        #   IVar Dict diagram
        # ===========================
        #
        #   ivar_dicts = {
        #       key: {
        #           ivar: {
        #               # Meta
        #               name: key/origin
        #               fitness:
        #               type:
        #               # Args
        #               key_1: {
        #                   default: val
        #               }, ...,
        #               key_n: {
        #                   default: val
        #               }
        #           }
        #       }
        #   }
        #
        # ===========================

        def suggest_ivar(spread_results, _i=0, _u=0):
            """Shape of spread results:
            {key1: {
                'ivar': {
                    key1: ...,
                    key2: ...,
                    ...
                    keyN: ...,
                }
                'fitness': float
            }, key2...key_n, origin}"""

            main_ivar_dict = spread_results['origin']
            new_ivar = main_ivar_dict['ivar'].copy()
            step_size = OPTIMISATION_SETTINGS['arg_step_size']  # Size of each full step (x=3,y=4 => d=5)

            diff_dict = {
                # key: {fitness deviation/val deviation}
            }

            # Get 'origin'/centre point result
            for key in spread_results.keys():
                if key == 'origin':
                    spread_results[key].update({
                        'fitness_diff': 0,
                        'val_diff': 0,
                    })
                    continue

                spread_results[key].update({
                    # Negative diff indicates a greater new result
                    'fitness_diff': spread_results[key]['fitness'] - main_ivar_dict['fitness'],
                    'val_diff': spread_results[key]['ivar'][key]['default'] - main_ivar_dict['ivar'][key]['default'],
                })

            normalisation_constant = 0
            for key in spread_results.keys():
                normalisation_constant += math.pow(spread_results[key]['fitness_diff'], 2)
            normalisation_constant = math.pow(normalisation_constant, 0.5)

            for key in spread_results.keys():
                if key == 'origin':
                    continue

                min_step = args_dict[key]['step_size']  # min size of step (in coordinate), if applicable
                # if args_dict['type'] == 'continuous':
                #     pass
                # elif args_dict['type'] == 'discrete':
                #     pass
                if min_step:
                    # if val is positive, (result is good) move forward in the current direction
                    # if val is negative, (result is bad) move away from the current direction
                    # 'move' is the number of steps to take
                    move = step_size * min_step \
                           * try_divide(spread_results[key]['fitness_diff'],
                                        normalisation_constant)
                else:  # No minimum step_size,
                    min_step = 1 / 100.0 * (args_dict[key]['range'][1] - args_dict[key]['step_size'][
                        0])  # Assume minimum step_size of 1/100
                    move = step_size * min_step \
                           * try_divide(spread_results[key]['fitness_diff'],
                                        normalisation_constant)

                if abs(move) < min_step:
                    move = try_divide(move * min_step, abs(move))

                #  sgn(val) represents the original direction. a negative 'move' moves in the opposite direction
                new_ivar[key]['default'] += move * try_sgn(spread_results[key]['val_diff'])
                #  If outside range
                range = args_dict[key]['range']
                if new_ivar[key]['default'] > range[1]:
                    new_ivar[key]['default'] = range[1]
                elif new_ivar[key]['default'] < range[0]:
                    new_ivar[key]['default'] = range[0]

            # return new ivar
            return {
                'ivar': new_ivar,
                'name': F'suggest_{_i}_{_u}'
            }

        def random_ivar(_i=0):
            """Create a random initial starting ivar"""
            _ivar = {}
            for key in args_dict.keys():
                arg_range = args_dict[key]['range']
                step = args_dict[key]['step_size']
                r = random.random()
                # find closest step
                if step == 0:
                    value = r * (arg_range[1] - arg_range[0]) + arg_range[0]
                else:
                    target = r * (arg_range[1] - arg_range[0]) + arg_range[0]
                    steps = (arg_range[1] - arg_range[0]) // step
                    # Binary search
                    top_step, btm_step = steps, 0
                    while True:
                        c_step = (top_step + btm_step) // 2  # 0: c_step = steps // 2
                        value = step * c_step + arg_range[0]
                        if abs(value - target) <= step or top_step == btm_step:
                            break
                        elif value > target:
                            if top_step == c_step:
                                c_step = btm_step
                            top_step = c_step
                        elif value < target:
                            if btm_step == c_step:
                                c_step = top_step
                            btm_step = c_step
                _ivar.update({
                    key: {
                        'default': value,
                    },
                })
            ivar_dict = {
                'ivar': _ivar,
                'name': F'random_{_i}'
            }
            return ivar_dict

        def ivar_spread(ivar, _i=0, _u=0):
            """Create ivar_spread from starting ivar origin"""
            # Ivar itself
            ivar_key_tuples = {
                'origin': {
                    'ivar': ivar,
                    'name': F'origin_{_i}_{_u}'
                },
            }
            # Test hypersphere around ivar
            for key in args_dict.keys():
                _ivar = copy.deepcopy(ivar)
                # _ivar['name'] = key  # name is F'spread_{_i}_{_u}', tuple-key is key
                range = args_dict[key]['range']
                curr = ivar[key]['default']
                step = args_dict[key]['step_size']

                # If variability low, increase step
                var = args_dict[key]['variability']
                if var == 0:
                    pass

                # Roll r
                r = random.random()
                t = -1  # Decrease
                if r > 0.5:
                    t = 1  # Increase

                if curr == range[1]:  # Extreme right, decrease regardless
                    _ivar[key]['default'] -= step
                elif curr == range[0]:  # Extreme left, increase regardless
                    _ivar[key]['default'] += step
                _ivar[key]['default'] += step * t
                if curr >= range[1]:  # If reach over the right, set to right bound
                    _ivar[key]['default'] -= range[1]
                elif curr <= range[0]:  # If reach over the left, set to left
                    _ivar[key]['default'] += range[0]

                ivar_key_tuples.update({
                    key: {
                        'ivar': _ivar,
                    }
                })
                # Get name
                ivar_key_tuples[key]['name'] = F'spread_{key}_{_i}_{_u}'
            return ivar_key_tuples

        def get_fitness_score(result):
            # return result['balance'] / result['capital']
            if 'growth_factor' in result:
                return result['growth_factor']
            return 0

        def get_test_result(_ivar):
            _result, _meta = self.test(ta_name, _ivar, ds_names, '', False, False)
            # On any test, add it to fitness and ivar collection
            _score = get_fitness_score(_result)
            fitness_collection.append(_score)
            ivar_collection.append(_ivar)
            return _result[-1], _score  # final only

        def get_ivar_distance(ivar, ivar2):
            """Distance between ivar and ivar 2."""
            dist = 0
            for key in ivar:
                dist += math.pow((ivar2[key] - ivar[key]) / args_dict[key]['step_size'], 2)  # geometric distance
            return math.pow(dist, 0.5)

        def get_distance_ivar(ivar, ivar2):
            """Gets direction ivar dict from ivar to ivar 2"""
            dist = {}
            for key in ivar:
                dist.update({
                    key: ivar2[key] - ivar[key]
                })
            return dist

        # == Geometric util ==

        def check_if_near(ivar, ivar_list):
            """Check if ivar is contained within the step-size (alpha) hyperspheres of past ivars."""
            for _ivar in ivar_list:
                dist = get_ivar_distance(ivar, _ivar)
                if dist < step_size:
                    return True
            return False

        def check_if_going_to(ivar, new_ivar, ivar_list):
            """Check if ivar is approaching a hypersphere"""
            dist = get_distance_ivar(ivar, new_ivar)  # calculate trajectory
            # if check_if_near(new_ivar, ivar_list):
            #     return True, 1
            m1 = 0
            m2 = step_size * approach_size  # vector scalar lies between m1 and m2
            for key in ivar:
                # For each key/coordinate
                val_close = m1 * dist[key] + ivar[key]
                val_far = m2 * dist[key] + ivar[key]
                found = True
                # Check if any ivar value lies in the trajectory * some scalar in (m1, m2)
                for _ivar in ivar_list:
                    if val_close < _ivar[key] < val_far:  # Scalar in (m1, m2) must work for all coordinates
                        # adjust min m1 and max m2
                        _dist = _ivar[key] - ivar[key]
                        s = 1  # If _ivar[key] is on the right, deduct step_size to get closer side of range
                        if _ivar[key] < ivar[key]:
                            s = -1  # Else, increase step_size to get closer side, vice versa
                        # Determine scalar range for all components
                        _m1 = (_dist - s * args_dict['step_size']) / dist[key]
                        _m2 = (_dist + s * args_dict['step_size']) / dist[key]
                        # Continuously squeeze range (m1, m2)
                        if _m1 > m1:
                            m1 = _m1
                        if _m2 < m2:
                            m2 = _m2
                    else:  # All keys need to be within range!
                        found = False
                if not found or m1 == m2:
                    return False, 0
            m = m1
            return True, m

        for i in range(runs):

            # Random starting point
            if i == 0:
                pass  # ivar = init_ivar
            else:
                ivar_dict = random_ivar(i)
            test_result, fitness = get_test_result(ivar_dict['ivar'])

            # run counter
            u = 0

            while True:  # Run until terminate conditions

                ivar_result_tuples = {}  # {key: {ivar, profit, fitness_diff, val_diff}}
                # Get spread to analyse
                ivar_tuples_to_test = ivar_spread(
                    ivar_dict['ivar'], i, u)  # Initial ivar outside loop or new_ivar from previous loop
                # Get spread results
                for key in ivar_tuples_to_test.keys():
                    if key == 'origin':
                        ivar_dict.update({
                            'fitness': fitness,
                        })
                        ivar_result_tuples[key] = ivar_dict
                        continue

                    ivar = ivar_tuples_to_test[key]['ivar']
                    ivar_dict = ivar_tuples_to_test[key]
                    test_result, _fitness = get_test_result(ivar)

                    ivar_dict.update({
                        'fitness': _fitness,
                    })
                    ivar_result_tuples[key] = ivar_dict

                # Determine next move based on deltas
                new_ivar_dict = suggest_ivar(ivar_result_tuples, i, u)
                new_test_result, new_fitness = get_test_result(new_ivar_dict['ivar'])

                # Compare new ivar to old ivar
                diff = new_fitness - fitness

                # === Try Terminate ===
                out = False  # Termination condition
                # Check divergence for cycles
                pass
                # If reflected (Most 90% ivars reflected)
                reflect_n = 0
                for key in new_ivar_dict['ivar']:
                    if key == 'name' or key == "fitness" or key == "type":
                        continue
                    sgn = math.copysign(1, ivar[key]['default'])
                    new_sgn = math.copysign(1, new_ivar_dict['ivar'][key]['default'])
                    if sgn != new_sgn:
                        reflect_n += 1
                if reflect_n > (0.9 * len(ivar.keys())):  # ===== OUT =====
                    out = True
                # If barely any change
                if diff < 0:
                    pass
                # If too many rounds
                if u > max_depth:  # ===== OUT =====
                    out = True
                u += 1

                # Terminate
                if out:
                    final_ivar_results.append({
                        'ivar': new_ivar_dict['ivar'],
                        'fitness': new_fitness,
                        'type': 'wander',
                        'name': F'suggest_{i}_{u}'
                    })
                    break

                # ivar = adjust_ivar(results)
                ivar_dict = new_ivar_dict
                fitness = new_fitness
                # exploration
                # # future
                pass
                # for optim_vector in optim_field:
                #     pass
                pass
                # # exploitation
                pass
                # # future
                # candidates = []
                pass
                # # choose among candidates
                # for candidate in candidates:
                #     pass
                pass

                # ===== Plot =====
                # Plot new best result from spread
                if canvas:
                    plot_optimisations(ax, ivar_result_tuples)
                    # for ivar_result in ivar_result_tuples:
                    #     plot_optimisations()
                    # Update plot every pass
                    canvas.draw()
                    PyQt5.QtWidgets.QApplication.processEvents()

            # Modify p_window
            self.p_bar_2.setValue(i + 1 / runs)

        # Descent Trajectory finals # Easier!
        trimmed_ivar_results = []
        final_fitness_scores = [d['fitness'] for d in final_ivar_results]
        top_fitness_score, btm_fitness_score, avg_fitness_score = \
            try_max(final_fitness_scores), try_min(final_fitness_scores), try_mean(final_fitness_scores)
        std_fitness_score = try_stdev(final_fitness_scores)

        # Trim bad results/Collect top results
        final_ivar_results = sorted(final_ivar_results, key=lambda x: x['fitness'], reverse=True)
        for ivar_result in final_ivar_results:
            if len(final_ivar_results) <= 5:
                break
            if not in_std_range(ivar_result['fitness'], avg_fitness_score, std_fitness_score, 3):
                final_ivar_results.remove(ivar_result)
        for i in range(min(len(final_ivar_results), top_n)):  # top n, index 0 is highest
            trimmed_ivar_results.append(final_ivar_results[i])

        # Trim unusual results (Too high etc.)
        pass

        # Add non-final trajectories that scored highly
        pass

        # Create result dict
        result_dict = {
            'high': top_fitness_score,  # from picked ivars
            'low': btm_fitness_score,
            'average': avg_fitness_score,
            'std_deviation': std_fitness_score,
            # ----
            'average_overall': try_mean(fitness_collection),
            'low_overall': try_min(fitness_collection),
            'total_tests': len(fitness_collection),
            'total_runs': runs,
            'data': ', '.join(ds_names)[:-2],
            # ----
        }
        # For each key in ivar, display top (picked) value:
        top_ivar = final_ivar_results[0]['ivar']
        top_score = final_ivar_results[0]['fitness']
        for key in top_ivar.keys():
            # Register #1 value from optimisation
            result_dict.update(
                {'top_' + key: top_ivar[key]['default']}
            )
        result_dict.update({
            'top_fitness': top_score,
        })

        # Average 'destination' value per ivar
        for key in final_ivar_results[0]['ivar'].keys():
            result_dict[F'average_{key}'] = 0
        for final_ivar_result in final_ivar_results:
            for key in final_ivar_result['ivar'].keys():
                if key in ['name', 'fitness', 'type']:
                    continue  # Current ver. IVars will not contain 'name'
                result_dict[F'average_{key}'] += final_ivar_result['ivar'][key]['default'] / len(final_ivar_results)

        # Create Meta and Result
        meta = {
            'start': self.start_time,
            'end': self.end_time,
            'time_taken': self.start_time - self.end_time,
        }
        if meta_store:
            meta.update(self.robot.get_robot_meta())
        _ivar = {}
        optim_meta = create_optim_meta(optim_name, ivar, self.xvar, meta, meta_store)
        optim_result = create_optim_result(optim_name, result_dict, meta['name'])

        # Create new IVar and insert
        for ivar_result in trimmed_ivar_results:
            ivar_result['name'] = optim_name + '_' + ivar_result['name']
        insert_ivars(ta_name, trimmed_ivar_results)

        optim_series = create_optim_series(optim_name, fitness_collection, ivar_collection)
        write_optim_series(optim_name, optim_series, meta['name'])

        # Save optimisation file
        if store:
            # Write Meta and Result
            write_optim_meta(optim_name, optim_meta, meta['name'])
            write_optim_result(optim_name, optim_result, meta['name'])

        return trimmed_ivar_results


def get_optimisation_types(self):
    return ['block', 'random_descent', 'random', 'bayesian', 'evolution']


#  General

def load_test_result_list():
    folder = 'static/results/evaluation'
    # Get list of folders
    folders = []
    for folder in folders:
        pass
    # Get list of files that end with .csv
    tr_list = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.csv')]
    return tr_list


def load_optim_result_list():
    folder = 'static/results/optimisation'
    # Get list of files that end with .csv
    or_list = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.csv')]
    return or_list

#  Utility
