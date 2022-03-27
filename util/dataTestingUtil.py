import datetime
import os
from datetime import datetime, timedelta
import math
from math import *
from os import listdir
from os.path import isfile, join
from statistics import stdev
from typing import List
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import QProgressBar, QWidget

from robot.abstract.robot import robot
from settings import EVALUATION_FOLDER, OPTIMISATION_FOLDER
from util.dataRetrievalUtil import load_dataset, load_df, get_computer_specs, number_of_datafiles, retrieve, try_stdev
from util.langUtil import craft_instrument_filename, strtodatetime, try_key, remove_special_char, strtotimedelta, \
    try_divide, try_max, try_mean, get_test_name, get_file_name

import numpy as np
from sklearn.linear_model import LinearRegression

#  Robot
from robot import FMACDRobot


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
        #
        'profit_factor': 0,
        'recovery_factor': 0,
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
        'average_profit_Trade': 0,
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

    summary_dict = base_summary_dict()
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
    summary_dict['ticks'] = -1  # not applicable here
    #
    summary_dict['total_profit'] = profit_d[-1]
    gross_profit = 0
    gross_loss = 0
    for signal in signals:
        if signal['net'] > 0:
            gross_profit += signal['net']
        elif signal['net'] < 0:
            gross_profit += signal['net']
    summary_dict['gross_profit'] = gross_profit
    summary_dict['gross_loss'] = gross_loss

    # deepest loss in drawdown - stats
    mean = 0
    gmean = 0
    for signal in signals:
        mean += signal['net']
        gmean *= signal['net']
    summary_dict['AHPR'] = try_divide(mean, l)
    summary_dict['GHPR'] = math.pow(gmean, try_divide(1, l))

    summary_dict['profit_factor'] = try_divide(summary_dict['total_profit'], summary_dict['gross_profit'])
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
    summary_dict['average_profit_Trade'] = average_profit / l3
    summary_dict['largest_loss_trade'] = largest_loss
    summary_dict['average_loss_trade'] = average_loss / l3

    # summary_dict['longest_trade'] = 0
    # summary_dict['longest_profit_trade'] = 0
    # summary_dict['longest_loss_trade'] = 0
    # summary_dict['average_trade_period'] = 0

    if 'Datetime' in df.columns:  # -1 - -2 doesn't work
        interval = (strtodatetime(df.iloc[1].Datetime) - strtodatetime(df.iloc[0].Datetime)).total_seconds()
    elif 'Date' in df.columns:
        interval = (strtodatetime(df.iloc[1].Date) - strtodatetime(df.iloc[0].Date)).total_seconds()
    else:
        col = df.columns[0]
        interval = (strtodatetime(df.iloc[1][col]) - strtodatetime(df.iloc[0][col])).total_seconds()

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
        period_to_net += net / period
        if signal['net'] > 0:
            period_to_gross += net / period
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
            summary_dict[key] += summary_dict_list[i][key]
    for key in summary_dict:
        if len(summary_dict_list) > 0:
            summary_dict[key] /= len(summary_dict_list)
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
    total_instruments = 0
    for i in range(len(summary_dict_list)):
        if i == 0:
            continue
        for key in summary_dict_list[i]:
            final_summary_dict[key] += summary_dict_list[i][key] * summary_dict_list[i]['n_instruments']
        total_instruments += summary_dict_list[i]['n_instruments']

    for key in final_summary_dict:
        if not key.lower() == 'dataset':
            final_summary_dict[key] = try_divide(final_summary_dict[key], total_instruments)

    final_summary_dict.update({
        'n_datasets': len(summary_dict_list),
        'dataset': 'Total'
    })

    return final_summary_dict


def create_test_meta(test_name, ivar, xvar, other):
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
    meta.update(get_computer_specs())

    return meta


# Forex type

def get_optimised_robot_list():
    folder = F'{OPTIMISATION_FOLDER}'
    os.makedirs(folder, exist_ok=True)
    folders = os.listdir(F'{folder}')
    return folders


def get_tested_robot_list():
    folder = F'{EVALUATION_FOLDER}'
    os.makedirs(folder, exist_ok=True)
    folders = os.listdir(F'{folder}')
    return folders


def get_tests_list(robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(F'{folder}')
    _files = []
    for file in files:
        if not get_file_name(file).endswith('__meta'):
            _files.append(get_test_name(file))
    return _files


def create_test_result(test_name: str, summary_dict_list, meta_df: pd.DataFrame):
    folder = F'static/results/evaluation'
    result_path = F'{folder}/{test_name}.csv'
    meta_path = F'{folder}/{test_name}__meta.csv'

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

    summary_df.to_csv(result_path)
    meta_df.to_csv(meta_path)


def write_test_result(test_name, summary_dicts: List, robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    path = F'{folder}/{test_name}.csv'

    if not os.path.exists(folder):
        os.makedirs(folder)

    sdfs = []
    for summary_dict in summary_dicts:
        sdfs.append(summary_dict)
    sdfs = pd.DataFrame(sdfs)
    sdfs.to_csv(path)

    # summary_df = pd.DataFrame(sdfs[0])
    # for i in range(sdfs):
    #     if i != 0:
    #         summary_df = summary_df.append(pd.DataFrame(sdfs[i]))

    print(F'Writing test result at {path}')
    return sdfs


def write_test_meta(test_name, meta_dict, robot_name: str):
    folder = F'{EVALUATION_FOLDER}/{robot_name}'
    meta_path = F'{folder}/{test_name}__meta.csv'

    if not os.path.exists(folder):
        os.makedirs(folder)

    meta = pd.DataFrame([meta_dict])
    meta.to_csv(meta_path)
    print(F'Writing test result at {meta_path}')
    return meta


def load_test_result(test_name: str, robot_name: str):
    folder = F'{EVALUATION_FOLDER}'
    if not test_name.endswith('.csv'):
        test_name += '.csv'
    result_path = F'{folder}/{robot_name}/{test_name}'

    trdf = pd.read_csv(result_path, index_col=0)
    return trdf


def load_test_meta(meta_name: str, robot_name: str):
    folder = F'{EVALUATION_FOLDER}'
    if not meta_name.endswith('.csv'):
        meta_name += '.csv'
    meta_path = F'{folder}/{robot_name}/{meta_name}'

    tmdf = pd.read_csv(meta_path, index_col=0)
    return tmdf


# Stock type


class DataTester:

    def __init__(self, xvar):

        self.robot = None
        self.xvar = xvar
        self.progress_bar = None
        self.start_time = None
        self.end_time = None

    def bind_progress_bar(self, p_bar: QProgressBar, p_window: QWidget):
        self.p_bar = p_bar
        self.p_window = p_window

    # Test test_result, result_meta '{robot_name}__{ivar_name}__{test_name}.csv',
    # '{robot_name}__{ivar_name}__{test_name}__meta.csv'

    def test(self, ta_name: str, ivar: List[float], ds_names: List[str], test_name: str):

        self.start_time = datetime.now()

        ta_name = remove_special_char(ta_name)
        # need to un-df ivar!
        print('Going to test: ' + F'{ta_name}.{ta_name}({ivar}) with i:{ivar}, x:{self.xvar}')
        self.robot = eval(F'{ta_name}.{ta_name}({ivar}, {self.xvar})')

        if self.p_bar:
            self.p_bar.setMaximum(number_of_datafiles(ds_names) + 1)
            self.p_bar.setValue(0)
            self.p_window.show()

        # Testing util functions
        def test_dataset(ds_name):
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
                self.p_window.setWindowTitle(F'Testing against {d_name}')
                stats_dict, signals_dict, robot_dict = test_data(df, row['symbol'], row['interval'], row['period'])
                summary_dict_list.append(create_summary_df_from_list(stats_dict, signals_dict, df))

            return aggregate_summary_df_in_dataset(ds_name, summary_dict_list)

        def test_data(df, sym, interval, period):

            # Set robot to testing mode
            print(F"Robot {self.robot} starting test {sym}-{interval}-{period}...")
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

        if self.p_bar:
            self.p_window.setWindowTitle('Writing results...')

        self.end_time = datetime.now()

        # Create Meta and Result
        meta = {
            'start': self.start_time,
            'end': self.end_time,
            'time_taken': self.start_time - self.end_time,
        }
        meta.update(self.robot.get_robot_meta())
        _ivar = {
            'name': ivar[0]
        }
        for i in range(1, len(ivar)):
            _ivar.update({
                F'arg{i}': ivar[i]
            })

        test_meta = create_test_meta(test_name, _ivar, self.xvar, meta)  # Returns dict
        test_result = full_summary_dict_list

        # Write Meta and Result
        write_test_meta(test_name, meta, meta['name'])
        write_test_result(test_name, test_result, meta['name'])

        if self.p_bar:
            self.p_bar.setValue(self.p_bar.maximum())
            self.p_window.setWindowTitle('Done! Close this window.')

        return test_result, test_meta

    def simulate_single(self, ta_name, ivar, df_name, canvas):
        pass

    def print_results(self):
        pass

    # full test_data, produces more data '{robot_name}__{ivar_name}__{optim_name}.csv',
    # '{robot_name}__{ivar_name}__{optim_name}__meta.csv'
    # ivar_choices: [ivar1[], ivar2[], ivar3[]...]

    def single_test(self):
        pass

    # Optimise, optim_result, optim_meta


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
