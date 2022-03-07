import datetime
from datetime import datetime, timedelta
import math
from math import *
from os import listdir
from os.path import isfile, join
from statistics import stdev
from typing import List

import pandas as pd
from PyQt5.QtWidgets import QProgressBar

from robot.abstract.robot import robot
from settings import EVALUATION_FOLDER
from util.dataRetrievalUtil import load_dataset, load_df, get_computer_specs
from util.langUtil import craft_instrument_filename, strtodatetime, try_key, remove_special_char

import numpy as np
from sklearn.linear_model import LinearRegression


#  Robot

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


def create_summary_df_from_list(profit_d, equity_d, signals, df, additional={}):
    """Takes in the profit-loss dataframe, buy-sell signals,
    produces the data-1 summary stat dictionary"""

    summary_dict = {
        'period': 0,
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
        if signal['pos'].lower() == "long":
            long_signals.add(signal)
        else:
            short_signals.add(signal)

    l = len(complete_signals)
    l2 = len(incomplete_signals)
    l3 = len(profit_signals)
    l4 = len(loss_signals)
    l5 = len(long_signals)
    l6 = len(short_signals)

    summary_dict['period'] = strtodatetime(df.at(0, 'datetime')) - strtodatetime(df.at(len(df), 'datetime'))
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
        gmean *= signal['net']  # must convert from pips to sgd todo
    summary_dict['AHPR'] = mean / l
    summary_dict['GHPR'] = math.pow(gmean, 1 / l)

    summary_dict['profit_factor'] = summary_dict['total_profit'] / summary_dict['gross_profit']
    # summary_dict['recovery_factor'] = summary_dict['total_profit'] / summary_dict['max_drawdown']  # Implemented below

    summary_dict['total_trades'] = l
    summary_dict['total_deals'] = l  # + unclosed deals

    prev_d = 0
    drawdown, drawdowns = False, []
    drawdown_l, lowest_drawdown = 0, 0
    for d in profit_d:  # Detect drawdowns
        if drawdown:
            if d < prev_d:
                drawdown_l += 1
                if d < lowest_drawdown:
                    lowest_drawdown = d
            else:
                drawdowns.append({
                    'length': drawdown_l,
                    'depth': prev_d - lowest_drawdown,
                    'ledge': prev_d,
                })
                # reset
                prev_d = d
                drawdown = False
                drawdown_l, lowest_drawdown = 0, 0
        if d < prev_d:
            drawdown = True
            lowest_drawdown = d
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
        avg_len += drawdown['len']
        if greatest_len < drawdown['len']:
            greatest_len = drawdown['len']
    p_min = min(profit_d)
    summary_dict['balance_drawdown_abs'] = profit_d[0] - p_min
    summary_dict['balance_drawdown_max'] = greatest_depth
    min_drawdown = [dd for dd in drawdowns if dd['depth'] == p_min]
    if min_drawdown:
        summary_dict['balance_drawdown_rel'] = min_drawdown[0]['ledge'] - p_min
    else:
        summary_dict['balance_drawdown_rel'] = profit_d[0] - p_min
    summary_dict['balance_drawdown_avg'] = avg_depth / len(drawdowns)
    summary_dict['balance_drawdown_len_avg'] = avg_len / len(drawdowns)
    summary_dict['balance_drawdown_len_max'] = greatest_len

    prev_d = 0
    drawdown, drawdowns = False, []
    drawdown_l, lowest_drawdown = 0, 0
    for d in equity_d:  # Detect drawdowns
        if drawdown:
            if d < prev_d:
                drawdown_l += 1
                if d < lowest_drawdown:
                    lowest_drawdown = d
            else:
                drawdowns.append({
                    'length': drawdown_l,
                    'depth': prev_d - lowest_drawdown,
                    'ledge': prev_d,
                })
                # reset
                prev_d = d
                drawdown = False
                drawdown_l, lowest_drawdown = 0, 0
        if d < prev_d:
            drawdown = True
            lowest_drawdown = d
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
        avg_len += drawdown['len']
        if greatest_len < drawdown['len']:
            greatest_len = drawdown['len']
    p_min = min(profit_d)
    summary_dict['equity_drawdown_abs'] = profit_d[0] - p_min
    summary_dict['equity_drawdown_max'] = greatest_depth
    min_drawdown = [dd for dd in drawdowns if dd['depth'] == p_min]
    if min_drawdown:
        summary_dict['equity_drawdown_rel'] = min_drawdown[0]['ledge'] - p_min
    else:
        summary_dict['equity_drawdown_rel'] = profit_d[0] - p_min
    summary_dict['equity_drawdown_avg'] = avg_depth / len(drawdowns)
    summary_dict['equity_drawdown_len_avg'] = avg_len / len(drawdowns)
    summary_dict['equity_drawdown_len_max'] = greatest_len

    expected_payoff = 0
    for signal in complete_signals:
        expected_payoff += signal['net']
    summary_dict['expected_payoff'] = summary_dict['total_profit'] / l
    summary_dict['sharpe_ratio'] = 0  # No risk-free proxy
    summary_dict['standard_deviation'] = 0
    summary_dict['LR_correlation'] = 0  # no line correlation yet
    summary_dict['LR_standard_error'] = 0
    summary_dict['recovery_factor'] = summary_dict['total_profit'] / summary_dict['max_drawdown']

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

    period_to_net, period_to_gross, avg_period = 0, 0, 0
    avg_profit_period, avg_loss_period = 0, 0
    longest_length, shortest_length = timedelta(), timedelta()
    for signal in signal:
        net = abs(signal['net'])
        period = (strtodatetime(signal['end']) - strtodatetime(signal['start'])).total_seconds()
        period_to_net += net / period
        avg_period += period
        if signal['net'] > 0:
            avg_profit_period += period
        else:
            avg_loss_period += period
        if longest_length < period:
            longest_length = period
        if shortest_length > period:
            shortest_length = period
        if signal['net'] > 0:
            period_to_gross += net / period

    summary_dict['period_to_profit'] = period_to_net / l
    summary_dict['period_to_gross'] = period_to_gross / l3

    summary_dict['longest_trade_length'] = longest_length
    summary_dict['shortest_trade_length'] = shortest_length
    summary_dict['average_trade_length'] = avg_period / l
    summary_dict['average_profit_length'] = avg_profit_period / l3
    summary_dict['average_loss_length'] = avg_loss_period / l4

    winning, losing = False, False
    win_length, lose_length = 0, 0
    win_amount, lose_amount = 0, 0
    win_streak, lose_streak = [], []
    for signal in signals:
        if signal['net'] > 0:
            if winning:
                win_length += 1
                win_amount += signal['net']
            else:
                winning = True
                win_length += 1
                win_amount += signal['net']

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
                lose_length += 1
                lose_amount += signal['net']

            winning = False
            win_streak.append({
                'length': win_length,
                'amount': win_amount,
            })
            win_length, win_amount = 0, 0

    mean = lambda list: sum(list) / len(list)
    summary_dict['max_consecutive_wins'] = max([s['length'] for s in win_streak])
    summary_dict['max_consecutive_profit'] = max([s['amount'] for s in win_streak])
    summary_dict['avg_consecutive_wins'] = mean([s['length'] for s in win_streak])
    summary_dict['avg_consecutive_profit'] = mean([s['amount'] for s in win_streak])

    summary_dict['max_consecutive_losses'] = max([s['length'] for s in lose_streak])
    summary_dict['max_consecutive_loss'] = max([s['amount'] for s in lose_streak])
    summary_dict['avg_consecutive_losses'] = mean([s['length'] for s in lose_streak])
    summary_dict['avg_consecutive_loss'] = mean([s['amount'] for s in lose_streak])

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

    summary_dict = summary_dict_list[0]
    for i in range(summary_dict_list):
        if i == 0:
            continue
        for key in summary_dict_list[i]:
            summary_dict[key] += summary_dict[i][key]
    for key in summary_dict:
        summary_dict[key] /= len(summary_dict_list)

    final_summary_dict = {
        'n_instruments': len(summary_dict_list),
        'standard_deviation_profit': stdev(profits),
        #
        'dataset': ds_name,
    }

    final_summary_dict.update(summary_dict)

    return final_summary_dict


def aggregate_summary_df_in_datasets(summary_dict_list: List):
    """Aggregate summary and dataset summaries"""

    summary_dict = summary_dict_list[0]
    total_instruments = 0
    for i in range(summary_dict_list):
        if i == 0:
            continue
        for key in summary_dict_list[i]:
            summary_dict[key] += summary_dict[i][key] * summary_dict[i]['n_instruments']
        total_instruments += summary_dict[i]['n_instruments']

    for key in summary_dict:
        if not key.lower() == 'dataset':
            summary_dict[key] /= total_instruments

    final_summary_dict = {
        'n_datasets': len(summary_dict_list),
        'dataset': 'Total'
    }

    final_summary_dict.update(summary_dict)
    summary_dict_list.append(final_summary_dict)

    return summary_dict_list


def create_test_meta(test_name, ivar, xvar):
    """Test meta file contains test name, ivar (dict version) used, start and end date etc
    most importantly, xvar (dict version) attributes"""
    meta = {
        'datetime': datetime.now(),
        'end': datetime.now(),
        'time_taken': 0,
        # # XVar
        # 'lag': xvar['lag'],  # ms
        # 'starting_capital': xvar['starting_capital'],
        # 'leverage': xvar['leverage'],
        # 'currency_count': xvar['currency'],  # pips
        # 'type': xvar['type'],  # aka singular/multi
        # 'dataset_type': xvar['dataset_type'],  # forex-leaning, etc.
        # # Also in meta/result file name
        # 'test_name': 0,
        # 'robot_name': 0,
        #
        # # IVar
        # Robot meta output
        'robot_meta': {}
    }
    # IVar Dict
    meta.update(ivar)
    # XVar Dict
    meta.update(xvar)
    # Computer specs
    meta.update(get_computer_specs())

    return meta


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


def write_test_result(summary_dicts: List):
    folder = EVALUATION_FOLDER
    name = ""
    path = F'{folder}/{name}.csv'

    sdfs = []
    for summary_dict in summary_dicts:
        sdfs.append(pd.DataFrame(summary_dict))

    summary_df = pd.DataFrame(sdfs[0])
    for i in range(sdfs):
        if i != 0:
            summary_df.append(pd.DataFrame(sdfs[i]))

    return sdfs


def load_test_result(test_name: str):
    folder = F'static/results/evaluation'
    result_path = F'{folder}/{test_name}.csv'

    trdf = pd.read_csv(result_path)
    return trdf


def load_test_meta(meta_name: str):
    folder = F'static/results/evaluation'
    meta_path = F'{folder}/{meta_name}.csv'

    tmdf = pd.read_csv(meta_path)
    return tmdf


class DataTester:

    def __init__(self, robot_name, ivar, xvar):
        self.robot_name = remove_special_char(robot_name)
        self.ivar = ivar
        self.xvar = xvar
        eval(F'{robot_name}.{robot_name}({ivar})')

        self.progress_bar = None

    def bind_progress_bar(self, p_bar: QProgressBar):
        self.progress_bar = p_bar

    # Test test_result, result_meta '{robot_name}__{ivar_name}__{test_name}.csv',
    # '{robot_name}__{ivar_name}__{test_name}__meta.csv'

    def test_dataset(self, ta_name: str, ivar: List[float], ds_names: List[str]):
        self.robot = eval(F'{ta_name}.{ta_name}({ivar})')

        # Testing starts here!
        for ds_name in ds_names:
            self.test_dataset(ds_name)

        def test_dataset(self, ds_name):
            dsdf = load_dataset(ds_name)
            for index, row in dsdf.iterrows():
                test_data(craft_instrument_filename(row['symbol'], row['interval'], row['period']))

        def test_data(self, d_name):
            df = load_df(d_name)

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
    # Get list of files that end with .csv
    tr_list = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.csv')]
    return tr_list


def load_optim_result_list():
    folder = 'static/results/optimisation'
    # Get list of files that end with .csv
    or_list = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.csv')]
    return or_list

#  Utility
