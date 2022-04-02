"""FMACD Robot"""
import math
from typing import List

import pandas as pd
import talib
from dateutil import parser

from robot.abstract.robot import robot
from datetime import datetime, timedelta

# Pre-Retrieve done by DataTester

# Definitions:
#
# Check - Indicator result: long, short or none
# Signal - Dictionary, trade deal = { type, vol (amount), start, end, }


# This FMACD algorithm uses MACD crossing as a trigger signal,
# and MACD hist and SMA200 as secondary conditions.
#
#
from util.langUtil import strtotimedelta, get_instrument_type, strtodatetime
from util.robotDataUtil import generate_base_signal_dict


class FMACDRobot(robot):
    """A simple robot to test the TradingHunter suite."""

    IVAR_STEP = 0.05
    N_ARGS = 2
    ARGS_STR = ['stop_loss', 'take_profit', 'fast_period', 'slow_period', 'signal_period', 'sma_period']
    ARGS_DEFAULT = [1, 1.5, 12, 26, 9, 200]
    ARGS_RANGE = [[0.01, 10], [0.01, 10],
                  [10, 15], [22, 30],
                  [8, 10], [150, 250], ]
    ARGS_DICT = {
        # Main, optimisable
        'profit_loss_ratio': {
            'default': 1.5,
            'range': [0.75, 4],
            'step_size': 0.1,  # Default step size
        },
        'fast_period': {
            'default': 12,
            'range': [0, 1],
            'step_size': 0.1,
        },
        'slow_period': {
            'default': 26,
            'range': [0, 1],
            'step_size': 0.1,
        },
        'signal_period': {
            'default': 9,
            'range': [0, 1],
            'step_size': 0.1,
        },
        'sma_period': {
            'default': 200,
            'range': [0, 1],
            'step_size': 0.1,
        },
    }
    OTHER_ARGS_DICT = {
        'left_peak': {
            'default': 2,
            'range': [1, 4],
            'step_size': 1,  # Default step size
        },
        'right_peak': {
            'default': 2,
            'range': [1, 4],
            'step_size': 1,
        },
        'look_back': {
            'default': 20,
            'range': [10, 30],
            'step_size': 1,
        },
        'peak_order': {
            'default': 1,
            'range': [1, 5],
            'step_size': 1,
        },
        'lots_per_k': {
            'default': 0.01,
            'range': [0.001, 0.1],
            'step_size': 0.001,
        },
        'stop_loss_amp': {
            'default': 1.1,
            'range': [0.9, 2],
            'step_size': 0.001,
        },
        'stop_loss_flat_amp': {  # in pips
            'default': 0,
            'range': [-100, 100],
            'step_size': 1,
        },
    }
    # Retrieve Prep
    PREPARE_PERIOD = 200

    # Other
    MARGIN_RISK_PER_TRADE = 0
    LOSS_PERCENTAGE = 0

    # Plotting variables
    PLOT_NO = [0, 0, 1]  # (Trailing)

    VERSION = '0.1'
    NAME = 'FMACDRobot'

    def __init__(self, ivar=ARGS_DEFAULT, xvar={}):
        """XVar variables should be numbers or strings.
        e.g. leverage must be a number (100), not '1:100'.
        instrument_type should be ...(pip value)
        instrument_type should be ...(forex)
        commission should be a number"""

        # == IVar ==
        if len(ivar) == 2:
            self.ivar = ivar
        else:
            self.ivar = FMACDRobot.ARGS_DEFAULT
        self.ivar_range = FMACDRobot.ARGS_RANGE

        # == Data Meta ==
        self.symbol = ""
        self.period = timedelta()
        self.interval = timedelta()
        self.instrument_type = get_instrument_type(self.symbol)
        # == XVar ==
        self.xvar = xvar
        self.lag = xvar['lag']  # unused
        self.starting_capital = xvar['capital']
        self.leverage = xvar['leverage']
        self.instrument_type = xvar['instrument_type']
        # self.currency = xvar['currency']
        self.commission = xvar['commission']
        self.contract_size = xvar['contract_size']

        # == Main Args ==
        self.profit_loss_ratio = self.ARGS_DICT['profit_loss_ratio']['default']
        self.fast_period = self.ARGS_DICT['fast_period']['default']
        self.slow_period = self.ARGS_DICT['slow_period']['default']
        self.signal_period = self.ARGS_DICT['signal_period']['default']
        self.sma_period = self.ARGS_DICT['sma_period']['default']
        # == Other Args ==
        self.look_back = self.OTHER_ARGS_DICT['look_back']['default']
        self.left_peak = self.OTHER_ARGS_DICT['left_peak']['default']
        self.right_peak = self.OTHER_ARGS_DICT['right_peak']['default']
        self.lot_per_k = self.OTHER_ARGS_DICT['lots_per_k']['default']
        self.stop_loss_amp = self.OTHER_ARGS_DICT['stop_loss_amp']['default']
        self.stop_loss_flat_amp = self.OTHER_ARGS_DICT['stop_loss_flat_amp']['default']

        # == Preparation ==
        self.prepare_period = self.PREPARE_PERIOD  # Because of SMA200
        self.instrument_type = "Forex"  # Default

        # == Indicator Data ==
        self.indicators = {
            # Non-critical (If no checks produced, ignore)
            'SMA5': pd.DataFrame(),
            'SMA50': pd.DataFrame(),
            'SMA200': pd.DataFrame(),
            'SMA200_HIGH': pd.DataFrame(),
            'EMA200': pd.DataFrame(),
            # Critical (No check = Fail)
            'MACD_HIST': pd.DataFrame(),
            'MACD_DF': pd.DataFrame(),
            # Signal generators
            'MACD': pd.DataFrame(),
            'MACD_SIGNAL': pd.DataFrame(),
        }
        # self.indicators_start = {
        #     # Used to align indicator indices between and with data
        #     'SMA5': 0,
        #     'SMA50': 0,
        #     'SMA200': 0,
        #     'SMA200_HIGH': 0,
        #     'EMA200': 0,
        #     # Critical (No check = Fail)
        #     'MACD_HIST': 0,
        #     'MACD_DF': 0,
        #     # Signal generators
        #     'MACD': 0,
        #     'MACD_SIGNAL': 0,
        #     # Use length differences instead. If length = 0, indicator has no start
        # }
        self.new_indicators = {}
        # == Signals ==
        self.signals = []  # { Standard_Dict, FMACD_Specific_Dict }
        self.open_signals = []  # Currently open signals
        self.new_signals = [] # By definition, open
        # self.failed_signals = []  # So that the robot doesn't waste time on failed signals;
        # failure saved for analysis. Add failed signals directly to signals...
        # Failed signals should still calculate stop-loss and take-profit for hindsight
        # analysis.

        # == Statistical Data ==
        self.balance = []  # Previous Balance OR Starting capital, + Realised P/L OR Equity - Unrealised P/L OR
        #                       Free_Margin + Margin - Unrealised P/L
        self.free_balance = []  # Previous F.Balance or Starting capital, +- Realised P/L + Buy-ins OR Curr Balance -
        #                                                                                       (Margin) Cur.Buy-ins
        #                     OR Free_Margin - Unrealised P/L
        # Buy-in: Forex Margin OR Stock Asset price - Liability price
        # Unrealised P/L: - Buy-in + Close-pos/Sell-in - Close-pos

        self.profit = []  # Realised P/L
        self.unrealised_profit = []  # Unrealised P/L
        self.gross_profit = []  # Cumulative Realised Gross Profit
        self.gross_loss = []  # Cumulative Realised Gross Loss

        self.asset = []  # Open Long Position price
        self.liability = []  # Open Short Position Price
        self.short_margin, self.long_margin = [], []  # Total Short/Long Margins
        self.margin = []  # Margin (Max(Long_Margin), Max(Short_Margin))
        self.free_margin = []  # Balance - Margin + Unrealised P/L

        self.equity = []  # Free_Balance + Asset (Long) - Liabilities (Short) OR Forex Free_Margin + Margin
        self.margin_level = []  # Equity / Margin * 100%, Otherwise 0%
        self.stat_datetime = []

        # == Data ==
        self.df = pd.DataFrame()
        self.last = pd.DataFrame()
        self.last_date = None
        # self.con_df = pd.DataFrame()  # df w.r.t to data (old data to build indicators not included)

        # == Robot Status ==
        self.test_mode = False
        self.test_idx = 0
        self.market_active = False  # FMACDRobot is not aware of actual money
        self.started = False

        # == Testing Only ==
        self.indicators_test = {
            'SMA200': pd.DataFrame(),
            'SMA5': pd.DataFrame(),
            'EMA': pd.DataFrame(),
            'MACD': pd.DataFrame(),
            'MACD_SIGNAL': pd.DataFrame(),
            'MACD_HIST': pd.DataFrame(),
            'MACD_DF': pd.DataFrame(),
        }
        self.df_test = pd.DataFrame()

    def reset(self, ivar=ARGS_DEFAULT, xvar={}):
        if not xvar:
            xvar = self.xvar
        self.__init__(ivar, xvar)

    def start(self, symbol: str, interval: str, period: str, pre_data: pd.DataFrame(), test_mode=False):
        """Begin by understanding the incoming data. Setup data will be sent
        E.g. If SMA-200 is needed, at the minimum, the past 400 data points should be known.
        old_data = retrieve()
        From then on, the robot receives data realtime - simulated by feeding point by point. (candlestick)
        data_meta: Symbol, Period, Interval, xvar variables, ...

        After running .start(), run .retrieve_prepare() with the appropriate data
        (same symbols/interval, period = PREPARE_PERIOD)

        Output: Data statistics start from 1 step before the first datapoint (in .next())
        """
        self.reset()
        self.test_mode = test_mode

        # == Data Meta ==
        self.symbol = symbol
        self.period = strtotimedelta(period)
        self.interval = interval
        self.instrument_type = get_instrument_type(symbol)

        # == Prepare ==
        self.df = pre_data
        # Set up Indicators
        self.build_indicators()  # Runs indicators after

        # == Statistical Data (Setup) ==
        self.balance.append(self.starting_capital)
        self.free_balance.append(self.starting_capital)

        self.profit.append(0)
        self.unrealised_profit.append(0)
        self.gross_profit.append(0)
        self.gross_loss.append(0)

        self.asset.append(0)
        self.liability.append(0)
        self.short_margin.append(0)
        self.long_margin.append(0)
        self.margin.append(0)
        self.free_margin.append(0)

        self.equity.append(self.starting_capital)
        self.margin_level.append(0)
        # First datetime data point is relative to the data. If none, fill in post-run
        if len(pre_data) > 0 and not self.test_mode:
            self.stat_datetime.append(strtodatetime(pre_data.index[-1]))
            # if 'Datetime' in pre_data.columns:
            #     self.stat_datetime.append(parser.parse(pre_data.iloc[-1].Datetime))
            # else:
            #     self.stat_datetime.append(parser.parse(pre_data.iloc[-1].Date))
        else:
            self.stat_datetime.append(None)  # Fill in manually later.

        # == Robot Status ==
        self.started = True

    def apply_xvar(self, xvar={}):

        self.lag = xvar['lag']
        self.starting_capital = xvar['capital']
        self.leverage = xvar['leverage']
        self.instrument_type = xvar['instrument_type']
        # self.currency = xvar['currency']  # from symbol
        self.commission = xvar['commission']
        self.contract_size = xvar['contract_size']

    def apply_other_args(self):

        self.profit_loss_ratio = self.ARGS_DICT['profit_loss_ratio']['default']
        self.fast_period = self.ARGS_DICT['fast_period']['default']
        self.slow_period = self.ARGS_DICT['slow_period']['default']
        self.signal_period = self.ARGS_DICT['signal_period']['default']
        self.sma_period = self.ARGS_DICT['sma_period']['default']

        self.look_back = self.OTHER_ARGS_DICT['look_back']['default']
        self.left_peak = self.OTHER_ARGS_DICT['left_peak']['default']
        self.right_peak = self.OTHER_ARGS_DICT['right_peak']['default']
        self.lot_per_k = self.OTHER_ARGS_DICT['lots_per_trade']['default']

    def next(self, candlesticks: pd.DataFrame):
        """When the next candlestick comes. Just in case candlesticks were
        'dropped' in the process, use next() with all the new candlesticks at once.
        Only the last candlestick will be processed and no signals will be processed
        for the 'missed' candlesticks. Stats will track regardless.
        """

        # == Step 1: Update data =============

        if self.test_mode:
            pass
        else:
            self.df = self.df.append(candlesticks)
            # Update indicators
            self.build_indicators()

        self.last = candlesticks.iloc[-1]
        self.last_date = candlesticks.index[-1]
        # if 'Datetime' in candlesticks.columns:
        #     self.last_date = candlesticks.iloc[-1]['Datetime']
        # elif 'Date' in candlesticks.columns:
        #     self.last_date = candlesticks.iloc[-1]['Date']
        # else:
        #     col = self.df.columns[0]
        #     self.last_date = candlesticks.iloc[-1][col]

        # == Step 2: Update Stats =============

        # Calculate stat data values
        for i in range(len(candlesticks)):
            self.next_statistics(candlesticks.iloc[i])

        # == Step 3: Analyse Graph =============
        # Do nothing

        # == Step 4: Signals =============

        # ==    a1: Check to close deals
        for signal in self.open_signals:
            sgn = math.copysign(1, signal['vol'])  # +ve for long, -ve for short
            stop, take = signal['stop_loss'], signal['take_profit']
            # Stop-loss OR Take-profit
            if sgn * self.last.Close <= sgn * stop or sgn * self.last.Close >= sgn * take:
                self.close_signal(signal)

        # == =   b1: Check to create signals =============

        # Generate signals
        signal = self.generate_signal()

        # == =   b2: Check indicators =============

        #  Check indicators
        if signal:  # Type- 0: None; 1: Long; 2: Short
            checks = self.check_indicators(signal['type'])
            check = True
            for _check in checks:
                if not _check:
                    check = False
                    break

            # == =   b3: Confirm Signal =============

            # Create signal, virtual if failed check

            self.create_signal(signal, check)

        # == Step 5: Cleanup =============
        # Do nothing

    def finish(self):

        if not self.stat_datetime[0]:
            self.stat_datetime[0] = self.stat_datetime[1] - strtotimedelta(self.interval)
        self.started = False

        robot_dict = {
            # 'time_taken': 0,
            'name': self.NAME,
            'version': self.VERSION,
        }
        stats_dict = {
            'profit': self.profit,
            'unrealised_profit': self.unrealised_profit,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            # £
            'asset': self.asset,
            'liability': self.liability,
            'short_margin': self.short_margin,
            'long_margin': self.long_margin,
            'margin': self.margin,
            'free_margin': self.free_margin,
            # £
            'equity': self.equity,
            'margin_level': self.margin_level,
            'datetime': self.stat_datetime,
        }
        signals_dict = {
            'signals': self.signals,
            'open_signals': self.open_signals,
        }

        return stats_dict, signals_dict, robot_dict

    def get_robot_meta(self):

        robot_dict = {
            # 'time_taken': 0,
            'name': self.NAME,
            'version': self.VERSION,
        }
        return robot_dict

    # External control

    def test(self):
        """Same as .text_next() but produces less result data. Robots in general
        either do a scan for signals (fast) or check for signals without scan if possible (faster)"""
        length = len(self.df)
        start = self.prepare_period
        if start > length:
            start = 0
        self.test_idx = start
        # self.indicator_idx = 0
        for i in range(start, length):
            self.next(self.df[self.test_idx: self.test_idx + 1])
            self.test_idx = i

    def sim_start(self):
        # self.test_mode = True
        # # Since sim, test from 0.
        # self.test_idx = 0
        pass

    def sim_next(self, candlesticks: pd.DataFrame):
        self.next(candlesticks)

    # Retrieve results

    def get_data(self):
        return self.df

    def get_time_data(self):
        return self.df.index

    def get_profit(self):
        return self.profit_d, self.equity_d

    def get_signals(self):
        return self.signals

    # Temperature

    def get_signals(self):
        return self.signals, self.open_signals

    def get_instructions(self):
        return [
            {
                'index': 1,
                'data': self.indicators['MACD'],
                'type': 'macd_hist',
                'colour': 'blue',
                # Placeholder, for multi-data plotting
                'other_data': None
            },
            {
                'index': 1,
                'data': self.indicators['MACD_SIGNAL'],
                'type': 'line',
                'colour': 'red',
            },
            {
                'index': 1,
                'data': self.indicators['MACD_HIST'],
                'type': 'macd_hist',
                'colour': 'na',
            },
            {
                'index': 0,
                'data': self.indicators['SMA200'],
                'type': 'line',
                'colour': 'blue',
            },
        ]

    def get_new_signals(self):
        pass

    def get_new_instructions(self):
        pass

    # Indicator (Indicators give go-long or go-short suggestions. They DO NOT give signals)

    def rebuild_macd(self, period):
        if len(self.df.Close) < 0:
            return
        self.indicators['MACD'], self.indicators['MACD_SIGNAL'], self.indicators['MACD_HIST'] = \
            talib.MACD(self.df.Close, fastperiod=self.fast_period,
                       slowperiod=self.slow_period, signalperiod=self.signal_period)
        self.indicators['MACD_DF'] = pd.DataFrame(index=self.df.index,
                                                  data={"macd": self.indicators['MACD'],
                                                        "macd_signal": self.indicators['MACD_SIGNAL'],
                                                        "macdhist": self.indicators['MACD_HIST'], })

    def rebuild_sma(self, period):
        self.indicators['SMA5'] = talib.SMA(self.df.Close, timeperiod=2)
        self.indicators['SMA50'] = talib.SMA(self.df.Close, timeperiod=50)
        self.indicators['SMA200'] = talib.SMA(self.df.Close, timeperiod=200)
        self.indicators['SMA200_HIGH'] = talib.SMA(self.df.High, timeperiod=200)
        self.indicators['EMA200'] = talib.EMA(self.df.Close, timeperiod=200)

    def build_indicators(self):

        period = self.prepare_period
        # Calculate indicators for whole period - Use for plotting after
        self.rebuild_macd(period)
        self.rebuild_sma(period)

    def match_indicators(self):
        """If indicators are only built for past N days or may be offset from stat_data,
        indexed data (without datetime), especially excluding non-trading days will not match.
        This function will fill-zeroes such that the data (df) and the stat_data matches when plotted.
        A datetime axis label list will be generated too."""

        # Find maximum period

        # Get list of dates in (active) df (stat_datetime),

        # For each date in indicator, if not in stat_datetime, do "fill values" (extend the number)

        # Ideally, all intervals should be uniform. Otherwise, it may not be indexed anyway. (0,1,2,3 -> mon, tues, ...)

        # Now, for each date in stat_datetime, if not in indicator, do "fill values" (fill 0)

        # start_date, end_date = stat_datetime[0], stat_datetime[-1]

        pass

    # Analysis

    def find_peaks(self, idx=0):
        if self.test_mode:
            idx += self.test_idx
        for i in range(idx, idx + self.look_back):
            pass

    # Check

    def check_indicators(self, type):
        """Returns a list of integer-bools according to the signal-generating indicators.
        0: False, 1: Long, 2: Short"""
        return {
            'MACD_HIST': self.check_macd_hist(type),
            'SMA': self.check_sma(type),
        }

    def check_macd(self):
        """Generates signal"""
        rev_idx = -2
        if self.test_mode:
            rev_idx = self.test_idx - len(self.df)

        if len(self.indicators['MACD']) > 2 and len(self.indicators['MACD_SIGNAL']) > 2:
            if self.indicators['MACD'].iloc[rev_idx] > self.indicators['MACD_SIGNAL'].iloc[rev_idx]:
                if self.indicators['MACD'].iloc[rev_idx + 1] < self.indicators['MACD_SIGNAL'].iloc[rev_idx + 1]:
                    return 1
            else:
                if self.indicators['MACD'].iloc[rev_idx + 1] > self.indicators['MACD_SIGNAL'].iloc[rev_idx + 1]:
                    return 2
        return 0

    def check_macd_hist(self, _type):
        """Checks Signal"""
        if self.test_mode:
            rev_idx = self.test_idx - len(self.df)
            if self.indicators['MACD_HIST'].iloc[rev_idx] > 0:
                if self.indicators['MACD_HIST'].iloc[rev_idx] > self.indicators['MACD_HIST'].iloc[rev_idx + 1]:
                    return _type == 1
            else:
                if self.indicators['MACD_HIST'].iloc[rev_idx] < self.indicators['MACD_HIST'].iloc[rev_idx + 1]:
                    return _type == 2

        if self.indicators['MACD_HIST'].iloc[-1] > 0:
            if self.indicators['MACD_HIST'].iloc[-1] > self.indicators['MACD_HIST'].iloc[-2]:
                return _type == 1
        else:
            if self.indicators['MACD_HIST'].iloc[-1] < self.indicators['MACD_HIST'].iloc[-2]:
                return _type == 2
        return 0

    def check_sma(self, _type):
        """Checks Signal"""
        if self.test_mode:
            rev_idx = self.test_idx - len(self.df)
            if self.indicators['SMA200'].iloc[rev_idx] > self.last.Close:
                return _type == 1
            else:
                return _type == 2

        if len(self.indicators['SMA200']) > 0:
            return 0
        if self.indicators['SMA200'].iloc[-1] > self.last.Close:
            return _type == 1
        else:
            return _type == 2
        return 0

    def check_sma_bundle(self):
        pass

    # Signal (Signal scripts give buy/sell signals. Does not handle stop-loss or take-profit etc.)

    def generate_signal(self):
        type = self.check_macd()
        if type:
            signal = generate_base_signal_dict()
            # Assign dict values
            signal['type'] = type
            signal['start'] = self.last_date
            signal['vol'] = self.assign_lots(type)
            sgn = math.copysign(1, signal['vol'])
            signal['leverage'] = self.xvar['leverage']
            signal['margin'] = sgn * signal['vol'] / self.xvar['leverage'] * self.xvar['contract_size']
            signal['open_price'] = self.last.Close
            # signal['virtual'] = True  # Determined after 'create_signal'
            # Generate stop loss and take profit
            signal['stop_loss'] = self.get_stop_loss(type)
            signal['take_profit'] = self.get_take_profit(signal['stop_loss'])
            return signal
        return None

    def create_signal(self, signal, check):
        """Create signal based on signal dict 'mold' input and add into open signals"""
        if check:
            sgn = math.copysign(1, signal['vol'])
            self.add_margin(-sgn, signal['margin'])
            self.calc_equity()
            signal['virtual'] = False
        signal['virtual'] = not check

        self.open_signals.append(signal)

    def partial_close_signal(self, signal, check):
        # Split signal into 2 signals
        # todo
        # Add signals back
        signal1, signal2 = 0, 0
        self.open_signals.remove(signal)
        self.open_signals.append(signal1)
        self.open_signals.append(signal2)
        # Close one signal
        self.close_signal(signal2)

    def close_signal(self, signal):

        # Realise Profit/Loss
        action = (signal['open_price'] - self.last.Close) * signal['vol'] * self.leverage
        signal['end'] = self.last_date
        signal['close_price'] = self.last.Close
        sgn = math.copysign(1, signal['vol'])
        signal['net'] = action

        if not signal['virtual']:
            # Release margin, release unrealised P/L
            self.add_profit(action)
            self.add_margin(sgn, signal['margin'])
            self.calc_equity()

        # Add signal to signals, remove from open signals
        self.open_signals.remove(signal)
        self.signals.append(signal)

    def confirm_signal(self, check, signal):
        pass

    def macd_signal(self):

        pass

    # Statistic update

    def get_current_equity(self):
        """Connect to platform to query current positions."""
        pass

    def add_profit(self, profit):
        self.unrealised_profit[-1] -= profit
        self.profit[-1] += profit
        if profit > 0:
            self.gross_profit[-1] += profit
        else:
            self.gross_loss[-1] -= profit
        pass

    def add_margin(self, sgn, margin):
        """Adding margins reduce free margin"""
        if sgn > 0:
            self.long_margin[-1] -= margin
        else:
            self.short_margin[-1] -= margin
        self.margin[-1] = max([self.short_margin[-1], self.long_margin[-1]])

    def calc_equity(self):
        self.free_balance[-1] = self.balance[-1] - self.margin[-1]
        # Note, unrealised_profit not updated yet -
        self.free_margin[-1] = self.free_balance[-1] + self.unrealised_profit[-1]
        self.equity[-1] = self.free_margin[-1] + self.margin[-1]
        # Assets and Liabilities untouched
        if self.margin[-1]:
            self.margin_level[-1] = self.equity[-1] / self.margin[-1] * 100
        else:
            self.margin_level[-1] = 0

    # Risk Management

    def assign_lots(self, type):
        sgn = 1
        if type == 2:
            sgn = -1
        vol = sgn * self.lot_per_k * self.free_margin[-1] / 1000  # vol
        if sgn * vol < 0.01:
            vol = sgn * 0.01  # Min 0.01 lot
        return vol

    def next_statistics(self, candlestick):
        self.balance.append(self.balance[-1])
        self.free_balance.append(self.balance[-1])

        self.profit.append(self.profit[-1])
        self.unrealised_profit.append(self.unrealised_profit[-1])
        self.gross_profit.append(self.gross_profit[-1])
        self.gross_loss.append(self.gross_loss[-1])

        self.short_margin.append(self.short_margin[-1])
        self.long_margin.append(self.long_margin[-1])
        self.asset.append(self.asset[-1])
        self.liability.append(self.liability[-1])
        for signal in self.open_signals:
            if signal['type'] == 'short':
                # Calculate margin
                self.short_margin.append(signal['margin'])
            else:  # == 'long'
                self.long_margin.append(signal['margin'])
            # Calculate unrealised P/L
            self.unrealised_profit[-1] += math.copysign(
                (candlestick.Close - signal['open_price']) * signal['vol'], signal['vol'])

        self.margin.append(max([self.short_margin[-1], self.long_margin[-1]]))
        self.free_balance[-1] -= self.margin[-1]
        self.free_margin.append(self.balance[-1] - self.margin[-1] + self.unrealised_profit[-1])

        # self.equity.append(self.asset[-1] - self.liability[-1])
        self.equity.append(self.free_margin[-1] + self.margin[-1])
        if self.margin[-1] == 0:
            self.margin_level.append(0)
        else:
            self.margin_level.append(self.equity[-1] / self.margin[-1] * 100)
        self.stat_datetime.append(parser.parse(self.last_date))

    # Stop loss and Take profit

    def get_stop_loss(self, type):
        rev_idx = 0
        if self.test_mode:
            rev_idx = self.test_idx - len(self.df)

        length, length2 = self.look_back, 0
        if len(self.df) < length:
            length = self.df

        turn = self.last.Close
        max_loops = 3
        while turn == self.last.Close:
            for i in range(1 + length2 - rev_idx, 1 + length2 + length - rev_idx):
                if type == 1:
                    if self.df.Close.iloc[-i] < turn:
                        turn = self.df.Close.iloc[-i]  # Get low
                elif type == 2:
                    if self.df.Close.iloc[-i] > turn:
                        turn = self.df.Close.iloc[-i]  # Get high
            length2 += length
            max_loops -= 1
            if max_loops <= 0:
                break

        # Amplify:
        diff = turn - self.last.Close
        turn = self.last.Close + diff * self.stop_loss_amp
        if type == 1:
            turn -= self.stop_loss_flat_amp / 10000
        elif type == 2:
            turn += self.stop_loss_flat_amp / 10000

        return turn

    def get_take_profit(self, stop_loss):
        diff = self.last.Close - stop_loss
        return self.last.Close + diff

    # Optimisation

    def step_ivar(self, idx, up=True):
        if len(self.ivar_range) >= idx:
            i = -1
            if up:
                i = 1
            self.ivar[idx] += (self.ivar_range[idx][1] - self.ivar[idx]) * robot.IVAR_STEP * i

    # Utility

    def close_trade(self):

        # Calculate profit
        pass

    def get_concurr_data(self):
        return self.df[self.df.index.isin(self.stat_datetime)]

    def set_multiple_ivars(self, ivars):
        self.ivars = ivars
