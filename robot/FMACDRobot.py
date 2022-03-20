"""FMACD Robot"""
import math
from typing import List

import pandas as pd
import talib

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
from util.langUtil import strtotimedelta, get_instrument_type
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
            'default': 0.001,
            'range': [0.0001, 0.01],
            'step_size': 0.0001,
        },
    }
    # Retrieve Prep
    PREPARE_PERIOD = 200

    # Other
    MARGIN_RISK_PER_TRADE = 0
    LOSS_PERCENTAGE = 0
    # OTHER_ARGS_STR = ['left_peak', 'right_peak', 'look_back', 'lots_per_k']
    # # OTHER_ARGS_STR = ['takeprofit_2', 'takeprofit_2_ratio', 'capital_ratio', 'lot_size']
    # OTHER_ARGS_DEFAULT = [2, 2, 20, 0.001]

    # Signal Definition:
    #  {}: 'net', 'start', 'end', 'type (short/long)', 'vol', 'type', 'equity',
    #  + 'macd_fail', 'sma_fail', 'ema_fail',

    def __init__(self, ivar=ARGS_DEFAULT, xvar={}):
        """XVar variables should be numbers or strings.
        e.g. leverage must be a number (100), not '1:100'.
        currency_type should be ...(pip value)
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
        self.lag = xvar['lag']  # unused
        self.starting_capital = xvar['capital']
        self.leverage = xvar['leverage']
        self.currency_type = xvar['currency_type']  # todo load_currency_type_suggestions
        self.currency = xvar['currency']  # here too! add
        self.commission = xvar['commission']
        self.contract_size = xvar['contract_size']  # here too!

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
        self.lot_per_trade = self.OTHER_ARGS_DICT['lots_per_trade']['default']

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
        self.indicators_start = {
            # Used to align indicator indices between and with data
            'SMA5': 0,
            'SMA50': 0,
            'SMA200': 0,
            'SMA200_HIGH': 0,
            'EMA200': 0,
            # Critical (No check = Fail)
            'MACD_HIST': 0,
            'MACD_DF': 0,
            # Signal generators
            'MACD': 0,
            'MACD_SIGNAL': 0,
            # Use length differences instead. If length = 0, indicator has no start
        }
        # == Signals ==
        self.signals = []  # { Standard_Dict, FMACD_Specific_Dict }
        self.open_signals = []  # Currently open signals
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

    # def reset(self, ivar=ARGS_DEFAULT, xvar={}):
    #
    #     # == IVar ==
    #     if len(ivar) == 2:
    #         self.ivar = ivar
    #     else:
    #         self.ivar = FMACDRobot.ARGS_DEFAULT
    #     self.ivar_range = FMACDRobot.ARGS_RANGE
    #
    #     # == Data Meta ==
    #     self.symbol = ""
    #     self.period = timedelta()
    #     self.interval = timedelta()
    #     self.instrument_type = get_instrument_type(self.symbol)
    #     # == XVar ==
    #     self.apply_xvar(xvar)
    #     self.apply_other_args()
    #
    #     # == Preparation ==
    #     self.prepare_period = 200
    #     self.instrument_type = "Forex"
    #
    #     # == Indicator Data ==
    #     self.indicators = {
    #         # Non-critical (If no checks produced, ignore)
    #         'SMA5': pd.DataFrame(),
    #         'SMA50': pd.DataFrame(),
    #         'SMA200': pd.DataFrame(),
    #         'SMA200_HIGH': pd.DataFrame(),
    #         'EMA200': pd.DataFrame(),
    #         # Critical (No check = Fail)
    #         'MACD_HIST': pd.DataFrame(),
    #         'MACD_DF': pd.DataFrame(),
    #         # Signal generators
    #         'MACD': pd.DataFrame(),
    #         'MACD_SIGNAL': pd.DataFrame(),
    #     }
    #     # == Signals ==
    #     self.signals = []  # { Standard_Dict, FMACD_Specific_Dict }
    #     self.open_signals = []
    #
    #     # == Statistical Data ==
    #     self.balance = []  # Previous Balance OR Starting capital, + Realised P/L OR Equity - Unrealised P/L OR
    #     #                       Free_Margin + Margin - Unrealised P/L
    #     self.free_balance = []  # Previous F.Balance or Starting capital, +- Realised P/L + Buy-ins OR Curr Balance -
    #     #                                                                                               Cur.Buy-ins
    #     #                     OR Free_Margin - Unrealised P/L
    #     # Buy-in: Forex Margin OR Stock Asset price - Liability price
    #     # Unrealised P/L: - Buy-in + Close-pos/Sell-in - Close-pos
    #
    #     self.profit = []  # Realised P/L
    #     self.unrealised_profit = []  # Unrealised P/L
    #     self.gross_profit = []  # Cumulative Realised Gross Profit
    #     self.gross_loss = []  # Cumulative Realised Gross Loss
    #
    #     self.asset = []  # Open Long Position price
    #     self.liability = []  # Open Short Position Price
    #     self.short_margin, self.long_margin = [], []  # Total Short/Long Margins
    #     self.margin = []  # Margin (Max(Long_Margin), Max(Short_Margin))
    #     self.free_margin = []  # Balance - Margin + Unrealised P/L
    #
    #     self.equity = []  # Asset (Long) - Liabilities (Short) OR Forex Free_Margin + Margin
    #     self.margin_level = []  # Equity / Margin * 100%, Otherwise 0%
    #     self.stat_datetime = []
    #
    #     # == Data ==
    #     self.df = pd.DataFrame()
    #     self.last = pd.DataFrame()
    #     # self.con_df = pd.DataFrame()  # df w.r.t to data (old data to build indicators not included)
    #
    #     # == Robot Status ==
    #     self.test_mode = False
    #     self.market_active = False  # FMACDRobot is not aware of actual money
    #     self.started = False
    #
    #     # == Testing Only ==
    #     self.indicators_test = {
    #         'SMA200': pd.DataFrame(),
    #         'SMA5': pd.DataFrame(),
    #         'EMA': pd.DataFrame(),
    #         'MACD': pd.DataFrame(),
    #         'MACD_SIGNAL': pd.DataFrame(),
    #         'MACD_HIST': pd.DataFrame(),
    #         'MACD_DF': pd.DataFrame(),
    #     }
    #     self.df_test = pd.DataFrame()

    # ======= Start =======

    def reset(self, ivar=ARGS_DEFAULT, xvar={}):
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
        if len(pre_data) > 0:
            self.stat_datetime.append(pre_data[-1:].index)
        else:
            self.stat_datetime.append(None)  # Fill in manually later.

        # == Robot Status ==
        self.started = True

    def apply_xvar(self, xvar={}):

        self.lag = xvar['lag']
        self.starting_capital = xvar['capital']
        self.leverage = xvar['leverage']
        self.currency_type = xvar['currency_type']
        self.currency = xvar['currency']
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
        self.lot_per_trade = self.OTHER_ARGS_DICT['lots_per_trade']['default']

    # def test_start(self, df):
    #     self.df_test = df
    #     self.build_indicators()
    #     for row, index in self.df_test.iterrows():
    #         self.test_next(row)

    # Just make data get appended on Next().

    # if test: want to test all one-shot. no downloads if possible
    # if 200 > total period of data,
    # Simply have no pre_period

    def next(self, candlesticks: pd.DataFrame):
        """When the next candlestick comes. Just in case candlesticks were
        'dropped' in the process, use next() with all the new candlesticks at once.
        Only the last candlestick will be processed and no signals will be processed
        for the 'missed' candlesticks. Stats will track regardless.
        """

        # == Step 1: Update data =============

        self.df = self.df.append(candlesticks)
        self.last = candlesticks[-1:]

        # == Step 2: Update Stats =============

        # Calculate stat data values
        for i in range(len(candlesticks)):
            self.next_statistics()
        # update indicators
        self.build_indicators()

        # == Step 3: Analyse Graph =============
        # Do nothing

        # == Step 4: Signals =============

        # ==    a1: Check to close deals
        for signal in self.open_signals:
            sgn = math.copysign(signal['vol'], 1)  # +ve for long, -ve for short
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

    def test(self):
        """Same as .text_next() but produces less result data. Robots in general
        either do a scan for signals (fast) or check for signals without scan if possible (faster)"""
        length = len(self.df)
        start = length - self.prepare_period
        if start < 0:
            start = 0
        self.test_idx = start
        self.indicator_idx = 0
        for i in range(start, length):
            self.test_next()
            self.test_idx = i

    def test_next(self):
        """When the next candlestick comes. Just in case candlesticks were
        'dropped' in the process, use next() with all the new candlesticks at once.
        Only the last candlestick will be processed and no signals will be processed
        for the 'missed' candlesticks. Stats will track regardless.

        test_idx is incremented in self.test()
        """

        # == Step 1: Simulate Update data =============

        self.last = self.df[self.test_idx: self.text_idx + 1]

        # == Step 2: Update Stats =============

        # Calculate stat data values
        self.next_statistics(self.last)
        # Skip updating indicators

        # == Step 3: Analyse Graph =============
        # Do nothing

        # == Step 4: Signals =============

        # ==    a1: Check to close deals
        for signal in self.open_signals:
            sgn = math.copysign(signal['vol'], 1)  # +ve for long, -ve for short
            stop, take = signal['stop_loss'], signal['take_profit']
            # Stop-loss OR Take-profit
            if sgn * self.last.Close <= sgn * stop or sgn * self.last.Close >= sgn * take:
                self.close_signal(signal)

        # == =   b1: Check to create signals =============

        # Generate signals
        signal = self.generate_signal()  # todo do not check [-1]. check idx!

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

    # ======= End =======

    def finish(self):

        if not self.stat_datetime[0]:
            self.stat_datetime[0] = self.stat_datetime[1] - strtotimedelta(self.interval)
        self.started = False

        stats_dict = {
            'profit': self.profit,
            'unrealised_profit': self.profit,
            'gross_profit': self.profit,
            'gross_loss': self.profit,
            # £
            'asset': self.profit,
            'liability': self.profit,
            'short_margin': self.profit,
            'long_margin': self.profit,
            'margin': self.profit,
            'free_margin': self.profit,
            # £
            'equity': self.profit,
            'margin_level': self.profit,
            'stat_datetime': self.profit,
        }
        signals_dict = {
            'signals': self.signals,
            'open_signals': self.open_signals,
        }

        return stats_dict, signals_dict

    # Retrieve results

    def get_data(self):
        return self.df

    def get_time_data(self):
        return self.df.index

    def get_profit(self):
        return self.profit_d, self.equity_d

    def get_signals(self):
        return self.signals

    def get_curr_data_time(self):
        return self.last.index[0]

    # Indicator (Indicators give go-long or go-short suggestions. They DO NOT give signals)

    def rebuild_macd(self, period):
        self.indicators['MACD'], self.indicators['MACD_SIGNAL'], self.indicators['MACD_HIST'] = \
            talib.MACD(self.df, fastperiod=self.fast_period,
                       slowperiod=self.slow_period, signalperiod=self.signal_period)
        self.indicators['MACD_DF'] = pd.DataFrame(index=self.df.index,
                                                  data={"macd": self.indicators['MACD'],
                                                        "macd_signal": self.indicators['MACD_SIGNAL'],
                                                        "macdhist": self.indicators['MACD_HIST'], })

        # build sma200 sma5

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
        # todo skip sat sun
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

    def calc_stoploss(self):
        pass

        # find n1, n2 peak in last n3 bars
        # otherwise, look further n3 * 0.5 bars and take lowest regardless

        # After finding the peak, set 10% further.
        # if TOO low, ignore this step

    # Check

    def check_indicators(self, type):
        """Returns a list of integer-bools according to the signal-generating indicators.
        0: False, 1: Long, 2: Short"""
        return {
            'MACD_HIST': self.check_macd_hist(type),
            'SMA': self.check_sma(type),
        }

    def check_macd(self):
        if self.test_mode:  # todo how to get relative indexes?
            if self.indicators['MACD'][-2] > self.indicators['MACD_SIGNAL'][-2]:
                if self.indicators['MACD'][-1] < self.indicators['MACD_SIGNAL'][-1]:
                    return 1
            else:
                if self.indicators['MACD'][-1] > self.indicators['MACD_SIGNAL'][-1]:
                    return 2
        if len(self.indicators['MACD']) > 2 and len(self.indicators['MACD_SIGNAL']) > 2:
            if self.indicators['MACD'][-2] > self.indicators['MACD_SIGNAL'][-2]:
                if self.indicators['MACD'][-1] < self.indicators['MACD_SIGNAL'][-1]:
                    return 1
            else:
                if self.indicators['MACD'][-1] > self.indicators['MACD_SIGNAL'][-1]:
                    return 2
        return 0

    def check_macd_hist(self, type):
        if self.indicators['MACD_HIST'][-1] > 0:
            if self.indicators['MACD_HIST'][-1] > self.indicators['MACD_HIST'][-2]:
                return type == 1
        else:
            if self.indicators['MACD_HIST'][-1] < self.indicators['MACD_HIST'][-2]:
                return type == 2
        return 0

    def check_sma(self, type):
        if len(self.indicators['SMA200']) > 0:
            return 0
        if self.indicators['SMA200'][-1] > self.df['close'][-1]:
            return type == 1
        else:
            return type == 2
        return 0

    def check_sma_bundle(self):
        pass

    # Signal (Signal scripts give buy/sell signals. Does not handle stop-loss or take-profit etc.)

    def generate_signal(self):
        type = self.check_macd()
        if not type:
            signal = generate_base_signal_dict()
            # Assign dict values
            signal['type'] = type
            signal['start'] = self.last.index[-1]
            signal['vol'] = self.assign_lots(type)
            signal['leverage'] = self.xvar['leverage']
            signal['margin'] = signal['vol'] / self.xvar['leverage'] * self.xvar['contract_size']
            signal['open_price'] = self.last.Close[-1]
            # signal['virtual'] = True  # Determined after 'create_signal'
            # Generate stop loss and take profit
            signal['stop_loss'] = self.get_stop_loss(type)
            signal['take_profit'] = self.get_take_profit(signal['stop_loss'])
            return signal
        return None

    def create_signal(self, signal, check):

        if check:
            sgn = math.copysign(signal['vol'], 1)
            self.add_margin(-sgn, signal['margin'])
            self.calc_equity()
            signal['virtual'] = False
        signal['virtual'] = not check

        self.open_signals.append(signal)

    def close_signal(self, signal):

        if not signal['virtual']:
            # Realise Profit/Loss
            action = (signal['open_price'] - self.last.Close[-1]) * signal['vol'] * self.leverage
            signal['end'] = self.last.index[-1]
            sgn = math.copysign(signal['vol'], 1)

            # Release margin, release unrealised P/L
            self.add_profit(action)
            self.add_margin(sgn, signal['margin'])
            self.calc_equity()

        # Add signal to signals, remove from open signals
        self.open_signals.remove(signal)
        self.signals.add(signal)

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
        self.profit += profit
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
            self.short_magin[-1] -= margin
        self.margin[-1] = max([self.short_margin[-1], self.long_margin[-1]])

    def calc_equity(self, profit):
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
        return sgn * 0.01 * self.free_margin[-1] / 1000  # vol

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
                (candlestick[-1:].Close - signal['start_price']) * signal['vol'], signal['vol'])

        self.margin.append(max([self.short_margin[-1], self.long_margin[-1]]))
        self.free_balance[-1] -= self.margin[-1]
        self.free_margin.append(self.balance[-1] - self.margin[-1] + self.unrealised_profit[-1])

        # self.equity.append(self.asset[-1] - self.liability[-1])
        self.equity.append(self.free_margin[-1] + self.margin[-1])
        if self.margin == 0:
            self.margin_level.append(0)
        else:
            self.margin_level.append(self.equity / self.margin * 100)
        self.stat_datetime.append(candlestick[-1:].index)

    # Stop loss and Take profit

    def get_stop_loss(self, type):
        length, length2 = self.look_back, 0
        if len(self.df < length):
            length = self.df

        turn = self.last.Close
        while turn == self.last.Close:
            for i in range(length2, length):
                if type == 1:
                    if self.df[-i] < turn:
                        turn = self.df[-i]  # Get low
                elif type == 2:
                    if self.df[-i] > turn:
                        turn = self.df[-i]  # Get high
            length2 += length
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

    def close_signal(self, signal):
        self.open_signals.remove(signal)
        self.signals.append(signal)
