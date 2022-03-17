"""FMACD Robot"""
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
    IVAR_STEP = 0.05
    N_ARGS = 2
    ARGS_STR = ['stop_loss', 'take_profit', 'fast_period', 'slow_period', 'signal_period', 'sma_period']
    # OTHER_ARGS_STR = ['takeprofit_2', 'takeprofit_2_ratio', 'capital_ratio']
    ARGS_DEFAULT = [1, 1.5, 12, 26, 9, 200]
    ARGS_RANGE = [[0.01, 10], [0.01, 10],
                  [10, 15], [22, 30],
                  [8, 10], [150, 250], ]
    # Retrieve Prep
    PREPARE_PERIOD = 200

    # Other
    CAPITAL_PER_TRADE = 0  # todo determine with pops later
    LOSS_PERCENTAGE = 0

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
        self.currency_type = xvar['currency']
        # self.commission = xvar['commission']

        # == Preparation ==
        self.prepare_period = 200
        self.instrument_type = "Forex"

        # == Indicator Data ==
        self.indicators = {
            'SMA200': pd.DataFrame(),
            'SMA5': pd.DataFrame(),
            'EMA': pd.DataFrame(),
            'MACD': pd.DataFrame(),
            'MACD_SIGNAL': pd.DataFrame(),
            'MACD_HIST': pd.DataFrame(),
            'MACD_DF': pd.DataFrame(),
        }
        # == Signals ==
        self.signals = []  # { Standard_Dict, FMACD_Specific_Dict }

        # == Statistical Data ==
        self.balance = []  # Previous Balance OR Starting capital, + Realised P/L OR Equity - Unrealised P/L OR
        #                       Free_Margin + Margin - Unrealised P/L
        self.free_balance = []  # Previous F.Balance or Starting capital, +- Realised P/L + Buy-ins OR Curr Balance -
        #                                                                                               Cur.Buy-ins
        #                     OR Free_Margin - Unrealised P/L
        # Buy-in: Forex Margin OR Stock Asset price - Liability price
        # Unrealised P/L: - Buy-in + Close-pos/Sell-in - Close-pos

        self.profit = []  # Realised P/L
        self.unrealised_profit = []  # Unrealised P/L
        self.gross_profit = []  # Cumulative Realised Gross Profit
        self.gross_loss_data = []  # Cumulative Realised Gross Loss

        self.asset = []  # Open Long Position price
        self.liability = []  # Open Short Position Price
        self.short_margin, self.long_margin = [], []  # Total Short/Long Margins
        self.margin = []  # Margin (Max(Long_Margin), Max(Short_Margin))
        self.free_margin = []  # Balance - Margin + Unrealised P/L

        self.equity = []  # Asset (Long) - Liabilities (Short) OR Forex Free_Margin + Margin
        self.margin_level = []  # Equity / Margin * 100%, Otherwise 0%
        self.stat_datetime = []

        # == Data ==
        self.df = pd.DataFrame()
        self.last = pd.DataFrame()
        # self.con_df = pd.DataFrame()  # df w.r.t to data (old data to build indicators not included)

        # == Robot Status ==
        self.test_mode = False
        self.market_active = False  # FMACDRobot is not aware of actual money
        self.started = False
        self.last_date = datetime.now()

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
        self.currency_type = xvar['currency']
        # self.commission = xvar['commission']

        # == Preparation ==
        self.prepare_period = 200
        self.instrument_type = "Forex"

        # == Indicator Data ==
        self.indicators = {
            'SMA200': pd.DataFrame(),
            'SMA5': pd.DataFrame(),
            'EMA': pd.DataFrame(),
            'MACD': pd.DataFrame(),
            'MACD_SIGNAL': pd.DataFrame(),
            'MACD_HIST': pd.DataFrame(),
            'MACD_DF': pd.DataFrame(),
        }
        # == Signals ==
        self.signals = []  # { Standard_Dict, FMACD_Specific_Dict }

        # == Statistical Data ==
        self.balance = []  # Previous Balance OR Starting capital, + Realised P/L OR Equity - Unrealised P/L OR
        #                       Free_Margin + Margin - Unrealised P/L
        self.free_balance = []  # Previous F.Balance or Starting capital, +- Realised P/L + Buy-ins OR Curr Balance -
        #                                                                                               Cur.Buy-ins
        #                     OR Free_Margin - Unrealised P/L
        # Buy-in: Forex Margin OR Stock Asset price - Liability price
        # Unrealised P/L: - Buy-in + Close-pos/Sell-in - Close-pos

        self.profit = []  # Realised P/L
        self.unrealised_profit = []  # Unrealised P/L
        self.gross_profit = []  # Cumulative Realised Gross Profit
        self.gross_loss_data = []  # Cumulative Realised Gross Loss

        self.asset = []  # Open Long Position price
        self.liability = []  # Open Short Position Price
        self.short_margin, self.long_margin = [], []  # Total Short/Long Margins
        self.margin = []  # Margin (Max(Long_Margin), Max(Short_Margin))
        self.free_margin = []  # Balance - Margin + Unrealised P/L

        self.equity = []  # Asset (Long) - Liabilities (Short) OR Forex Free_Margin + Margin
        self.margin_level = []  # Equity / Margin * 100%, Otherwise 0%
        self.stat_datetime = []

        # == Data ==
        self.df = pd.DataFrame()
        self.last = pd.DataFrame()
        # self.con_df = pd.DataFrame()  # df w.r.t to data (old data to build indicators not included)

        # == Robot Status ==
        self.test_mode = False
        self.market_active = False  # FMACDRobot is not aware of actual money
        self.started = False
        self.last_date = datetime.now()

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

    # ======= Start =======

    def start(self, symbol: str, interval: str, period: str, pre_data: pd.DataFrame()):
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
        self.gross_loss_data.append(0)

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

        self.df.append(candlesticks)
        self.last = candlesticks[-1:]
        last_value = self.last.Stop

        # Update missed data-points
        for i in range(len(candlesticks)):
            self.next_stats(candlesticks[i:i + 1])

        # Calculate stat data values

        profit = 0
        liquid_assets = 0
        self.profit.append(profit)
        self.asset.append(liquid_assets)

        # update indicators
        self.build_indicators()
        self.calculate_equity()

        # close open positions
        open_positions = [_signal for _signal in self.signals if not _signal['end']]
        for i in range(open_positions):
            signal = open_positions[i]

            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

            # use data[-1] as latest

            if signal['type'] == 1:
                if last_value < stop_loss or last_value > take_profit:
                    signal = self.close_signal(signal)
                    open_positions[i] = signal
            elif signal['type'] == 2:
                if last_value > stop_loss or last_value < take_profit:
                    signal = self.close_signal(signal)
                    open_positions[i] = signal

        # check signals

        # Step 1: Check for Signal

        # Step 2: Check for indicator support

        # Step 3: Confirm Signal

        check_dict = self.check_indicators()  # indicator check dictionary
        f_check = check_dict.values()[0]
        for key in check_dict.keys():
            if not f_check:
                continue
            if f_check != check_dict[key]:
                f_check = 0
        # make signals
        self.create_signal(f_check, check_dict, candlesticks[-1])

        # Update profit/Equity record
        # self.assign_equity()  # Assign final values

    def test_next(self, candlestick: pd.DataFrame):
        """Same as .next() but does not recalculate indicators at every step."""
        pass

    def speed_test(self):
        """Same as .text_next() but produces less result data. Robots in general
        either do a scan for signals (fast) or check for signals without scan if possible (faster)"""
        pass

    # ======= End =======

    def finish(self):

        if not self.stat_datetime[0]:
            self.stat_datetime[0] = self.stat_datetime[1] - strtotimedelta(self.interval)

        # remove first variable off data variables
        # self.balance = self.balance[1:]
        # self.profit = self.profit[1:]
        # self.equity = self.equity[1:]
        # self.gross_profit = self.gross_profit[1:]
        # self.asset = self.asset[1:]
        pass

    def on_complete(self):
        pass
        # self.profit_df = create_profit_df_from_list(self.profit, self.asset)
        # self.signal_df = create_signal_df_from_list(self.completed_signals, self.signals)
        # self.summary_df = create_summary_df_from_list(self.profit, self.asset, self.completed_signals)

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

    def calc_macd(self):

        length = []
        max_length = max(length)

        df_max = 0

        self.indicators['MACD'], self.indicators['MACD_SIGNAL'], self.indicators['MACD_HIST'] = \
            talib.MACD(self.df, fastperiod=12, slowperiod=26, signalperiod=9)
        self.indicators['MACD_DF'] = pd.DataFrame(index=self.df.index,
                                                  data={"macd": self.indicators['MACD'],
                                                        "macd_signal": self.indicators['MACD_SIGNAL'],
                                                        "macdhist": self.indicators['MACD_HIST'], })
        self.indicators['SMA5'] = talib.SMA(self.df, timeperiod=2)
        self.indicators['SMA200'] = talib.SMA(self.df, timeperiod=200)

        # build sma200 sma5

    def build_indicators(self):
        self.calc_macd

    # Check

    def check_indicators(self):
        """Returns a list of integer-bools according to the signal-generating indicators.
        0: False, 1: Long, 2: Short"""
        return {
            'MACD': self.check_macd(),
            'MACD_HIST': self.check_macd_hist(),
            'SMA': self.check_sma(),
        }

    def check_macd(self):
        if len(self.indicators['MACD']) > 2 and len(self.indicators['MACD_SIGNAL']) > 2:
            if self.indicators['MACD'][-2] > self.indicators['MACD_SIGNAL'][-2]:
                if self.indicators['MACD'][-1] < self.indicators['MACD_SIGNAL'][-1]:
                    return 1
            else:
                if self.indicators['MACD'][-1] > self.indicators['MACD_SIGNAL'][-1]:
                    return 2
        return 0

    def check_macd_hist(self):
        if self.indicators['MACD_HIST'][-1] > 0:
            if self.indicators['MACD_HIST'][-1] > self.indicators['MACD_HIST'][-2]:
                return 1
        else:
            if self.indicators['MACD_HIST'][-1] < self.indicators['MACD_HIST'][-2]:
                return 2
        return 0

    def check_sma(self):
        if self.indicators['SMA200'][-1] > self.df['close'][-1]:
            return 1
        else:
            return 2

    # Signal (Signal scripts give buy/sell signals. Does not handle stop-loss or take-profit etc.)

    def create_signal(self, check):

        start = self.get_curr_data_time()
        start_price = self.latest_d['close'][0]
        amount = self.assign_capital()
        self.free_margin -= amount

        if check:
            signal = generate_base_signal_dict()
            signal['type'] = check
            signal['start'] = start
            signal['start_price'] = start_price
            signal['vol'] = amount / start_price
            # when you long, you deduct margin and gain asset value = vol*curr
            # you also lose the initial margin. gain it back to free_margin/balance on close
            # note! todo gain * leverage, lose * leverage

            # when you short, you deduct initial margin too. your asset value is negative (decreasing is better)
            # asset value then adds how much you sold. eg. +1.5 * 100 as asset, lose margin, -X * 100.
            # Equity = Margin + (Sold_Rate - Buy_Rate) * vol * lev at every second (SHORT)
            # Equity = Margin + (Sell_Rate - Bought_Rate) * vol * lev (LONG)
            # Buy_Rate and Sell_Rate changes per tick. Bought_Rate digs into balance.
            # Sold_Rate should raise free_margin, but it doesn't

            # Equity = Margin + Balance' + ... So we increase Balance? But we do not receive cash! it is an asset!
            # Equity = Margin + Balance' + Sell_Price * vol * lev

            margin = amount
            if check == 1:  # long
                self.trade += margin
            else:  # short
                self.trade -= margin
                # self.equity -= margin  # Equity calculated separately
            self.signals.append(signal)
            # Adjust equity and capital
            # 'type': None,
            # 'start': None,
            # 'end': None,
            # 'vol': None,  # +ve for long, -ve for short
            # 'net': None,
            # 'leverage': None,
            # # P/L values
            # 'initial_margin': None,
            # 'start_price': None,  # Price on open
            # 'end_price': None,

    def close_signal(self, signal):

        end_price = self.latest_d['close'][0]
        if signal['type'] == 1:  # long
            signal['net'] = signal['vol'] * (signal['start_price'] - end_price)
        else:  # short
            signal['net'] = signal['vol'] * (end_price - signal['start_price'])
        signal['end'] = self.latest_d['datetime'][0]
        signal['end_price'] = end_price
        self.trade -= signal['net']
        self.free_margin += signal['net']

        return signal

    def confirm_signal(self, check, signal):
        pass

    # Statistic update

    def assign_capital(self, check):
        assigned = self.free_margin * 0.1
        external_assigner = False
        if external_assigner:
            pass
        return assigned

    # def calculate_equity(self):
    #     _equity = 0
    #     for signal in self.signals:
    #         if not signal['end']:
    #             if signal['type'] == 1:  # long
    #                 _equity += signal['vol'] * self.latest_d['close'].tolist()[0]
    #             elif signal['type'] == 2:  # short
    #                 _equity -= signal['vol'] * self.latest_d['close'].tolist()[0]
    #     self.equity = self.free_margin + _equity
    #     self.trade = _equity
    #     # todo calculate curr_values

    def next_statistics(self):
        self.balance.append(self.balance[-1])
        self.free_balance.append(self.free_balance[-1])

        self.profit.append(self.profit[-1])
        # self.unrealised_profit.append(self.unrealised_profit[-1])
        self.gross_profit.append(self.gross_profit[-1])
        self.gross_loss_data.append(self.gross_loss_data[-1])

        self.short_margin.append(0)
        self.long_margin.append(0)
        for signal in self.signals:
            if not signal['end']:
                pass

        self.margin.append(max([self.short_margin[-1], self.long_margin[-1]]))
        self.free_margin.append(self.balance[-1] - self.margin[-1] + self.unrealised_profit[-1])

    # Trade properties

    def next_stats(self, candle):
        # todo
        pass

    # Optimisation

    def step_ivar(self, idx, up=True):
        if len(self.ivar_range) >= idx:
            i = -1
            if up:
                i = 1
            self.ivar[idx] += (self.ivar_range[idx][1] - self.ivar[idx]) * robot.IVAR_STEP * i

    # Utility

    def if_go(self):
        # Check if [-1] and [-2] sma exists for the df
        pass

    def sell(self, ind):
        # self.signals = [signal for signal in self.signals if signal['type'] == 'Buy']
        signal = self.signals[ind]
        del self.signals[ind]
        completed_signal = {

        }
        self.completed_signals.append(completed_signal)

    def add_signal(self):
        # if signals were = { 'date': [], 'vol': [] }
        # checking for a signal would be troublesome!
        # more convenient if for signal in signals!
        # therefore signals = [{ 'date': date, 'vol': vol, ...}...]
        pass

    def check_stop_loss(self):
        pass

    def check_profit_loss(self):
        pass

    def close_trade(self):

        # Calculate profit
        pass

    def get_concurr_data(self):
        return self.df[self.df.index.isin(self.stat_datetime)]
