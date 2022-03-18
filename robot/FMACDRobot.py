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
    # Retrieve Prep
    PREPARE_PERIOD = 200

    # Other
    CAPITAL_PER_TRADE = 0  # todo determine with pops later
    LOSS_PERCENTAGE = 0
    OTHER_ARGS_STR = ['left_peak', 'right_peak', 'look_back']
    # OTHER_ARGS_STR = ['takeprofit_2', 'takeprofit_2_ratio', 'capital_ratio', 'lot_size']
    OTHER_ARGS_DEFAULT = [2, 2, 20]

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
        self.contract_size = xvar['contract_size']  # here too! add

        # == Preparation ==
        self.prepare_period = self.PREPARE_PERIOD # Because of SMA200
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
        # == Signals ==
        self.signals = []  # { Standard_Dict, FMACD_Specific_Dict }
        self.open_signals = []

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

        # ==    a: Check to close deals
        for signal in self.open_signals:

            sgn = math.copysign(signal['vol'], 1)  # +ve for long, -ve for short
            stop, take = signal['stop_loss'], signal['take_profit']
            # Stop-loss OR Take-profit
            if sgn * self.last.Close <= sgn * stop or sgn * self.last.Close >= sgn * take:
                self.close_signal(signal)
        # todo here =======================================================
        # == =   b1: Check to create signals =============

        # Generate signals
        signal = self.generate_signal()
        action = (signal['open_price'] - self.last.Close) * signal['vol'] * self.leverage
        if self.instrument_type == "Forex":
            # Differentiate here
            pass
        else:
            if sgn:
                self.asset[-1] += signal['vol'] * self.last.Close
            else:
                self.liability[-1] += signal['vol'] * self.last.Close
            pass
        margin = signal['vol'] * self.contract_size / self.leverage

        # == =   b2: Check indicators =============

        #  Check indicators
        if signal:
            pass

            # == =   b3: Confirm Signal =============

            # if okay, deduct assets and call open_signals
            pass

        # ==    c: =============

        # Imagine if this is real: Money - use API to get, Open_Signals, do store - but made AND call to server!
        # reconfigure later -
        # Closing signals might be done differently, so do a diff check algorithm!

        # == Step 5: Cleanup =============
        # Do nothing

        # delete useless methods!

        # do test version

        # close open positions
        open_positions = [_signal for _signal in self.signals if not _signal['end']]
        for i in range(open_positions):
            signal = open_positions[i]

            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

            # use data[-1] as latest
            last_value = self.last.Close
            if signal['type'] == 1:
                if last_value < stop_loss or last_value > take_profit:
                    signal = self.close_signal(signal)
                    open_positions[i] = signal
            elif signal['type'] == 2:
                if last_value > stop_loss or last_value < take_profit:
                    signal = self.close_signal(signal)
                    open_positions[i] = signal

        # check signals

        # how todo

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

    def rebuild_macd(self, period):
        self.indicators['MACD'], self.indicators['MACD_SIGNAL'], self.indicators['MACD_HIST'] = \
            talib.MACD(self.df, fastperiod=12, slowperiod=26, signalperiod=9)
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

    def generate_signal(self, check):




        return None

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

        # Realise Profit/Loss
        action = (signal['open_price'] - self.last.Close[-1]) * signal['vol'] * self.leverage
        signal['end'] = self.last.index[-1]

#         File "pandas/_libs/tslibs/timestamps.pyx", line 348, in pandas._libs.tslibs.timestamps._Timestamp.__sub__
# TypeError: Timestamp subtraction must have the same timezones or no timezones

        sgn = math.copysign(signal['vol'], 1)  # +ve for long, -ve for short
        stop, take = signal['stop_loss'], signal['take_profit']
        # Stop-loss OR Take-profit
        if sgn * self.last.Close <= sgn * stop or sgn * self.last.Close >= sgn * take:
            self.close_signal(signal)

        end_price = self.latest_d['close'][0]
        if signal['type'] == 1:  # long
            signal['net'] = signal['vol'] * (signal['start_price'] - end_price)
        else:  # short
            signal['net'] = signal['vol'] * (end_price - signal['start_price'])
        signal['end'] = self.latest_d['datetime'][0]
        signal['end_price'] = end_price
        self.trade -= signal['net']
        self.free_margin += signal['net']
        # if self.instrument_type == "Forex":
        #     # Differentiate here
        #     pass
        # else:
        #     if sgn:
        #         self.asset[-1] += signal['vol'] * self.last.Close
        #     else:
        #         self.liability[-1] += signal['vol'] * self.last.Close
        #     pass


        # VOL * CONTRACT_SIZE / LEVERAGE = 1 * 100,000 / 100 = 1,000 (MARGIN)
        # n_lots * size/lot / leverage

        # PROFITLOSS = VOL * PRICE_ACTION * LEVERAGE
        #              n_lots * $ * leverage

        return signal

    def confirm_signal(self, check, signal):
        pass

    def macd_signal(self):




        pass

    # Statistic update

    def assign_equity(self, check):
        assigned = self.free_margin * 0.1
        external_assigner = False
        if external_assigner:
            pass
        return assigned

    def add_equity(self, margin):
        pass

    def add_margin(self, margin):
        pass

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

    def next_statistics(self, candlestick):
        self.balance.append(self.balance[-1])
        self.free_balance.append(self.free_balance[-1])

        self.profit.append(self.profit[-1])
        self.unrealised_profit.append(0)
        self.gross_profit.append(self.gross_profit[-1])
        self.gross_loss_data.append(self.gross_loss_data[-1])

        self.short_margin.append(0)
        self.long_margin.append(0)
        self.asset.append(0)
        self.liability.append(0)
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
        self.free_margin.append(self.balance[-1] - self.margin[-1] + self.unrealised_profit[-1])

        # self.equity.append(self.asset[-1] - self.liability[-1])
        self.equity.append(self.free_margin[-1] + self.marin[-1])
        if self.margin == 0:
            self.margin_level.append(0)
        else:
            self.margin_level.append(self.equity / self.margin * 100)
        self.stat_datetime.append(candlestick[-1:].index)

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

    def close_signal(self, signal):
        self.open_signals.remove(signal)
        self.signals.append(signal)
