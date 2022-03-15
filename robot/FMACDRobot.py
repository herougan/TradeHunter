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
from util.langUtil import strtotimedelta
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

    # Signal Definition:
    #  {}: 'net', 'start', 'end', 'type (short/long)', 'vol', 'type', 'equity',
    #  + 'macd_fail', 'sma_fail', 'ema_fail',

    def __init__(self, ivar=ARGS_DEFAULT, xvar={}):
        if len(ivar) == 2:
            self.ivar = ivar
        else:
            self.ivar = FMACDRobot.ARGS_DEFAULT
        self.ivar_range = FMACDRobot.ARGS_RANGE
        self.steps = 0

        # External attributes
        self.symbol = ""
        self.period = timedelta()
        self.interval = timedelta()

        self.lag = xvar['lag']  # unused
        self.starting_capital = xvar['capital']
        self.leverage = xvar['leverage']
        self.currency_type = xvar['currency']

        # Preparation attributes
        self.prepare_period = 200

        # Indicator Data
        self.indicators = {
            'SMA200': pd.DataFrame(),
            'SMA5': pd.DataFrame(),
            'EMA': pd.DataFrame(),
            'MACD': pd.DataFrame(),
            'MACD_SIGNAL': pd.DataFrame(),
            'MACD_HIST': pd.DataFrame(),
            'MACD_DF': pd.DataFrame(),
        }

        # Tracking variables
        self.free_margin = self.starting_capital
        self.equity = self.free_margin
        # self.assets = 0  # Value of open long positions
        # self.liabilities = 0  # Value of open short positions
        # Statistical Data
        self.con_df = pd.DataFrame()  # df w.r.t to data (old data to build indicators not included)
        self.balance, self.curr_balance = [], 0  # = Previous Balance or Starting capital +- Closed P/L
        self.free_capital, self.curr_free_capital = [], 0  # Balance - Long/Short Buy-ins
        # for forex, buying incurs a "margin" as a cost, so free_margin is equity without margin
        # and free_capital should be balance without margin.
        # The asset value of a short or long is merely the cost differential (can be negative)
        # Balance 100,000, put Margin 2,000 in. free_capital is 98,000. If unrealised P/L = 1,000, Equity is 101,000*
        # Free_Margin is 101,000 - 2,000 = 99,000. Closing the deal, Balance += 1,000.
        # Buildable
        self.profit_data, self.curr_profit = [], 0  # free_margin only
        self.equity_data, self.curr_equity = [], 0  # assets - liabilities + free_margin
        self.gross_profit_data, self.curr_gross_profit = [], 0  # Profit-only from trades
        self.gross_loss_data, self.curr_gross_loss = [], 0  # Loss-only from trades
        self.asset_data, self.curr_asset = [], 0  # Unrealised P/L a.k.a Asset value
        # Operational variables
        self.signals = []
        self.signal_headers = []
        self.completed_signals = []

        """
        For forex:
        
        Free Margin = Remaining Capital
        Margin (/signal) = 
        
        For stock:
        
        Free Margin =
        Margin (/signal) =
        """

        # Data
        self.df = {}
        self.latest_d = pd.DataFrame()

        # Status
        self.test_mode = False
        self.market_active = False
        self.started = False
        self.last_date = datetime.now()

        # ==========Testing Only===========#
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

    def reset(self):
        self.steps = 0

        self.profit_df = []
        self.equity_df = []
        self.signal_df = []

    def test_mode(self):
        self.test_mode = True

    # ======= Start =======

    def start(self, symbol: str, interval: str, period: str):
        """Begin by understanding the incoming data. Setup data will be sent
        E.g. If SMA-200 is needed, at the minimum, the past 400 data points should be known.
        old_data = retrieve()
        From then on, the robot receives data realtime - simulated by feeding point by point. (candlestick)
        data_meta: Symbol, Period, Interval, xvar variables, ...

        Output: Data statistics start from 1 step before the first datapoint (in .next())
        """
        self.symbol = symbol
        self.period = strtotimedelta(period)
        self.interval = interval

        self.reset()
        self.interval = interval
        self.retrieve_prepare()

        # Set up Indicators
        self.build_indicators()

        # Setup trackers
        self.balance.append(self.starting_capital)
        # self.balance, self.curr_balance = [], 0  # = Previous Balance or Starting capital +- Closed P/L
        # self.profit_data, self.curr_profit = [], 0  # todo
        # self.equity_data, self.curr_equity = [], 0  # todo
        # self.gross_profit_data, self.curr_gross_profit = [], 0  # todo
        # self.asset_data, self.curr_asset = [], 0  # Unrealised P/L a.k.a Asset value

        self.started = True

        # If testing
        if self.test_mode:
            pass

    def retrieve_prepare(self, df):

        self.df = df

    def next(self, candlesticks: pd.DataFrame):
        """Same as above, but only when updates are missed so the whole backlog would
         included. Only candlestick_list[-1] will be measured for signals. The time has set
         for the rest but they are needed to calculate the indicators."""

        self.df.append(candlesticks)
        self.last = candlesticks[-1:]
        # last_date = self.last.index[0]  # date
        # open = self.last.Open
        last_value = self.last.Stop

        # Update missed datapoints
        for i in range(len(candlesticks) - 1):
            candle = candlesticks.iloc[i]
            self.next_stats()

        profit = 0
        liquid_assets = 0
        self.profit_data.append(profit)
        self.asset_data.append(liquid_assets)

        # update indicators
        self.build_indicators()
        self.calculate_equity()

        # close open positions
        open_positions = [_signal for _signal in self.signals if not _signal['end']]
        for i in range(open_positions):
            signal = open_positions[i]

            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

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

        # update profit/requity record
        self.assign_equity()  # Assign final values

    # ======= End =======

    def finish(self):

        # remove first variable off data variables
        # self.balance = self.balance[1:]
        # self.profit_data = self.profit_data[1:]
        # self.equity_data = self.equity_data[1:]
        # self.gross_profit_data = self.gross_profit_data[1:]
        # self.asset_data = self.asset_data[1:]
        pass

    def on_complete(self):
        pass
        # self.profit_df = create_profit_df_from_list(self.profit_data, self.asset_data)
        # self.signal_df = create_signal_df_from_list(self.completed_signals, self.signals)
        # self.summary_df = create_summary_df_from_list(self.profit_data, self.asset_data, self.completed_signals)

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

    # Capital management

    def assign_capital(self, check):
        assigned = self.free_margin * 0.1
        return assigned

    def calculate_equity(self):
        _equity = 0
        for signal in self.signals:
            if not signal['end']:
                if signal['type'] == 1:  # long
                    _equity += signal['vol'] * self.latest_d['close'].tolist()[0]
                elif signal['type'] == 2:  # short
                    _equity -= signal['vol'] * self.latest_d['close'].tolist()[0]
        self.equity = self.free_margin + _equity
        self.trade = _equity
        # todo calculate curr_values

    def assign_equity(self):
        # assign curr_values
        pass
        self.profit_data.append(self.net_profit)
        self.equity_data.append(self.equity)

    # Trade properties

    def next_stats(self):
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
