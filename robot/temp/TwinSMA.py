"""TwinSMA Robot"""
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


# This TwinSMA algorithm uses SMA crossing as a trigger signal,
#
#
from settings import IVarType
from util.langUtil import strtotimedelta, get_instrument_type, strtodatetime
from util.robotDataUtil import generate_base_signal_dict


class TwinSMA(robot):
    """A simple robot to test the TradingHunter suite."""

    # N_ARGS = 2
    # ARGS_STR = ['stop_loss', 'take_profit', 'fast_period', 'slow_period', 'signal_period', 'sma_period']
    # ARGS_DEFAULT = [1, 1.5, 12, 26, 9, 200]
    # ARGS_RANGE = [[0.01, 10], [0.01, 10],
    #               [10, 15], [22, 30],
    #               [8, 10], [150, 250], ]
    ARGS_DICT = {
        # Main, optimisable
        'profit_loss_ratio': {
            'default': 1.5,
            'range': [0.75, 3],
            'step_size': 0.1,  # Default step size
            'type': IVarType.CONTINUOUS,
        },
        'sma_base_period': {
            'default': 200,
            'range': [100, 300],
            'step_size': 1,
            'type': IVarType.DISCRETE,
        },
        'sma_period_step': {
            'default': 200,
            'range': [100, 300],
            'step_size': 1,
            'type': IVarType.DISCRETE,
        },
        'sma_base_step': {
            'default': 200,
            'range': [100, 300],
            'step_size': 1,
            'type': IVarType.DISCRETE,
        },
        'sma_coeff_step': {
            'default': 200,
            'range': [100, 300],
            'step_size': 1,
            'type': IVarType.DISCRETE,
        },
        'number_of_sma': {
            'default': 200,
            'range': [100, 300],
            'step_size': 1,
            'type': IVarType.DISCRETE,
        },
    }
    OTHER_ARGS_DICT = {
        'fast_period': {
            'default': 12,
            'range': [10, 15],
            'step_size': 0.1,
        },
        'slow_period': {
            'default': 26,
            'range': [25, 30],
            'step_size': 0.1,
        },
        'signal_period': {
            'default': 9,
            'range': [8, 10],
            'step_size': 1,
        },
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
            'range': [0.0005, 0.1],
            'step_size': 0.0001,
        },
        'stop_loss_amp': {
            'default': 1.05,
            'range': [0.9, 2],
            'step_size': 0.001,
        },
        'stop_loss_flat_amp': {  # in pips
            'default': 0,
            'range': [-100, 100],
            'step_size': 1,
        },
        'amp_constant': {
            'default': 0,
            'range': [-100, 100],
            'step_size': 1,
            'type': 'continuous',
        },
        'amp_types': {
            'default': 0,
            'range': [0, 1, 2, 3, 5, 10],
            'step_size': 1,
            'type': 'array',
        },
        'amp_bins': {
            'default': 0,
            'range': [-100, 100],
            'step_size': 1,
            'type': 'discrete',
        }
    }
    # Retrieve Prep
    PREPARE_PERIOD = 200

    # Other
    MARGIN_RISK_PER_TRADE = 0
    LOSS_PERCENTAGE = 0

    # Plotting variables
    PLOT_NO = [0, 0, 1]  # (Trailing)

    VERSION = '0.1'
    NAME = 'TwinSMA'

    def __init__(self, ivar=ARGS_DICT, xvar={}):
        """XVar variables should be numbers or strings.
        e.g. leverage must be a number (100), not '1:100'.
        instrument_type should be ...(pip value)
        instrument_type should be ...(forex)
        commission should be a number"""

        # == IVar ==
        if len(ivar.keys()) >= len(self.ARGS_DICT.keys()):
            self.ivar = ivar
        else:
            self.ivar = TwinSMA.ARGS_DICT
        self.ivar_range = {}

        # == Data Meta ==
        self.symbol = ""
        self.period = timedelta()
        self.interval = timedelta()
        self.instrument_type = get_instrument_type(self.symbol)
        self.lot_size = 100000

        # == XVar ==
        self.xvar = xvar
        self.lag = xvar['lag']  # unused
        self.starting_capital = xvar['capital']
        self.leverage = xvar['leverage']
        self.instrument_type = xvar['instrument_type']
        # self.currency = xvar['currency']
        self.commission = xvar['commission']
        self.contract_size = xvar['contract_size']
        if 'lot_size' in xvar:
            self.lot_size = xvar['lot_size']

        # == Main Args ==
        self.profit_loss_ratio = self.ivar['profit_loss_ratio']['default']
        self.sma_period = self.ivar ['sma_period']['default']
        # (Demoted)
        self.fast_period = self.OTHER_ARGS_DICT['fast_period']['default']
        self.slow_period = self.OTHER_ARGS_DICT['slow_period']['default']
        self.signal_period = self.OTHER_ARGS_DICT['signal_period']['default']
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
        self.indicators_done = {
            # Non-critical
            'SMA5': False,
            'SMA50': False,
            'SMA200': False,
            'SMA200_HIGH': False,
            'EMA200': False,
            # Critical
            'MACD_HIST': False,
            'MACD_DF': False,
            # Signal generators
            'MACD': False,
            'MACD_SIGNAL': False,
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
        self.signals = []  # { Standard_Dict, }
        self.open_signals = []  # Currently open signals
        self.new_closed_signals = []  # Deleted on each 'next'. For easy runtime analysis
        self.new_open_signals = []
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
        self.virtual_profit = []

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
        self.market_active = False
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

    def reset(self, ivar=ARGS_DICT, xvar={}):
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
        self.reset(self.ivar, self.xvar)
        self.test_mode = test_mode

        # == Data Meta ==
        self.symbol = symbol
        self.period = strtotimedelta(period)
        self.interval = interval
        self.instrument_type = get_instrument_type(symbol)
        if self.xvar['instrument_type'] != "Forex":
            self.lot_size = 100
        else:
            self.lot_size = 100000
        if self.symbol == "JPY=X":
            self.lot_size = 1000

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
        self.virtual_profit.append(0)

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
