"""DistSMA Robot"""
import math
from datetime import timedelta

import pandas as pd

from robot.abstract.robot import robot
from settings import IVarType
from util.langUtil import get_instrument_type, strtotimedelta, strtodatetime


class DistSMA(robot):
    """A robot that trades when the current price is heavily against the current trend.
    SMAs of different periods are employed for: Measuring trend or variability and determining
    pullback chance. Each SMA gives a score and the sum of these are measured against the criterion
    """

    VERSION = '0.1'
    NAME = 'DistSMA'

    ARGS_DICT = {
        # Main, optimisable
        'profit_loss_ratio': {
            'default': 1.5,
            'range': [0.75, 3],
            'step_size': 0.1,  # Default step size
            'type': IVarType.CONTINUOUS,
        },
        'sma_base_period': {
            'default': 10,
            'range': [3, 50],
            'step_size': 1,
            'type': IVarType.DISCRETE,
            'comment': 'Base SMA represents the fastest moving average. All following SMAs'
                       'are slower and have weaker coefficients. These coefficients determine'
                       'how the weight of SMA pullback predictions. The total sums of which'
                       ', if pass some criterion, returns a pullback signal.',
        },
        'sma_period_step': {
            'default': 1.5,
            'range': [1.1, 5],
            'step_size': 0.1,
            'type': IVarType.CONTINUOUS,
            'comment': 'Multiplier, for which following SMA periods will be calculated'
        },
        'sma_base_coeff': {
            'default': 1,
            'range': [0.1, 10],
            'step_size': 0.1,
            'type': IVarType.CONTINUOUS,
        },
        'sma_coeff_step': {
            'default': 0.9,
            'range': [0.1, 1.1],
            'step_size': 0.05,
            'type': IVarType.CONTINUOUS,
            'comment': 'Multiplier, for which following SMA prediction weights will be calculated'
        },
        'sma_count': {
            'default': 5,
            'range': [3, 20],
            'step_size': 1,
            'type': IVarType.DISCRETE,
        },
        # Variability parameters
        'variability_constant_1': {
            'default': 1,
            'range': [1, 10],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
        }
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

    def __init__(self, ivar=ARGS_DICT, xvar={}):
        """XVar, IVar"""

        # == Ivar ==
        if ivar is not None:
            self.ivar = ivar
        else:
            self.ivar = self.ARGS_DICT

        # == Meta ==
        self.symbol = ""
        self.period = timedelta()
        self.interval = timedelta()
        self.instrument_type = get_instrument_type(self.symbol)
        self.lot_size = 100000
        self.pip_size = 0.0001

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
        self.sma_base_period = self.ivar['sma_base_period']['default']
        self.sma_base_coeff = self.ivar['sma_base_coeff']['default']
        self.sma_count = self.ivar['sma_count']['default']
        self.sma_period_step = self.ivar['sma_period_step']['default']
        self.sma_coeff_step = self.ivar['sma_coeff_step']['default']
        # SMA arrays
        self.sma_periods, self.sma_coeffs = [], []
        for i in range(self.sma_count):
            self.sma_periods.append(self.sma_base_period * math.pow(self.sma_period_step, i))
            self.sma_coeffs.append(self.sma_base_coeff * math.pow(self.sma_coeff_step, i))
        self.variability_constant = self.ivar['variability_constant']['default']
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

        # == Indicators ==
        self.sma_indicators = []
        self.sma_variability_indicator = pd.DataFrame()

        # == Signals ==
        self.signals = []  # { Standard_Dict, FMACD_Specific_Dict }
        self.open_signals = []  # Currently open signals
        self.new_closed_signals = []  # Deleted on each 'next'. For easy runtime analysis
        self.new_open_signals = []

        # == Statistical Data ==
        self.balance = []
        self.free_balance = []

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

        self.variability_scores = []
        self.pullback_scores = []

        # == Data ==
        self.df = pd.DataFrame()

        # == Testing ==
        self.test_mode = False
        self.test_idx = 0
        self.df_test = pd.DataFrame()

    def reset(self, ivar=ARGS_DICT, xvar={}):
        if not xvar:
            xvar = self.xvar
        self.__init__(ivar, xvar)

    def start(self, symbol: str, interval: str, period: str, pre_data: pd.DataFrame(), test_mode=False):
        """Pre_data of length PREPARE_PERIOD required. If SMA period exceeds PREPARE_PERIOD, largest SMA period preferred.
        (base * pow(step, count)) in the IVar"""
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
            self.pip_size = 0.01
        else:
            self.pip_size = 0.0001

        # == Prepare ==
        self.df = pre_data
        # Set up Indicators
        self.build_indicators()

        # == Stats Setup ==
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

        self.variability_scores.append(0)
        self.pullback_scores.append(0)
        if len(pre_data) > 0 and not self.test_mode:
            self.stat_datetime.append(strtodatetime(pre_data.index[-1]))
        else:  # todo dont do this
            self.stat_datetime.append(None)

        # == Robot Status ==
        self.started = True


    def next(self):
        pass

    def build_indicators(self):
        pass

    def get_pullback_score(self):

        # (1) Determine Variability
        sma_values = 0

        # (2) Determine Pullback chance
        for sma in self.sma_indicators:
            pass

        # (*3) Compare criterion
        # later

        # (4) Determine variability trend (advanced)

















    def finish(self):
        pass