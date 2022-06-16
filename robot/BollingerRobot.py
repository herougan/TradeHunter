"""BollingerRobot Robot"""
import math
from datetime import timedelta

import pandas as pd
from dateutil import parser

from robot.abstract.robot import robot
from settings import IVarType
from util.langUtil import get_instrument_type, strtotimedelta, strtodatetime
from util.mathUtil import try_mean, try_int, try_width, try_min, \
    try_max, try_sgn, try_divide
from util.robotDataUtil import generate_base_signal_dict



class BollingerRobot(robot):
    """A robot that uses bollinger bands as signal generation
    """

    VERSION = '0.1'
    NAME = 'BollingerRobot'

    ARGS_DICT = {
        # Main, optimisable
        'profit_loss_ratio': {
            'default': 1.5,
            'range': [0.75, 3],
            'step_size': 0.1,  # Default step size
            'type': IVarType.CONTINUOUS,
        },
        'sma_base_period': {
            'default': 20,
            'range': [3, 60],
            'step_size': 1,
            'type': IVarType.DISCRETE,
            'comment': 'Base SMA represents the fastest moving average. All following SMAs'
                       'are slower and have weaker coefficients. These coefficients determine'
                       'how the weight of SMA pullback predictions. The total sums of which'
                       ', if pass some criterion, returns a pullback signal.',
        },
        'sma_period_step': {
            'default': 1.3,
            'range': [0.5, 3],
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
            'default': 6,
            'range': [3, 20],
            'step_size': 1,
            'type': IVarType.DISCRETE,
        },
        # Variability parameters
        'variability_constant': {
            'default': 1,
            'range': [1, 10],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
        },
        'variability_requirement': {
            'default': 3,
            'range': [1, 5],
            'step_size': 0.01,
            'type': IVarType.CONTINUOUS,
        },
        'stop_loss_variability': {
            'default': 1.5,
            'range': [0.5, 5],
            'step_size': 0.01,
            'type': IVarType.CONTINUOUS,
        },
        'ma_type': {
            'default': 0,
            'range': ['Simple', 'Exponential', 'Static'],
            'type': IVarType.ENUM,
        },
        #
        'grid_length': {
            'default': 10,
            'range': [5, 20],
            'type': IVarType.CONTINUOUS,
        },
        'split_count': {
            'default': 3,
            'range': [1, 10],
            'type': IVarType.DISCRETE,
        },
        'split_step': {
            'default': 1.1,
            'range': [0.5, 2],
            'type': IVarType.CONTINUOUS,
        },
    }
    OTHER_ARGS_DICT = {
        'fast_period': {
            'default': 12,
            'range': [10, 15],
            'step_size': 0.1,
            'type': IVarType.CONTINUOUS,
        },
        'slow_period': {
            'default': 26,
            'range': [25, 30],
            'step_size': 0.1,
            'type': IVarType.CONTINUOUS,
        },
        'signal_period': {
            'default': 9,
            'range': [8, 10],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
        },
        'left_peak': {
            'default': 2,
            'range': [1, 4],
            'step_size': 1,  # Default step size
            'type': IVarType.CONTINUOUS,
        },
        'right_peak': {
            'default': 2,
            'range': [1, 4],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
        },
        'look_back': {
            'default': 20,
            'range': [10, 30],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
        },
        'peak_order': {
            'default': 1,
            'range': [1, 5],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
        },
        'lots_per_k': {
            'default': 0.0005,
            'range': [0.0001, 0.01],
            'step_size': 0.0001,
            'type': IVarType.CONTINUOUS,
        },
        'stop_loss_amp': {
            'default': 1.05,
            'range': [0.9, 2],
            'step_size': 0.001,
            'type': IVarType.CONTINUOUS,
        },
        'stop_loss_flat_amp': {  # in pips
            'default': 0,
            'range': [-100, 100],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
        },
        'amp_constant': {
            'default': 0,
            'range': [-100, 100],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
        },
        'amp_types': {
            'default': 0,
            'range': [0, 1, 2, 3, 5, 10],
            'step_size': 1,
            'type': IVarType.ARRAY,
        },
        'amp_bins': {
            'default': 0,
            'range': [-100, 100],
            'step_size': 1,
            'type': IVarType.CONTINUOUS,
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
        self.last = None
        self.last_date = None
        self.last_var = None
        # signal management
        self.bought = []

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
        self.variability_constant = self.ivar['variability_constant']['default']
        self.variability_requirement = self.ivar['variability_requirement']['default']
        self.stop_loss_variability = self.ivar['stop_loss_variability']['default']
        self.ma_type = self.ivar['ma_type']['default']
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
        for i in range(self.sma_count):
            self.sma_indicators.append(pd.DataFrame())
            self.sma_periods.append(int(self.sma_base_period * math.pow(self.sma_period_step, i)))
            self.sma_coeffs.append(int(self.sma_base_coeff * math.pow(self.sma_coeff_step, i)))
        # Auxillary indicators
        self.sma_variability_indicator, self.sma_variability_trend = pd.DataFrame(), pd.DataFrame()
        self.sma_trend_predictor, self.sma_average = pd.DataFrame(), pd.DataFrame()  # ~of the largest period SMA used
        self.sma_average_up_1, self.sma_average_up_2, self.sma_average_down_1, self.sma_average_down_2 \
            = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # == Signals ==
        self.signals = []  # { Standard_Dict, FMACD_Specific_Dict }
        self.open_signals = []  # Currently open signals
        self.new_closed_signals = []  # Deleted on each 'next'. For easy runtime analysis
        self.new_open_signals = []

        # == Statistical Data ==
        self.balance = []
        self.free_balance = []
        #
        self.profit = []  # Realised P/L
        self.unrealised_profit = []  # Unrealised P/L
        self.gross_profit = []  # Cumulative Realised Gross Profit
        self.gross_loss = []  # Cumulative Realised Gross Loss
        self.virtual_profit = []
        #
        self.asset = []  # Open Long Position price
        self.liability = []  # Open Short Position Price
        self.short_margin, self.long_margin = [], []  # Total Short/Long Margins
        self.margin = []  # Margin (Max(Long_Margin), Max(Short_Margin))
        self.free_margin = []  # Balance - Margin + Unrealised P/L
        #
        self.equity = []  # Free_Balance + Asset (Long) - Liabilities (Short) OR Forex Free_Margin + Margin
        self.margin_level = []  # Equity / Margin * 100%, Otherwise 0%
        self.stat_datetime = []
        #
        self.variability_scores = []  # See self.sma_variability_indicator
        self.pullback_scores = []

        # == Data ==
        self.df = pd.DataFrame()

        # == Logical ==
        self.last_bought = 0

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
        # Set up Indicators for pre_data
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

        # pre-Setup indicators
        _index = list(pre_data.index)
        self.stat_datetime = _index

        # == Robot Status ==
        self.started = True

    def apply_xvar(self, xvar):

        self.lag = xvar['lag']
        self.starting_capital = xvar['capital']
        self.leverage = xvar['leverage']
        self.instrument_type = xvar['instrument_type']
        # self.currency = xvar['currency']  # from symbol
        self.commission = xvar['commission']
        self.contract_size = xvar['contract_size']
        if 'lot_size' in xvar:
            self.lot_size = xvar['lot_size']
        if 'pip_size' in xvar:
            self.pip_size = xvar['pip_size']

    def next(self, candlesticks: pd.DataFrame()):

        # == Step 1: Update data ==
        if not self.test_mode:
            self.df = pd.concat([self.df, candlesticks])
            self.build_indicators()
        self.last = candlesticks.iloc[-1]
        self.last_date = candlesticks.index[-1]

        # == Step 2: Update stats ==
        for i in range(len(candlesticks)):
            self.next_statistics(candlesticks.iloc[i])

        # == Step 3: Analyse ==

        # == Step 4: Signals ==

        # 4a - Check current signals
        if len(self.df) > self.sma_periods[-1]:  # Wait until all SMAs summoned
            for signal in self.open_signals:
                self.adjust_stop_take(signal)
                self.close_signal(signal)  # Try to close signal
        # 4b - Insure signals  #  Triple buys with diff risk

        # 4c - Create new signals
            sign = self.check_indicators()
            if sign:  # returns potential signal sign 0: No,
                # 1: Long, 2: Short, 3: Long Virtual, 4: Short Virtual (Only for testing)
                signal = self.generate_signal(sign)
                # todo 1) cooldown (n/or)
                # 2) ONLY if even better deal (n/o)
                # 3) Increase volume based on probability
                # 4) Saving old trades, martingale OR pushing profit line
                # 5) Moving stop-take
                #- ---- -- -- --
                # 5) Pullback score on PROJECTED* direction. if market trending downwards,
                # being 3-score below is not enough (var + trend_score), while being above only 1-2-score
                # could be
                # 6) Let go old trades... if super old, close asap, if old, find next best moment
                # run the reversal alg to find next best time to sell. if projected time is too far,
                # close immediately.
                if signal is not None:
                    self.open_signals.append(signal)

        # todo work on: handling open signals

    def finish(self):
        pass

    # Simulation

    def test(self):
        pass

    def sim_next(self, candlesticks: pd.DataFrame()):
        """Full run once to prevent repetitive computation during testing."""
        self.next(candlesticks)

    # Temperature

    def get_signals(self):
        return self.signals, self.open_signals

    def get_profit(self):
        return self.profit, self.equity, self.balance, self.margin

    def get_instructions(self):
        SMA = []
        i = 0
        for indicator in self.sma_indicators:
            # ratio = i / len(self.sma_indicators)
            # red, green, blue = 1 - ratio, 1 - ratio, ratio
            # i += 1
            # _col = (red, green, blue)
            # # cancelled
            _col = 'darkred'
            SMA.append({
                'index': 0,
                'data': indicator,
                'type': 'line',
                'colour': _col,
                'name': 'SMA',
            })

        return SMA + [
            # {
            #     'index': 0,
            #     'data': self.sma_variability_indicator,  # .copy(deep=True)
            #     'type': 'line',
            #     'colour': 'blue',
            #     'name': 'SMA_v',
            # },
            {
                'index': 1,
                'data': self.sma_variability_trend,
                'type': 'line',
                'colour': 'blue',
                'name': 'SMA_t',
            },
            {
                'index': 1,
                'data': self.sma_trend_predictor,
                'type': 'line',
                'colour': 'red',
                'name': 'SMA_p',
            },
            # Average lines and sma-width deviation
            {
                'index': 0,
                'data': self.sma_average,
                'type': 'line',
                'colour': 'darkturquoise',
                'linestyle': 'dashed',
                'name': 'SMA_mean',
            },
            {
                'index': 0,
                'data': self.sma_average_up_1,
                'type': 'line',
                'colour': 'darkcyan',
                'linestyle': 'dashed',
                'name': 'SMA_mean',
            },
            {
                'index': 0,
                'data': self.sma_average_up_2,
                'type': 'line',
                'colour': 'darkslategrey',
                'linestyle': 'dashed',
                'name': 'SMA_mean',
            },
            {
                'index': 0,
                'data': self.sma_average_down_1,
                'type': 'line',
                'colour': 'darkcyan',
                'linestyle': 'dashed',
                'name': 'SMA_mean',
            },
            {
                'index': 0,
                'data': self.sma_average_down_2,
                'type': 'line',
                'colour': 'darkslategrey',
                'linestyle': 'dashed',
                'name': 'SMA_mean',
            },
        ]

    # Indicator

    def build_indicators(self):
        """Appends data starting from """
        # i-th self.sma_indicators is the i-th SMA
        for i in range(len(self.sma_indicators)):
            _period = self.sma_periods[i]
            _data, _index = [], []
            # len(sma)+period is the next index, sum to just before u+1 from u+1-period
            for u in range(len(self.sma_indicators[i]), len(self.df)):
                _index.append(self.df.index[u])
                if u + 1 >= _period:
                    _data.append(try_mean(self.df.Close[u + 1 - _period: u + 1].values))
                else:  # Not enough datapoints for given SMA period
                    _data.append(None)
            # Extend SMA
            _indicator = pd.DataFrame(index=_index, data={'Value': _data})
            self.sma_indicators[i] = pd.concat([self.sma_indicators[i], _indicator])
        # Auxillary indicators: Update sma_variability
        v_data, avg_data, t_data, p_data, _index = [], [], [], [], []
        avg_up_1, avg_up_2, avg_down_1, avg_down_2 = [], [], [], []
        for i in range(len(self.sma_variability_indicator), len(self.sma_indicators[-1])):
            # Length of sma variability indicator is the same as the other sma stats
            _index.append(self.sma_indicators[-1].index[i])
            _values = [indicator.Value.iloc[-1] for indicator in self.sma_indicators]
            p_data.append(5000)
            t_data.append(5000)
            # Avg and var values
            v_data.append(try_width(_values))
            avg_data.append(try_mean(_values))
            # + 1* variability, + 2*, -1*, -2*
            avg_up_1.append(avg_data[-1] + v_data[-1])
            avg_up_2.append(avg_data[-1] + 2 * v_data[-1])
            avg_down_1.append(avg_data[-1] - v_data[-1])
            avg_down_2.append(avg_data[-1] - 2 * v_data[-1])
        self.sma_variability_indicator = pd.concat([self.sma_variability_indicator, pd.DataFrame(
            index=_index,
            data=v_data
        )])
        self.sma_variability_trend = pd.concat([self.sma_variability_trend, pd.DataFrame(
            index=_index,
            data=t_data
        )])
        self.sma_trend_predictor = pd.concat([self.sma_trend_predictor, pd.DataFrame(
            index=_index,
            data=p_data,  # Use where the trend is pointing to. maybe test without this first.
        )])
        # Avg indicators
        self.sma_average = pd.concat([self.sma_average, pd.DataFrame(
            index=_index,
            data=avg_data,
        )])
        self.sma_average_up_1 = pd.concat([self.sma_average_up_1, pd.DataFrame(
            index=_index,
            data=avg_up_1,
        )])
        self.sma_average_up_2 = pd.concat([self.sma_average_up_2, pd.DataFrame(
            index=_index,
            data=avg_up_2,
        )])
        self.sma_average_down_1 = pd.concat([self.sma_average_down_1, pd.DataFrame(
            index=_index,
            data=avg_down_1,
        )])
        self.sma_average_down_2 = pd.concat([self.sma_average_down_2, pd.DataFrame(
            index=_index,
            data=avg_down_2,
        )])
        self.last_var = v_data[-1]

    def get_pullback_score(self):

        # (1) Determine Variability
        var = self.sma_variability_indicator.iloc[-1, 0]  # todo dev too high number

        # (2) Determine Pullback distance in terms of the variability indicator
        avg = self.sma_average.iloc[-1, 0]
        dist = self.last.Close - avg
        # How many 'sma_width' deviations away?
        if var is None or avg is None:
            dev = None
        else:
            dev = try_divide(dist, var)
        # todo is using var smart? higher var means sudden jumps with periods captured within the sma periods
        # var does not capture consistency in variability. if variability is high,
        # buying low on a down swing is more likely to sell high eg.
        # there is that intuition that variability SUPPORTS swing trading.
        # need to study shoulder cup etc methods in the future

        return dev

    def check_indicators(self):
        score = self.get_pullback_score()
        if score is None:
            return 0
        sgn = try_sgn(score)

        # (*3) Compare criterion
        var_req = self.variability_requirement
        var_req_sub = var_req * 0.75
        if abs(score) > var_req:
            if sgn > 0:
                return 2
            else:
                return 1
        elif abs(score) > var_req_sub:
            if sgn > 0:
                return 4
            else:
                return 3

        # (4) Determine variability trend (advanced)
        # -
        return 0

    # Analysis

    def get_supports(self):
        pass

    # Signal

    def create_signal(self, signal, check):
        pass

    def generate_signal(self, sign):
        signal = generate_base_signal_dict()
        # Assign dict values
        virtual = False
        if sign == 1:
            type = 1
        elif sign == 2:
            type = 2
        elif sign == 3:
            type = 1
            virtual = True
        elif sign == 4:
            type = 2
            virtual = True
        else:
            return None
        signal['type'] = type
        signal['start'] = self.last_date
        signal['vol'] = self.assign_lots(type)  # -ve lots = selling
        sgn = math.copysign(1, signal['vol'])
        signal['leverage'] = self.xvar['leverage']
        # Action *= Leverage; Margin /= Leverage
        signal['margin'] = sgn * signal['vol'] / self.xvar['leverage'] * self.lot_size
        signal['open_price'] = self.last.Close
        # Generate stop loss and take profit
        signal['stop_loss'], signal['baseline'] = self.get_stop_loss(type)
        signal['take_profit'] = self.get_take_profit(signal['stop_loss'])
        signal['virtual'] = virtual
        return signal

    def close_signal(self, signal):
        # Checking algorithm here
        sgn = math.copysign(1, signal['vol'])  # +ve for long, -ve for short
        stop, take = signal['stop_loss'], signal['take_profit']
        # Stop-loss OR Take-profit
        # todo closing wrongly
        if sgn * self.last.Close <= sgn * stop or sgn * self.last.Close >= sgn * take:  # Go-ahead
            # Realise Profit/Loss
            action = (self.last.Close - signal['open_price'])
            profit = action * signal['vol'] * self.leverage * self.lot_size
            # self.xvar['contract_size']
            signal['end'] = self.last_date
            signal['close_price'] = self.last.Close
            sgn = math.copysign(1, signal['vol'])
            signal['net'] = profit

            if not signal['virtual']:
                # Release margin, release unrealised P/L
                self.add_profit(profit)
                self.add_margin(sgn, signal['margin'])
                self.calc_equity()
            else:
                self.add_virtual_profit(profit)

            # Add signal to signals, remove from open signals
            self.open_signals.remove(signal)
            self.signals.append(signal)

    # Statistic update

    def add_profit(self, profit):
        # self.unrealised_profit[-1] -= profit
        self.balance[-1] += profit
        self.margin[-1] += profit
        self.profit[-1] += profit
        if profit > 0:
            self.gross_profit[-1] += profit
        else:
            self.gross_loss[-1] -= profit

    def add_virtual_profit(self, profit):
        self.virtual_profit[-1] += profit

    def add_margin(self, vol, margin):
        """Adding margins reduce free margin. sgn is the sign of signal volume"""
        if vol > 0:
            self.long_margin[-1] += margin
        else:
            self.short_margin[-1] += margin
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

    def adjust_stop_take(self, signal):
        pass

    def split_signal(self, signal, _signal):
        """Creates new signal which takes volume defined by _signal away from original
        signal."""
        pass

    def get_stop_loss(self, type):
        if type == 1:  # Buy
            return - self.last_var * self.stop_loss_variability + self.last.Close, self.last_date
        else:  # Sell
            return self.last_var * self.stop_loss_variability + self.last.Close, self.last_date

    def get_take_profit(self, stop_loss):
        diff = self.last.Close - stop_loss
        return self.last.Close + diff * self.profit_loss_ratio

    # Technical
    def next_statistics(self, candlestick):
        self.balance.append(self.balance[-1])
        self.free_balance.append(self.balance[-1])

        self.profit.append(self.profit[-1])
        self.unrealised_profit.append(0)
        self.gross_profit.append(self.gross_profit[-1])
        self.gross_loss.append(self.gross_loss[-1])
        self.virtual_profit.append(self.virtual_profit[-1])

        self.short_margin.append(0)
        self.long_margin.append(0)
        self.asset.append(self.asset[-1])
        self.liability.append(self.liability[-1])
        for signal in self.open_signals:
            if signal['virtual']:
                continue
            if signal['type'] == 'short':
                # Calculate margin
                self.short_margin[-1] += signal['margin']
            else:  # == 'long'
                self.long_margin[-1] += signal['margin']
            # Calculate unrealised P/L
            self.unrealised_profit[-1] += (candlestick.Close - signal['open_price']) * signal['vol'] * self.lot_size
            # math.copysign(
            # (candlestick.Close - signal['open_price']) * signal['vol'] * self.lot_size, signal['vol'])

        self.margin.append(max([self.short_margin[-1], self.long_margin[-1]]))
        self.free_balance[-1] -= self.margin[-1]
        self.free_margin.append(self.balance[-1] - self.margin[-1] + self.unrealised_profit[-1])

        # self.equity.append(self.asset[-1] - self.liability[-1])
        self.equity.append(self.free_margin[-1] + self.margin[-1])
        if self.margin[-1] == 0:
            self.margin_level.append(0)
        else:
            self.margin_level.append(self.equity[-1] / self.margin[-1] * 100)  # %
        self.stat_datetime.append(parser.parse(self.last_date))

        self.pullback_scores.append(self.get_pullback_score())
        # self.variability_scores.append(self.sma_variability_indicator.data[-1])
        if len(self.sma_variability_indicator) > 0:
            self.variability_scores.append(self.sma_variability_indicator.iloc[-1, 0])
        else:
            self.variability_scores.append(0)

    def assign_lots(self, _type, catalyst=1):
        sgn = 1
        if _type == 2:
            sgn = -1
        vol = sgn * self.lot_per_k * self.free_margin[-1] / 1000  # vol
        vol *= catalyst
        if sgn * vol < 0.01:
            vol = sgn * 0.01  # Min 0.01 lot
        return vol




























    def te_mp(self):
        pass
