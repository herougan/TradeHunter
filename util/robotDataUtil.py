from settings import IVarType


def create_signal_dict(check, **keyargs):
    """Central method to create signals."""
    pass


def get_arg_range(args, args_range, idx):
    """Get arg_range[i] given args and idx. In the case where
    there are not enough variables for args, a placeholder range
    will be generated,"""
    if idx < len(args_range):
        return args_range[idx]
    elif idx < len(args):
        return [args[idx] * 0.5, args[idx] * 2]
    else:
        return []


def get_arg_dict_step_size(robot):
    args_dict = robot.ARGS_DICT
    defaults = []
    for key in args_dict.keys():
        defaults.append(args_dict[key]['step_size'])

    return defaults


def get_arg_dict_range(robot):
    args_dict = robot.ARGS_DICT
    defaults = []
    for key in args_dict.keys():
        defaults.append(args_dict[key]['range'])

    return defaults


def get_arg_dict_defaults(robot):
    args_dict = robot.ARGS_DICT
    defaults = []
    for key in args_dict.keys():
        defaults.append(args_dict[key]['default'])

    return defaults


def generate_base_signal_dict():
    signal = {
        'type': None,  # 0: None/Error, 1: Long, 2: Short
        'start': None,
        'end': None,
        'vol': None,  # +ve for long, -ve for short (.# of lots)
        'net': None,
        'leverage': None,
        # P/L values
        'margin': None,
        'open_price': None,  # Price on open
        'close_price': None,  # Price on close
        # Success/Failure
        'virtual': True,  # Order success/indicator success
        # Misc_Fail variables here:  # Describe which indicators failed
        # MACD_Hist: False
        # MACD_Hist_weight: 0.5
    }
    return signal


def get_default(arg):
    type = arg['type']
    if type == IVarType.ENUM:
        pass
    if type == IVarType.ENUM:
        pass
    if type == IVarType.ENUM:
        pass
    if type == IVarType.ENUM:
        pass
    if type == IVarType.ENUM:
        pass
    if type == IVarType.ENUM:
        pass
    if type == IVarType.ENUM:
        pass
    if type == IVarType.ENUM:
        pass
    if type == IVarType.ENUM:
        pass
    pass


def get_ivar_val_by_percentage(args_dict, percentage, key):
    """Returns closest 'step' to percentage in args range. E.g. X% refers to the
    nearest step between range[0] and range[1] where step / (range[1] - range[0]) approx.= X%"""
    pass