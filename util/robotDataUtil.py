

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