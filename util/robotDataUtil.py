

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
        'type': None,
        'start': None,
        'end': None,
        'vol': None,  # +ve for long, -ve for short
        'net': None,
        'leverage': None,
        # P/L values
        'initial_margin': None,
        'start_price': None,  # Price on open
        'end_price': None,  # Price on close
        # Calculated values
        # 'asset_value':    vol * start_price - vol * end_price; same for long and short
        # 'base cost':      vol * start_price
        # transaction 'net'/realised P/L =
        #                   asset_value - base_price
        #                   (-ve sign flips base_price value and asset_value for short)
    }
    return signal