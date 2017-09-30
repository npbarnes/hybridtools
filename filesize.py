import os
from math import log

def dir_size(path):
    """Get the total size of the contents of the specified directory.
    Does not include FAT size; only the apparent number of bytes.
    """
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except FileNotFoundError:
                pass
    return total

def human_size(num_bytes):
    """Take a positive size in bytes and convert to a human readable string"""
    unit_list = list(zip(['bytes', 'kB', 'MB', 'GB', 'TB', 'PB'], [0, 0, 1, 2, 2, 2]))
    if num_bytes == 0:
        return '0 bytes'
    elif num_bytes == 1:
        return '1 byte'
    elif num_bytes > 1:
        exponent = min(int(log(num_bytes,1024)), len(unit_list) - 1)
        quotient = float(num_bytes) / 1024**exponent
        unit, num_decimals = unit_list[exponent]
        format_string = '{:.%sf} {}' % (num_decimals)
        return format_string.format(quotient, unit)
    else:
        raise ValueError('num_bytes must be positive')
