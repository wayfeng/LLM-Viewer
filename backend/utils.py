import functools

PRINT_PARAMS_ENABLED = False  # Set to False to disable printing
def print_params(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if PRINT_PARAMS_ENABLED:
            print(f"Called {func.__name__} with:")
            print("  args:", args)
            print("  kwargs:", kwargs)
        return func(*args, **kwargs)
    return wrapper

def str_number(num, unit=""):
    if num > 1e14:
        return f"{num/1e12:.0f}T{unit}"
    elif num > 1e12:
        return f"{num/1e12:.1f}T{unit}"
    elif num>1e11:
        return f"{num/1e9:.0f}G{unit}"
    elif num > 1e9:
        return f"{num/1e9:.1f}G{unit}"
    elif num > 1e8:
        return f"{num/1e6:.0f}M{unit}"
    elif num > 1e6:
        return f"{num/1e6:.1f}M{unit}"
    elif num > 1e5:
        return f"{num/1e3:.0f}K{unit}"
    elif num > 1e3:
        return f"{num/1e3:.1f}K{unit}"
    elif num >= 1:
        return f"{num:.1f}{unit}"
    else:
        return f"{num:.2f}{unit}"

def str_number_time(num):
    if num >= 1:
        return f"{num:.1f}"
    elif num > 1e-3:
        return f"{num*1e3:.1f}m"
    elif num > 1e-6:
        return f"{num*1e6:.1f}u"
    elif num > 1e-9:
        return f"{num*1e9:.1f}n"
    else:
        return f"{num:.0f}"