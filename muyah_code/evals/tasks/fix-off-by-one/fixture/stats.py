def mean(values):
    return sum(values) / len(values)


def moving_average(values, window):
    """Average of each consecutive run of `window` values."""
    return [mean(values[i:i + window]) for i in range(len(values) - window)]
