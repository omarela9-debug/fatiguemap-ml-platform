def normalize_signal(value, min_value, max_value):
    """
    Normalizes a sensor value to a 0.0-1.0 range.
    """

    if max_value == min_value:
        return 0.0

    return round((value - min_value) / (max_value - min_value), 3)
