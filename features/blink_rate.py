def calculate_blink_rate(blink_count, time_window_seconds):
    """
    Calculates blinks per minute.
    """

    if time_window_seconds == 0:
        return 0.0

    return round((blink_count / time_window_seconds) * 60, 2)
