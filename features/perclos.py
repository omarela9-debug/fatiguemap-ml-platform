def calculate_perclos(closed_eye_frames, total_frames):
    """
    Calculates PERCLOS / frame closure ratio.
    """

    if total_frames == 0:
        return 0.0

    return round(closed_eye_frames / total_frames, 3)
