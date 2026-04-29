def detect_head_nod(current_pitch, previous_pitch, threshold=8.0):
    """
    Detects a possible head nod event using pitch angle change.
    """

    pitch_change = abs(current_pitch - previous_pitch)

    if pitch_change >= threshold:
        return True

    return False
