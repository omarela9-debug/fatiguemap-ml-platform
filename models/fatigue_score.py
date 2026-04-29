def calculate_fatigue_score(perclos, blink_rate, head_nod_count):
    """
    Computes a baseline fatigue score from multimodal features.
    Score ranges from 0.0 to 1.0.
    """

    perclos_weight = 0.45
    blink_weight = 0.25
    nod_weight = 0.30

    normalized_blink = min(blink_rate / 30, 1.0)
    normalized_nods = min(head_nod_count / 10, 1.0)

    score = (
        perclos_weight * perclos +
        blink_weight * normalized_blink +
        nod_weight * normalized_nods
    )

    return round(min(score, 1.0), 3)
