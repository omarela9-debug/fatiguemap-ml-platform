class BaselineFatigueModel:
    """
    Rule-based baseline model for real-time fatigue estimation.
    This is designed as a first version before replacing it with
    a trained ML model.
    """

    def __init__(self):
        self.weights = {
            "perclos": 0.45,
            "blink_rate": 0.25,
            "head_nod_count": 0.30
        }

    def predict(self, perclos, blink_rate, head_nod_count):
        normalized_blink = min(blink_rate / 30, 1.0)
        normalized_nods = min(head_nod_count / 10, 1.0)

        score = (
            self.weights["perclos"] * perclos +
            self.weights["blink_rate"] * normalized_blink +
            self.weights["head_nod_count"] * normalized_nods
        )

        return round(min(score, 1.0), 3)
