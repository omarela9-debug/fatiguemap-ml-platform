import json
import os
from collections import deque

import cv2
import numpy as np


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "eye_classifier_model.json")


class EyeStateClassifier:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.labels = []
        self.image_size = (96, 48)
        self.mean = None
        self.std = None
        self.centroids = {}
        self.enabled = False
        self.error = None
        self.load()

    def load(self):
        if not os.path.exists(self.model_path):
            self.error = f"model not found: {self.model_path}"
            return

        try:
            with open(self.model_path) as f:
                self.model = json.load(f)

            self.labels = list(self.model["labels"])
            self.image_size = tuple(self.model.get("image_size", [96, 48]))
            self.mean = np.array(self.model["mean"], dtype=np.float32)
            self.std = np.array(self.model["std"], dtype=np.float32)
            self.std[self.std < 1e-6] = 1.0
            self.centroids = {
                label: np.array(value, dtype=np.float32)
                for label, value in self.model["centroids"].items()
            }
            self.enabled = True
            self.error = None
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    def predict(self, roi):
        if not self.enabled:
            return {
                "available": False,
                "state": "unavailable",
                "confidence": 0.0,
                "probs": {},
                "error": self.error,
            }

        features = extract_features_from_image(roi, self.image_size)
        x = (features - self.mean) / self.std
        distances = {}

        for label, centroid in self.centroids.items():
            distances[label] = float(np.linalg.norm(x - centroid))

        state = min(distances, key=distances.get)
        distance_values = np.array(list(distances.values()), dtype=np.float32)
        centered = distance_values - float(np.min(distance_values))
        temperature = max(1e-6, float(np.std(distance_values)) * 0.75)
        logits = -centered / temperature
        logits -= float(np.max(logits))
        exp_logits = np.exp(logits)
        labels = list(distances.keys())
        total = float(np.sum(exp_logits))
        probs = {
            label: float(value / total)
            for label, value in zip(labels, exp_logits)
        }

        return {
            "available": True,
            "state": state,
            "confidence": probs.get(state, 0.0),
            "probs": probs,
            "error": None,
        }


class RollingPerclos:
    def __init__(self, window_sec=30.0):
        self.window_sec = window_sec
        self.samples = deque()

    def update(self, timestamp, closed_prob):
        self.samples.append((timestamp, float(np.clip(closed_prob, 0.0, 1.0))))
        cutoff = timestamp - self.window_sec

        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

        return self.value()

    def value(self):
        if not self.samples:
            return 0.0

        return 100.0 * sum(value for _, value in self.samples) / len(self.samples)

    def clear(self):
        self.samples.clear()


def extract_features_from_image(image, image_size=(96, 48)):
    if image is None:
        raise ValueError("missing classifier image")

    gray = np.asarray(image)

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray[:, :, :3], cv2.COLOR_RGB2GRAY)

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 95)
    height, _ = blurred.shape
    top = blurred[: height // 3]
    middle = blurred[height // 3 : 2 * height // 3]
    bottom = blurred[2 * height // 3 :]
    dark_mask = blurred < np.percentile(blurred, 20)

    features = [
        float(blurred.mean()),
        float(blurred.std()),
        float(np.percentile(blurred, 5)),
        float(np.percentile(blurred, 10)),
        float(np.percentile(blurred, 25)),
        float(np.percentile(blurred, 50)),
        float(np.percentile(blurred, 75)),
        float(np.percentile(blurred, 90)),
        float(np.percentile(blurred, 95)),
        float(top.mean()),
        float(middle.mean()),
        float(bottom.mean()),
        float(top.std()),
        float(middle.std()),
        float(bottom.std()),
        float(np.count_nonzero(edges) / edges.size),
        float(np.count_nonzero(dark_mask) / dark_mask.size),
    ]

    hist = cv2.calcHist([blurred], [0], None, [16], [0, 256]).flatten()
    hist = hist / max(1.0, float(hist.sum()))
    features.extend(float(x) for x in hist)

    return np.array(features, dtype=np.float32)
