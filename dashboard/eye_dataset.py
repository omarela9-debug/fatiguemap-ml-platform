import csv
import os
import re
import threading
import time
from datetime import datetime

import cv2


ALLOWED_LABELS = {"open", "closed", "blink", "bad", "other"}


def _safe_name(value, fallback):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = text.strip("_")
    return text or fallback


class EyeDatasetRecorder:
    def __init__(self, base_dir="eye_dataset"):
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(os.path.dirname(__file__), base_dir)

        self.base_dir = base_dir
        self.lock = threading.Lock()
        self.active = False
        self.session_dir = None
        self.metadata_path = None
        self.metadata_file = None
        self.writer = None
        self.label = "open"
        self.note = ""
        self.frame_count = 0
        self.saved_count = 0
        self.error_count = 0
        self.max_fps = 12.0
        self.last_saved_at = 0.0
        self.save_raw = False
        self.save_roi = True
        self.session_started_at = None

    def start(self, label="open", note="", session_name=None, max_fps=12.0, save_raw=False, save_roi=True):
        label = self._normalize_label(label)
        max_fps = self._normalize_fps(max_fps)
        session_label = _safe_name(session_name, "eye_session")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with self.lock:
            self.stop_locked()

            self.label = label
            self.note = str(note or "")
            self.max_fps = max_fps
            self.save_raw = bool(save_raw)
            self.save_roi = bool(save_roi)
            self.frame_count = 0
            self.saved_count = 0
            self.error_count = 0
            self.last_saved_at = 0.0
            self.session_started_at = time.time()
            self.session_dir = os.path.join(self.base_dir, f"{timestamp}_{session_label}")

            for item in ALLOWED_LABELS:
                os.makedirs(os.path.join(self.session_dir, item), exist_ok=True)

            self.metadata_path = os.path.join(self.session_dir, "metadata.csv")
            self.metadata_file = open(self.metadata_path, "w", newline="")
            self.writer = csv.DictWriter(
                self.metadata_file,
                fieldnames=[
                    "frame_id",
                    "timestamp",
                    "label",
                    "note",
                    "camera_id",
                    "image_path",
                    "raw_path",
                    "roi_path",
                    "width",
                    "height",
                    "roi_x1",
                    "roi_y1",
                    "roi_x2",
                    "roi_y2",
                ],
            )
            self.writer.writeheader()
            self.active = True

            return self.status_locked()

    def stop(self):
        with self.lock:
            self.stop_locked()
            return self.status_locked()

    def stop_locked(self):
        self.active = False

        if self.metadata_file is not None:
            self.metadata_file.flush()
            self.metadata_file.close()

        self.metadata_file = None
        self.writer = None

    def set_label(self, label, note=""):
        label = self._normalize_label(label)

        with self.lock:
            self.label = label
            self.note = str(note or "")
            return self.status_locked()

    def record(self, image, record, processor):
        now = time.time()

        with self.lock:
            if not self.active:
                return

            min_interval = 1.0 / max(0.1, self.max_fps)

            if now - self.last_saved_at < min_interval:
                return

            label = self.label
            note = self.note
            session_dir = self.session_dir
            save_raw = self.save_raw
            save_roi = self.save_roi
            writer = self.writer
            self.frame_count += 1
            frame_id = self.frame_count
            self.last_saved_at = now

        try:
            arr = processor._to_numpy(image)
            gray = processor._to_gray_uint8(arr)
            roi, roi_bounds = processor.crop_eye_roi_with_bounds(gray)
            camera_id = str(getattr(record, "camera_id", "unknown"))
            label_dir = os.path.join(session_dir, label)
            file_stem = f"{frame_id:06d}_{int(now * 1000)}"
            raw_path = ""
            roi_path = ""
            image_path = ""

            if save_raw:
                raw_path = os.path.join(label_dir, f"{file_stem}_raw.png")
                cv2.imwrite(raw_path, gray)
                image_path = raw_path

            if save_roi:
                roi_path = os.path.join(label_dir, f"{file_stem}_roi.png")
                cv2.imwrite(roi_path, roi)
                image_path = roi_path

            if not image_path:
                roi_path = os.path.join(label_dir, f"{file_stem}_roi.png")
                cv2.imwrite(roi_path, roi)
                image_path = roi_path

            x1, y1, x2, y2 = roi_bounds

            with self.lock:
                if self.writer is None or self.writer is not writer:
                    return

                self.writer.writerow({
                    "frame_id": frame_id,
                    "timestamp": now,
                    "label": label,
                    "note": note,
                    "camera_id": camera_id,
                    "image_path": os.path.relpath(image_path, session_dir),
                    "raw_path": os.path.relpath(raw_path, session_dir) if raw_path else "",
                    "roi_path": os.path.relpath(roi_path, session_dir) if roi_path else "",
                    "width": gray.shape[1],
                    "height": gray.shape[0],
                    "roi_x1": x1,
                    "roi_y1": y1,
                    "roi_x2": x2,
                    "roi_y2": y2,
                })

                self.saved_count += 1

                if self.saved_count % 20 == 0:
                    self.metadata_file.flush()
        except Exception as exc:
            with self.lock:
                self.error_count += 1

            if self.error_count <= 5:
                print("[EyeDataset] record error:", exc)

    def status(self):
        with self.lock:
            return self.status_locked()

    def status_locked(self):
        return {
            "active": self.active,
            "session_dir": self.session_dir,
            "metadata_path": self.metadata_path,
            "label": self.label,
            "note": self.note,
            "frame_count": self.frame_count,
            "saved_count": self.saved_count,
            "error_count": self.error_count,
            "max_fps": self.max_fps,
            "save_raw": self.save_raw,
            "save_roi": self.save_roi,
            "session_started_at": self.session_started_at,
        }

    def _normalize_label(self, label):
        label = _safe_name(label, "other")

        if label not in ALLOWED_LABELS:
            raise ValueError(f"label must be one of {sorted(ALLOWED_LABELS)}")

        return label

    def _normalize_fps(self, value):
        try:
            fps = float(value)
        except (TypeError, ValueError):
            fps = 12.0

        return max(1.0, min(30.0, fps))
