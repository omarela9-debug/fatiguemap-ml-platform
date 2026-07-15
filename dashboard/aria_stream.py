from collections import deque
import os
import cv2
import numpy as np
import time
import math
import threading

from eye_dataset import EyeDatasetRecorder
from eye_classifier import EyeStateClassifier, RollingPerclos

try:
    import aria.sdk as aria
except ImportError:
    aria = None


DEBUG_EYE_WINDOWS = os.getenv("FATIGUEMAP_EYE_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

RAW_DATA_KEYS = [
    "raw_imu_timestamp_ns",
    "raw_imu_timestamp_sec",
    "raw_imu_index",
    "raw_accel_x_msec2",
    "raw_accel_y_msec2",
    "raw_accel_z_msec2",
    "raw_gyro_x_radsec",
    "raw_gyro_y_radsec",
    "raw_gyro_z_radsec",
    "raw_mag_x_tesla",
    "raw_mag_y_tesla",
    "raw_mag_z_tesla",
    "raw_eye_timestamp_ns",
    "raw_eye_camera_id",
    "raw_eye_width",
    "raw_eye_height",
    "raw_eye_channels",
    "raw_eye_dtype",
    "raw_eye_mean",
    "raw_eye_std",
    "raw_eye_frame_count",
]


def _clamp(value, low, high):
    return max(low, min(high, value))


def accel_to_pitch_roll(ax, ay, az):
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    return pitch, roll


class RollingMeanWindow:
    def __init__(self, window_sec=30.0):
        self.window_sec = window_sec
        self.samples = deque()

    def update(self, timestamp, value):
        self.samples.append((timestamp, float(value)))
        self._trim(timestamp)
        return self.value()

    def value(self, timestamp=None):
        if timestamp is not None:
            self._trim(timestamp)

        if not self.samples:
            return 0.0

        return sum(value for _, value in self.samples) / len(self.samples)

    def count(self):
        return len(self.samples)

    def clear(self):
        self.samples.clear()

    def _trim(self, timestamp):
        cutoff = timestamp - self.window_sec

        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()


class EyeFrameProcessor:
    def __init__(self, debug_windows=False):
        self.debug_windows = debug_windows
        self.debug_disabled = False
        self.frame_index = 0
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, image):
        self.frame_index += 1

        try:
            arr = self._to_numpy(image)
            gray = self._to_gray_uint8(arr)
            roi = self._crop_eye_roi(gray)

            if roi is None or roi.size == 0:
                return self._result("empty_roi", frame_ok=False)

            enhanced = self.clahe.apply(roi)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            sharpened = cv2.addWeighted(enhanced, 1.55, blurred, -0.55, 0)
            denoised = cv2.medianBlur(sharpened, 5)
            edges = cv2.Canny(denoised, 35, 95)

            mask, threshold = self._segment_visible_iris_or_pupil(denoised)
            contour, visible_area, confidence = self._best_visible_region(mask, denoised)
            eyelid_rows = self._estimate_eyelid_rows(edges)
            mask_area_ratio = float(np.count_nonzero(mask) / max(1, mask.size))

            status = "ok" if contour is not None else "no_valid_pupil"
            segmentation_ok = contour is not None and confidence >= 0.35

            overlay = self._make_overlay(
                roi=roi,
                mask=mask,
                contour=contour,
                eyelid_rows=eyelid_rows,
                visible_area=visible_area,
                confidence=confidence,
            )
            self._show_debug(roi, enhanced, edges, mask, overlay)

            return {
                **self._result(status, frame_ok=True),
                "segmentation_ok": segmentation_ok,
                "visible_area": float(visible_area),
                "confidence": float(confidence),
                "threshold": int(threshold),
                "roi_mean": float(np.mean(roi)),
                "roi_std": float(np.std(roi)),
                "edge_density": float(np.count_nonzero(edges) / max(1, edges.size)),
                "mask_area_ratio": mask_area_ratio,
            }
        except Exception as exc:
            return self._result(f"error:{exc}", frame_ok=False)

    def _result(self, status, frame_ok):
        return {
            "frame_ok": frame_ok,
            "segmentation_ok": False,
            "visible_area": 0.0,
            "confidence": 0.0,
            "threshold": 0,
            "roi_mean": 0.0,
            "roi_std": 0.0,
            "edge_density": 0.0,
            "mask_area_ratio": 0.0,
            "status": status,
        }

    def _to_numpy(self, image):
        if image is None:
            raise ValueError("missing image")

        arr = image.to_numpy_array() if hasattr(image, "to_numpy_array") else image
        arr = np.asarray(arr)

        if arr.size == 0:
            raise ValueError("empty image array")

        arr = np.squeeze(arr)

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)

        if arr.ndim not in (2, 3):
            raise ValueError(f"unsupported image shape {arr.shape}")

        return arr

    def _to_gray_uint8(self, arr):
        if arr.dtype != np.uint8:
            arr_float = arr.astype(np.float32)
            finite = np.isfinite(arr_float)

            if not np.any(finite):
                raise ValueError("image contains no finite values")

            arr_float = np.where(finite, arr_float, 0)
            max_value = float(np.max(arr_float))

            if max_value <= 1.5:
                arr_float *= 255.0
            else:
                arr_float = cv2.normalize(arr_float, None, 0, 255, cv2.NORM_MINMAX)

            arr = np.clip(arr_float, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return arr.copy()

        channels = arr.shape[2]

        if channels == 1:
            return arr[:, :, 0].copy()

        if channels == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)

        if channels >= 3:
            return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)

        raise ValueError(f"unsupported channel count {channels}")

    def _crop_eye_roi(self, gray):
        roi, _ = self.crop_eye_roi_with_bounds(gray)
        return roi

    def crop_eye_roi_with_bounds(self, gray):
        h, w = gray.shape[:2]

        if h < 20 or w < 20:
            return gray.copy(), (0, 0, w, h)

        y1 = int(h * 0.25)
        y2 = int(h * 0.82)
        x1 = int(w * 0.12)
        x2 = int(w * 0.88)

        if y2 <= y1 or x2 <= x1:
            return gray.copy(), (0, 0, w, h)

        return gray[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def pupil_area_from_frame(self, frame_u8, debug_out_path=None):
        self.frame_index += 1

        if frame_u8 is None:
            return False, np.nan, {}

        if frame_u8.ndim == 3:
            gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_u8

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        h, w = gray.shape[:2]
        x0 = int(w * 0.10)
        x1 = int(w * 0.90)
        y0 = int(h * 0.10)
        y1 = int(h * 0.90)
        roi = gray[y0:y1, x0:x1]

        if roi.size == 0 or min(roi.shape[:2]) < 3:
            return False, np.nan, {"roi": (x0, y0, x1, y1)}

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi = clahe.apply(roi)
        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        min_dim = min(roi_blur.shape[:2])
        block_size = min(31, min_dim if min_dim % 2 == 1 else min_dim - 1)
        block_size = max(3, block_size)
        bw = cv2.adaptiveThreshold(
            roi_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            7,
        )
        bw = cv2.morphologyEx(
            bw,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        bw = cv2.morphologyEx(
            bw,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            bw,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        best = None
        best_score = -1.0
        roi_h, roi_w = roi.shape[:2]

        for c in contours:
            area = float(cv2.contourArea(c))

            if area < 30:
                continue

            if area > 0.35 * roi_h * roi_w:
                continue

            peri = float(cv2.arcLength(c, True))

            if peri <= 0:
                continue

            circularity = 4.0 * np.pi * area / (peri * peri + 1e-9)

            if circularity < 0.08:
                continue

            x, y, ww, hh = cv2.boundingRect(c)
            border_touch = int(
                x <= 1
                or y <= 1
                or (x + ww) >= (roi_w - 2)
                or (y + hh) >= (roi_h - 2)
            )
            border_penalty = 0.6 if border_touch else 1.0
            cx = x + ww / 2.0
            cy = y + hh / 2.0
            dx = (cx - roi_w / 2.0) / (roi_w / 2.0)
            dy = (cy - roi_h / 2.0) / (roi_h / 2.0)
            center_penalty = float(np.exp(-0.8 * (dx * dx + dy * dy)))
            score = area * circularity * border_penalty * center_penalty

            if score > best_score:
                best_score = score
                best = (c, area, circularity, (x, y, ww, hh), (cx, cy))

        show_debug = (
            self.debug_windows
            and not self.debug_disabled
            and self.frame_index % 3 == 0
        )
        meta = {
            "roi": (x0, y0, x1, y1),
            "mask_area_ratio": float(np.count_nonzero(bw) / max(1, bw.size)),
        }

        if best is None:
            if debug_out_path or show_debug:
                dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 0, 255), 2)

                if debug_out_path:
                    cv2.imwrite(debug_out_path, dbg)

                if show_debug:
                    self._show_pupil_debug(roi, bw, dbg)

            return False, np.nan, meta

        c, area, circ, (bx, by, bww, bhh), (cx, cy) = best
        meta.update({
            "area": area,
            "circularity": circ,
            "score": best_score,
            "bbox_roi": (bx, by, bww, bhh),
            "centroid_roi": (float(cx), float(cy)),
        })

        if debug_out_path or show_debug:
            dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 0, 255), 2)
            bx_full = x0 + bx
            by_full = y0 + by
            cv2.rectangle(
                dbg,
                (bx_full, by_full),
                (bx_full + bww, by_full + bhh),
                (0, 255, 0),
                2,
            )
            cv2.circle(dbg, (int(x0 + cx), int(y0 + cy)), 4, (255, 0, 0), -1)
            cv2.putText(
                dbg,
                f"area={area:.0f} circ={circ:.2f} score={best_score:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if debug_out_path:
                cv2.imwrite(debug_out_path, dbg)

            if show_debug:
                self._show_pupil_debug(roi, bw, dbg)

        return True, float(area), meta

    def _show_pupil_debug(self, enhanced_roi, mask, overlay):
        try:
            cv2.imshow("FatigueMap pupil enhanced ROI", enhanced_roi)
            cv2.imshow("FatigueMap pupil mask", mask)
            cv2.imshow("FatigueMap pupil overlay", overlay)
            cv2.waitKey(1)
        except Exception as exc:
            print("[EyeTrack] disabling pupil debug windows:", exc)
            self.debug_disabled = True

    def _segment_visible_iris_or_pupil(self, img):
        mean = float(np.mean(img))
        std = float(np.std(img))
        dark_threshold = int(_clamp(min(np.percentile(img, 18), mean - 0.35 * std), 3, 115))
        dark_mask = cv2.inRange(img, 0, dark_threshold)

        otsu_threshold, otsu_mask = cv2.threshold(
            img,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        percentile_threshold = int(_clamp(np.percentile(img, 32), 8, 135))
        percentile_mask = cv2.inRange(img, 0, percentile_threshold)

        min_dim = max(3, min(img.shape[:2]))
        block_size = min(31, min_dim if min_dim % 2 == 1 else min_dim - 1)
        block_size = max(3, block_size)

        adaptive_mask = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            4,
        )

        strict_mask = cv2.bitwise_and(otsu_mask, adaptive_mask)
        mask = cv2.bitwise_or(cv2.bitwise_and(strict_mask, percentile_mask), dark_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel, iterations=1)

        return mask, min(int(otsu_threshold), percentile_threshold, dark_threshold)

    def _best_visible_region(self, mask, img):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0, 0.0

        h, w = img.shape[:2]
        roi_area = h * w
        min_area = max(10.0, roi_area * 0.0004)
        max_area = roi_area * 0.10
        center_x = w / 2.0
        center_y = h / 2.0
        best = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)

            if cw < 3 or ch < 3:
                continue

            aspect = cw / max(1.0, float(ch))

            if aspect < 0.35 or aspect > 2.85:
                continue

            extent = area / max(1.0, float(cw * ch))

            if extent < 0.18 or extent > 0.96:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0 else 4.0 * np.pi * area / (perimeter * perimeter)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = 0.0 if hull_area <= 0 else area / hull_area

            if circularity < 0.08 or solidity < 0.35:
                continue

            contour_center_x = x + cw / 2.0
            contour_center_y = y + ch / 2.0
            distance = math.hypot(contour_center_x - center_x, contour_center_y - center_y)
            max_distance = math.hypot(center_x, center_y)
            centrality = 1.0 - _clamp(distance / max(1.0, max_distance), 0.0, 1.0)
            area_ratio = area / max(1.0, roi_area)
            size_score = 1.0 - _clamp(abs(area_ratio - 0.035) / 0.10, 0.0, 1.0)
            confidence = _clamp(
                0.28 * circularity + 0.24 * solidity + 0.22 * centrality + 0.16 * extent + 0.10 * size_score,
                0.0,
                1.0,
            )
            score = area * (0.35 + confidence)

            if score > best_score:
                best_score = score
                best = (contour, area, confidence)

        if best is None:
            return None, 0.0, 0.0

        return best

    def _estimate_eyelid_rows(self, edges):
        h, _ = edges.shape[:2]

        if h < 8:
            return None

        row_scores = np.count_nonzero(edges, axis=1)
        threshold = max(3, int(np.percentile(row_scores, 85)))
        top_half = row_scores[: h // 2]
        bottom_half = row_scores[h // 2 :]

        if top_half.size == 0 or bottom_half.size == 0:
            return None

        top = int(np.argmax(top_half))
        bottom = int(np.argmax(bottom_half) + h // 2)

        if row_scores[top] < threshold and row_scores[bottom] < threshold:
            return None

        return top, bottom

    def _make_overlay(self, roi, mask, contour, eyelid_rows, visible_area, confidence):
        overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        tint = np.zeros_like(overlay)
        tint[:, :, 1] = mask
        overlay = cv2.addWeighted(overlay, 0.85, tint, 0.30, 0)

        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 1)

            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(overlay, ellipse, (255, 180, 0), 1)

        if eyelid_rows is not None:
            top, bottom = eyelid_rows
            cv2.line(overlay, (0, top), (overlay.shape[1] - 1, top), (0, 200, 255), 1)
            cv2.line(overlay, (0, bottom), (overlay.shape[1] - 1, bottom), (0, 200, 255), 1)

        cv2.putText(
            overlay,
            f"area={visible_area:.0f} conf={confidence:.2f}",
            (6, max(14, overlay.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return overlay

    def _show_debug(self, roi, enhanced, edges, mask, overlay):
        if not self.debug_windows or self.debug_disabled:
            return

        if self.frame_index % 3 != 0:
            return

        try:
            cv2.imshow("FatigueMap raw eye ROI", roi)
            cv2.imshow("FatigueMap enhanced eye ROI", enhanced)
            cv2.imshow("FatigueMap eye edges", edges)
            cv2.imshow("FatigueMap iris/pupil mask", mask)
            cv2.imshow("FatigueMap eye overlay", overlay)
            cv2.waitKey(1)
        except Exception as exc:
            print("[EyeTrack] disabling debug windows:", exc)
            self.debug_disabled = True


class ImuObserver:
    def __init__(
        self,
        shared_state,
        eye_dataset_recorder=None,
        eye_classifier=None,
        imu_index="auto",
    ):
        self.state = shared_state
        self._printed_debug = False
        self.requested_imu_idx = self._parse_imu_index(imu_index)
        self.selected_imu_idx = None if self.requested_imu_idx == "auto" else self.requested_imu_idx
        self._imu_auto_stats = {}
        self._imu_auto_started_at = None
        self._gravity_xyz = None
        self._gravity_tau_sec = 0.18
        self.microsleep_count = 0
        self.microsleep_active = False
        self.eye_closed_start_time = None
        self.eye_processor = EyeFrameProcessor(debug_windows=DEBUG_EYE_WINDOWS)
        self.eye_dataset_recorder = eye_dataset_recorder
        self.eye_classifier = eye_classifier

    def on_streaming_client_failure(self, reason, message):
        print(f"[IMU observer] streaming failure: {reason} | {message}")

    def on_imu_received(self, motion_data, imu_idx):
        imu_idx = self._normalize_imu_index(imu_idx)
        samples = motion_data if isinstance(motion_data, (list, tuple)) else [motion_data]
        for sample in samples:
            if self._should_process_imu_sample(sample, imu_idx):
                self._update_from_motion_sample(sample, imu_idx)

    def on_image_received(self, image, record):
        if aria is not None and getattr(record, "camera_id", None) != aria.CameraId.EyeTrack:
            return

        now = time.time()

        if self.eye_dataset_recorder is not None:
            self.eye_dataset_recorder.record(image, record, self.eye_processor)

        pupil_result = self._detect_pupil(image, record)
        classifier_result = self._classify_eye(image)
        self._update_eye_state(now, pupil_result, classifier_result)

    def _detect_pupil(self, image, record=None):
        try:
            arr = self.eye_processor._to_numpy(image)
            gray = self.eye_processor._to_gray_uint8(arr)
            found, area, meta = self.eye_processor.pupil_area_from_frame(gray)
            camera_id = getattr(record, "camera_id", None)
            timestamp_ns = (
                getattr(record, "capture_timestamp_ns", None)
                or getattr(record, "timestamp_ns", None)
                or getattr(record, "tracking_timestamp_ns", None)
            )
            channels = gray.shape[2] if gray.ndim == 3 else 1

            return {
                "frame_ok": True,
                "found": bool(found and np.isfinite(area)),
                "area": float(area) if np.isfinite(area) else np.nan,
                "meta": meta,
                "gray": gray,
                "raw_eye_timestamp_ns": timestamp_ns,
                "raw_eye_camera_id": None if camera_id is None else str(camera_id),
                "raw_eye_width": int(gray.shape[1]),
                "raw_eye_height": int(gray.shape[0]),
                "raw_eye_channels": int(channels),
                "raw_eye_dtype": str(gray.dtype),
                "gray_mean": float(np.mean(gray)),
                "gray_std": float(np.std(gray)),
            }
        except Exception as exc:
            return {
                "frame_ok": False,
                "found": False,
                "area": np.nan,
                "meta": {},
                "gray": None,
                "raw_eye_timestamp_ns": None,
                "raw_eye_camera_id": None,
                "raw_eye_width": None,
                "raw_eye_height": None,
                "raw_eye_channels": None,
                "raw_eye_dtype": None,
                "error": str(exc),
                "gray_mean": 0.0,
                "gray_std": 0.0,
            }

    def _classify_eye(self, image):
        if self.eye_classifier is None or not self.eye_classifier.enabled:
            return None

        try:
            arr = self.eye_processor._to_numpy(image)
            gray = self.eye_processor._to_gray_uint8(arr)
            roi, _ = self.eye_processor.crop_eye_roi_with_bounds(gray)
            return self.eye_classifier.predict(roi)
        except Exception as exc:
            return {
                "available": False,
                "state": "error",
                "confidence": 0.0,
                "probs": {},
                "error": str(exc),
            }

    def _update_eye_state(self, now, pupil_result, classifier_result=None):
        with self.state["lock"]:
            self.state["last_eye_frame_time"] = now

            if self.state["first_eye_frame_time"] is None:
                self.state["first_eye_frame_time"] = now

            if not pupil_result.get("frame_ok", False):
                self.state["eye_frame_errors"] += 1
                self.state["eye_status"] = pupil_result.get("error", "invalid_frame")
                return

            self.state["total_frames"] += 1
            self.state["eye_roi_mean"] = pupil_result.get("gray_mean", 0.0)
            self.state["eye_roi_std"] = pupil_result.get("gray_std", 0.0)
            self.state["raw_eye_timestamp_ns"] = pupil_result.get("raw_eye_timestamp_ns")
            self.state["raw_eye_camera_id"] = pupil_result.get("raw_eye_camera_id")
            self.state["raw_eye_width"] = pupil_result.get("raw_eye_width")
            self.state["raw_eye_height"] = pupil_result.get("raw_eye_height")
            self.state["raw_eye_channels"] = pupil_result.get("raw_eye_channels")
            self.state["raw_eye_dtype"] = pupil_result.get("raw_eye_dtype")
            self.state["raw_eye_mean"] = pupil_result.get("gray_mean", 0.0)
            self.state["raw_eye_std"] = pupil_result.get("gray_std", 0.0)
            self.state["raw_eye_frame_count"] = self.state["total_frames"]

            gray = pupil_result.get("gray")

            if isinstance(gray, np.ndarray):
                self.state["latest_eye_gray"] = gray.copy()

            pupil_found = bool(pupil_result.get("found", False))
            pupil_area = pupil_result.get("area", np.nan)
            pupil_meta = pupil_result.get("meta", {})
            pupil_score = float(pupil_meta.get("score", 0.0))
            pupil_circularity = float(pupil_meta.get("circularity", 0.0))
            pupil_signal_ok = (
                pupil_found
                and np.isfinite(pupil_area)
                and pupil_area > 0
                and pupil_score >= 10.0
                and pupil_circularity >= 0.08
            )
            self.state["pupil_found"] = pupil_found
            self.state["pupil_area_px"] = float(pupil_area) if np.isfinite(pupil_area) else 0.0
            self.state["pupil_score"] = pupil_score
            self.state["pupil_circularity"] = pupil_circularity
            self.state["eye_segmentation_ok"] = pupil_signal_ok
            self.state["eye_confidence"] = pupil_circularity
            self.state["eye_mask_area_ratio"] = pupil_meta.get("mask_area_ratio", 0.0)
            self.state["visible_iris_area"] = self.state["pupil_area_px"]

            classifier_available = bool(classifier_result and classifier_result.get("available"))

            if classifier_result is not None:
                state = classifier_result.get("state", "unknown")
                probs = classifier_result.get("probs", {})
                self.state["eye_classifier_available"] = classifier_available
                self.state["eye_classifier_state"] = state
                self.state["eye_classifier_confidence"] = classifier_result.get("confidence", 0.0)
                self.state["eye_open_prob"] = probs.get("open", 0.0)
                self.state["eye_closed_prob"] = probs.get("closed", 0.0)
                self.state["eye_blink_prob"] = probs.get("blink", 0.0)
                self.state["eye_classifier_error"] = classifier_result.get("error")
            area_ref = self.state.get("area_ref", 0.0)
            closure = self.state.get("closure", 0.0)
            closed80 = False
            raw_state = "unknown"
            blink_event_state = "unknown"
            source = "unknown"
            fallback_state = self._classifier_fallback_state(classifier_result)
            classifier_open_strong = self._classifier_open_confident(classifier_result)

            if pupil_signal_ok:
                self.state["missing_pupil_frames"] = 0
                self.state["weak_pupil_frames"] = 0
                self.state["pupil_area_history"].append(float(pupil_area))
                area_samples = list(self.state["pupil_area_history"])
                area_ref = float(np.percentile(area_samples, 90))
                self.state["area_ref"] = area_ref

                if area_ref > 1.0 and len(area_samples) >= 5:
                    closure = float(np.clip(1.0 - (float(pupil_area) / area_ref), 0.0, 1.0))
                    closed80 = closure >= 0.80
                    source = "pupil"

                    if fallback_state in ("closed", "blink") and closure >= 0.45:
                        closure = max(closure, 0.85)
                        closed80 = True
                        source = "pupil_classifier"

                    raw_state = "closed" if closed80 else ("partial" if closure >= 0.45 else "open")

                    if closure >= 0.70:
                        blink_event_state = "closed"
                    elif closure >= 0.45:
                        blink_event_state = "ambiguous"
                    else:
                        blink_event_state = "open"
                    self.state["eye_calibrated"] = True
                else:
                    raw_state = "open"
                    blink_event_state = "open"
                    source = "pupil_calibrating"
                    self.state["eye_calibrated"] = False
            else:
                if pupil_found:
                    self.state["weak_pupil_frames"] += 1
                    self.state["missing_pupil_frames"] = 0
                else:
                    self.state["missing_pupil_frames"] += 1
                    self.state["weak_pupil_frames"] = 0

                raw_state = fallback_state
                source = "classifier_fallback" if raw_state != "unknown" else "ignored_missing_pupil"

                if raw_state == "closed":
                    closure = 1.0
                    closed80 = True
                    blink_event_state = "closed"
                elif raw_state == "blink":
                    closure = 0.85
                    closed80 = True
                    blink_event_state = "closed"
                elif raw_state == "open":
                    closure = 0.0
                    closed80 = False
                    blink_event_state = "open"
                elif (
                    self.state.get("eye_calibrated")
                    and self.state.get("area_ref", 0.0) > 1.0
                    and not classifier_open_strong
                    and (self.state["missing_pupil_frames"] + self.state["weak_pupil_frames"]) >= 2
                ):
                    closure = 1.0
                    closed80 = True
                    raw_state = "closed"
                    blink_event_state = "closed"
                    source = "pupil_lost_after_calibration"
                else:
                    closure = 0.0
                    closed80 = False
                    blink_event_state = "unknown"

            is_valid_eye_signal = 1.0 if raw_state != "unknown" else 0.0
            valid_eye_fraction = self.state["valid_eye_window"].update(now, is_valid_eye_signal)
            episode_closed = self._update_blink_state(blink_event_state, now)

            if episode_closed and self.state.get("eye_calibrated"):
                closure = max(closure, 0.85)
                closed80 = True
                raw_state = "closed"

            perclos80_true = self.state["perclos80_window"].update(
                now,
                1.0 if closed80 else 0.0,
            ) * 100.0
            perclos_continuous_pct = self.state["perclos_continuous_window"].update(
                now,
                closure,
            ) * 100.0
            closure_mean_pct = self.state["closure_mean_window"].update(now, closure) * 100.0
            final_state = self._update_smoothed_state(raw_state)

            self.state["closure"] = closure
            self.state["closure_pct"] = closure * 100.0
            self.state["closed80"] = closed80
            self.state["perclos80_true"] = perclos80_true
            self.state["perclos_continuous_pct"] = perclos_continuous_pct
            self.state["perclos_source"] = source
            self.state["valid_eye_fraction"] = valid_eye_fraction
            self.state["closure_mean_pct"] = closure_mean_pct
            self.state["raw_eye_state"] = raw_state
            self.state["final_smoothed_eye_state"] = final_state
            self.state["baseline_iris_area"] = area_ref
            self.state["baseline_pupil_area"] = area_ref
            self.state["visible_iris_ratio"] = float(np.clip(1.0 - closure, 0.0, 1.0))
            self.state["true_perclos_raw"] = closure * 100.0
            self.state["true_perclos"] = closure * 100.0
            self.state["perclos"] = perclos_continuous_pct / 100.0
            self.state["eye_closed"] = final_state in ("closed", "blink")
            self.state["eye_status"] = f"{source}_{final_state}"
            self._update_microsleep_state(now, self.state["eye_closed"])

            if self.state["eye_closed"]:
                self.state["closed_frames"] += 1
            else:
                self.state["open_frames"] += 1

            self.state["was_eye_closed"] = self.state["eye_closed"]
            total = max(1, self.state["total_frames"])
            self.state["frame_closure_ratio"] = self.state["closed_frames"] / total

    def _update_microsleep_state(self, current_time, eye_closed):
        if eye_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
                self.state["eye_closed_start_time"] = current_time
                self.state["microsleep_closed_duration_sec"] = 0.0
                print("[Microsleep] START")

            closure_duration = current_time - self.eye_closed_start_time
            self.state["microsleep_closed_duration_sec"] = closure_duration

            if (
                closure_duration >= self.state["microsleep_threshold_sec"]
                and not self.microsleep_active
            ):
                self.microsleep_count += 1
                self.microsleep_active = True
                self.state["microsleep_count"] = self.microsleep_count
                self.state["microsleep_active"] = True
                print(f"[Microsleep] DETECTED ({closure_duration:.2f}s closed)")

            return

        if self.eye_closed_start_time is not None or self.microsleep_active:
            print("[Microsleep] END")

        self.eye_closed_start_time = None
        self.microsleep_active = False
        self.state["eye_closed_start_time"] = None
        self.state["microsleep_active"] = False
        self.state["microsleep_closed_duration_sec"] = 0.0

    def reset_microsleep_state(self):
        self.microsleep_count = 0
        self.microsleep_active = False
        self.eye_closed_start_time = None

    def _classifier_fallback_state(self, classifier_result):
        if not classifier_result or not classifier_result.get("available"):
            return "unknown"

        state = classifier_result.get("state", "unknown")
        confidence = classifier_result.get("confidence", 0.0)
        probs = classifier_result.get("probs", {})
        open_prob = probs.get("open", 0.0)
        closed_prob = probs.get("closed", 0.0)
        blink_prob = probs.get("blink", 0.0)

        if state == "closed" and (confidence >= 0.55 or closed_prob >= 0.60):
            return "closed"

        if state == "blink" and (confidence >= 0.55 or blink_prob >= 0.60):
            return "blink"

        if state == "open" and (confidence >= 0.50 or open_prob >= 0.55):
            return "open"

        return "unknown"

    def _classifier_open_confident(self, classifier_result):
        if not classifier_result or not classifier_result.get("available"):
            return False

        probs = classifier_result.get("probs", {})
        return (
            classifier_result.get("state") == "open"
            and (
                classifier_result.get("confidence", 0.0) >= 0.60
                or probs.get("open", 0.0) >= 0.65
            )
        )

    def _update_smoothed_state(self, raw_state):
        if raw_state == "unknown":
            return self.state.get("final_smoothed_eye_state", "unknown")

        if raw_state == self.state.get("candidate_eye_state"):
            self.state["candidate_eye_count"] += 1
        else:
            self.state["candidate_eye_state"] = raw_state
            self.state["candidate_eye_count"] = 1

        required = 1 if raw_state == "blink" else 2

        if (
            self.state.get("final_smoothed_eye_state") == "unknown"
            or self.state["candidate_eye_count"] >= required
        ):
            self.state["final_smoothed_eye_state"] = raw_state

        return self.state["final_smoothed_eye_state"]

    def _update_blink_state(self, eye_state, now):
        phase = self.state.get("blink_phase", "open")
        started_at = self.state.get("blink_started_at")

        if eye_state in ("closed", "blink"):
            if phase == "open":
                self.state["blink_phase"] = "closed_seen"
                self.state["blink_started_at"] = now
                self.state["blink_closed_frame_count"] = 1
                self.state["blink_open_frame_count"] = 0
                return True

            if phase == "closed_seen":
                self.state["blink_closed_frame_count"] += 1
                self.state["blink_open_frame_count"] = 0

                if (
                    started_at is not None
                    and now - started_at > self.state["blink_max_closed_sec"]
                ):
                    self.state["blink_phase"] = "long_closed"

                return True

            if phase == "long_closed":
                self.state["blink_open_frame_count"] = 0
                return True

        if eye_state == "ambiguous":
            if phase in ("closed_seen", "long_closed"):
                self.state["blink_open_frame_count"] = 0

                if (
                    phase == "closed_seen"
                    and started_at is not None
                    and now - started_at > self.state["blink_max_closed_sec"]
                ):
                    self.state["blink_phase"] = "long_closed"

                return True

            return False

        if phase in ("closed_seen", "long_closed"):
            self.state["blink_open_frame_count"] += 1
            open_confirmed = (
                self.state["blink_open_frame_count"]
                >= self.state["blink_open_confirm_frames"]
            )

            if not open_confirmed:
                return True

            closed_duration = 0.0 if started_at is None else now - started_at
            closed_frames = self.state.get("blink_closed_frame_count", 0)
            is_blink_duration = (
                phase == "closed_seen"
                and closed_frames >= self.state["blink_min_closed_frames"]
                and closed_duration <= self.state["blink_max_closed_sec"]
            )

            if (
                is_blink_duration
                and now - self.state.get("last_blink_time", 0.0) >= self.state["blink_cooldown_sec"]
            ):
                self.state["blink_count"] += 1
                self.state["last_blink_time"] = now

            self.state["blink_phase"] = "open"
            self.state["blink_started_at"] = None
            self.state["blink_closed_frame_count"] = 0
            self.state["blink_open_frame_count"] = 0
            return False

        return False

    def _parse_imu_index(self, imu_index):
        if imu_index is None:
            return "auto"

        value = str(imu_index).strip().lower()

        if value in ("", "auto", "first"):
            return "auto"

        try:
            return int(value)
        except ValueError:
            print(f"[IMU observer] invalid ARIA_IMU_INDEX={imu_index!r}; using auto")
            return "auto"

    def _normalize_imu_index(self, imu_idx):
        try:
            return int(imu_idx)
        except Exception:
            return imu_idx

    def _extract_accel_xyz(self, sample):
        accel = (
            getattr(sample, "accel_msec2", None)
            or getattr(sample, "accel", None)
            or getattr(sample, "accelerometer", None)
        )
        return self._to_xyz(accel)

    def _should_process_imu_sample(self, sample, imu_idx):
        if self.requested_imu_idx != "auto":
            should_process = imu_idx == self.selected_imu_idx

            with self.state["lock"]:
                self.state["imu_last_idx"] = imu_idx

                if should_process:
                    self.state["imu_selected_idx"] = self.selected_imu_idx
                else:
                    self.state["imu_skipped_samples"] += 1

            return should_process

        if self.selected_imu_idx is None:
            accel_xyz = self._extract_accel_xyz(sample)
            self._update_imu_auto_selection(imu_idx, accel_xyz)

        should_process = self.selected_imu_idx is not None and imu_idx == self.selected_imu_idx

        with self.state["lock"]:
            self.state["imu_last_idx"] = imu_idx
            self.state["imu_selected_idx"] = (
                "auto_pending" if self.selected_imu_idx is None else self.selected_imu_idx
            )

            if not should_process:
                self.state["imu_skipped_samples"] += 1

        return should_process

    def _update_imu_auto_selection(self, imu_idx, accel_xyz):
        if accel_xyz is None:
            return

        now = time.time()

        if self._imu_auto_started_at is None:
            self._imu_auto_started_at = now

        ax, ay, az = accel_xyz
        mag = math.sqrt(ax * ax + ay * ay + az * az)

        if mag <= 1e-6:
            return

        stats = self._imu_auto_stats.setdefault(
            imu_idx,
            {
                "count": 0,
                "x_fraction_sum": 0.0,
            },
        )
        stats["count"] += 1
        stats["x_fraction_sum"] += abs(ax) / mag

        total_samples = sum(item["count"] for item in self._imu_auto_stats.values())
        elapsed = now - self._imu_auto_started_at

        if len(self._imu_auto_stats) >= 2 and (total_samples >= 40 or elapsed >= 0.35):
            self.selected_imu_idx = min(
                self._imu_auto_stats,
                key=lambda idx: (
                    self._imu_auto_stats[idx]["x_fraction_sum"]
                    / max(1, self._imu_auto_stats[idx]["count"])
                ),
            )
            print(f"[IMU observer] auto-selected IMU index: {self.selected_imu_idx}")
            return

        if elapsed >= 0.5 and self._imu_auto_stats:
            self.selected_imu_idx = max(
                self._imu_auto_stats,
                key=lambda idx: self._imu_auto_stats[idx]["count"],
            )
            print(f"[IMU observer] auto-selected only observed IMU index: {self.selected_imu_idx}")

    def _smooth_gravity(self, accel_xyz, dt):
        if self._gravity_xyz is None:
            self._gravity_xyz = accel_xyz
            return self._gravity_xyz

        alpha = dt / (self._gravity_tau_sec + dt) if dt > 0 else 0.04
        alpha = _clamp(alpha, 0.02, 0.35)
        px, py, pz = self._gravity_xyz
        ax, ay, az = accel_xyz
        self._gravity_xyz = (
            px + alpha * (ax - px),
            py + alpha * (ay - py),
            pz + alpha * (az - pz),
        )
        return self._gravity_xyz

    def reset_imu_filter(self):
        self._gravity_xyz = None

    def _update_from_motion_sample(self, sample, imu_idx):
        now = time.time()

        if not self._printed_debug:
            self._printed_debug = True
            print("[IMU observer] first sample type:", type(sample))
            print("[IMU observer] sample attrs:", [a for a in dir(sample) if not a.startswith("_")])
            print(f"[IMU observer] requested IMU index: {self.requested_imu_idx}")

        ts = (
            getattr(sample, "capture_timestamp_ns", None)
            or getattr(sample, "timestamp_ns", None)
            or getattr(sample, "tracking_timestamp_ns", None)
        )

        ts_sec = ts / 1e9 if ts is not None else now

        accel = (
            getattr(sample, "accel_msec2", None)
            or getattr(sample, "accel", None)
            or getattr(sample, "accelerometer", None)
        )

        gyro = (
            getattr(sample, "gyro_radsec", None)
            or getattr(sample, "gyro", None)
            or getattr(sample, "gyroscope", None)
        )

        mag = (
            getattr(sample, "mag_tesla", None)
            or getattr(sample, "mag", None)
            or getattr(sample, "magnetometer", None)
        )

        accel_xyz = self._to_xyz(accel)
        gyro_xyz = self._to_xyz(gyro)
        mag_xyz = self._to_xyz(mag)

        with self.state["lock"]:
            prev_t = self.state["last_ts"]
            dt = 0.0 if prev_t is None else max(0.0, min(ts_sec - prev_t, 0.1))
            self.state["last_ts"] = ts_sec
            self.state["imu_last_idx"] = imu_idx
            self.state["imu_selected_idx"] = self.selected_imu_idx
            self.state["imu_processed_samples"] += 1
            self.state["raw_imu_timestamp_ns"] = ts
            self.state["raw_imu_timestamp_sec"] = ts_sec
            self.state["raw_imu_index"] = imu_idx

            if accel_xyz is not None:
                ax, ay, az = accel_xyz
                self.state["raw_accel_x_msec2"] = ax
                self.state["raw_accel_y_msec2"] = ay
                self.state["raw_accel_z_msec2"] = az
            else:
                self.state["raw_accel_x_msec2"] = None
                self.state["raw_accel_y_msec2"] = None
                self.state["raw_accel_z_msec2"] = None

            if gyro_xyz is not None:
                gx, gy, gz = gyro_xyz
                self.state["raw_gyro_x_radsec"] = gx
                self.state["raw_gyro_y_radsec"] = gy
                self.state["raw_gyro_z_radsec"] = gz
            else:
                self.state["raw_gyro_x_radsec"] = None
                self.state["raw_gyro_y_radsec"] = None
                self.state["raw_gyro_z_radsec"] = None

            if mag_xyz is not None:
                mx, my, mz = mag_xyz
                self.state["raw_mag_x_tesla"] = mx
                self.state["raw_mag_y_tesla"] = my
                self.state["raw_mag_z_tesla"] = mz
            else:
                self.state["raw_mag_x_tesla"] = None
                self.state["raw_mag_y_tesla"] = None
                self.state["raw_mag_z_tesla"] = None

            if gyro_xyz is not None and dt > 0:
                _, _, gz = gyro_xyz
                self.state["yaw_deg"] += math.degrees(gz * dt)

            if accel_xyz is not None:
                ax, ay, az = accel_xyz
                self.state["accel_mag"] = math.sqrt(ax * ax + ay * ay + az * az)
                raw_pitch, raw_roll = accel_to_pitch_roll(ax, ay, az)
                self.state["imu_pitch_raw_deg"] = raw_pitch
                self.state["imu_roll_raw_deg"] = raw_roll
                gravity_xyz = self._smooth_gravity(accel_xyz, dt)
                pitch_now, roll_now = accel_to_pitch_roll(*gravity_xyz)
                self.state["imu_pitch_gravity_deg"] = pitch_now
                self.state["imu_roll_gravity_deg"] = roll_now
                self.state["pitch_deg"] = pitch_now - self.state["imu_pitch_zero_deg"]
                self.state["roll_deg"] = roll_now - self.state["imu_roll_zero_deg"]

                pitch_now = self.state["pitch_deg"]
                last_pitch = self.state["last_pitch_for_nod"]

                if now - self.state["last_pitch_sample_time"] >= 0.15:
                    if abs(pitch_now - last_pitch) > 3.0 and now - self.state["last_nod_time"] > 0.8:
                        self.state["nod_count"] += 1
                        self.state["last_nod_time"] = now

                    self.state["last_pitch_for_nod"] = pitch_now
                    self.state["last_pitch_sample_time"] = now

    def _to_xyz(self, value):
        if value is None:
            return None

        try:
            if len(value) >= 3:
                return float(value[0]), float(value[1]), float(value[2])
        except Exception:
            pass

        x = getattr(value, "x", None)
        y = getattr(value, "y", None)
        z = getattr(value, "z", None)

        if x is not None and y is not None and z is not None:
            return float(x), float(y), float(z)

        return None

class RealAriaStream:
    def __init__(
        self,
        profile_name=None,
        imu_index="auto",
        imu_pitch_zero_deg=0.0,
        imu_roll_zero_deg=0.0,
    ):
        if aria is None:
            raise RuntimeError("aria.sdk is required for RealAriaStream; set USE_MOCK_DATA=True to run without glasses")

        self.stream_profile_name = profile_name or "sdk_default"
        self.stream_profile_error = None
        self.shared_state = {
            "lock": threading.Lock(),
            "stream_start_time": time.time(),
            "stream_profile_name": self.stream_profile_name,
            "stream_profile_error": self.stream_profile_error,
            "last_ts": None,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "accel_mag": 9.81,
            "imu_selected_idx": str(imu_index),
            "imu_last_idx": None,
            "imu_processed_samples": 0,
            "imu_skipped_samples": 0,
            "imu_pitch_raw_deg": 0.0,
            "imu_roll_raw_deg": 0.0,
            "imu_pitch_gravity_deg": 0.0,
            "imu_roll_gravity_deg": 0.0,
            "imu_pitch_zero_deg": float(imu_pitch_zero_deg),
            "imu_roll_zero_deg": float(imu_roll_zero_deg),
            "imu_neutral_calibrated": bool(imu_pitch_zero_deg or imu_roll_zero_deg),
            **{key: None for key in RAW_DATA_KEYS},
            "raw_eye_frame_count": 0,
            "latest_eye_gray": None,
            "nod_count": 0,
            "last_nod_time": 0.0,
            "last_pitch_for_nod": 0.0,
            "last_pitch_sample_time": 0.0,
            "blink_count": 0,
            "closed_frames": 0,
            "open_frames": 0,
            "total_frames": 0,
            "microsleep_count": 0,
            "microsleep_active": False,
            "microsleep_threshold_sec": 0.5,
            "eye_closed_start_time": None,
            "microsleep_closed_duration_sec": 0.0,
            "perclos": 0.0,
            "perclos80_true": 0.0,
            "perclos_continuous_pct": 0.0,
            "perclos_source": "waiting_for_eye_frame",
            "true_perclos": 0.0,
            "true_perclos_raw": 0.0,
            "closure": 0.0,
            "closure_pct": 0.0,
            "closed80": False,
            "pupil_found": False,
            "pupil_area_px": 0.0,
            "pupil_score": 0.0,
            "pupil_circularity": 0.0,
            "area_ref": 0.0,
            "pupil_area_history": deque(maxlen=1800),
            "perclos80_window": RollingMeanWindow(window_sec=30.0),
            "perclos_continuous_window": RollingMeanWindow(window_sec=30.0),
            "closure_mean_window": RollingMeanWindow(window_sec=10.0),
            "valid_eye_window": RollingMeanWindow(window_sec=30.0),
            "valid_eye_fraction": 0.0,
            "closure_mean_pct": 0.0,
            "raw_eye_state": "unknown",
            "final_smoothed_eye_state": "unknown",
            "candidate_eye_state": "unknown",
            "candidate_eye_count": 0,
            "blink_phase": "open",
            "blink_started_at": None,
            "blink_closed_frame_count": 0,
            "blink_open_frame_count": 0,
            "blink_max_closed_sec": 1.25,
            "blink_min_closed_frames": 1,
            "blink_open_confirm_frames": 4,
            "missing_pupil_frames": 0,
            "weak_pupil_frames": 0,
            "visible_iris_area": 0.0,
            "baseline_iris_area": 0.0,
            "baseline_pupil_area": 0.0,
            "visible_iris_ratio": 1.0,
            "baseline_samples": deque(maxlen=90),
            "eye_closed": False,
            "eye_calibrated": False,
            "eye_segmentation_ok": False,
            "eye_status": "waiting_for_eye_frame",
            "eye_confidence": 0.0,
            "eye_roi_mean": 0.0,
            "eye_roi_std": 0.0,
            "eye_edge_density": 0.0,
            "eye_mask_area_ratio": 0.0,
            "eye_frame_errors": 0,
            "eye_debug_enabled": DEBUG_EYE_WINDOWS,
            "eye_stream_status": "waiting_for_eye_track",
            "first_eye_frame_time": None,
            "last_eye_frame_time": None,
            "last_blink_time": 0.0,
            "closed_started_at": None,
            "blink_counted_for_closure": False,
            "blink_min_closed_sec": 0.035,
            "blink_cooldown_sec": 0.15,
            "frame_closure_ratio": 0.0,
            "was_eye_closed": False,
            "eye_classifier_available": False,
            "eye_classifier_state": "unavailable",
            "eye_classifier_confidence": 0.0,
            "eye_classifier_error": None,
            "eye_closed_prob": 0.0,
            "eye_open_prob": 0.0,
            "eye_blink_prob": 0.0,
            "classifier_perclos_window": RollingPerclos(window_sec=30.0),
        }
        self.eye_dataset_recorder = EyeDatasetRecorder()
        self.eye_classifier = EyeStateClassifier()

        self.device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        self.device_client.set_client_config(client_config)

        print("[RealAriaStream] Connecting to glasses over USB...")
        self.device = self.device_client.connect()
        print(f"[RealAriaStream] Connected to serial: {self.device.info.serial}")

        self.streaming_manager = self.device.streaming_manager

        streaming_config = aria.StreamingConfig()
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
        if profile_name:
            try:
                streaming_config.profile_name = profile_name
                print(f"[RealAriaStream] Streaming profile requested: {profile_name}")
            except Exception as exc:
                self.stream_profile_name = "sdk_default"
                self.stream_profile_error = str(exc)
                self.shared_state["stream_profile_name"] = self.stream_profile_name
                self.shared_state["stream_profile_error"] = self.stream_profile_error
                print(f"[RealAriaStream] Could not set streaming profile {profile_name}: {exc}")
        else:
            print("[RealAriaStream] Streaming profile: SDK default")
        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config

        self.streaming_client = self.streaming_manager.streaming_client

        sub_config = self.streaming_client.subscription_config
        sub_config.subscriber_data_type = aria.StreamingDataType.Imu | aria.StreamingDataType.EyeTrack
        
        queue_size = aria.MessageQueueSizeMap()
        queue_size[aria.StreamingDataType.Imu] = 1
        queue_size[aria.StreamingDataType.EyeTrack] = 1
        sub_config.message_queue_size = queue_size

        sub_config.subscriber_name = "fatigue-dashboard-imu"
        self.streaming_client.subscription_config = sub_config

        self.observer = ImuObserver(
            self.shared_state,
            self.eye_dataset_recorder,
            self.eye_classifier,
            imu_index=imu_index,
        )
        self.streaming_client.set_streaming_client_observer(self.observer)

        print("[RealAriaStream] Starting stream...")
        self.streaming_manager.start_streaming()
        self.streaming_client.subscribe()
        print("[RealAriaStream] IMU subscription active.")

    def get_frame(self):
        now = time.time()

        with self.shared_state["lock"]:
            yaw = round(self.shared_state["yaw_deg"], 2)
            pitch = round(self.shared_state["pitch_deg"], 2)
            roll = round(self.shared_state["roll_deg"], 2)
            accel_mag = self.shared_state["accel_mag"]
            imu_selected_idx = self.shared_state["imu_selected_idx"]
            imu_last_idx = self.shared_state["imu_last_idx"]
            imu_processed_samples = self.shared_state["imu_processed_samples"]
            imu_skipped_samples = self.shared_state["imu_skipped_samples"]
            imu_pitch_raw_deg = self.shared_state["imu_pitch_raw_deg"]
            imu_roll_raw_deg = self.shared_state["imu_roll_raw_deg"]
            imu_pitch_gravity_deg = self.shared_state["imu_pitch_gravity_deg"]
            imu_roll_gravity_deg = self.shared_state["imu_roll_gravity_deg"]
            imu_pitch_zero_deg = self.shared_state["imu_pitch_zero_deg"]
            imu_roll_zero_deg = self.shared_state["imu_roll_zero_deg"]
            imu_neutral_calibrated = self.shared_state["imu_neutral_calibrated"]
            nod_count = self.shared_state["nod_count"]
            blink_count = self.shared_state["blink_count"]
            closed_frames = self.shared_state["closed_frames"]
            open_frames = self.shared_state["open_frames"]
            total_frames = self.shared_state["total_frames"]
            microsleep_count = self.shared_state["microsleep_count"]
            microsleep_active = self.shared_state["microsleep_active"]
            microsleep_closed_duration_sec = self.shared_state["microsleep_closed_duration_sec"]
            perclos = self.shared_state["perclos"]
            eye_closed = self.shared_state["eye_closed"]
            frame_closure_ratio = self.shared_state["frame_closure_ratio"]
            true_perclos = self.shared_state["true_perclos"]
            true_perclos_raw = self.shared_state["true_perclos_raw"]
            visible_iris_area = self.shared_state["visible_iris_area"]
            baseline_iris_area = self.shared_state["baseline_iris_area"]
            visible_iris_ratio = self.shared_state["visible_iris_ratio"]
            eye_calibrated = self.shared_state["eye_calibrated"]
            eye_segmentation_ok = self.shared_state["eye_segmentation_ok"]
            eye_status = self.shared_state["eye_status"]
            eye_confidence = self.shared_state["eye_confidence"]
            eye_roi_mean = self.shared_state["eye_roi_mean"]
            eye_roi_std = self.shared_state["eye_roi_std"]
            eye_edge_density = self.shared_state["eye_edge_density"]
            eye_mask_area_ratio = self.shared_state["eye_mask_area_ratio"]
            eye_frame_errors = self.shared_state["eye_frame_errors"]
            eye_debug_enabled = self.shared_state["eye_debug_enabled"]
            stream_profile_name = self.shared_state["stream_profile_name"]
            stream_profile_error = self.shared_state["stream_profile_error"]
            stream_start_time = self.shared_state["stream_start_time"]
            first_eye_frame_time = self.shared_state["first_eye_frame_time"]
            last_eye_frame_time = self.shared_state["last_eye_frame_time"]
            eye_classifier_available = self.shared_state["eye_classifier_available"]
            eye_classifier_state = self.shared_state["eye_classifier_state"]
            eye_classifier_confidence = self.shared_state["eye_classifier_confidence"]
            eye_classifier_error = self.shared_state["eye_classifier_error"]
            eye_closed_prob = self.shared_state["eye_closed_prob"]
            eye_open_prob = self.shared_state["eye_open_prob"]
            eye_blink_prob = self.shared_state["eye_blink_prob"]
            pupil_found = self.shared_state["pupil_found"]
            pupil_area_px = self.shared_state["pupil_area_px"]
            pupil_score = self.shared_state["pupil_score"]
            pupil_circularity = self.shared_state["pupil_circularity"]
            area_ref = self.shared_state["area_ref"]
            closure_pct = self.shared_state["closure_pct"]
            closed80 = self.shared_state["closed80"]
            perclos80_true = self.shared_state["perclos80_true"]
            perclos_continuous_pct = self.shared_state["perclos_continuous_pct"]
            perclos_source = self.shared_state["perclos_source"]
            raw_eye_state = self.shared_state["raw_eye_state"]
            final_smoothed_eye_state = self.shared_state["final_smoothed_eye_state"]
            valid_eye_fraction = self.shared_state["valid_eye_fraction"]
            closure_mean_pct = self.shared_state["closure_mean_pct"]
            fatigue_score = min(1.0, max(0.0, abs(pitch) / 45.0 + abs(roll) / 45.0))
            raw_fields = {key: self.shared_state.get(key) for key in RAW_DATA_KEYS}

        elapsed_eye_sec = 0.0
        eye_data_age_ms = None
        stream_uptime_sec = max(0.0, now - stream_start_time)
        eye_start_delay_sec = None
        eye_wait_sec = 0.0

        if first_eye_frame_time is not None and last_eye_frame_time is not None:
            elapsed_eye_sec = max(0.0, last_eye_frame_time - first_eye_frame_time)
            eye_start_delay_sec = max(0.0, first_eye_frame_time - stream_start_time)
        else:
            eye_wait_sec = stream_uptime_sec

        blink_rate_bpm = 0.0
        eye_fps_est = 0.0

        if elapsed_eye_sec >= 1.0:
            blink_rate_bpm = (blink_count / elapsed_eye_sec) * 60.0
            eye_fps_est = total_frames / elapsed_eye_sec

        if last_eye_frame_time is not None:
            eye_data_age_ms = (now - last_eye_frame_time) * 1000.0

        if last_eye_frame_time is None:
            eye_stream_status = "waiting_for_eye_track"
        elif eye_data_age_ms is not None and eye_data_age_ms > 1500:
            eye_stream_status = "eye_track_stale"
        elif not eye_calibrated:
            eye_stream_status = "calibrating"
        else:
            eye_stream_status = "live"

        return {
            "timestamp": now,
            **raw_fields,
            "stream_uptime_sec": round(stream_uptime_sec, 2),
            "stream_profile_name": stream_profile_name,
            "stream_profile_error": stream_profile_error,
            "eye_stream_status": eye_stream_status,
            "eye_wait_sec": round(eye_wait_sec, 2),
            "eye_start_delay_sec": None if eye_start_delay_sec is None else round(eye_start_delay_sec, 2),
            "eye_fps_est": round(eye_fps_est, 2),
            "blink_rate_bpm": round(blink_rate_bpm, 2),
            "blink_count": blink_count,
            "perclos": round(perclos * 100, 2),
            "perclos80_true": round(perclos80_true, 2),
            "perclos_continuous_pct": round(perclos_continuous_pct, 2),
            "perclos_source": perclos_source,
            "true_perclos": round(true_perclos, 2),
            "true_perclos_raw": round(true_perclos_raw, 2),
            "closure_pct": round(closure_pct, 2),
            "closed80": closed80,
            "pupil_found": pupil_found,
            "pupil_area_px": round(pupil_area_px, 2),
            "pupil_score": round(pupil_score, 2),
            "pupil_circularity": round(pupil_circularity, 3),
            "area_ref": round(area_ref, 2),
            "raw_eye_state": raw_eye_state,
            "final_smoothed_eye_state": final_smoothed_eye_state,
            "valid_eye_fraction": round(valid_eye_fraction, 3),
            "closure_mean_pct": round(closure_mean_pct, 2),
            "visible_iris_area": round(visible_iris_area, 2),
            "baseline_iris_area": round(baseline_iris_area, 2),
            "visible_iris_ratio": round(visible_iris_ratio * 100, 2),
            "eye_calibrated": eye_calibrated,
            "eye_segmentation_ok": eye_segmentation_ok,
            "eye_status": eye_status,
            "eye_confidence": round(eye_confidence, 3),
            "eye_roi_mean": round(eye_roi_mean, 2),
            "eye_roi_std": round(eye_roi_std, 2),
            "eye_edge_density": round(eye_edge_density, 4),
            "eye_mask_area_ratio": round(eye_mask_area_ratio, 4),
            "eye_frame_errors": eye_frame_errors,
            "eye_debug_enabled": eye_debug_enabled,
            "eye_data_age_ms": None if eye_data_age_ms is None else round(eye_data_age_ms, 1),
            "eye_classifier_available": eye_classifier_available,
            "eye_classifier_state": eye_classifier_state,
            "eye_classifier_confidence": round(eye_classifier_confidence, 3),
            "eye_classifier_error": eye_classifier_error,
            "eye_closed_prob": round(eye_closed_prob, 3),
            "eye_open_prob": round(eye_open_prob, 3),
            "eye_blink_prob": round(eye_blink_prob, 3),
            "closed_frames": closed_frames,
            "open_frames": open_frames,
            "total_frames": total_frames,
            "microsleep_count": microsleep_count,
            "microsleep_active": microsleep_active,
            "microsleep_closed_duration_sec": round(microsleep_closed_duration_sec, 2),
            "eye_closed": eye_closed,
            "frame_closure_ratio": round(frame_closure_ratio * 100, 2),
            "fatigue_score": round(fatigue_score, 3),
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "imu_selected_idx": imu_selected_idx,
            "imu_last_idx": imu_last_idx,
            "imu_processed_samples": imu_processed_samples,
            "imu_skipped_samples": imu_skipped_samples,
            "imu_pitch_raw_deg": round(imu_pitch_raw_deg, 2),
            "imu_roll_raw_deg": round(imu_roll_raw_deg, 2),
            "imu_pitch_gravity_deg": round(imu_pitch_gravity_deg, 2),
            "imu_roll_gravity_deg": round(imu_roll_gravity_deg, 2),
            "imu_pitch_zero_deg": round(imu_pitch_zero_deg, 2),
            "imu_roll_zero_deg": round(imu_roll_zero_deg, 2),
            "imu_neutral_calibrated": imu_neutral_calibrated,
            "nod_detected": False,
            "nod_count_override": nod_count,
            "accel_mag": round(accel_mag, 3),
        }

    def get_raw_snapshot(self):
        with self.shared_state["lock"]:
            return {
                "imu": {
                    "timestamp_ns": self.shared_state.get("raw_imu_timestamp_ns"),
                    "timestamp_sec": self.shared_state.get("raw_imu_timestamp_sec"),
                    "imu_index": self.shared_state.get("raw_imu_index"),
                    "accel_msec2": [
                        self.shared_state.get("raw_accel_x_msec2"),
                        self.shared_state.get("raw_accel_y_msec2"),
                        self.shared_state.get("raw_accel_z_msec2"),
                    ],
                    "gyro_radsec": [
                        self.shared_state.get("raw_gyro_x_radsec"),
                        self.shared_state.get("raw_gyro_y_radsec"),
                        self.shared_state.get("raw_gyro_z_radsec"),
                    ],
                    "mag_tesla": [
                        self.shared_state.get("raw_mag_x_tesla"),
                        self.shared_state.get("raw_mag_y_tesla"),
                        self.shared_state.get("raw_mag_z_tesla"),
                    ],
                    "processed_samples": self.shared_state.get("imu_processed_samples", 0),
                    "skipped_samples": self.shared_state.get("imu_skipped_samples", 0),
                },
                "eye": {
                    "timestamp_ns": self.shared_state.get("raw_eye_timestamp_ns"),
                    "camera_id": self.shared_state.get("raw_eye_camera_id"),
                    "width": self.shared_state.get("raw_eye_width"),
                    "height": self.shared_state.get("raw_eye_height"),
                    "channels": self.shared_state.get("raw_eye_channels"),
                    "dtype": self.shared_state.get("raw_eye_dtype"),
                    "mean": self.shared_state.get("raw_eye_mean"),
                    "std": self.shared_state.get("raw_eye_std"),
                    "frame_count": self.shared_state.get("raw_eye_frame_count", 0),
                },
            }

    def latest_eye_png(self):
        with self.shared_state["lock"]:
            latest_eye = self.shared_state.get("latest_eye_gray")

            if latest_eye is None:
                return None

            frame = latest_eye.copy()

        ok, encoded = cv2.imencode(".png", frame)

        if not ok:
            return None

        return encoded.tobytes()

    def reset_nods(self):
        with self.shared_state["lock"]:
            self.shared_state["nod_count"] = 0
            self.shared_state["last_nod_time"] = 0.0
            self.shared_state["last_pitch_for_nod"] = self.shared_state["pitch_deg"]
            self.shared_state["last_pitch_sample_time"] = 0.0

    def start_eye_dataset_recording(self, **kwargs):
        return self.eye_dataset_recorder.start(**kwargs)

    def set_eye_dataset_label(self, label, note=""):
        return self.eye_dataset_recorder.set_label(label, note)

    def stop_eye_dataset_recording(self):
        return self.eye_dataset_recorder.stop()

    def eye_dataset_status(self):
        return self.eye_dataset_recorder.status()

    def calibrate_imu_neutral(self):
        with self.shared_state["lock"]:
            self.shared_state["imu_pitch_zero_deg"] = self.shared_state["imu_pitch_gravity_deg"]
            self.shared_state["imu_roll_zero_deg"] = self.shared_state["imu_roll_gravity_deg"]
            self.shared_state["pitch_deg"] = 0.0
            self.shared_state["roll_deg"] = 0.0
            self.shared_state["last_pitch_for_nod"] = 0.0
            self.shared_state["last_pitch_sample_time"] = 0.0
            self.shared_state["imu_neutral_calibrated"] = True

            return {
                "imu_selected_idx": self.shared_state["imu_selected_idx"],
                "imu_pitch_zero_deg": round(self.shared_state["imu_pitch_zero_deg"], 2),
                "imu_roll_zero_deg": round(self.shared_state["imu_roll_zero_deg"], 2),
                "imu_neutral_calibrated": True,
            }

    def reset_all(self):
        with self.shared_state["lock"]:
            self.shared_state["yaw_deg"] = 0.0
            self.shared_state["pitch_deg"] = 0.0
            self.shared_state["roll_deg"] = 0.0
            self.shared_state["last_ts"] = None
            self.shared_state["imu_last_idx"] = None
            self.shared_state["imu_processed_samples"] = 0
            self.shared_state["imu_skipped_samples"] = 0
            self.shared_state["imu_pitch_raw_deg"] = 0.0
            self.shared_state["imu_roll_raw_deg"] = 0.0
            self.shared_state["imu_pitch_gravity_deg"] = 0.0
            self.shared_state["imu_roll_gravity_deg"] = 0.0
            for key in RAW_DATA_KEYS:
                self.shared_state[key] = None

            self.shared_state["raw_eye_frame_count"] = 0
            self.shared_state["latest_eye_gray"] = None
            self.shared_state["nod_count"] = 0
            self.shared_state["last_nod_time"] = 0.0
            self.shared_state["last_pitch_for_nod"] = 0.0
            self.shared_state["last_pitch_sample_time"] = 0.0

            self.shared_state["blink_count"] = 0
            self.shared_state["closed_frames"] = 0
            self.shared_state["open_frames"] = 0
            self.shared_state["total_frames"] = 0
            self.shared_state["microsleep_count"] = 0
            self.shared_state["microsleep_active"] = False
            self.shared_state["eye_closed_start_time"] = None
            self.shared_state["microsleep_closed_duration_sec"] = 0.0
            self.shared_state["stream_start_time"] = time.time()
            self.shared_state["perclos"] = 0.0
            self.shared_state["perclos80_true"] = 0.0
            self.shared_state["perclos_continuous_pct"] = 0.0
            self.shared_state["perclos_source"] = "waiting_for_eye_frame"
            self.shared_state["true_perclos"] = 0.0
            self.shared_state["true_perclos_raw"] = 0.0
            self.shared_state["closure"] = 0.0
            self.shared_state["closure_pct"] = 0.0
            self.shared_state["closed80"] = False
            self.shared_state["pupil_found"] = False
            self.shared_state["pupil_area_px"] = 0.0
            self.shared_state["pupil_score"] = 0.0
            self.shared_state["pupil_circularity"] = 0.0
            self.shared_state["area_ref"] = 0.0
            self.shared_state["pupil_area_history"].clear()
            self.shared_state["perclos80_window"].clear()
            self.shared_state["perclos_continuous_window"].clear()
            self.shared_state["closure_mean_window"].clear()
            self.shared_state["valid_eye_window"].clear()
            self.shared_state["valid_eye_fraction"] = 0.0
            self.shared_state["closure_mean_pct"] = 0.0
            self.shared_state["raw_eye_state"] = "unknown"
            self.shared_state["final_smoothed_eye_state"] = "unknown"
            self.shared_state["candidate_eye_state"] = "unknown"
            self.shared_state["candidate_eye_count"] = 0
            self.shared_state["blink_phase"] = "open"
            self.shared_state["blink_started_at"] = None
            self.shared_state["blink_closed_frame_count"] = 0
            self.shared_state["blink_open_frame_count"] = 0
            self.shared_state["missing_pupil_frames"] = 0
            self.shared_state["weak_pupil_frames"] = 0
            self.shared_state["visible_iris_area"] = 0.0
            self.shared_state["baseline_iris_area"] = 0.0
            self.shared_state["baseline_pupil_area"] = 0.0
            self.shared_state["visible_iris_ratio"] = 1.0
            self.shared_state["baseline_samples"].clear()
            self.shared_state["eye_closed"] = False
            self.shared_state["eye_calibrated"] = False
            self.shared_state["eye_segmentation_ok"] = False
            self.shared_state["eye_status"] = "waiting_for_eye_frame"
            self.shared_state["eye_confidence"] = 0.0
            self.shared_state["eye_roi_mean"] = 0.0
            self.shared_state["eye_roi_std"] = 0.0
            self.shared_state["eye_edge_density"] = 0.0
            self.shared_state["eye_mask_area_ratio"] = 0.0
            self.shared_state["eye_frame_errors"] = 0
            self.shared_state["eye_stream_status"] = "waiting_for_eye_track"
            self.shared_state["first_eye_frame_time"] = None
            self.shared_state["last_eye_frame_time"] = None
            self.shared_state["last_blink_time"] = 0.0
            self.shared_state["closed_started_at"] = None
            self.shared_state["blink_counted_for_closure"] = False
            self.shared_state["frame_closure_ratio"] = 0.0
            self.shared_state["was_eye_closed"] = False
            self.shared_state["eye_classifier_available"] = False
            self.shared_state["eye_classifier_state"] = "unavailable"
            self.shared_state["eye_classifier_confidence"] = 0.0
            self.shared_state["eye_classifier_error"] = None
            self.shared_state["eye_closed_prob"] = 0.0
            self.shared_state["eye_open_prob"] = 0.0
            self.shared_state["eye_blink_prob"] = 0.0
            self.shared_state["classifier_perclos_window"].clear()
            self.shared_state.pop("eye_brightness_baseline", None)

        if hasattr(self, "observer"):
            self.observer.reset_imu_filter()
            self.observer.reset_microsleep_state()
 
    def close(self):
        try:
            self.eye_dataset_recorder.stop()
        except Exception as e:
            print("[EyeDataset] stop warning:", e)

        try:
            print("[RealAriaStream] Unsubscribing...")
            self.streaming_client.unsubscribe()
        except Exception as e:
            print("[RealAriaStream] unsubscribe warning:", e)

        try:
            print("[RealAriaStream] Stopping stream...")
            self.streaming_manager.stop_streaming()
        except Exception as e:
            print("[RealAriaStream] stop_streaming warning:", e)

        try:
            print("[RealAriaStream] Disconnecting device...")
            self.device_client.disconnect(self.device)
        except Exception as e:
            print("[RealAriaStream] disconnect warning:", e)
