import math
import random
import time


class MockAriaStream:
    def __init__(self):
        self.start_time = time.time()
        self.nod_count = 0
        self.microsleep_count = 0
        self.microsleep_active = False
        self.eye_closed_start_time = None
        self.raw_snapshot = {
            "imu": {},
            "eye": {},
        }
        self.eye_dataset = {
            "active": False,
            "session_dir": None,
            "metadata_path": None,
            "label": "open",
            "note": "",
            "frame_count": 0,
            "saved_count": 0,
            "error_count": 0,
            "max_fps": 12.0,
            "save_raw": False,
            "save_roi": True,
            "session_started_at": None,
        }

    def get_frame(self):
        t = time.time() - self.start_time

        yaw = 15 * math.sin(0.5 * t)
        pitch = 8 * math.sin(0.8 * t)
        roll = 5 * math.sin(0.3 * t)

        blink_rate = 14 + 4 * math.sin(0.25 * t) + random.uniform(-1.0, 1.0)
        blink_rate = max(0, round(blink_rate, 2))

        # Simulate blink count increasing over time
        blink_count = int((t * blink_rate) / 60)

        # Simulate eye state (blinking)
        eye_closed = random.random() < 0.08  # 8% chance eye closed per frame
        now = time.time()

        if eye_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now

            microsleep_duration = now - self.eye_closed_start_time

            if microsleep_duration >= 0.5 and not self.microsleep_active:
                self.microsleep_count += 1
                self.microsleep_active = True
        else:
            self.eye_closed_start_time = None
            self.microsleep_active = False
            microsleep_duration = 0.0

        visible_iris_ratio = random.uniform(0.02, 0.18) if eye_closed else random.uniform(0.86, 1.0)
        baseline_iris_area = 1000.0
        visible_iris_area = baseline_iris_area * visible_iris_ratio
        true_perclos = round(100 * (1 - visible_iris_ratio), 2)
        pupil_found = not eye_closed or random.random() < 0.45
        perclos_source = "pupil" if pupil_found else "classifier_fallback"

        # Frame counters
        total_frames = int(t * 30)
        closed_frames = int(total_frames * 0.08)
        open_frames = total_frames - closed_frames

        frame_closure_ratio = closed_frames / total_frames if total_frames > 0 else 0

        fatigue_score = 0.35 + 0.15 * math.sin(0.15 * t) + random.uniform(-0.03, 0.03)
        fatigue_score = max(0, min(1, round(fatigue_score, 3)))
        nod_detected = False
        raw_accel = [
            round(9.81 * math.sin(math.radians(pitch)), 5),
            round(9.81 * math.sin(math.radians(roll)), 5),
            round(9.81 * math.cos(math.radians(pitch)), 5),
        ]
        raw_gyro = [
            round(0.02 * math.sin(0.8 * t), 5),
            round(0.02 * math.sin(0.3 * t), 5),
            round(0.02 * math.sin(0.5 * t), 5),
        ]
        raw_mag = [None, None, None]
        raw_eye_width = 320
        raw_eye_height = 240
        raw_eye_timestamp_ns = int(time.time() * 1e9)
        raw_imu_timestamp_ns = raw_eye_timestamp_ns

        self.raw_snapshot = {
            "imu": {
                "timestamp_ns": raw_imu_timestamp_ns,
                "timestamp_sec": raw_imu_timestamp_ns / 1e9,
                "imu_index": "mock",
                "accel_msec2": raw_accel,
                "gyro_radsec": raw_gyro,
                "mag_tesla": raw_mag,
                "processed_samples": int(t * 100),
                "skipped_samples": 0,
            },
            "eye": {
                "timestamp_ns": raw_eye_timestamp_ns,
                "camera_id": "mock_eye_track",
                "width": raw_eye_width,
                "height": raw_eye_height,
                "channels": 1,
                "dtype": "uint8",
                "mean": 80,
                "std": 20,
                "frame_count": total_frames,
            },
        }

        return {
            "timestamp": time.time(),
            "raw_imu_timestamp_ns": raw_imu_timestamp_ns,
            "raw_imu_timestamp_sec": raw_imu_timestamp_ns / 1e9,
            "raw_imu_index": "mock",
            "raw_accel_x_msec2": raw_accel[0],
            "raw_accel_y_msec2": raw_accel[1],
            "raw_accel_z_msec2": raw_accel[2],
            "raw_gyro_x_radsec": raw_gyro[0],
            "raw_gyro_y_radsec": raw_gyro[1],
            "raw_gyro_z_radsec": raw_gyro[2],
            "raw_mag_x_tesla": raw_mag[0],
            "raw_mag_y_tesla": raw_mag[1],
            "raw_mag_z_tesla": raw_mag[2],
            "raw_eye_timestamp_ns": raw_eye_timestamp_ns,
            "raw_eye_camera_id": "mock_eye_track",
            "raw_eye_width": raw_eye_width,
            "raw_eye_height": raw_eye_height,
            "raw_eye_channels": 1,
            "raw_eye_dtype": "uint8",
            "raw_eye_mean": 80,
            "raw_eye_std": 20,
            "raw_eye_frame_count": total_frames,
            "stream_uptime_sec": round(t, 2),
            "stream_profile_name": "mock",
            "stream_profile_error": None,
            "eye_stream_status": "live",
            "eye_wait_sec": 0.0,
            "eye_start_delay_sec": 0.0,
            "eye_fps_est": 30.0,

            "blink_rate_bpm": blink_rate,
            "blink_count": blink_count,
            "perclos": true_perclos,
            "perclos80_true": true_perclos,
            "perclos_continuous_pct": true_perclos,
            "perclos_source": perclos_source,
            "true_perclos": true_perclos,
            "true_perclos_raw": true_perclos,
            "closure_pct": true_perclos,
            "closed80": true_perclos >= 80,
            "pupil_found": pupil_found,
            "pupil_area_px": round(visible_iris_area, 2) if pupil_found else 0.0,
            "pupil_score": 420.0 if pupil_found else 0.0,
            "pupil_circularity": 0.72 if pupil_found else 0.0,
            "area_ref": baseline_iris_area,
            "raw_eye_state": "closed" if eye_closed else "open",
            "final_smoothed_eye_state": "closed" if eye_closed else "open",
            "valid_eye_fraction": 1.0,
            "closure_mean_pct": true_perclos,
            "visible_iris_area": round(visible_iris_area, 2),
            "baseline_iris_area": baseline_iris_area,
            "visible_iris_ratio": round(visible_iris_ratio * 100, 2),
            "eye_calibrated": True,
            "eye_segmentation_ok": True,
            "eye_status": "mock",
            "eye_confidence": 0.9,
            "eye_roi_mean": 80,
            "eye_roi_std": 20,
            "eye_edge_density": 0.05,
            "eye_mask_area_ratio": 0.04,
            "eye_frame_errors": 0,
            "eye_data_age_ms": 0,
            "eye_debug_enabled": False,
            "eye_classifier_available": True,
            "eye_classifier_state": "closed" if eye_closed else "open",
            "eye_classifier_confidence": 0.9,
            "eye_classifier_error": None,
            "eye_closed_prob": 0.9 if eye_closed else 0.05,
            "eye_open_prob": 0.05 if eye_closed else 0.9,
            "eye_blink_prob": 0.05,
            "frame_closure_ratio": round(frame_closure_ratio * 100, 2),
            "eye_closed": eye_closed,
            "closed_frames": closed_frames,
            "open_frames": open_frames,
            "total_frames": total_frames,
            "microsleep_count": self.microsleep_count,
            "microsleep_active": self.microsleep_active,
            "microsleep_closed_duration_sec": round(microsleep_duration, 2),

            "fatigue_score": fatigue_score,

            "yaw_deg": round(yaw, 2),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "imu_selected_idx": "mock",
            "imu_last_idx": "mock",
            "imu_processed_samples": int(t * 100),
            "imu_skipped_samples": 0,
            "imu_pitch_raw_deg": round(pitch, 2),
            "imu_roll_raw_deg": round(roll, 2),
            "imu_pitch_gravity_deg": round(pitch, 2),
            "imu_roll_gravity_deg": round(roll, 2),
            "imu_pitch_zero_deg": 0.0,
            "imu_roll_zero_deg": 0.0,
            "imu_neutral_calibrated": True,

            "nod_detected": nod_detected,
            "nod_count_override": self.nod_count,
            "accel_mag": 9.81 + random.uniform(-0.08, 0.08),
        }

    def reset_nods(self):
        self.nod_count = 0

    def reset_all(self):
        self.start_time = time.time()
        self.nod_count = 0
        self.microsleep_count = 0
        self.microsleep_active = False
        self.eye_closed_start_time = None

    def start_eye_dataset_recording(self, **kwargs):
        self.eye_dataset.update({
            "active": True,
            "label": kwargs.get("label", "open"),
            "note": kwargs.get("note", ""),
            "max_fps": kwargs.get("max_fps", 12.0),
            "save_raw": kwargs.get("save_raw", False),
            "save_roi": kwargs.get("save_roi", True),
            "session_started_at": time.time(),
        })
        return dict(self.eye_dataset)

    def set_eye_dataset_label(self, label, note=""):
        self.eye_dataset["label"] = label
        self.eye_dataset["note"] = note
        return dict(self.eye_dataset)

    def stop_eye_dataset_recording(self):
        self.eye_dataset["active"] = False
        return dict(self.eye_dataset)

    def eye_dataset_status(self):
        return dict(self.eye_dataset)

    def get_raw_snapshot(self):
        return dict(self.raw_snapshot)

    def latest_eye_png(self):
        return None
