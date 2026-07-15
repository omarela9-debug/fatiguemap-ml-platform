import os
import csv
from datetime import datetime
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
import time
import atexit

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

from config import (
    ARIA_IMU_INDEX,
    ARIA_IMU_PITCH_ZERO_DEG,
    ARIA_IMU_ROLL_ZERO_DEG,
    ARIA_STREAM_PROFILE,
    HEART_RATE_BAUD,
    HEART_RATE_SERIAL_PORT,
    HEART_RATE_STALE_SEC,
    UPDATE_HZ,
    USE_MOCK_DATA,
)
from mock_stream import MockAriaStream
from aria_stream import RealAriaStream
from metrics import LiveMetrics

app = Flask(__name__)
CORS(app)

heart_rate_lock = threading.Lock()
heart_rate_bpm = None
heart_rate_connected = False
heart_rate_timestamp = None
heart_rate_port = None
heart_rate_raw_line = None


class HeartRateSerialReader:
    def __init__(self, port="", baud=115200, stale_sec=5.0):
        self.port = port
        self.baud = baud
        self.stale_sec = stale_sec
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.serial_conn = None

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self._close_serial()

    def snapshot(self):
        global heart_rate_connected

        now = time.time()

        with heart_rate_lock:
            connected = (
                heart_rate_connected
                and heart_rate_timestamp is not None
                and now - heart_rate_timestamp <= self.stale_sec
            )

            if not connected:
                heart_rate_connected = False

            return {
                "heart_rate_bpm": heart_rate_bpm,
                "heart_rate_connected": connected,
                "heart_rate_timestamp": heart_rate_timestamp,
                "heart_rate_port": heart_rate_port,
                "heart_rate_raw_line": heart_rate_raw_line,
            }

    def _run(self):
        if serial is None:
            print("[HeartRate] pyserial not installed; heart rate disabled.")
            return

        while not self.stop_event.is_set():
            ports = self._candidate_ports()

            if not ports:
                self._mark_disconnected()
                time.sleep(2.0)
                continue

            for port in ports:
                if self.stop_event.is_set():
                    break

                if self._read_port(port):
                    break

            time.sleep(0.5)

    def _candidate_ports(self):
        if self.port:
            return [self.port]

        if list_ports is None:
            return []

        ports = list(list_ports.comports())
        preferred = []
        fallback = []

        for item in ports:
            device = item.device
            lower = device.lower()

            if any(token in lower for token in ("usbmodem", "usbserial", "ttyacm", "ttyusb")):
                preferred.append(device)
            else:
                fallback.append(device)

        return preferred or fallback

    def _read_port(self, port):
        try:
            self.serial_conn = serial.Serial(port, self.baud, timeout=1)
            print(f"[HeartRate] Connected to {port} at {self.baud} baud")

            while not self.stop_event.is_set():
                raw = self.serial_conn.readline()

                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()

                if line:
                    self._handle_line(line, port)

            return True
        except Exception as exc:
            print(f"[HeartRate] serial error on {port}: {exc}")
            self._mark_disconnected()
            return False
        finally:
            self._close_serial()

    def _handle_line(self, line, port):
        global heart_rate_bpm, heart_rate_connected, heart_rate_timestamp, heart_rate_port, heart_rate_raw_line

        with heart_rate_lock:
            heart_rate_raw_line = line

        parts = [part.strip() for part in line.split(",")]

        if len(parts) != 2 or parts[0].upper() != "HR":
            return

        try:
            bpm = int(float(parts[1]))
        except ValueError:
            return

        if bpm < 35 or bpm > 220:
            return

        with heart_rate_lock:
            heart_rate_bpm = bpm
            heart_rate_connected = True
            heart_rate_timestamp = time.time()
            heart_rate_port = port
            heart_rate_raw_line = line

    def _mark_disconnected(self):
        global heart_rate_connected

        with heart_rate_lock:
            heart_rate_connected = False

    def _close_serial(self):
        if self.serial_conn is None:
            return

        try:
            self.serial_conn.close()
        except Exception:
            pass
        finally:
            self.serial_conn = None


metrics = LiveMetrics()
stream = (
    MockAriaStream()
    if USE_MOCK_DATA
    else RealAriaStream(
        profile_name=ARIA_STREAM_PROFILE,
        imu_index=ARIA_IMU_INDEX,
        imu_pitch_zero_deg=ARIA_IMU_PITCH_ZERO_DEG,
        imu_roll_zero_deg=ARIA_IMU_ROLL_ZERO_DEG,
    )
)
heart_rate_reader = HeartRateSerialReader(
    port=HEART_RATE_SERIAL_PORT,
    baud=HEART_RATE_BAUD,
    stale_sec=HEART_RATE_STALE_SEC,
)
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

run_filename = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S.csv")
run_path = os.path.join(RUNS_DIR, run_filename)

csv_file = open(run_path, "w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "timestamp",
    "stream_uptime_sec",
    "stream_profile_name",
    "stream_profile_error",
    "eye_stream_status",
    "eye_wait_sec",
    "eye_start_delay_sec",
    "eye_fps_est",
    "heart_rate_bpm",
    "heart_rate_connected",
    "heart_rate_timestamp",
    "heart_rate_port",
    "heart_rate_raw_line",
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
    "blink_rate_bpm",
    "blink_count",
    "perclos",
    "perclos80_true",
    "perclos_continuous_pct",
    "perclos_source",
    "true_perclos",
    "true_perclos_raw",
    "closure_pct",
    "closed80",
    "pupil_found",
    "pupil_area_px",
    "pupil_score",
    "pupil_circularity",
    "area_ref",
    "raw_eye_state",
    "final_smoothed_eye_state",
    "valid_eye_fraction",
    "closure_mean_pct",
    "visible_iris_area",
    "baseline_iris_area",
    "visible_iris_ratio",
    "eye_calibrated",
    "eye_segmentation_ok",
    "eye_status",
    "eye_confidence",
    "eye_mask_area_ratio",
    "eye_data_age_ms",
    "eye_classifier_state",
    "eye_classifier_confidence",
    "eye_closed_prob",
    "eye_open_prob",
    "eye_blink_prob",
    "closed_frames",
    "open_frames",
    "total_frames",
    "microsleep_count",
    "microsleep_active",
    "microsleep_closed_duration_sec",
    "eye_closed",
    "fatigue_score",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "imu_selected_idx",
    "imu_last_idx",
    "imu_processed_samples",
    "imu_skipped_samples",
    "imu_pitch_raw_deg",
    "imu_roll_raw_deg",
    "imu_pitch_gravity_deg",
    "imu_roll_gravity_deg",
    "imu_pitch_zero_deg",
    "imu_roll_zero_deg",
    "imu_neutral_calibrated",
    "nod_count_override",
    "accel_mag",
])
csv_writer.writeheader()

print(f"[Logger] Saving run to {run_path}")


def cleanup():
    heart_rate_reader.stop()
    if hasattr(stream, "close"):
        stream.close()
    csv_file.close()


atexit.register(cleanup)
heart_rate_reader.start()

log_count = 0

def update_loop():
    interval = 1.0 / UPDATE_HZ
    while True:
        try:
            frame = stream.get_frame()
        except Exception as e:
            print("[Backend] stream frame error:", e)
            time.sleep(interval)
            continue

        frame.update(heart_rate_reader.snapshot())
        metrics.update(frame)

        csv_writer.writerow({
            "timestamp": frame.get("timestamp"),
            "stream_uptime_sec": frame.get("stream_uptime_sec"),
            "stream_profile_name": frame.get("stream_profile_name"),
            "stream_profile_error": frame.get("stream_profile_error"),
            "eye_stream_status": frame.get("eye_stream_status"),
            "eye_wait_sec": frame.get("eye_wait_sec"),
            "eye_start_delay_sec": frame.get("eye_start_delay_sec"),
            "eye_fps_est": frame.get("eye_fps_est"),
            "heart_rate_bpm": frame.get("heart_rate_bpm"),
            "heart_rate_connected": frame.get("heart_rate_connected"),
            "heart_rate_timestamp": frame.get("heart_rate_timestamp"),
            "heart_rate_port": frame.get("heart_rate_port"),
            "heart_rate_raw_line": frame.get("heart_rate_raw_line"),
            "raw_imu_timestamp_ns": frame.get("raw_imu_timestamp_ns"),
            "raw_imu_timestamp_sec": frame.get("raw_imu_timestamp_sec"),
            "raw_imu_index": frame.get("raw_imu_index"),
            "raw_accel_x_msec2": frame.get("raw_accel_x_msec2"),
            "raw_accel_y_msec2": frame.get("raw_accel_y_msec2"),
            "raw_accel_z_msec2": frame.get("raw_accel_z_msec2"),
            "raw_gyro_x_radsec": frame.get("raw_gyro_x_radsec"),
            "raw_gyro_y_radsec": frame.get("raw_gyro_y_radsec"),
            "raw_gyro_z_radsec": frame.get("raw_gyro_z_radsec"),
            "raw_mag_x_tesla": frame.get("raw_mag_x_tesla"),
            "raw_mag_y_tesla": frame.get("raw_mag_y_tesla"),
            "raw_mag_z_tesla": frame.get("raw_mag_z_tesla"),
            "raw_eye_timestamp_ns": frame.get("raw_eye_timestamp_ns"),
            "raw_eye_camera_id": frame.get("raw_eye_camera_id"),
            "raw_eye_width": frame.get("raw_eye_width"),
            "raw_eye_height": frame.get("raw_eye_height"),
            "raw_eye_channels": frame.get("raw_eye_channels"),
            "raw_eye_dtype": frame.get("raw_eye_dtype"),
            "raw_eye_mean": frame.get("raw_eye_mean"),
            "raw_eye_std": frame.get("raw_eye_std"),
            "raw_eye_frame_count": frame.get("raw_eye_frame_count"),
            "blink_rate_bpm": frame.get("blink_rate_bpm"),
            "blink_count": frame.get("blink_count"),
            "perclos": frame.get("perclos"),
            "perclos80_true": frame.get("perclos80_true"),
            "perclos_continuous_pct": frame.get("perclos_continuous_pct"),
            "perclos_source": frame.get("perclos_source"),
            "true_perclos": frame.get("true_perclos"),
            "true_perclos_raw": frame.get("true_perclos_raw"),
            "closure_pct": frame.get("closure_pct"),
            "closed80": frame.get("closed80"),
            "pupil_found": frame.get("pupil_found"),
            "pupil_area_px": frame.get("pupil_area_px"),
            "pupil_score": frame.get("pupil_score"),
            "pupil_circularity": frame.get("pupil_circularity"),
            "area_ref": frame.get("area_ref"),
            "raw_eye_state": frame.get("raw_eye_state"),
            "final_smoothed_eye_state": frame.get("final_smoothed_eye_state"),
            "valid_eye_fraction": frame.get("valid_eye_fraction"),
            "closure_mean_pct": frame.get("closure_mean_pct"),
            "visible_iris_area": frame.get("visible_iris_area"),
            "baseline_iris_area": frame.get("baseline_iris_area"),
            "visible_iris_ratio": frame.get("visible_iris_ratio"),
            "eye_calibrated": frame.get("eye_calibrated"),
            "eye_segmentation_ok": frame.get("eye_segmentation_ok"),
            "eye_status": frame.get("eye_status"),
            "eye_confidence": frame.get("eye_confidence"),
            "eye_mask_area_ratio": frame.get("eye_mask_area_ratio"),
            "eye_data_age_ms": frame.get("eye_data_age_ms"),
            "eye_classifier_state": frame.get("eye_classifier_state"),
            "eye_classifier_confidence": frame.get("eye_classifier_confidence"),
            "eye_closed_prob": frame.get("eye_closed_prob"),
            "eye_open_prob": frame.get("eye_open_prob"),
            "eye_blink_prob": frame.get("eye_blink_prob"),
            "closed_frames": frame.get("closed_frames"),
            "open_frames": frame.get("open_frames"),
            "total_frames": frame.get("total_frames"),
            "microsleep_count": frame.get("microsleep_count"),
            "microsleep_active": frame.get("microsleep_active"),
            "microsleep_closed_duration_sec": frame.get("microsleep_closed_duration_sec"),
            "eye_closed": frame.get("eye_closed"),
            "fatigue_score": frame.get("fatigue_score"),
            "yaw_deg": frame.get("yaw_deg"),
            "pitch_deg": frame.get("pitch_deg"),
            "roll_deg": frame.get("roll_deg"),
            "imu_selected_idx": frame.get("imu_selected_idx"),
            "imu_last_idx": frame.get("imu_last_idx"),
            "imu_processed_samples": frame.get("imu_processed_samples"),
            "imu_skipped_samples": frame.get("imu_skipped_samples"),
            "imu_pitch_raw_deg": frame.get("imu_pitch_raw_deg"),
            "imu_roll_raw_deg": frame.get("imu_roll_raw_deg"),
            "imu_pitch_gravity_deg": frame.get("imu_pitch_gravity_deg"),
            "imu_roll_gravity_deg": frame.get("imu_roll_gravity_deg"),
            "imu_pitch_zero_deg": frame.get("imu_pitch_zero_deg"),
            "imu_roll_zero_deg": frame.get("imu_roll_zero_deg"),
            "imu_neutral_calibrated": frame.get("imu_neutral_calibrated"),
            "nod_count_override": frame.get("nod_count_override"),
            "accel_mag": frame.get("accel_mag"),
        })

        global log_count
        log_count += 1

        if log_count % 20 == 0:
           csv_file.flush()

        time.sleep(interval)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/live")
def api_live():
    return jsonify(metrics.latest_payload())


@app.route("/api/raw/latest")
def api_raw_latest():
    raw = {}

    if hasattr(stream, "get_raw_snapshot"):
        raw = stream.get_raw_snapshot()

    return jsonify({
        "status": "ok",
        "run_csv": run_path,
        "raw": {
            **raw,
            "heart_rate": heart_rate_reader.snapshot(),
        },
        "eye_png_url": "/api/raw/eye.png",
    })


@app.route("/api/raw/eye.png")
def api_raw_eye_png():
    if not hasattr(stream, "latest_eye_png"):
        return jsonify({"status": "unavailable"}), 501

    png = stream.latest_eye_png()

    if not png:
        return jsonify({"status": "waiting_for_eye_frame"}), 404

    return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store"})


@app.route("/api/reset_nods", methods=["POST"])
def api_reset_nods():
    metrics.reset_nods()

    if hasattr(stream, "reset_nods"):
        stream.reset_nods()

    return jsonify({"status": "ok"})


@app.route("/api/reset_all", methods=["POST"])
def api_reset_all():
    metrics.reset_all()

    if hasattr(stream, "reset_all"):
        stream.reset_all()

    return jsonify({"status": "ok"})


@app.route("/api/calibrate_imu_neutral", methods=["POST"])
def api_calibrate_imu_neutral():
    if not hasattr(stream, "calibrate_imu_neutral"):
        return jsonify({"status": "unavailable"}), 501

    return jsonify({"status": "ok", "imu": stream.calibrate_imu_neutral()})


@app.route("/api/eye_dataset/status")
def api_eye_dataset_status():
    if not hasattr(stream, "eye_dataset_status"):
        return jsonify({"status": "unavailable"}), 501

    return jsonify({"status": "ok", "dataset": stream.eye_dataset_status()})


@app.route("/api/eye_dataset/start", methods=["POST"])
def api_eye_dataset_start():
    if not hasattr(stream, "start_eye_dataset_recording"):
        return jsonify({"status": "unavailable"}), 501

    payload = request.get_json(silent=True) or {}

    try:
        status = stream.start_eye_dataset_recording(
            label=payload.get("label", "open"),
            note=payload.get("note", ""),
            session_name=payload.get("session_name"),
            max_fps=payload.get("max_fps", payload.get("fps", 12.0)),
            save_raw=payload.get("save_raw", False),
            save_roi=payload.get("save_roi", True),
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    return jsonify({"status": "ok", "dataset": status})


@app.route("/api/eye_dataset/label", methods=["POST"])
def api_eye_dataset_label():
    if not hasattr(stream, "set_eye_dataset_label"):
        return jsonify({"status": "unavailable"}), 501

    payload = request.get_json(silent=True) or {}

    try:
        status = stream.set_eye_dataset_label(
            label=payload.get("label", "other"),
            note=payload.get("note", ""),
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    return jsonify({"status": "ok", "dataset": status})


@app.route("/api/eye_dataset/stop", methods=["POST"])
def api_eye_dataset_stop():
    if not hasattr(stream, "stop_eye_dataset_recording"):
        return jsonify({"status": "unavailable"}), 501

    return jsonify({"status": "ok", "dataset": stream.stop_eye_dataset_recording()})

if __name__ == "__main__":
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
