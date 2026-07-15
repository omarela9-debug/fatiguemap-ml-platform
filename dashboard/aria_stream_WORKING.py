import time
import math
import threading

import aria.sdk as aria


class ImuObserver:
    def __init__(self, shared_state):
        self.state = shared_state
        self._printed_debug = False

    def on_streaming_client_failure(self, reason, message):
        print(f"[IMU observer] streaming failure: {reason} | {message}")

    def on_imu_received(self, motion_data, imu_idx):
        """
        Docs say:
          on_imu_received(motion_data: List[MotionData], imu_idx: int)
        We handle both a list payload and single-item payload defensively.
        """
        samples = motion_data if isinstance(motion_data, (list, tuple)) else [motion_data]

        for sample in samples:
            self._update_from_motion_sample(sample, imu_idx)

    def _update_from_motion_sample(self, sample, imu_idx):
        now = time.time()

        if not self._printed_debug:
            self._printed_debug = True
            print("[IMU observer] first sample type:", type(sample))
            print("[IMU observer] sample attrs:", [a for a in dir(sample) if not a.startswith("_")])

        # Try a few likely timestamp field names
        ts = (
            getattr(sample, "capture_timestamp_ns", None)
            or getattr(sample, "timestamp_ns", None)
            or getattr(sample, "tracking_timestamp_ns", None)
        )

        if ts is not None:
            ts_sec = ts / 1e9
        else:
            ts_sec = now

        # Try likely accel field names
        accel = (
            getattr(sample, "accel_msec2", None)
            or getattr(sample, "accel", None)
            or getattr(sample, "accelerometer", None)
        )

        # Try likely gyro field names
        gyro = (
            getattr(sample, "gyro_radsec", None)
            or getattr(sample, "gyro", None)
            or getattr(sample, "gyroscope", None)
        )

        accel_xyz = self._to_xyz(accel)
        gyro_xyz = self._to_xyz(gyro)

        with self.state["lock"]:
            prev_t = self.state["last_ts"]
            dt = 0.0 if prev_t is None else max(0.0, min(ts_sec - prev_t, 0.1))
            self.state["last_ts"] = ts_sec

            if gyro_xyz is not None and dt > 0:
                gx, gy, gz = gyro_xyz  # rad/s
                # Simple gyro integration -> degrees
                self.state["roll_deg"] += math.degrees(gx * dt)
                self.state["pitch_deg"] += math.degrees(gy * dt)
                self.state["yaw_deg"] += math.degrees(gz * dt)

            if accel_xyz is not None:
                ax, ay, az = accel_xyz
                accel_mag = math.sqrt(ax * ax + ay * ay + az * az)
                self.state["accel_mag"] = accel_mag

                # crude nod heuristic from pitch motion
                pitch_now = self.state["pitch_deg"]
                last_pitch = self.state["last_pitch_for_nod"]
                if abs(pitch_now - last_pitch) > 8.0 and (now - self.state["last_nod_time"]) > 0.8:
                    self.state["nod_count"] += 1
                    self.state["last_nod_time"] = now
                self.state["last_pitch_for_nod"] = pitch_now

    def _to_xyz(self, value):
        if value is None:
            return None

        # numpy-like / list-like
        try:
            if len(value) >= 3:
                return float(value[0]), float(value[1]), float(value[2])
        except Exception:
            pass

        # object with x,y,z
        x = getattr(value, "x", None)
        y = getattr(value, "y", None)
        z = getattr(value, "z", None)
        if x is not None and y is not None and z is not None:
            return float(x), float(y), float(z)

        return None


class RealAriaStream:
    def __init__(self):
        self.shared_state = {
            "lock": threading.Lock(),
            "last_ts": None,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "accel_mag": 9.81,
            "nod_count": 0,
            "last_nod_time": 0.0,
            "last_pitch_for_nod": 0.0,
        }

        self.device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        self.device_client.set_client_config(client_config)

        print("[RealAriaStream] Connecting to glasses over USB...")
        self.device = self.device_client.connect()
        print(f"[RealAriaStream] Connected to serial: {self.device.info.serial}")

        self.streaming_manager = self.device.streaming_manager

        streaming_config = aria.StreamingConfig()
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config

        self.streaming_client = self.streaming_manager.streaming_client

        sub_config = self.streaming_client.subscription_config
        sub_config.subscriber_data_type = aria.StreamingDataType.Imu
        queue_size = aria.MessageQueueSizeMap()
        sub_config.message_queue_size = queue_size
        sub_config.subscriber_name = "fatigue-dashboard-imu"
        self.streaming_client.subscription_config = sub_config

        self.observer = ImuObserver(self.shared_state)
        self.streaming_client.set_streaming_client_observer(self.observer)

        print("[RealAriaStream] Starting stream...")
        self.streaming_manager.start_streaming()
        self.streaming_client.subscribe()
        print("[RealAriaStream] IMU subscription active.")

    def get_frame(self):
        with self.shared_state["lock"]:
            yaw = round(self.shared_state["yaw_deg"], 2)
            pitch = round(self.shared_state["pitch_deg"], 2)
            roll = round(self.shared_state["roll_deg"], 2)
            accel_mag = self.shared_state["accel_mag"]
            nod_count = self.shared_state["nod_count"]

        # Temporary placeholders until we add real eye/blink data
        blink_rate = 0.0
        fatigue_score = min(1.0, max(0.0, abs(pitch) / 45.0 + abs(roll) / 45.0))

        return {
            "timestamp": time.time(),
            "blink_rate_bpm": round(blink_rate, 2),
            "fatigue_score": round(fatigue_score, 3),
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "nod_detected": False,  # nod_count is tracked internally for now
            "nod_count_override": nod_count,
            "accel_mag": round(accel_mag, 3),
        }

    def close(self):
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
