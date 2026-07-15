import os


USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
UPDATE_HZ = 30

# Project Aria Gen 1 profiles:
# profile18: streaming-optimized default, EyeTrack 320x240 at 10 FPS
# profile28: higher EyeTrack rate, 320x240 at 60 FPS
# profile16: highest EyeTrack rate, 640x480 at 90 FPS, heavier CPU/device load
ARIA_STREAM_PROFILE = os.getenv("ARIA_STREAM_PROFILE", "profile28").strip() or "profile18"

# Project Aria exposes two IMU streams. "auto" picks one stream and ignores the
# other so pitch/roll do not jump between two different IMU coordinate frames.
ARIA_IMU_INDEX = os.getenv("ARIA_IMU_INDEX", "auto").strip() or "auto"

ARIA_IMU_PITCH_ZERO_DEG = float(os.getenv("ARIA_IMU_PITCH_ZERO_DEG", "0"))
ARIA_IMU_ROLL_ZERO_DEG = float(os.getenv("ARIA_IMU_ROLL_ZERO_DEG", "0"))

HEART_RATE_SERIAL_PORT = os.getenv("HEART_RATE_SERIAL_PORT", "").strip()
HEART_RATE_BAUD = int(os.getenv("HEART_RATE_BAUD", "115200"))
HEART_RATE_STALE_SEC = float(os.getenv("HEART_RATE_STALE_SEC", "5.0"))
