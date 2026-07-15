from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time
import atexit

from config import UPDATE_HZ, USE_MOCK_DATA
from mock_stream import MockAriaStream
from aria_stream import RealAriaStream
from metrics import LiveMetrics

app = Flask(__name__)
CORS(app)

metrics = LiveMetrics()
stream = MockAriaStream() if USE_MOCK_DATA else RealAriaStream()


def cleanup():
    if hasattr(stream, "close"):
        stream.close()


atexit.register(cleanup)


def update_loop():
    interval = 1.0 / UPDATE_HZ
    while True:
        frame = stream.get_frame()
        metrics.update(frame)
        time.sleep(interval)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/live")
def api_live():
    return jsonify(metrics.latest_payload())


if __name__ == "__main__":
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
