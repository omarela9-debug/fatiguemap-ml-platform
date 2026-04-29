from flask import Flask, jsonify
from flask_cors import CORS

from models.fatigue_score import calculate_fatigue_score

app = Flask(__name__)
CORS(app)

@app.route("/health")
def health():
    return jsonify({"status": "running"})

@app.route("/metrics")
def metrics():
    sample_data = {
        "perclos": 0.32,
        "blink_rate": 22,
        "head_nod_count": 3
    }

    fatigue_score = calculate_fatigue_score(
        sample_data["perclos"],
        sample_data["blink_rate"],
        sample_data["head_nod_count"]
    )

    return jsonify({
        **sample_data,
        "fatigue_score": fatigue_score
    })

if __name__ == "__main__":
    app.run(debug=True)
