"""Minimal REST API that serves and updates the robot telemetry stored in data.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request

APP_ROOT = Path(__file__).parent
DATA_PATH = APP_ROOT / "data2.json"

app = Flask(__name__)


def load_data() -> dict:
    """Read the latest snapshot from disk."""
    if not DATA_PATH.exists():
        raise FileNotFoundError("data.json is missing")
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def persist_data(payload: dict) -> None:
    """Save the payload as pretty-printed JSON."""
    DATA_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def current_timestamp() -> str:
    """Return an ISO8601 UTC timestamp with a Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@app.after_request
def add_cors_headers(response):
    """Allow the React frontend (or other clients) to hit the API directly."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "Hoop telemetry API",
            "endpoints": ["/data", "/upload"],
            "timestamp": current_timestamp(),
        }
    )


@app.route("/data", methods=["GET"])
def get_data():
    try:
        snapshot = load_data()
    except FileNotFoundError:
        abort(404, description="data.json not found on the server")
    return jsonify(snapshot)


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_data():
    if request.method == "OPTIONS":
        # Allow preflight requests without touching the file on disk.
        return ("", 204)

    payload = request.get_json(silent=True)
    if payload is None:
        abort(400, description="Missing or invalid JSON payload")

    payload.setdefault("timestamp", current_timestamp())
    persist_data(payload)

    return jsonify({"status": "ok", "message": "Data stored successfully"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": current_timestamp()})


if __name__ == "__main__":
    # Expose the API on all interfaces so the React app (or other devices) can reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
