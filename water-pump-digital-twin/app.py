import os
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_file

import database
import mqtt_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_SORT_KEYS"] = False


def get_latest_state_payload() -> Dict[str, Any]:
    memory_state = mqtt_client.get_current_state()
    latest = memory_state.get("latest") if memory_state else None
    if latest is None:
        latest = database.get_latest_state()
        if latest:
            memory_state = {
                "timestamp": latest.get("timestamp"),
                "temperature": latest.get("temperature"),
                "current": latest.get("current"),
                "vibration_rms": latest.get("vibration_rms"),
                "acceleration_x": latest.get("acceleration_x"),
                "acceleration_y": latest.get("acceleration_y"),
                "acceleration_z": latest.get("acceleration_z"),
                "health_score": latest.get("health_score"),
                "fault_label": latest.get("fault_label"),
                "pump_status": latest.get("pump_status"),
                "last_seen": latest.get("timestamp"),
                "online": True,
                "connection_status": "ONLINE",
                "seconds_since_update": 0.0,
                "latest": latest,
            }
            return memory_state
        return {
            "online": False,
            "connection_status": "OFFLINE",
            "seconds_since_update": None,
            "latest": None,
            "timestamp": None,
            "temperature": None,
            "current": None,
            "vibration_rms": None,
            "acceleration_x": None,
            "acceleration_y": None,
            "acceleration_z": None,
            "health_score": None,
            "fault_label": "NO DATA",
            "pump_status": "UNKNOWN",
            "predicted_class": None,
            "predicted_condition": None,
            "confidence": None,
            "inference_location": None,
            "last_seen": None,
        }

    payload = {
        "online": memory_state.get("online", False),
        "connection_status": memory_state.get("connection_status", "OFFLINE"),
        "seconds_since_update": memory_state.get("seconds_since_update"),
        "last_seen": memory_state.get("last_seen"),
        "latest": latest,
    }
    for key in [
        "timestamp",
        "temperature",
        "current",
        "vibration_rms",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
        "health_score",
        "fault_label",
        "pump_status",
        "predicted_class",
        "predicted_condition",
        "confidence",
        "inference_location",
        "sensor_fault",
        "disconnected_sensor",
    ]:
        payload[key] = latest.get(key) if latest else None
    return payload


@app.get("/")
def index():
    return send_file(os.path.join(BASE_DIR, "templates", "index.html"))


@app.get("/api/state")
def api_state():
    return jsonify(get_latest_state_payload())


@app.get("/api/history")
def api_history():
    limit = request.args.get("limit", default=60, type=int)
    start = request.args.get("start", default=None, type=float)
    end = request.args.get("end", default=None, type=float)
    return jsonify(database.get_history(limit=limit, start=start, end=end))


@app.get("/api/alerts")
def api_alerts():
    limit = request.args.get("limit", default=20, type=int)
    return jsonify(database.get_recent_events(limit=limit))


@app.get("/api/health")
def api_health():
    state = get_latest_state_payload()
    latest = state.get("latest")
    return jsonify(
        {
            "health_score": latest.get("health_score") if latest else None,
            "fault_label": latest.get("fault_label") if latest else "NO DATA",
            "pump_status": latest.get("pump_status") if latest else "UNKNOWN",
            "connection_status": state.get("connection_status", "OFFLINE"),
        }
    )


if __name__ == "__main__":
    database.init_database()
    mqtt_client.start_mqtt_client()
    app.run(host="0.0.0.0", port=5000, debug=False)
