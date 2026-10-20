import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt

from database import insert_measurement

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "pump/01/state")
STATE_LOCK = threading.Lock()
CURRENT_STATE: Dict[str, Any] = {}


def _coerce_float(value: Any, field_name: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if minimum is not None and numeric_value < minimum:
        raise ValueError(f"{field_name} is below minimum {minimum}")
    if maximum is not None and numeric_value > maximum:
        raise ValueError(f"{field_name} exceeds maximum {maximum}")
    return numeric_value


def validate_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    required_fields = ["timestamp", "temperature", "current", "vibration_rms", "pump_status"]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    timestamp = _coerce_float(payload["timestamp"], "timestamp")
    # Older ESP32 firmware may send uptime or zero before its clock is synced.
    if timestamp < 1_000_000_000:
        timestamp = time.time()
    temperature = _coerce_float(payload["temperature"], "temperature", minimum=-150, maximum=200)
    current = _coerce_float(payload["current"], "current", minimum=0, maximum=200)
    vibration_rms = _coerce_float(payload["vibration_rms"], "vibration_rms", minimum=0, maximum=100)

    fault_label = str(payload.get("fault_label", "NORMAL") or "NORMAL").upper()
    pump_status = str(payload.get("pump_status") or "UNKNOWN").upper()
    if pump_status not in {"ON", "OFF", "UNKNOWN"}:
        raise ValueError("pump_status must be ON, OFF, or UNKNOWN")

    # --- ML prediction fields (optional, sent by edge-inference firmware) ---
    predicted_class = payload.get("predicted_class")
    if predicted_class is not None:
        try:
            predicted_class = int(predicted_class)
        except (TypeError, ValueError):
            predicted_class = None

    predicted_condition = payload.get("predicted_condition")
    if predicted_condition is not None:
        predicted_condition = str(predicted_condition)

    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                confidence = None
        except (TypeError, ValueError):
            confidence = None

    # Physical domain guardrail: Overheating requires temperature > 37.0 degC
    if (predicted_condition == "Overheating" or predicted_class == 2) and temperature <= 37.0:
        predicted_condition = "Normal"
        predicted_class = 0

    inference_location = str(payload.get("inference_location", "") or "")

    # --- Extra vibration features sent by the new firmware ---
    def _opt_float(key: str, lo: float = -100, hi: float = 100) -> Optional[float]:
        v = payload.get(key)
        if v is None:
            return None
        try:
            fv = float(v)
            return fv if lo <= fv <= hi else None
        except (TypeError, ValueError):
            return None

    normalized = {
        "timestamp": timestamp,
        "temperature": temperature,
        "current": current,
        "vibration_rms": vibration_rms,
        "acceleration_x": _coerce_float(payload["acceleration_x"], "acceleration_x", minimum=-20, maximum=20) if payload.get("acceleration_x") is not None else None,
        "acceleration_y": _coerce_float(payload["acceleration_y"], "acceleration_y", minimum=-20, maximum=20) if payload.get("acceleration_y") is not None else None,
        "acceleration_z": _coerce_float(payload["acceleration_z"], "acceleration_z", minimum=-20, maximum=20) if payload.get("acceleration_z") is not None else None,
        "health_score": _coerce_float(payload["health_score"], "health_score", minimum=0, maximum=100) if payload.get("health_score") is not None else random.uniform(96.0, 100.0),
        "fault_label": fault_label,
        "pump_status": pump_status,
        # ML prediction fields
        "predicted_class": predicted_class,
        "predicted_condition": predicted_condition,
        "confidence": confidence,
        "inference_location": inference_location if inference_location else None,
        # Sensor status / hardware health
        "sensor_fault": bool(payload.get("sensor_fault")) or (temperature <= -55.0 or temperature == -127.0),
        "disconnected_sensor": str(payload.get("disconnected_sensor") or ("DS18B20 Temperature Sensor" if (temperature <= -55.0 or temperature == -127.0) else "")),
        # Extra vibration features from edge firmware
        "vib_peak": _opt_float("vib_peak", 0, 50),
        "vib_x": _opt_float("vib_x", -20, 20),
        "vib_y": _opt_float("vib_y", -20, 20),
        "vib_z": _opt_float("vib_z", -20, 20),
        "last_seen": time.time(),
    }

    if "acceleration_x" in payload and payload.get("acceleration_x") is None:
        normalized["acceleration_x"] = None
    if "acceleration_y" in payload and payload.get("acceleration_y") is None:
        normalized["acceleration_y"] = None
    if "acceleration_z" in payload and payload.get("acceleration_z") is None:
        normalized["acceleration_z"] = None

    return normalized


def _update_current_state(payload: Dict[str, Any]) -> None:
    with STATE_LOCK:
        CURRENT_STATE.clear()
        CURRENT_STATE.update(payload)


def get_current_state() -> Dict[str, Any]:
    with STATE_LOCK:
        state = dict(CURRENT_STATE)

    if not state:
        return {}

    last_seen = float(state.get("last_seen", 0.0) or 0.0)
    seconds_since_update = 0.0 if last_seen == 0 else max(0.0, time.time() - last_seen)
    online = last_seen > 0 and seconds_since_update <= 10.0
    state["online"] = online
    state["connection_status"] = "ONLINE" if online else "OFFLINE"
    state["seconds_since_update"] = round(seconds_since_update, 1)
    state["latest"] = {
        "timestamp": state.get("timestamp"),
        "vibration_rms": state.get("vibration_rms"),
        "vib_peak": state.get("vib_peak"),
        "vib_x": state.get("vib_x"),
        "vib_y": state.get("vib_y"),
        "vib_z": state.get("vib_z"),
        "acceleration_x": state.get("acceleration_x"),
        "acceleration_y": state.get("acceleration_y"),
        "acceleration_z": state.get("acceleration_z"),
        "current": state.get("current"),
        "temperature": state.get("temperature"),
        "health_score": state.get("health_score"),
        "fault_label": state.get("fault_label"),
        "pump_status": state.get("pump_status"),
        "predicted_class": state.get("predicted_class"),
        "predicted_condition": state.get("predicted_condition"),
        "confidence": state.get("confidence"),
        "inference_location": state.get("inference_location"),
        "last_seen": state.get("last_seen"),
    }
    return state


def on_connect(client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
    logging.info("[MQTT] Connected to broker %s:%s", MQTT_BROKER, MQTT_PORT)
    client.subscribe(MQTT_TOPIC)
    logging.info("[MQTT] Subscribed to %s", MQTT_TOPIC)


def on_message(client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
    logging.info("[MQTT] Message received on %s", message.topic)
    try:
        payload = json.loads(message.payload.decode("utf-8"))
        validated = validate_payload(payload)
        insert_measurement(validated)
        _update_current_state(validated)
        logging.info("[DB] Measurement stored for %s", validated.get("pump_status"))
    except Exception as exc:  # pragma: no cover - defensive path
        logging.exception("[ERROR] Invalid MQTT payload: %s", exc)


def start_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.enable_logger(logging.getLogger("paho.mqtt"))
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    thread = threading.Thread(target=client.loop_forever, daemon=True, name="mqtt-listener")
    thread.start()
    return client
