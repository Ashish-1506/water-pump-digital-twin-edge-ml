import os
import sqlite3
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "pump.db")


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                vibration_rms REAL,
                acceleration_x REAL,
                acceleration_y REAL,
                acceleration_z REAL,
                current REAL,
                temperature REAL,
                health_score REAL,
                fault_label TEXT,
                pump_status TEXT,
                predicted_class INTEGER,
                predicted_condition TEXT,
                confidence REAL,
                inference_location TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fault_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                fault_label TEXT,
                health_score REAL,
                temperature REAL,
                current REAL,
                vibration_rms REAL,
                action TEXT
            )
            """
        )

        sensor_columns = [
            column[1] for column in connection.execute("PRAGMA table_info(sensor_history)").fetchall()
        ]
        for column_name in [
            "vibration_rms",
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "current",
            "temperature",
            "health_score",
            "fault_label",
            "pump_status",
            "predicted_class",
            "predicted_condition",
            "confidence",
            "inference_location",
        ]:
            if column_name not in sensor_columns:
                connection.execute(
                    f"ALTER TABLE sensor_history ADD COLUMN {column_name} REAL"
                )

        event_columns = [
            column[1] for column in connection.execute("PRAGMA table_info(fault_events)").fetchall()
        ]
        for column_name in [
            "timestamp",
            "fault_label",
            "health_score",
            "temperature",
            "current",
            "vibration_rms",
            "action",
        ]:
            if column_name not in event_columns:
                connection.execute(
                    f"ALTER TABLE fault_events ADD COLUMN {column_name} REAL"
                )

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensor_history_timestamp ON sensor_history(timestamp)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_fault_events_timestamp ON fault_events(timestamp)"
        )
        connection.commit()


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    keys = set(row.keys())
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "vibration_rms": row["vibration_rms"] if "vibration_rms" in keys else None,
        "acceleration_x": row["acceleration_x"] if "acceleration_x" in keys else None,
        "acceleration_y": row["acceleration_y"] if "acceleration_y" in keys else None,
        "acceleration_z": row["acceleration_z"] if "acceleration_z" in keys else None,
        "current": row["current"] if "current" in keys else None,
        "temperature": row["temperature"] if "temperature" in keys else None,
        "health_score": row["health_score"] if "health_score" in keys else None,
        "fault_label": row["fault_label"] if "fault_label" in keys else None,
        "pump_status": row["pump_status"] if "pump_status" in keys else None,
        "predicted_class": row["predicted_class"] if "predicted_class" in keys else None,
        "predicted_condition": row["predicted_condition"] if "predicted_condition" in keys else None,
        "confidence": row["confidence"] if "confidence" in keys else None,
        "inference_location": row["inference_location"] if "inference_location" in keys else None,
    }


def event_row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "fault_label": row["fault_label"],
        "health_score": row["health_score"],
        "temperature": row["temperature"],
        "current": row["current"],
        "vibration_rms": row["vibration_rms"],
        "action": row["action"],
    }


def insert_measurement(payload: Dict[str, Any]) -> Dict[str, Any]:
    reading = {
        "timestamp": float(payload.get("timestamp", 0.0) or 0.0),
        "vibration_rms": payload.get("vibration_rms"),
        "acceleration_x": payload.get("acceleration_x"),
        "acceleration_y": payload.get("acceleration_y"),
        "acceleration_z": payload.get("acceleration_z"),
        "current": payload.get("current"),
        "temperature": payload.get("temperature"),
        "health_score": payload.get("health_score"),
        "fault_label": str(payload.get("fault_label", "NORMAL") or "NORMAL").upper(),
        "pump_status": str(payload.get("pump_status", "UNKNOWN") or "UNKNOWN").upper(),
        "predicted_class": payload.get("predicted_class"),
        "predicted_condition": payload.get("predicted_condition"),
        "confidence": payload.get("confidence"),
        "inference_location": payload.get("inference_location"),
    }

    with get_connection() as connection:
        previous = connection.execute(
            "SELECT * FROM sensor_history ORDER BY id DESC LIMIT 1"
        ).fetchone()

        cursor = connection.execute(
            """
            INSERT INTO sensor_history (
                timestamp, vibration_rms, acceleration_x, acceleration_y, acceleration_z,
                current, temperature, health_score, fault_label, pump_status,
                predicted_class, predicted_condition, confidence, inference_location
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reading["timestamp"],
                reading["vibration_rms"],
                reading["acceleration_x"],
                reading["acceleration_y"],
                reading["acceleration_z"],
                reading["current"],
                reading["temperature"],
                reading["health_score"],
                reading["fault_label"],
                reading["pump_status"],
                reading["predicted_class"],
                reading["predicted_condition"],
                reading["confidence"],
                reading["inference_location"],
            ),
        )
        connection.commit()

        current_fault = reading["fault_label"]
        current_status = reading["pump_status"]
        previous_fault = (previous["fault_label"] if previous else "NORMAL") or "NORMAL"
        previous_status = (previous["pump_status"] if previous else "UNKNOWN") or "UNKNOWN"

        fault_changed = (
            current_fault.upper() not in ("", "NORMAL")
            and (previous is None or current_fault.upper() != previous_fault.upper())
        )
        status_changed = previous is not None and current_status.upper() != previous_status.upper()

        if fault_changed or status_changed:
            action = "RELAY_OFF" if current_status.upper() == "OFF" else "STATE_CHANGE"
            connection.execute(
                """
                INSERT INTO fault_events (
                    timestamp, fault_label, health_score, temperature,
                    current, vibration_rms, action
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading["timestamp"],
                    current_fault,
                    reading["health_score"],
                    reading["temperature"],
                    reading["current"],
                    reading["vibration_rms"],
                    action,
                ),
            )
            connection.commit()

        return reading


def get_latest_state() -> Optional[Dict[str, Any]]:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM sensor_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row_to_dict(row)


def get_history(limit: int = 60, start: Optional[float] = None, end: Optional[float] = None) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit or 60), 500))
    query = "SELECT * FROM sensor_history WHERE 1=1"
    params: List[Any] = []
    if start is not None:
        query += " AND timestamp >= ?"
        params.append(float(start))
    if end is not None:
        query += " AND timestamp <= ?"
        params.append(float(end))
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [row_to_dict(row) for row in reversed(rows)]


def get_recent_events(limit: int = 20) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit or 20), 100))
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM fault_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [event_row_to_dict(row) for row in rows]


init_database()
