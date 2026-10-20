"""
serial_logger.py
================
High-reliability serial acquisition client for the ESP32 Water Pump sensor rig.

Features:
- Configurable COM port, baud rate, pump_id, and condition label.
- Supports the exact 9-column training CSV format:
    timestamp,vib_rms,vib_peak,vib_x,vib_y,vib_z,current,temperature,label
- Communicates session trigger to ESP32 firmware (sends label digit).
- Detects session completion marker: `# SESSION COMPLETE - Label: X`.
- Saves per-session CSV files (e.g., normal_session_YYYYMMDD_HHMMSS.csv) and metadata JSON.
- Auto-reconnects on transient serial disconnections.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None  # Handled gracefully with informative error message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SerialLogger")

# Label mappings
LABEL_TO_NAME = {
    0: "NORMAL",
    1: "DRY_RUN",
    2: "OVERLOAD",
    3: "BEARING_FAULT",
    4: "OVERHEATING",
}

NAME_TO_LABEL = {
    "NORMAL": 0,
    "0": 0,
    "DRY_RUN": 1,
    "1": 1,
    "OVERLOAD": 2,
    "2": 2,
    "BEARING_FAULT": 3,
    "HIGH_VIBRATION": 3,
    "3": 3,
    "OVERHEATING": 4,
    "4": 4,
}

VALID_LABELS = list(NAME_TO_LABEL.keys())

# Standard CSV Headers
RAW_CSV_HEADERS = [
    "timestamp",
    "vib_rms",
    "vib_peak",
    "vib_x",
    "vib_y",
    "vib_z",
    "current",
    "temperature",
    "label",
]


def load_calibration_config(config_path: Path) -> Dict[str, Any]:
    """Loads sensor calibration parameters from config file."""
    if not config_path.exists():
        logger.warning("Calibration file %s not found. Using defaults.", config_path)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_and_validate_record(
    line: str,
    calib_cfg: Optional[Dict[str, Any]] = None,
    default_label: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parses a single line emitted by firmware.
    Primary format (9 columns):
      timestamp,vib_rms,vib_peak,vib_x,vib_y,vib_z,current,temperature,label
    Returns parsed dictionary or None if header, comment, or malformed.
    """
    cleaned = line.strip()
    if not cleaned or cleaned.startswith("#"):
        return None

    # Ignore header line if emitted
    if cleaned.lower().startswith("timestamp,") or "vib_rms" in cleaned:
        return None

    parts = [p.strip() for p in cleaned.split(",")]

    # 9-column format
    if len(parts) >= 9:
        try:
            ts = int(parts[0])
            vib_rms = float(parts[1])
            vib_peak = float(parts[2])
            vib_x = float(parts[3])
            vib_y = float(parts[4])
            vib_z = float(parts[5])
            current = float(parts[6])
            temp = float(parts[7])
            label = int(parts[8])

            # Validation bounds
            if not (0.0 <= vib_rms <= 30.0):
                return None
            if not (0.0 <= vib_peak <= 30.0):
                return None
            if not (-20.0 <= vib_x <= 20.0 and -20.0 <= vib_y <= 20.0 and -20.0 <= vib_z <= 20.0):
                return None
            if not (0.0 <= current <= 50.0):
                return None
            if not (-20.0 <= temp <= 125.0):
                return None
            if not (0 <= label <= 4):
                return None

            return {
                "timestamp": ts,
                "vib_rms": round(vib_rms, 4),
                "vib_peak": round(vib_peak, 4),
                "vib_x": round(vib_x, 4),
                "vib_y": round(vib_y, 4),
                "vib_z": round(vib_z, 4),
                "current": round(current, 4),
                "temperature": round(temp, 2),
                "label": label,
            }
        except (ValueError, IndexError):
            return None

    # Legacy 6-column fallback: timestamp_ms,vib_magnitude,current_adc,current_A,temperature_C,calibrated_flag
    if len(parts) >= 5:
        try:
            ts = int(parts[0])
            vib = float(parts[1])
            curr_str = parts[3].upper()
            curr = 0.0 if curr_str in ("UNCALIBRATED", "NAN", "") else float(parts[3])
            temp = float(parts[4])
            lbl = default_label if default_label is not None else 0

            return {
                "timestamp": ts,
                "vib_rms": round(vib, 4),
                "vib_peak": round(vib, 4),
                "vib_x": 0.0,
                "vib_y": 0.0,
                "vib_z": round(vib, 4),
                "current": round(curr, 4),
                "temperature": round(temp, 2),
                "label": lbl,
            }
        except (ValueError, IndexError):
            return None

    return None


def run_logger(
    port: str,
    baud: int,
    pump_id: str,
    condition_label: str,
    notes: str,
    duration_sec: Optional[float],
    out_dir: Path,
    calib_config_path: Path,
) -> None:
    """Acquires and logs serial telemetry into session CSV and metadata files."""
    if serial is None:
        logger.error(
            "pyserial is not installed! Run `pip install -r requirements.txt` before proceeding."
        )
        sys.exit(1)

    norm_label_str = condition_label.upper()
    if norm_label_str not in NAME_TO_LABEL:
        raise ValueError(
            f"Invalid condition label '{condition_label}'. Must be one of {list(NAME_TO_LABEL.keys())}"
        )

    int_label = NAME_TO_LABEL[norm_label_str]
    label_name = LABEL_TO_NAME[int_label]

    out_dir.mkdir(parents=True, exist_ok=True)
    session_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_name = f"{label_name.lower()}_session_{session_ts}"
    csv_path = out_dir / f"{session_name}.csv"
    meta_path = out_dir / f"meta_{session_name}.json"

    calib_cfg = load_calibration_config(calib_config_path)

    logger.info("==================================================")
    logger.info("Starting Data Collection Session")
    logger.info("  Session Name    : %s", session_name)
    logger.info("  Pump ID         : %s", pump_id)
    logger.info("  Condition Label : %s (ID: %d)", label_name, int_label)
    logger.info("  Target Port     : %s @ %d baud", port, baud)
    logger.info("  Target CSV      : %s", csv_path)
    logger.info("==================================================")

    session_start_iso = datetime.now(timezone.utc).isoformat()
    total_records = 0
    malformed_records = 0
    disconnect_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RAW_CSV_HEADERS)
        writer.writeheader()
        csv_file.flush()

        start_time = time.time()
        ser: Optional[serial.Serial] = None

        try:
            while True:
                # Check duration limit
                elapsed = time.time() - start_time
                if duration_sec and elapsed >= duration_sec:
                    logger.info("Configured duration (%.1fs) reached. Stopping.", duration_sec)
                    break

                # Ensure connection
                if ser is None or not ser.is_open:
                    try:
                        logger.info("Connecting to %s...", port)
                        ser = serial.Serial(port, baud, timeout=1.0)
                        ser.reset_input_buffer()
                        time.sleep(1.5)  # Allow ESP32 reboot/settle
                        
                        # Trigger session on ESP32 by sending label digit
                        trigger_cmd = f"{int_label}\n"
                        ser.write(trigger_cmd.encode("utf-8"))
                        logger.info("Sent trigger command '%d' to ESP32.", int_label)
                    except (serial.SerialException, OSError) as e:
                        disconnect_count += 1
                        logger.warning("Port unavailable (%s). Retrying in 2 seconds...", e)
                        time.sleep(2.0)
                        continue

                # Read line
                try:
                    raw_bytes = ser.readline()
                except (serial.SerialException, OSError) as e:
                    logger.warning("Serial read error: %s. Reconnecting...", e)
                    disconnect_count += 1
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                    time.sleep(1.0)
                    continue

                if not raw_bytes:
                    continue

                try:
                    line_str = raw_bytes.decode("utf-8", errors="replace").strip()
                except Exception:
                    malformed_records += 1
                    continue

                if not line_str:
                    continue

                # Check for session completion message from ESP32
                if "# SESSION COMPLETE" in line_str:
                    logger.info("ESP32 reported session complete: %s", line_str)
                    break

                parsed = parse_and_validate_record(line_str, calib_cfg, default_label=int_label)
                if parsed is None:
                    # Ignore comment lines and header lines silently
                    if not line_str.startswith("#") and not line_str.lower().startswith("timestamp"):
                        malformed_records += 1
                        if malformed_records <= 5:
                            logger.warning("Rejected line sample [%d]: %r", malformed_records, line_str)
                    continue

                writer.writerow(parsed)
                total_records += 1

                if total_records % 20 == 0:
                    csv_file.flush()
                    logger.info(
                        "Logged %d rows | vib_rms=%.4f | curr=%.4f A | temp=%.2f C | label=%d",
                        total_records,
                        parsed["vib_rms"],
                        parsed["current"],
                        parsed["temperature"],
                        parsed["label"],
                    )

        except KeyboardInterrupt:
            logger.info("Session stopped cleanly by user interrupt (Ctrl+C).")
        finally:
            if ser is not None and ser.is_open:
                ser.close()
            csv_file.flush()

    session_end_iso = datetime.now(timezone.utc).isoformat()
    duration_actual = time.time() - start_time

    # Save metadata JSON
    metadata = {
        "session_name": session_name,
        "pump_id": pump_id,
        "condition_label": label_name,
        "label_id": int_label,
        "notes": notes,
        "start_time_utc": session_start_iso,
        "end_time_utc": session_end_iso,
        "duration_seconds": round(duration_actual, 2),
        "total_records": total_records,
        "malformed_records": malformed_records,
        "disconnect_count": disconnect_count,
        "port": port,
        "baud": baud,
        "csv_filename": csv_path.name,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("==================================================")
    logger.info("Session Finished Summary:")
    logger.info("  Total valid records logged : %d", total_records)
    logger.info("  Malformed lines rejected   : %d", malformed_records)
    logger.info("  Disconnects encountered    : %d", disconnect_count)
    logger.info("  CSV file saved             : %s", csv_path)
    logger.info("  Metadata file saved        : %s", meta_path)
    logger.info("==================================================")


def list_serial_ports() -> None:
    """Lists available COM ports on the system."""
    if serial is None:
        print("pyserial is not installed.")
        return
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial COM ports found on this system.")
    else:
        print("Available Serial Ports:")
        for p in ports:
            print(f"  - {p.device}: {p.description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESP32 Cyber-Physical Water Pump Serial Data Acquisition Logger"
    )
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g. COM3, COM4)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--pump-id", type=str, default="PUMP_01", help="Pump identifier")
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        choices=VALID_LABELS,
        help="Condition label (NORMAL, DRY_RUN, OVERLOAD, BEARING_FAULT, OVERHEATING or 0-4)",
    )
    parser.add_argument("--notes", type=str, default="", help="Operator experimental observations")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Optional auto-stop duration in seconds (default: waits for ESP32 completion or Ctrl+C)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "raw"),
        help="Directory to save raw CSVs",
    )
    parser.add_argument(
        "--calib-config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "config" / "sensor_calibration.json"),
        help="Path to sensor calibration JSON",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available COM ports and exit",
    )

    args = parser.parse_args()

    if args.list_ports:
        list_serial_ports()
        sys.exit(0)

    if args.label is None:
        parser.error("--label is required unless using --list-ports")

    run_logger(
        port=args.port,
        baud=args.baud,
        pump_id=args.pump_id,
        condition_label=args.label,
        notes=args.notes,
        duration_sec=args.duration_sec,
        out_dir=Path(args.out_dir),
        calib_config_path=Path(args.calib_config),
    )


if __name__ == "__main__":
    main()
