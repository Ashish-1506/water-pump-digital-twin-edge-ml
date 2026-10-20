# Industrial Water Pump Digital Twin & Edge Machine Learning

An end-to-end Industrial Internet of Things (IIoT) system for real-time condition monitoring, anomaly detection, edge fault classification, and digital twin virtualization of industrial water pumps.

The system performs **edge inference directly on an ESP32 microcontroller** using a trained Multi-Layer Perceptron (MLP) neural network, streams high-frequency telemetry and inference outcomes over MQTT, persists historical records in SQLite, and visualizes live operations on a modern web-based digital twin dashboard.

---

## Authors & Copyright

**Copyright (c) 2026 Ashish and Sumit. All Rights Reserved.**  
Developed by **Ashish** & **Sumit** for Industrial IoT & Cyber-Physical Systems Monitoring.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       PHYSICAL PUMP & SENSORS         |
                      |  - MPU-6050 (3-Axis Vibration)       |
                      |  - ACS712-20A (Current Sensor)        |
                      |  - DS18B20 (Digital Temperature)      |
                      |  - Active Buzzer & Safety Relay       |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       ESP32 EDGE COMPUTING NODE       |
                      |  1. Windowed Sample Acquisition       |
                      |  2. 7-Feature Vector Extraction       |
                      |  3. StandardScaler Normalization       |
                      |  4. Embedded MLP Neural Inference     |
                      |  5. 5-Second Fault Relay & Buzzer     |
                      |  6. Sensor Disconnect Diagnostics     |
                      +-------------------+-------------------+
                                          |
                                          | Wi-Fi / MQTT JSON
                                          v
                      +---------------------------------------+
                      |        MOSQUITTO MQTT BROKER          |
                      |            (Port: 1883)               |
                      |       Topic: pump/01/state            |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         DIGITAL TWIN BACKEND          |
                      |         (Python Flask & SQLite)       |
                      |  - Schema validation & guardrails     |
                      |  - Time-series metrics logging        |
                      |  - Safety event audit trail           |
                      |  - REST APIs (/api/state, /api/history)
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         WEB DIGITAL TWIN UI           |
                      |  - Real-time gauge metrics            |
                      |  - Edge ML prediction & confidence    |
                      |  - Chart.js dynamic signal telemetry  |
                      |  - Full-screen sensor blackout modal  |
                      +---------------------------------------+
```

---

## Hardware Specifications & Pinout

| Peripheral / Sensor | Interface / Pin | ESP32 GPIO | Operating Voltage | Notes |
|:---|:---|:---|:---|:---|
| **MPU-6050** | I2C SDA | **GPIO 21** | 3.3V | 3-axis accelerometer ($\pm 2g$) |
| **MPU-6050** | I2C SCL | **GPIO 22** | 3.3V | Hardware I2C bus (0x68) |
| **ACS712-20A** | Analog OUT | **GPIO 34** | 5V (Sensor) | ADC1_CH6 (Input only) via 10k/20k voltage divider (2/3 ratio) |
| **DS18B20** | 1-Wire DQ | **GPIO 4** | 3.3V | Requires 4.7kΩ pull-up resistor to 3.3V |
| **Safety Relay** | Digital IN | **GPIO 26** | 5V | Active-low trigger: cuts pump power during sustained fault |
| **Active Buzzer** | Digital + | **GPIO 27** | 3.3V / 5V | Audible alarm activated on 5s fault persistence |

---

## Repository Structure

```
WaterPumpMonitoring/
├── ml_model_training/                # Complete ML Training & Evaluation Pipeline
│   ├── dataset/                      # Multi-session labeled CSV telemetry files
│   ├── model_output/                 # Trained models, TFLite weights, scalers & figures
│   │   ├── pump_condition_mlp.tflite # Optimized 4-class TFLite model
│   │   ├── standard_scaler.pkl       # Fitted scikit-learn feature scaler
│   │   ├── random_forest.pkl         # Baseline ensemble comparator
│   │   ├── metrics.json              # Full evaluation report (Precision/Recall/F1)
│   │   └── report_figures/           # Confusion matrices and ROC curves
│   ├── train_water_pump.py           # End-to-end training, validation & TFLite export
│   └── generate_architecture_diagram.py
│
├── ml_data_collection/               # Sensor Data Acquisition & Raw Logging Pipeline
│   ├── config/                       # Feature extraction & calibration parameters
│   ├── firmware/                     # Serial logger firmware for collecting training sets
│   │   └── sensor_logger/
│   ├── scripts/                      # Data ingestion and merge utilities
│   │   ├── serial_logger.py          # High-speed serial stream recorder
│   │   └── combine_datasets.py       # Session aggregator
│   ├── tests/                        # Automated unit tests for packet parsing
│   └── requirements.txt              # ML dependencies (TensorFlow, scikit-learn, etc.)
│
├── water-pump-digital-twin/          # Digital Twin Web Application & ESP32 Firmware
│   ├── ESP32_MQTT_Publisher/         # Arduino IDE Edge Inference Sketch
│   │   ├── ESP32_MQTT_Publisher.ino  # Windowing, ML inference, MQTT publisher & safety
│   │   └── pump_model_data.h         # Embedded C++ neural network weights & scaler
│   ├── static/                       # UI styling and client-side logic
│   │   ├── app.js                    # Live polling, Chart.js traces, blackout modal
│   │   └── style.css                 # Dark industrial SCADA aesthetic
│   ├── templates/
│   │   └── index.html                # Digital twin dashboard layout
│   ├── app.py                        # Flask server & REST API
│   ├── database.py                   # SQLite schema, migrations & persistence
│   ├── mqtt_client.py                # MQTT subscriber, validator & state cache
│   ├── mqtt_debug_subscriber.py      # Diagnostic CLI subscriber
│   ├── requirements.txt              # Flask & Paho-MQTT dependencies
│   └── README.md                     # Digital twin specific guide
│
├── Test_All_sensors/                 # Hardware diagnostics sketch
│   └── Test_All_sensors.ino          # Step-by-step verification of MPU6050, ACS712 & DS18B20
│
├── mosquitto.conf                    # MQTT broker configuration (LAN listeners enabled)
├── .gitignore                        # Comprehensive rules excluding build & cache files
└── README.md                         # Project documentation
```

---

## Machine Learning Pipeline

### 1. Feature Engineering
The ESP32 captures sensor windows (50 samples for vibration, 30 for current) and computes a 7-dimensional feature vector:
1. `vib_rms` — Root Mean Square acceleration across all 3 axes ($g$)
2. `vib_peak` — Peak acceleration amplitude ($g$)
3. `vib_x` — Mean X-axis acceleration component ($g$)
4. `vib_y` — Mean Y-axis acceleration component ($g$)
5. `vib_z` — Mean Z-axis acceleration component ($g$)
6. `current` — Effective AC load current ($A$)
7. `temperature` — Pump casing temperature ($^\circ\text{C}$)

### 2. Operating Condition Classes
| Class ID | Condition | Signature Description |
|:---|:---|:---|
| **0** | **Normal** | Balanced vibration ($<1.5g$), nominal current ($0.8 - 1.5A$), temperature $<37^\circ\text{C}$ |
| **1** | **Dry Run** | Extremely low current ($<0.4A$), light vibration, pump spinning without fluid resistance |
| **2** | **Overheating** | Elevated casing temperature ($>37^\circ\text{C}$), progressive thermal drift |
| **3** | **Abnormal** | High vibration RMS ($>2.0g$), severe mechanical imbalance or bearing wear |

### 3. Embedded Inference on ESP32
Rather than relying on large external runtimes, the trained MLP model weights and biases are compiled directly into [pump_model_data.h](file:///water-pump-digital-twin/ESP32_MQTT_Publisher/pump_model_data.h) using standard C++ array structures. Inference takes less than **3 milliseconds** per evaluation on the ESP32 Xtensa core.

---

## Safety Automation & Hardware Protection

1. **5-Second Persistence Timer**:
   When a non-normal operating state (`Dry Run`, `Overheating`, or `Abnormal`) is detected, the ESP32 initiates a 5-second countdown. If the fault condition persists continuously for 5 seconds:
   - The **Safety Relay** (GPIO 26) triggers (Active-Low) to isolate pump power.
   - The **Audible Buzzer** (GPIO 27) sounds continuously.
   - If the system recovers to `Normal`, the relay resets and the buzzer silences immediately.

2. **Sensor Disconnection Blackout**:
   If any sensor is unplugged or fails (e.g. DS18B20 disconnected reporting $-127^\circ\text{C}$):
   - The ESP32 flags `sensor_fault: true` with the specific sensor name.
   - The Digital Twin UI triggers an immediate full-screen blackout overlay blocking all stale readings.
   - An alert modal displays the faulty sensor name with an interactive **"Retry Connection"** button.

3. **Physical Domain Guardrail**:
   `Overheating` predictions are validated against physical temperature measurements; an overheating flag is only confirmed if temperature exceeds $37^\circ\text{C}$, preventing false positives.

---

## Quick Start Guide

### 1. Prerequisites
- **Python**: Version 3.10 or 3.11 installed.
- **Mosquitto MQTT Broker**: Installed on host machine.
- **Arduino IDE**: With ESP32 Board Support and libraries (`PubSubClient`, `OneWire`, `DallasTemperature`).

### 2. Launch the MQTT Broker
Run Mosquitto using the provided configuration file (enables LAN connectivity for the ESP32):
```powershell
mosquitto -c mosquitto.conf -v
```

### 3. Launch the Digital Twin Dashboard
Install dependencies and run the Flask server:
```powershell
cd water-pump-digital-twin
pip install -r requirements.txt
python app.py
```
Open **`http://localhost:5000`** in your browser.

### 4. Flash the ESP32 Firmware
1. Open `water-pump-digital-twin/ESP32_MQTT_Publisher/ESP32_MQTT_Publisher.ino` in Arduino IDE.
2. In lines 9–11, configure your local Wi-Fi and host computer IP:
   ```cpp
   const char*     WIFI_SSID     = "YOUR_WIFI_SSID";
   const char*     WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   const IPAddress MQTT_BROKER(192, 168, 1, 100); // Your computer's IPv4
   ```
3. Connect your ESP32 via USB and upload the sketch.
4. Watch the Serial Monitor (115200 baud) for connection confirmation and telemetry broadcasts.

---

## License & Copyright

**Copyright (c) 2026 Ashish and Sumit.**  
All rights reserved. Released under the MIT Open Source License.
