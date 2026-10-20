# Water Pump Digital Twin with Edge Machine Learning Inference

An enterprise-grade, real-time industrial digital twin system for predictive maintenance, condition monitoring, anomaly detection, and automated safety isolation of industrial water pumps. 

This module features **on-device Edge ML inference directly on an ESP32 microcontroller**, a high-throughput **MQTT messaging backbone**, a **Flask REST & SQLite telemetry backend**, and a modern **dark-mode SCADA digital twin web dashboard**.

---

## Authors & Copyright

**Copyright (c) 2026 Ashish and Sumit. All Rights Reserved.**  
Developed by **Ashish** & **Sumit** for Industrial IoT & Cyber-Physical Systems Monitoring.

---

## 1. System Architecture

```text
[ Physical Pump & Sensors ]
   ├── MPU-6050 (3-Axis Accelerometer / Vibration)
   ├── ACS712-20A (Hall-Effect Current Sensor via 2/3 Voltage Divider)
   ├── DS18B20 (1-Wire High-Precision Digital Temperature Sensor)
   ├── Active-Low Safety Relay Module (Pump Power Interruption)
   └── Active Piezo Buzzer (Audible Fault Alarm)
              │
              ▼
[ ESP32 Edge Computing Node ]
   ├── 1. Windowed Acquisition (50 samples MPU-6050, 30 samples ACS712)
   ├── 2. 7-Feature Vector Extraction [vib_rms, vib_peak, vib_x, vib_y, vib_z, current, temp]
   ├── 3. On-Chip StandardScaler Normalization: z = (x - mean) / scale
   ├── 4. Embedded Multi-Layer Perceptron (MLP) Neural Network (pump_model_data.h)
   ├── 5. Physical Domain Thermal Guardrail (Overheating confirmed only if Temp > 37°C)
   ├── 6. 5-Second Persistence Fault Timer -> Activates Relay (LOW) & Buzzer (HIGH)
   └── 7. Sensor Diagnostic Routine (-127°C DS18B20 Disconnect Detection)
              │
              ▼ (Wi-Fi 802.11 b/g/n / MQTT JSON via PubSubClient)
[ Mosquitto MQTT Broker (Port 1883) ]
   └── Topic: pump/01/state
              │
              ▼ (Paho-MQTT Threaded Client)
[ Flask Backend & SQLite Database (pump.db) ]
   ├── Payload Validation & Type Coercion (_coerce_float with extended range)
   ├── Safety Audit Logging (Records fault transitions & emergency shutdowns)
   ├── Circular Time-Series Buffer for Dynamic Telemetry
   └── REST API Endpoints (/api/state, /api/history, /api/alerts)
              │
              ▼ (HTTP Polling / Dynamic DOM Updates)
[ SCADA Digital Twin Web Dashboard ]
   ├── Real-Time Radial Gauges (Vibration RMS, Current Draw, Casing Temperature)
   ├── Edge ML Prediction Panel (Condition Badge, Class Probabilities, Confidence)
   ├── Chart.js Dynamic Historical Waveform Traces (60-reading rolling window)
   ├── System Safety & Relay Status Card
   └── Full-Screen Pitch-Black Sensor Disconnect Overlay with Retry Action
```

---

## 2. Project Directory Structure

```text
water-pump-digital-twin/
├── ESP32_MQTT_Publisher/
│   ├── ESP32_MQTT_Publisher.ino      # Main ESP32 edge firmware (Windowing, ML, MQTT, Relay)
│   └── pump_model_data.h             # Embedded C++ neural network weights, biases & scaler
│
├── static/
│   ├── app.js                        # Digital twin dashboard polling, charts & modal logic
│   └── style.css                     # Dark industrial theme, glassmorphism, responsive grid
│
├── templates/
│   └── index.html                    # HTML5 semantic dashboard layout & blackout overlay
│
├── app.py                            # Flask web application & REST API controller
├── database.py                       # SQLite database manager (time-series & audit logs)
├── mqtt_client.py                    # MQTT subscriber thread, payload validator & cache
├── mqtt_debug_subscriber.py          # Diagnostic CLI utility to monitor raw MQTT packets
├── requirements.txt                  # Python dependencies (Flask, Paho-MQTT)
└── README.md                         # Digital twin documentation
```

### Relationship to Main Repository
- **`ml_model_training/`**: Contains the training pipeline that generates `pump_condition_mlp.tflite` and the model parameters embedded into `pump_model_data.h`.
- **`ml_data_collection/`**: Contains firmware and serial recording scripts used to harvest labeled sensor data from physical pump runs.
- **`Test_All_sensors/`**: Standalone sketch for hardware diagnostic tests before flashing the digital twin firmware.

---

## 3. Hardware Pinout & Wiring Specifications

| Peripheral | Sensor Pin | ESP32 GPIO | Operating Voltage | Interface Type | Wiring & Circuit Details |
|:---|:---|:---|:---|:---|:---|
| **MPU-6050** | SDA | **GPIO 21** | 3.3V | I2C Data | Connect directly to ESP32 hardware I2C SDA |
| **MPU-6050** | SCL | **GPIO 22** | 3.3V | I2C Clock | Connect directly to ESP32 hardware I2C SCL (Address: `0x68`) |
| **ACS712-20A** | OUT | **GPIO 34** | 5.0V (VCC) | Analog In (ADC1) | **Must use 10kΩ / 20kΩ voltage divider** (2/3 ratio) to step down sensor 0–5V to ESP32 ADC safe range (0–3.3V) |
| **DS18B20** | DQ | **GPIO 4** | 3.3V | 1-Wire Digital | Connect **4.7kΩ pull-up resistor** between DQ and 3.3V rail |
| **Safety Relay** | IN | **GPIO 26** | 5.0V | Digital Output | Active-Low relay module. LOW = Relay Energized (Cut Pump Power) / HIGH = Normal Pump Operation |
| **Active Buzzer**| + | **GPIO 27** | 3.3V / 5.0V | Digital Output | HIGH = Alarm On / LOW = Alarm Off |

---

## 4. Edge Machine Learning Pipeline

### Feature Extraction
The ESP32 samples 50 readings from the MPU-6050 accelerometer and 30 readings from the ACS712 current sensor over a continuous window. It calculates the following 7 features:
1. **`vib_rms`**: Root-mean-square acceleration across $X, Y, Z$ axes ($g$):
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (a_{x,i}^2 + a_{y,i}^2 + a_{z,i}^2)}$$
2. **`vib_peak`**: Peak acceleration magnitude observed in the window ($g$).
3. **`vib_x`**: Arithmetic mean of X-axis acceleration ($g$).
4. **`vib_y`**: Arithmetic mean of Y-axis acceleration ($g$).
5. **`vib_z`**: Arithmetic mean of Z-axis acceleration ($g$).
6. **`current`**: Mean effective current draw ($A$).
7. **`temperature`**: Instantaneous casing temperature from DS18B20 ($^\circ\text{C}$).

### Normalization
Features are normalized using on-chip StandardScaler parameters fitted during training:
$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

### Neural Network Forward Pass
The model embedded in `pump_model_data.h` is an optimized Multi-Layer Perceptron (MLP):
- **Input Layer**: 7 normalized features
- **Hidden Layer 1**: Dense (ReLU activation)
- **Hidden Layer 2**: Dense (ReLU activation)
- **Output Layer**: 4 neurons with Softmax activation function:
  $$P(\text{class } k) = \frac{e^{z_k}}{\sum_{j=1}^{4} e^{z_j}}$$

### Class Definitions
| Class ID | Operating Condition | Diagnostic Indicators |
|:---:|:---|:---|
| **0** | **Normal** | Vibration $< 1.5g$, Current $0.8 - 1.5A$, Temperature $< 37.0^\circ\text{C}$ |
| **1** | **Dry Run** | Current $< 0.4A$, low vibration, motor spinning with no fluid load |
| **2** | **Overheating** | Temperature $> 37.0^\circ\text{C}$, thermal drift under continuous duty |
| **3** | **Abnormal Operating Condition** | Vibration $> 2.0g$, mechanical looseness or bearing degradation |

---

## 5. Safety Automation & Protection System

### 1. 5-Second Non-Normal Persistence Timer
If the predicted condition shifts to any non-normal state (`Dry Run`, `Overheating`, or `Abnormal`):
- A 5-second countdown initiates locally on the ESP32.
- If the fault persists for **5 uninterrupted seconds**, the ESP32 trips the **Safety Relay** (`digitalWrite(RELAY_PIN, LOW)`) and triggers the **Audible Buzzer** (`digitalWrite(BUZZER_PIN, HIGH)`).
- If the state returns to `Normal` before or after triggering, the relay automatically deactivates (`HIGH`) and the buzzer turns off (`LOW`).

### 2. Sensor Disconnect Diagnostics & Dashboard Blackout
- When the DS18B20 temperature probe is disconnected, the DallasTemperature library reports `-127.0°C`.
- `mqtt_client.py` accommodates this value (`minimum=-150`) and flags `"sensor_fault": true` with `"disconnected_sensor": "DS18B20 Temperature Sensor"`.
- The Digital Twin UI immediately triggers a pitch-black screen overlay (`#sensor-blackout-overlay`), obscuring all telemetry gauges to prevent decisions based on stale/invalid data.
- An alert modal displays the missing sensor name and an interactive **"Retry Connection"** button.

### 3. Thermal Guardrail
To eliminate false positives from mechanical shock or high-load transients, an **Overheating** classification is verified against real thermal data: it is only accepted if casing temperature is strictly greater than **37.0°C**; otherwise, it is clamped to `Normal`.

---

## 6. MQTT Telemetry Protocol

**Topic**: `pump/01/state`  
**QoS**: 0 or 1  
**Payload Example**:
```json
{
  "device_id": "pump01",
  "timestamp": 1729267200,
  "vib_rms": 1.12,
  "vib_peak": 1.28,
  "vib_x": -0.82,
  "vib_y": 0.06,
  "vib_z": 0.42,
  "vibration_rms": 1.12,
  "acceleration_x": -0.82,
  "acceleration_y": 0.06,
  "acceleration_z": 0.42,
  "current": 0.38,
  "temperature": 33.7,
  "health_score": 95.0,
  "fault_label": "NORMAL",
  "pump_status": "ON",
  "predicted_class": 0,
  "predicted_condition": "Normal",
  "confidence": 0.96,
  "inference_location": "ESP32",
  "sensor_fault": false,
  "disconnected_sensor": ""
}
```

---

## 7. REST API Endpoints

| Endpoint | Method | Response Description |
|:---|:---:|:---|
| `/` | `GET` | Serves the HTML5 Digital Twin dashboard |
| `/api/state` | `GET` | Returns latest operational snapshot, online status, ML classifications, and sensor disconnect status |
| `/api/history?limit=60` | `GET` | Returns rolling time-series telemetry (temperature, current, vibration) for Chart.js |
| `/api/alerts?limit=20` | `GET` | Returns logged safety events, fault transitions, and emergency relay trips |

---

## 8. Installation & Execution Guide

### Prerequisites
1. **Python 3.10+ / 3.11+**
2. **Mosquitto MQTT Broker** installed and in system PATH
3. **Arduino IDE 2.x** with ESP32 board support installed

### Step 1: Start Mosquitto MQTT Broker
From the repository root, start Mosquitto using the provided configuration:
```powershell
mosquitto -c mosquitto.conf -v
```
*(Using `-c mosquitto.conf` ensures `listener 1883` and `allow_anonymous true` are enabled so remote ESP32 boards on your LAN can connect).*

### Step 2: Start Flask Digital Twin Server
Open a terminal and execute:
```powershell
cd water-pump-digital-twin
pip install -r requirements.txt
python app.py
```
Open **`http://localhost:5000`** in your web browser.

### Step 3: Flash the ESP32 Firmware
1. Open `water-pump-digital-twin/ESP32_MQTT_Publisher/ESP32_MQTT_Publisher.ino` in Arduino IDE.
2. In the configuration section (lines 9–11), enter your Wi-Fi details and computer IP:
   ```cpp
   const char*     WIFI_SSID     = "YOUR_WIFI_SSID";
   const char*     WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   const IPAddress MQTT_BROKER(192, 168, 1, 100); // Your computer's IPv4 address (from 'ipconfig')
   ```
3. Connect your ESP32 via USB, select **ESP32 Dev Module**, and upload.
4. Open the Serial Monitor at **115200 baud** to verify Wi-Fi and MQTT connection.

---

## 9. Troubleshooting Guide

| Problem | Root Cause | Solution |
|:---|:---|:---|
| ESP32 fails to connect to MQTT (`rc=-2`) | Mosquitto broker is not reachable from LAN or listening on localhost only | Run broker with `mosquitto -c mosquitto.conf -v`. Verify laptop IP with `ipconfig` and ensure Windows Firewall allows port 1883. |
| Dashboard shows "DS18B20 Disconnected" and screen turns black | DS18B20 1-Wire pin is floating or sensor disconnected (returns `-127.0°C`) | Check 4.7kΩ pull-up resistor between GPIO 4 and 3.3V. Reconnect sensor and click **"Retry Connection"** on dashboard. |
| Relay triggers continuously | Relay polarity mismatch | The firmware uses Active-Low logic (`LOW = ON`, `HIGH = OFF`). If using an Active-High relay module, invert the `digitalWrite` calls. |
| MPU-6050 reading is 0.0g across all axes | I2C wiring issue or address conflict | Verify SDA -> GPIO 21 and SCL -> GPIO 22. Check that AD0 pin is grounded (for 0x68 address). |

---

## 10. License & Copyright

**Copyright (c) 2026 Ashish and Sumit.**  
All rights reserved. Released under the MIT Open Source License.
