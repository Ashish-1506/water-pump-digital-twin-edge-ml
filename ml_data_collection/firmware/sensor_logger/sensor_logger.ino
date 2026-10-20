/**
 * @file sensor_logger.ino
 * @brief High-reliability raw sensor data acquisition and training logger for ESP32.
 * 
 * Hardware Setup:
 *  - ESP32-WROOM-32 (30-pin)
 *  - MPU6050 Accelerometer: SDA -> GPIO 21, SCL -> GPIO 22 (I2C 0x68)
 *  - DS18B20 Temperature:   DATA -> GPIO 4 (4.7k pull-up to 3.3V)
 *  - ACS712 Current Sensor: OUT -> GPIO 34 (ADC1_CH6, analog input only)
 * 
 * CSV Column Structure emitted:
 *  timestamp,vib_rms,vib_peak,vib_x,vib_y,vib_z,current,temperature,label
 * 
 * Label Reference:
 *  0: Normal
 *  1: Dry Run
 *  2: Overload
 *  3: Bearing / Vibration Fault
 *  4: Overheating
 */

#include <Arduino.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// -----------------------------------------------------------------------------
// Pin and I2C Definitions
// -----------------------------------------------------------------------------
constexpr uint8_t SDA_PIN   = 21;
constexpr uint8_t SCL_PIN   = 22;
constexpr uint8_t MPU_ADDR  = 0x68;

constexpr uint8_t TEMP_PIN  = 4;   // OneWire data pin
constexpr uint8_t ACS_PIN   = 34;  // Analog input (ADC1, GPIO 34)

// -----------------------------------------------------------------------------
// Sensor Calibration Parameters (Configurable)
// -----------------------------------------------------------------------------
constexpr float MPU_ACCEL_SCALE   = 16384.0f; // MPU6050 +/- 2g scale factor

// --- ACS712-20A Calibration Parameters ---
const float ACS_SENSITIVITY = 0.100;       // 100 mV/A for ACS712-20A (was 0.185 for 5A)
const float DIVIDER_RATIO   = 20.0 / 30.0; // 2/3 scaling ratio (10k / (10k + 20k) divider)
const float ACS_ZERO_VOLTS  = 2.364;       // Calibrated zero-current baseline at the sensor pin
const float ADC_VREF        = 3.3;
const float ADC_RESOLUTION  = 4095.0;

// -----------------------------------------------------------------------------
// Hardware Interfaces
// -----------------------------------------------------------------------------
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

// Built-in MPU6050 Wire Interface (no external library required)
class MPU6050_WireClient {
public:
  bool isAvailable = false;

  void begin() {
    Wire.begin(SDA_PIN, SCL_PIN);
    
    // Wake up MPU-6050 (clear SLEEP bit in PWR_MGMT_1)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B);
    Wire.write(0x00);
    byte err = Wire.endTransmission();
    isAvailable = (err == 0);

    // Set +/- 2g accelerometer scale
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C);
    Wire.write(0x00);
    Wire.endTransmission();
  }

  void getMotion6(int16_t* ax, int16_t* ay, int16_t* az, int16_t* gx, int16_t* gy, int16_t* gz) {
    if (!isAvailable) {
      *ax = 0; *ay = 0; *az = 16384;
      if (gx) *gx = 0; if (gy) *gy = 0; if (gz) *gz = 0;
      return;
    }

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B); // Start at ACCEL_XOUT_H
    Wire.endTransmission(false);
    Wire.requestFrom(static_cast<uint8_t>(MPU_ADDR), static_cast<uint8_t>(14));

    if (Wire.available() >= 14) {
      *ax = (Wire.read() << 8) | Wire.read();
      *ay = (Wire.read() << 8) | Wire.read();
      *az = (Wire.read() << 8) | Wire.read();
      Wire.read(); Wire.read(); // Skip temperature bytes (0x41, 0x42)
      int16_t rx = (Wire.read() << 8) | Wire.read();
      int16_t ry = (Wire.read() << 8) | Wire.read();
      int16_t rz = (Wire.read() << 8) | Wire.read();
      if (gx) *gx = rx;
      if (gy) *gy = ry;
      if (gz) *gz = rz;
    } else {
      *ax = 0; *ay = 0; *az = 16384;
      if (gx) *gx = 0; if (gy) *gy = 0; if (gz) *gz = 0;
    }
  }
};

MPU6050_WireClient mpu;

// -----------------------------------------------------------------------------
// Sensor Helper Functions
// -----------------------------------------------------------------------------

float readCurrent() {
  // Take multiple samples to filter ADC noise on the 20A sensor
  long sumADC = 0;
  for (int i = 0; i < 30; i++) {
    sumADC += analogRead(ACS_PIN);
    delayMicroseconds(100);
  }
  float avgADC = (float)sumADC / 30.0;

  // 1. Calculate the voltage physically present at GPIO 34
  float pinVoltage = (avgADC / 4095.0) * 3.3;

  // 2. Reconstruct the actual ACS712 module output before the voltage divider
  float sensorVoltage = pinVoltage / DIVIDER_RATIO;

  // 3. Compute current in Amperes using the calibrated baseline & 100 mV/A sensitivity
  float current_A = abs(sensorVoltage - ACS_ZERO_VOLTS) / ACS_SENSITIVITY;

  return current_A;
}

float readTemperature() {
  tempSensor.requestTemperatures();
  float temp = tempSensor.getTempCByIndex(0);
  if (temp == DEVICE_DISCONNECTED_C || temp < -40.0f || temp > 125.0f) {
    return 25.0f; // Safe fallback reading if sensor disconnected
  }
  return temp;
}

// -----------------------------------------------------------------------------
// Data Logging Function
// -----------------------------------------------------------------------------

void logTrainingData(int label) {
  // Print header only once
  Serial.println("timestamp,vib_rms,vib_peak,vib_x,vib_y,vib_z,current,temperature,label");

  unsigned long startTime = millis();
  unsigned long duration = 180000; // 3 minutes

  while (millis() - startTime < duration) {

    // ── Vibration: collect 50 samples ──────────────────
    int16_t ax, ay, az, gx, gy, gz;
    float sumRMS = 0, peak = 0;
    float last_x = 0, last_y = 0, last_z = 0;

    for (int i = 0; i < 50; i++) {
      mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
      float x = ax / 16384.0;
      float y = ay / 16384.0;
      float z = az / 16384.0;
      float mag = sqrt(x*x + y*y + z*z);
      sumRMS += mag * mag;
      if (mag > peak) peak = mag;
      last_x = x; last_y = y; last_z = z;
      delay(5);
    }
    float vib_rms  = sqrt(sumRMS / 50.0);
    float vib_peak = peak;

    // ── Current ────────────────────────────────────────
    float current = readCurrent();

    // ── Temperature ────────────────────────────────────
    float temperature = readTemperature();

    // ── Print CSV row ──────────────────────────────────
    Serial.print(millis());        Serial.print(",");
    Serial.print(vib_rms,  4);    Serial.print(",");
    Serial.print(vib_peak, 4);    Serial.print(",");
    Serial.print(last_x,   4);    Serial.print(",");
    Serial.print(last_y,   4);    Serial.print(",");
    Serial.print(last_z,   4);    Serial.print(",");
    Serial.print(current,  4);    Serial.print(",");
    Serial.print(temperature, 2); Serial.print(",");
    Serial.println(label);

    delay(250); // ~500ms total per row (50 samples × 5ms + 250ms)
  }

  Serial.println("# SESSION COMPLETE - Label: " + String(label));
}

void printMenu() {
  Serial.println(F("\n--- ESP32 Water Pump ML Data Logger ---"));
  Serial.println(F("Send a label digit (0-4) to start a 3-minute session:"));
  Serial.println(F("  0 -> Normal"));
  Serial.println(F("  1 -> Dry Run"));
  Serial.println(F("  2 -> Overload"));
  Serial.println(F("  3 -> Bearing / Vibration Fault"));
  Serial.println(F("  4 -> Overheating"));
  Serial.println(F("Waiting for command..."));
}

// -----------------------------------------------------------------------------
// Setup & Loop
// -----------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  mpu.begin();
  tempSensor.begin();

  Serial.println(F("# SYSTEM: Cyber-Physical Water Pump Telemetry Logger"));
  Serial.print(F("# MPU6050_STATUS: "));
  Serial.println(mpu.isAvailable ? F("OK") : F("ERROR"));

  printMenu();
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.peek();
    if (c >= '0' && c <= '4') {
      int label = Serial.parseInt();
      Serial.print(F("# Starting 3-minute recording for label: "));
      Serial.println(label);
      logTrainingData(label);
      printMenu();
    } else {
      // Discard newline or unhandled characters
      Serial.read();
    }
  }
}
