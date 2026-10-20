#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define MPU_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 22
#define TEMP_PIN 4
#define ACS_PIN 34

OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

// --- ACS712-20A & Voltage Divider Parameters ---
const float ACS_SENSITIVITY = 0.100;     // 100 mV/A for 20A module
const float DIVIDER_RATIO   = 20.0 / 30.0; // 2/3 ratio (10k/20k divider)
const float ACS_ZERO_VOLTS  = 2.364;     // Reconstructed baseline (1.576V / (2/3))

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission();

  tempSensor.begin();
  analogReadResolution(12);

  Serial.println("System Initialized with ACS712-20A & Protection Diode");
}

void readMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6);

  if (Wire.available() == 6) {
    int16_t rawX = (Wire.read() << 8) | Wire.read();
    int16_t rawY = (Wire.read() << 8) | Wire.read();
    int16_t rawZ = (Wire.read() << 8) | Wire.read();

    float ax = rawX / 16384.0;
    float ay = rawY / 16384.0;
    float az = rawZ / 16384.0;
    float magnitude = sqrt(ax * ax + ay * ay + az * az);

    Serial.print("VIBRATION | X=");
    Serial.print(ax, 3);
    Serial.print("g Y=");
    Serial.print(ay, 3);
    Serial.print("g Z=");
    Serial.print(az, 3);
    Serial.print("g MAG=");
    Serial.print(magnitude, 3);
    Serial.println("g");
  }
}

void readTemperature() {
  tempSensor.requestTemperatures();
  float temperature = tempSensor.getTempCByIndex(0);
  Serial.print("TEMPERATURE | ");
  if (temperature == DEVICE_DISCONNECTED_C) {
    Serial.println("ERROR");
  } else {
    Serial.print(temperature, 2);
    Serial.println(" C");
  }
}

void readCurrent() {
  // Take 30 samples to average out ADC noise
  long sumADC = 0;
  for (int i = 0; i < 30; i++) {
    sumADC += analogRead(ACS_PIN);
    delayMicroseconds(100);
  }
  float avgADC = (float)sumADC / 30.0;

  // Convert ADC count to voltage at GPIO 34
  float pinVoltage = (avgADC / 4095.0) * 3.3;

  // Reconstruct actual ACS712 module output before the voltage divider
  float sensorVoltage = pinVoltage / DIVIDER_RATIO;

  // Calculate current in Amperes
  float current_A = abs(sensorVoltage - ACS_ZERO_VOLTS) / ACS_SENSITIVITY;

  Serial.print("ACS712 | ADC=");
  Serial.print((int)avgADC);
  Serial.print(" | Sensor_V=");
  Serial.print(sensorVoltage, 3);
  Serial.print(" V | Current=");
  Serial.print(current_A, 2);
  Serial.println(" A");
}

void loop() {
  readMPU6050();
  readTemperature();
  readCurrent();
  Serial.println("------------------------------------");
  delay(500);
}