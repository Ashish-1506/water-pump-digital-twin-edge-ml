#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "pump_model_data.h"

const char*     WIFI_SSID     = "YOUR_WIFI_SSID";
const char*     WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const IPAddress MQTT_BROKER(192, 168, 1, 100);
const uint16_t  MQTT_PORT     = 1883;
const char*     MQTT_TOPIC    = "pump/01/state";
const char*     DEVICE_ID     = "pump01";

const uint8_t ACS712_PIN  = 34;
const uint8_t DS18B20_PIN = 4;
const uint8_t RELAY_PIN   = 26;
const uint8_t BUZZER_PIN  = 27;
const uint8_t SDA_PIN     = 21;
const uint8_t SCL_PIN     = 22;

const uint8_t MPU6050_ADDR        = 0x68;
const float   MPU6050_ACCEL_SCALE = 16384.0f;

const float ACS_SENSITIVITY = 0.100f;
const float DIVIDER_RATIO   = 20.0f / 30.0f;
const float ACS_ZERO_VOLTS  = 2.364f;
const float ADC_VREF        = 3.3f;
const float ADC_RESOLUTION  = 4095.0f;

const int VIB_SAMPLES = 50;

const int   NUM_FEATURES       = 7;
const float SCALER_MEAN[7]     = {
  1.7104240657887526f,
  1.9692557257249814f,
 -0.8696258426088532f,
  0.012625154009252914f,
  0.421926726308118f,
  0.578289903962893f,
 35.08227240074864f
};
const float SCALER_SCALE[7]    = {
  0.7602102320546072f,
  0.9137547753890416f,
  0.23965520285981018f,
  0.6119033119808688f,
  0.6470106262838639f,
  0.3575565818046133f,
  1.963440183125283f
};

const int   NUM_CLASSES        = 4;
const char* CLASS_LABELS[4]    = {
  "Normal",
  "Dry Run",
  "Overheating",
  "Abnormal Operating Condition"
};

const float RELAY_TEMP_LIMIT    = 55.0f;
const float RELAY_CURRENT_LIMIT = 3.5f;
const float RELAY_VIB_LIMIT     = 4.0f;

WiFiClient       wifiClient;
PubSubClient     mqttClient(wifiClient);
OneWire          oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

unsigned long lastPublishMs = 0;
const unsigned long PUBLISH_INTERVAL_MS = 1000;

unsigned long nonNormalStartMs = 0;
bool faultActive = false;

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
}

void connectMQTT() {
  while (!mqttClient.connected()) {
    if (!mqttClient.connect(DEVICE_ID)) delay(2000);
  }
}

void configureMPU6050() {
  Wire.beginTransmission(MPU6050_ADDR); Wire.write(0x6B); Wire.write(0x00); Wire.endTransmission(true);
  Wire.beginTransmission(MPU6050_ADDR); Wire.write(0x1C); Wire.write(0x00); Wire.endTransmission(true);
  Wire.beginTransmission(MPU6050_ADDR); Wire.write(0x1A); Wire.write(0x00); Wire.endTransmission(true);
}

bool readMPU6050Raw(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU6050_ADDR, (uint8_t)6, (uint8_t)true) != 6) return false;
  ax = (int16_t)((Wire.read() << 8) | Wire.read()) / MPU6050_ACCEL_SCALE;
  ay = (int16_t)((Wire.read() << 8) | Wire.read()) / MPU6050_ACCEL_SCALE;
  az = (int16_t)((Wire.read() << 8) | Wire.read()) / MPU6050_ACCEL_SCALE;
  return true;
}

bool extractVibrationFeatures(float &vib_rms, float &vib_peak,
                              float &vib_x,  float &vib_y, float &vib_z) {
  float sumSq = 0.0f, peak = 0.0f, lx = 0, ly = 0, lz = 0;
  bool mpuOk = true;
  for (int i = 0; i < VIB_SAMPLES; i++) {
    float ax, ay, az;
    if (!readMPU6050Raw(ax, ay, az)) {
      mpuOk = false;
      ax = 0; ay = 0; az = 0;
    }
    float mag = sqrtf(ax*ax + ay*ay + az*az);
    sumSq += mag * mag;
    if (mag > peak) peak = mag;
    lx = ax; ly = ay; lz = az;
    delay(5);
  }
  vib_rms  = sqrtf(sumSq / VIB_SAMPLES);
  vib_peak = peak;
  vib_x = lx; vib_y = ly; vib_z = lz;
  return mpuOk;
}

float readCurrent() {
  long sumADC = 0;
  for (int i = 0; i < 30; i++) { sumADC += analogRead(ACS712_PIN); delayMicroseconds(100); }
  float pinV  = ((float)sumADC / 30.0f / ADC_RESOLUTION) * ADC_VREF;
  float sensV = pinV / DIVIDER_RATIO;
  float curr  = fabsf(sensV - ACS_ZERO_VOLTS) / ACS_SENSITIVITY;
  if (curr < 0.0f || curr > 20.0f) return 0.0f;
  return curr;
}


const float DEFAULT_TEMPERATURE = 29.8f;

float readTemperature(bool &isConnected) {
  ds18b20.requestTemperatures();
  float t = ds18b20.getTempCByIndex(0);

  if (t == DEVICE_DISCONNECTED_C ||
      !isfinite(t) ||
      t <= -55.0f ||
      t >= 125.0f ||
      t < 0.0f ||
      t > 80.0f) {
    isConnected = false;
    return DEFAULT_TEMPERATURE;
  }

  isConnected = true;
  return t;
}


int runInference(const float raw[7], float &confidence) {
  float standardized[NUM_FEATURES];
  for (int i = 0; i < NUM_FEATURES; i++) {
    standardized[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
  }
  float probs[NUM_CLASSES] = {0.0f};
  PumpML::predict_probabilities(standardized, probs);

  int bestC = 0;
  float bestP = probs[0];
  for (int c = 1; c < NUM_CLASSES; c++) {
    if (probs[c] > bestP) {
      bestP = probs[c];
      bestC = c;
    }
  }
  confidence = bestP;
  return bestC;
}

bool shouldActivateRelay(float temp, float current, float vib_rms) {
  return (temp > RELAY_TEMP_LIMIT || current > RELAY_CURRENT_LIMIT || vib_rms > RELAY_VIB_LIMIT);
}

void publishState() {
  float vib_rms, vib_peak, vib_x, vib_y, vib_z;
  bool  mpuConnected  = extractVibrationFeatures(vib_rms, vib_peak, vib_x, vib_y, vib_z);
  float current       = readCurrent();
  bool  tempConnected = false;
  float temperature   = readTemperature(tempConnected);

  bool sensorFault = false;
  char disconnectedSensor[64] = "";
  if (!tempConnected) {
    sensorFault = true;
    strncpy(disconnectedSensor, "DS18B20 Temperature Sensor", sizeof(disconnectedSensor));
  } else if (!mpuConnected) {
    sensorFault = true;
    strncpy(disconnectedSensor, "MPU-6050 Accelerometer", sizeof(disconnectedSensor));
  }

  float raw[7] = { vib_rms, vib_peak, vib_x, vib_y, vib_z, current, (tempConnected ? temperature : 25.0f) };
  float confidence = 0.0f;
  int   predClass  = runInference(raw, confidence);

  if (predClass == 2 && temperature <= 37.0f) {
    predClass = 0;
  }

  bool hardTrip = shouldActivateRelay(temperature, current, vib_rms);
  bool isNotNormal = (predClass != 0) || hardTrip;

  unsigned long now = millis();
  if (isNotNormal) {
    if (nonNormalStartMs == 0) {
      nonNormalStartMs = now;
    } else if (now - nonNormalStartMs >= 5000) {
      faultActive = true;
    }
  } else {
    nonNormalStartMs = 0;
    faultActive = false;
  }

  if (faultActive){
    digitalWrite(RELAY_PIN, LOW);  
    digitalWrite(BUZZER_PIN, HIGH);
  } else {
    digitalWrite(RELAY_PIN, HIGH); 
    digitalWrite(BUZZER_PIN, LOW); 
  }

  const char* pumpStatus = faultActive ? "OFF" : "ON";

  float healthScore = constrain(
    100.0f - (temperature - 35.0f)*1.2f - current*30.0f - vib_rms*12.0f, 0.0f, 100.0f);

  const char* faultLabel = hardTrip ? "OVERLOAD" : CLASS_LABELS[predClass];

  char payload[700];
  snprintf(payload, sizeof(payload),
    "{"
    "\"device_id\":\"%s\","
    "\"timestamp\":%lu,"
    "\"vib_rms\":%.4f,"
    "\"vib_peak\":%.4f,"
    "\"vib_x\":%.4f,"
    "\"vib_y\":%.4f,"
    "\"vib_z\":%.4f,"
    "\"vibration_rms\":%.4f,"
    "\"acceleration_x\":%.4f,"
    "\"acceleration_y\":%.4f,"
    "\"acceleration_z\":%.4f,"
    "\"current\":%.4f,"
    "\"temperature\":%.2f,"
    "\"health_score\":%.1f,"
    "\"fault_label\":\"%s\","
    "\"pump_status\":\"%s\","
    "\"predicted_class\":%d,"
    "\"predicted_condition\":\"%s\","
    "\"confidence\":%.4f,"
    "\"inference_location\":\"ESP32\","
    "\"sensor_fault\":%s,"
    "\"disconnected_sensor\":\"%s\""
    "}",
    DEVICE_ID, (unsigned long)millis(),
    vib_rms, vib_peak, vib_x, vib_y, vib_z,
    vib_rms, vib_x, vib_y, vib_z,
    current, temperature,
    healthScore, faultLabel, pumpStatus,
    predClass, CLASS_LABELS[predClass], confidence,
    sensorFault ? "true" : "false", disconnectedSensor);

  mqttClient.publish(MQTT_TOPIC, payload);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Wire.begin(SDA_PIN, SCL_PIN);
  configureMPU6050();
  ds18b20.begin();

  connectWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setBufferSize(700);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqttClient.connected())        connectMQTT();
  mqttClient.loop();

  unsigned long now = millis();
  if (now - lastPublishMs >= PUBLISH_INTERVAL_MS) {
    lastPublishMs = now;
    publishState();
  }
}
