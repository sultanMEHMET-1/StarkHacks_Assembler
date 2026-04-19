#include <Arduino.h>
#include <Wire.h>

// === CONFIGURATION — EDIT THESE ===
#define SERIAL_BAUD 115200
#define MPU_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 22
#define LOOP_INTERVAL_MS 20UL         // 50Hz
#define COMPLEMENTARY_ALPHA 0.98f     // Gyro weight in complementary filter
#define GYRO_CALIBRATION_SAMPLES 200
#define GYRO_CALIBRATION_DELAY_MS 10
#define GYRO_SENSITIVITY 131.0f       // LSB/(deg/s) at +/-250 deg/s range

const int FINGER_PINS[] = {13, 12, 14, 27, 26};  // thumb, index, middle, ring, pinky
const int NUM_FINGERS = sizeof(FINGER_PINS) / sizeof(FINGER_PINS[0]);
// === END CONFIGURATION ===

// IMU state
int16_t AcX = 0;
int16_t AcY = 0;
int16_t AcZ = 0;
int16_t GyX = 0;
int16_t GyY = 0;
int16_t GyZ = 0;

float gyro_x_bias = 0.0f;
float gyro_y_bias = 0.0f;
float gyro_z_bias = 0.0f;

float roll = 0.0f;
float pitch = 0.0f;
float yaw = 0.0f;

unsigned long lastIMUTimeUs = 0;
unsigned long lastLoopTimeMs = 0;

int btn[NUM_FINGERS] = {0};

bool initMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1
  Wire.write(0x00);  // Wake up sensor
  if (Wire.endTransmission(true) != 0) {
    return false;
  }

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);  // GYRO_CONFIG
  Wire.write(0x00);  // +/-250 deg/s
  if (Wire.endTransmission(true) != 0) {
    return false;
  }

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);  // ACCEL_CONFIG
  Wire.write(0x00);  // +/-2g
  if (Wire.endTransmission(true) != 0) {
    return false;
  }

  return true;
}

bool readRaw() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  const uint8_t expectedBytes = 14;
  const uint8_t bytesRead = Wire.requestFrom((int)MPU_ADDR, (int)expectedBytes, (int)true);
  if (bytesRead != expectedBytes) {
    return false;
  }

  AcX = (int16_t)((Wire.read() << 8) | Wire.read());
  AcY = (int16_t)((Wire.read() << 8) | Wire.read());
  AcZ = (int16_t)((Wire.read() << 8) | Wire.read());

  (void)Wire.read();  // Temp high byte (unused)
  (void)Wire.read();  // Temp low byte (unused)

  GyX = (int16_t)((Wire.read() << 8) | Wire.read());
  GyY = (int16_t)((Wire.read() << 8) | Wire.read());
  GyZ = (int16_t)((Wire.read() << 8) | Wire.read());

  return true;
}

void calibrateGyro() {
  long sum_x = 0;
  long sum_y = 0;
  long sum_z = 0;
  int collected = 0;

  while (collected < GYRO_CALIBRATION_SAMPLES) {
    if (readRaw()) {
      sum_x += GyX;
      sum_y += GyY;
      sum_z += GyZ;
      collected++;
    }
    delay(GYRO_CALIBRATION_DELAY_MS);
  }

  gyro_x_bias = sum_x / (float)GYRO_CALIBRATION_SAMPLES;
  gyro_y_bias = sum_y / (float)GYRO_CALIBRATION_SAMPLES;
  gyro_z_bias = sum_z / (float)GYRO_CALIBRATION_SAMPLES;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);
  Serial.println("GloveIMU: JSONL on USB serial (115200 baud).");

  for (int i = 0; i < NUM_FINGERS; i++) {
    pinMode(FINGER_PINS[i], INPUT_PULLUP);
  }

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!initMPU6050()) {
    Serial.println("MPU-6050 init failed. Check wiring and power.");
  }

  Serial.println("Calibrating gyro - keep glove still...");
  calibrateGyro();
  Serial.print("Calibration done. Biases: ");
  Serial.print(gyro_x_bias, 2);
  Serial.print(", ");
  Serial.print(gyro_y_bias, 2);
  Serial.print(", ");
  Serial.println(gyro_z_bias, 2);

  if (readRaw()) {
    roll = atan2((float)AcY, (float)AcZ) * 180.0f / PI;
    pitch = atan2(-(float)AcX, sqrtf((float)AcY * AcY + (float)AcZ * AcZ)) * 180.0f / PI;
    yaw = 0.0f;
  }

  lastIMUTimeUs = micros();
  lastLoopTimeMs = millis();
}

void loop() {
  const unsigned long nowMs = millis();
  if (nowMs - lastLoopTimeMs < LOOP_INTERVAL_MS) {
    return;
  }
  lastLoopTimeMs = nowMs;

  if (!readRaw()) {
    Serial.println("IMU read failed.");
    return;
  }

  const unsigned long nowUs = micros();
  float dt = (nowUs - lastIMUTimeUs) / 1000000.0f;
  lastIMUTimeUs = nowUs;
  if (dt <= 0.0f || dt > 0.5f) {
    dt = LOOP_INTERVAL_MS / 1000.0f;
  }

  const float accel_roll = atan2((float)AcY, (float)AcZ) * 180.0f / PI;
  const float accel_pitch = atan2(-(float)AcX, sqrtf((float)AcY * AcY + (float)AcZ * AcZ)) * 180.0f / PI;

  const float gx = (GyX - gyro_x_bias) / GYRO_SENSITIVITY;
  const float gy = (GyY - gyro_y_bias) / GYRO_SENSITIVITY;
  const float gz = (GyZ - gyro_z_bias) / GYRO_SENSITIVITY;

  roll = COMPLEMENTARY_ALPHA * (roll + gx * dt) + (1.0f - COMPLEMENTARY_ALPHA) * accel_roll;
  pitch = COMPLEMENTARY_ALPHA * (pitch + gy * dt) + (1.0f - COMPLEMENTARY_ALPHA) * accel_pitch;
  yaw += gz * dt;

  for (int i = 0; i < NUM_FINGERS; i++) {
    btn[i] = (digitalRead(FINGER_PINS[i]) == LOW) ? 1 : 0;
  }

  char jsonBuf[128];
  snprintf(
      jsonBuf,
      sizeof(jsonBuf),
      "{\"roll\":%.1f,\"pitch\":%.1f,\"yaw\":%.1f,\"btn\":[%d,%d,%d,%d,%d],\"t\":%lu}\n",
      roll,
      pitch,
      yaw,
      btn[0],
      btn[1],
      btn[2],
      btn[3],
      btn[4],
      nowMs);

  Serial.print(jsonBuf);
}
