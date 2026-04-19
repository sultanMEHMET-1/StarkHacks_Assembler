# ESP32 Glove Firmware — Implementation Prompt

## Read This Entire Prompt Before Writing Any Code

You are building firmware for an ESP32 dev board mounted on a wearable glove. The ESP32 reads an MPU-6050 IMU (accelerometer + gyroscope) and 5 button inputs (one per finger), then streams the data over Bluetooth Classic SPP (Serial Port Profile) so a Python program on a nearby laptop can read it like a serial port.

This is a standalone project. It does NOT depend on the hand_localizer or arm_mover projects. It produces a Bluetooth serial stream that the hand_localizer will consume.

---

## Project Structure

```
glove_firmware/
├── README.md
├── platformio.ini              # PlatformIO build configuration
├── src/
│   └── main.cpp                # All firmware code in one file
```

Use PlatformIO with the Arduino framework for ESP32. If the implementer prefers Arduino IDE instead, that's fine — the code is the same, just the build system differs. But PlatformIO is preferred because it's reproducible.

### platformio.ini

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps =
    Wire
```

No external libraries needed. The ESP32's Bluetooth Classic support is built into the Arduino-ESP32 framework. The MPU-6050 is read via raw I2C register access (no library needed, we control the math ourselves).

---

## Hardware Connections

### MPU-6050 IMU

Connected via I2C:
- SDA → GPIO 21
- SCL → GPIO 22
- VCC → 3.3V
- GND → GND
- I2C address: 0x68 (default, AD0 pin low)

### Finger Buttons

5 momentary push buttons, one per finger. Each button connects its GPIO pin to GND when pressed. The ESP32's internal pull-up resistors are used (INPUT_PULLUP mode), so:
- Button NOT pressed → pin reads HIGH
- Button PRESSED → pin reads LOW

Pin assignments:
```
Thumb:   GPIO 13
Index:   GPIO 12
Middle:  GPIO 14
Ring:    GPIO 27
Pinky:   GPIO 26
```

These pin assignments must be easy to change. Put them in a clearly labeled array at the top of the file.

---

## What the Firmware Does

Every 20 milliseconds (50Hz), the firmware:

1. Reads raw accelerometer and gyroscope data from the MPU-6050 via I2C.
2. Computes roll and pitch from the accelerometer (gravity-based, no drift).
3. Integrates gyroscope Z-axis to compute yaw (will drift, that's accepted).
4. Applies a complementary filter to fuse accelerometer and gyroscope data for roll and pitch. This gives responsive angles (from gyro) that don't drift (corrected by accelerometer).
5. Reads the 5 button pins.
6. Sends one JSONL message over Bluetooth SPP containing all the data.

### Output Format

One JSON object per line, terminated with `\n`. Sent over Bluetooth Serial at 50Hz.

```json
{"roll":12.3,"pitch":-5.1,"yaw":45.2,"btn":[1,0,0,1,0],"t":12345}\n
```

Field definitions:

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `roll` | float | degrees | Rotation about the hand's forward axis. Hand tilts left = negative, right = positive. |
| `pitch` | float | degrees | Wrist flexion/extension. Wrist bent up (fingers pointing up) = positive, bent down = negative. |
| `yaw` | float | degrees | Rotation about the vertical axis. Integrated from gyro, will drift over time. |
| `btn` | array of 5 ints | 0 or 1 | Button states: [thumb, index, middle, ring, pinky]. 1 = pressed, 0 = not pressed. |
| `t` | unsigned long | milliseconds | `millis()` value at time of reading. The receiver can use this to detect gaps or resets. |

**Formatting rules:**
- Use 1 decimal place for roll, pitch, yaw (not 6). Saves bandwidth and serial buffer space.
- Use integers 0/1 for buttons, not true/false (shorter, easier to parse in Python).
- No spaces in the JSON. Every byte matters over Bluetooth Serial.
- Every message MUST end with exactly one `\n`. This is how the receiver knows a message is complete.
- No pretty-printing, no indentation, no extra newlines.

### Why JSONL over Bluetooth SPP

Bluetooth Classic SPP (Serial Port Profile) presents itself as a virtual serial port on the computer. The Python side opens it with `pyserial` exactly like opening a USB serial port. No Bluetooth-specific libraries needed on the computer side.

JSONL is human-readable (you can debug with a serial monitor), self-describing (field names are in the message), and trivially parsed in Python (`json.loads(line)`). The overhead of JSON vs binary at 50Hz and ~60 bytes per message is negligible.

**Do NOT use:**
- BLE (GATT services, characteristics, notifications — way more complex for no benefit at this data rate)
- ESP-NOW (requires a second ESP32 as receiver)
- WiFi/UDP (requires network configuration)
- Binary protocols (not debuggable with a serial monitor)
- Raw CSV (no field names, fragile parsing)

---

## IMU Math — This Is Important, Get It Right

### Reading raw data from MPU-6050

Read 14 bytes starting at register 0x3B:

```
Registers 0x3B-0x40: AcX_H, AcX_L, AcY_H, AcY_L, AcZ_H, AcZ_L (accelerometer)
Registers 0x41-0x42: Temp_H, Temp_L (temperature, ignore)
Registers 0x43-0x48: GyX_H, GyX_L, GyY_H, GyY_L, GyZ_H, GyZ_L (gyroscope)
```

Each value is a 16-bit signed integer (big-endian). Combine high and low bytes: `(high_byte << 8) | low_byte`, cast to `int16_t`.

### Accelerometer → roll and pitch

The accelerometer measures gravity. When the sensor is flat (screen up), gravity points along the Z axis. When tilted, the gravity vector shifts, and you can compute angles from it.

```cpp
float accel_roll  = atan2(AcY, AcZ) * 180.0 / PI;
float accel_pitch = atan2(-AcX, sqrt((float)AcY * AcY + (float)AcZ * AcZ)) * 180.0 / PI;
```

These are noisy but don't drift. They're the "absolute reference" that corrects the gyro over time.

### Gyroscope → angular rate

The gyroscope measures rotational velocity in degrees per second. At the default sensitivity (±250°/s), the raw value is divided by 131.0 to get °/s.

```cpp
float gyro_roll_rate  = GyX / 131.0;  // °/s
float gyro_pitch_rate = GyY / 131.0;  // °/s
float gyro_yaw_rate   = GyZ / 131.0;  // °/s
```

### Complementary filter for roll and pitch

The accelerometer is noisy but doesn't drift. The gyroscope is smooth but drifts. The complementary filter combines them:

```cpp
float dt = (micros() - lastTime) / 1000000.0;
lastTime = micros();

// Integrate gyro to get angle change
float gyro_roll  = roll  + gyro_roll_rate  * dt;
float gyro_pitch = pitch + gyro_pitch_rate * dt;

// Blend: 98% gyro (smooth, responsive) + 2% accelerometer (corrects drift)
roll  = 0.98 * gyro_roll  + 0.02 * accel_roll;
pitch = 0.98 * gyro_pitch + 0.02 * accel_pitch;
```

The 0.98/0.02 split is a starting point. Adjust if the angles drift too fast (increase accel weight) or are too jittery (increase gyro weight). The constants should be `#define` at the top of the file so they're easy to change.

### Yaw (no accelerometer correction possible)

Yaw cannot be corrected by the accelerometer because gravity doesn't change when you rotate around the vertical axis. Yaw is gyro-only, which means it drifts. For this application, that's accepted — the cube's AprilTag tracking provides absolute yaw, and the IMU's yaw is only used for wrist flexion offset calculation where short-term relative accuracy matters more than absolute accuracy.

```cpp
yaw += gyro_yaw_rate * dt;
```

### Gyroscope bias calibration at startup

Gyroscopes have a small constant bias (they report a nonzero rate even when stationary). At startup, the firmware should:

1. Print a message over Serial (USB, for debugging): "Keep glove still for calibration..."
2. Read 200 gyroscope samples over ~2 seconds.
3. Average the readings. This average is the bias.
4. Subtract the bias from all subsequent gyroscope readings.

```cpp
float gyro_x_bias = 0, gyro_y_bias = 0, gyro_z_bias = 0;

void calibrateGyro() {
    const int samples = 200;
    long sum_x = 0, sum_y = 0, sum_z = 0;
    for (int i = 0; i < samples; i++) {
        readRaw();
        sum_x += GyX;
        sum_y += GyY;
        sum_z += GyZ;
        delay(10);
    }
    gyro_x_bias = sum_x / (float)samples;
    gyro_y_bias = sum_y / (float)samples;
    gyro_z_bias = sum_z / (float)samples;
}
```

Then in the loop, subtract bias before computing rates:

```cpp
float gyro_roll_rate  = (GyX - gyro_x_bias) / 131.0;
float gyro_pitch_rate = (GyY - gyro_y_bias) / 131.0;
float gyro_yaw_rate   = (GyZ - gyro_z_bias) / 131.0;
```

---

## Bluetooth Setup

Use the `BluetoothSerial` class from the Arduino-ESP32 framework.

```cpp
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

void setup() {
    Serial.begin(115200);  // USB serial for debugging
    SerialBT.begin("GloveIMU");  // Bluetooth device name

    Serial.println("Bluetooth started. Pair with 'GloveIMU'.");
    // ... rest of setup
}
```

In the loop, send data over `SerialBT` (Bluetooth) AND optionally over `Serial` (USB) for debugging:

```cpp
// Build the JSON string
char jsonBuf[128];
snprintf(jsonBuf, sizeof(jsonBuf),
    "{\"roll\":%.1f,\"pitch\":%.1f,\"yaw\":%.1f,\"btn\":[%d,%d,%d,%d,%d],\"t\":%lu}\n",
    roll, pitch, yaw,
    btn[0], btn[1], btn[2], btn[3], btn[4],
    millis()
);

SerialBT.print(jsonBuf);  // Send over Bluetooth
Serial.print(jsonBuf);     // Echo to USB serial for debugging
```

**The Bluetooth device name "GloveIMU" must be consistent** because the Python side will search for this name to identify the correct serial port. Put it as a `#define` at the top.

---

## Timing

The main loop should run at 50Hz (every 20ms). Use `millis()` timing, not `delay()`:

```cpp
const unsigned long LOOP_INTERVAL_MS = 20;  // 50Hz
unsigned long lastLoopTime = 0;

void loop() {
    unsigned long now = millis();
    if (now - lastLoopTime < LOOP_INTERVAL_MS) return;
    lastLoopTime = now;

    // ... read IMU, read buttons, send data
}
```

This ensures consistent timing even if the IMU read or Bluetooth send takes a few milliseconds.

---

## Complete Program Structure

```cpp
#include <Arduino.h>
#include <Wire.h>
#include "BluetoothSerial.h"

// === CONFIGURATION — EDIT THESE ===
#define BT_DEVICE_NAME "GloveIMU"
#define MPU_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 22
#define LOOP_INTERVAL_MS 20          // 50Hz
#define COMPLEMENTARY_ALPHA 0.98     // Gyro weight in complementary filter
#define GYRO_CALIBRATION_SAMPLES 200
#define GYRO_SENSITIVITY 131.0       // LSB/(°/s) at ±250°/s range

const int FINGER_PINS[] = {13, 12, 14, 27, 26};  // thumb, index, middle, ring, pinky
const int NUM_FINGERS = 5;
// === END CONFIGURATION ===

BluetoothSerial SerialBT;

// IMU state
int16_t AcX, AcY, AcZ, GyX, GyY, GyZ;
float gyro_x_bias, gyro_y_bias, gyro_z_bias;
float roll, pitch, yaw;
unsigned long lastIMUTime;

// Button state
int btn[NUM_FINGERS];

// Loop timing
unsigned long lastLoopTime;

void readRaw() { /* I2C read of 14 bytes from MPU */ }
void calibrateGyro() { /* Average 200 readings for bias */ }

void setup() {
    Serial.begin(115200);
    SerialBT.begin(BT_DEVICE_NAME);

    // Init buttons
    for (int i = 0; i < NUM_FINGERS; i++) {
        pinMode(FINGER_PINS[i], INPUT_PULLUP);
    }

    // Init I2C and wake up MPU-6050
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B);  // Power management register
    Wire.write(0);     // Wake up
    Wire.endTransmission(true);

    // Calibrate gyroscope (device must be still)
    Serial.println("Calibrating gyro — keep glove still...");
    calibrateGyro();
    Serial.println("Calibration done.");

    lastIMUTime = micros();
    lastLoopTime = millis();

    roll = 0;
    pitch = 0;
    yaw = 0;
}

void loop() {
    unsigned long now = millis();
    if (now - lastLoopTime < LOOP_INTERVAL_MS) return;
    lastLoopTime = now;

    // Read IMU
    readRaw();

    // Compute dt
    float dt = (micros() - lastIMUTime) / 1000000.0;
    lastIMUTime = micros();

    // Accelerometer angles
    float accel_roll  = atan2(AcY, AcZ) * 180.0 / PI;
    float accel_pitch = atan2(-AcX, sqrt((float)AcY*AcY + (float)AcZ*AcZ)) * 180.0 / PI;

    // Gyro rates (bias-corrected)
    float gx = (GyX - gyro_x_bias) / GYRO_SENSITIVITY;
    float gy = (GyY - gyro_y_bias) / GYRO_SENSITIVITY;
    float gz = (GyZ - gyro_z_bias) / GYRO_SENSITIVITY;

    // Complementary filter for roll and pitch
    roll  = COMPLEMENTARY_ALPHA * (roll  + gx * dt) + (1.0 - COMPLEMENTARY_ALPHA) * accel_roll;
    pitch = COMPLEMENTARY_ALPHA * (pitch + gy * dt) + (1.0 - COMPLEMENTARY_ALPHA) * accel_pitch;

    // Yaw (gyro only, drifts)
    yaw += gz * dt;

    // Read buttons
    for (int i = 0; i < NUM_FINGERS; i++) {
        btn[i] = (digitalRead(FINGER_PINS[i]) == LOW) ? 1 : 0;
    }

    // Build and send JSON
    char jsonBuf[128];
    snprintf(jsonBuf, sizeof(jsonBuf),
        "{\"roll\":%.1f,\"pitch\":%.1f,\"yaw\":%.1f,\"btn\":[%d,%d,%d,%d,%d],\"t\":%lu}\n",
        roll, pitch, yaw,
        btn[0], btn[1], btn[2], btn[3], btn[4],
        millis()
    );

    SerialBT.print(jsonBuf);
    Serial.print(jsonBuf);  // Debug echo
}
```

---

## Verification Checklist

- [ ] ESP32 appears as "GloveIMU" in Bluetooth device list on laptop
- [ ] After pairing, a serial port appears (e.g., `/dev/rfcomm0` on Linux, `COM5` on Windows)
- [ ] Opening the port with a serial monitor shows JSONL lines at ~50Hz
- [ ] Each line is valid JSON with all fields: roll, pitch, yaw, btn, t
- [ ] Roll changes when tilting the hand left/right
- [ ] Pitch changes when flexing the wrist up/down
- [ ] Yaw changes when rotating the hand horizontally (will drift, that's expected)
- [ ] Roll and pitch are stable when the hand is held still (< 1° jitter)
- [ ] Roll and pitch return to ~0 when the hand is flat (accelerometer correction working)
- [ ] Buttons show 1 when pressed, 0 when released
- [ ] Each finger maps to the correct array index (thumb=0, index=1, etc.)
- [ ] Gyro calibration runs at startup (2 seconds of stillness)
- [ ] Data continues streaming after 5+ minutes (no buffer overflow, no Bluetooth disconnect)
- [ ] `t` field increases monotonically

---

## What Is NOT In Scope

- Magnetometer / compass (MPU-6050 doesn't have one)
- Quaternion output (Euler angles are fine for this application)
- OTA firmware updates
- Battery management
- Haptic feedback
- Flex sensors (buttons only for this milestone)
- Any communication with the arm mover (the ESP32 talks only to the hand localizer)

If you have questions, ask before building.