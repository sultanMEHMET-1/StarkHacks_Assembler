# Glove Firmware (ESP32 + MPU-6050 + USB serial)

Firmware for an ESP32 glove controller that reads:

- MPU-6050 IMU over I2C
- 5 finger buttons (active-low with internal pull-ups)

It streams one JSONL message every 20ms (50Hz) over **USB serial** (the port created when you connect the board with USB-C / USB, typically `/dev/ttyACM0` or `/dev/ttyUSB0` on Linux) at **115200 baud**.

## Output Format

Each line:

```json
{"roll":12.3,"pitch":-5.1,"yaw":45.2,"btn":[1,0,0,1,0],"t":12345}
```

Notes:

- One line per message (`\n` terminated).
- No spaces.
- `btn` order is `[thumb, index, middle, ring, pinky]`.
- `roll`/`pitch` use complementary filter (accel + gyro).
- `yaw` is gyro-integrated and will drift over time.

## Hardware Wiring

### MPU-6050

- SDA -> GPIO 21
- SCL -> GPIO 22
- VCC -> 3.3V
- GND -> GND
- AD0 low (I2C addr `0x68`)

### Buttons (to GND when pressed)

- Thumb -> GPIO 13
- Index -> GPIO 12
- Middle -> GPIO 14
- Ring -> GPIO 27
- Pinky -> GPIO 26

## Build and Flash (PlatformIO)

From this directory:

```bash
pio run
pio run --target upload
pio device monitor --baud 115200
```

At startup the firmware performs gyroscope bias calibration for about 2 seconds. Keep the glove still during this step.

## Reading data on the PC

Open the **USB serial device** with `pyserial`, `pio device monitor`, `minicom`, etc. at **115200** baud. No Bluetooth pairing is required.
