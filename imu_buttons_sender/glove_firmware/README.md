# Glove Firmware (ESP32 + MPU-6050 + USB serial)

Firmware for an ESP32 glove controller that reads a single button and streams its state at 50 Hz over USB serial.

## Output Format

Each line:

```json
{"btn":[1],"t":12345}
```

- `btn[0]`: `1` = pressed, `0` = released (active-low, internal pull-up)
- `t`: millis() timestamp on the ESP32
- 50 Hz, 115200 baud, `\n` terminated

## Hardware Wiring

### Button (to GND when pressed)

- Button -> GPIO 23
- Other side -> GND

No external resistor needed — GPIO 23 uses `INPUT_PULLUP`.

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
