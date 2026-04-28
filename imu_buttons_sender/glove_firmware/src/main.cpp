#include <Arduino.h>

// === CONFIGURATION — EDIT THESE ===
#define SERIAL_BAUD 115200
#define LOOP_INTERVAL_MS 20UL   // 50 Hz
#define BUTTON_PIN 23           // Active-low (INPUT_PULLUP); press = LOW = btn 1
// === END CONFIGURATION ===

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.println("GloveButton: JSONL on USB serial (115200 baud).");
}

static unsigned long lastLoopMs = 0;

void loop() {
  const unsigned long nowMs = millis();
  if (nowMs - lastLoopMs < LOOP_INTERVAL_MS) {
    return;
  }
  lastLoopMs = nowMs;

  const int btn = (digitalRead(BUTTON_PIN) == LOW) ? 1 : 0;

  char jsonBuf[48];
  snprintf(jsonBuf, sizeof(jsonBuf), "{\"btn\":[%d],\"t\":%lu}\n", btn, nowMs);
  Serial.print(jsonBuf);
}
