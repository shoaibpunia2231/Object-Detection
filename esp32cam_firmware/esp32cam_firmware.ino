#include <Wire.h>
#include <U8g2lib.h>

// SH1106 OLED using Software I2C
U8G2_SH1106_128X64_NONAME_F_SW_I2C u8g2(
  U8G2_R0,
  /* clock=*/ 22,
  /* data=*/ 21,
  /* reset=*/ U8X8_PIN_NONE
);

void setup() {

  Serial.begin(115200);

  // Start OLED
  u8g2.begin();

  // Clear Buffer
  u8g2.clearBuffer();

  // Font
  u8g2.setFont(u8g2_font_ncenB08_tr);

  // Print Text
  u8g2.drawStr(10, 25, "OLED Working");

  u8g2.drawStr(5, 50, "ESP32 Connected");

  // Send to Display
  u8g2.sendBuffer();
}

void loop() {

}