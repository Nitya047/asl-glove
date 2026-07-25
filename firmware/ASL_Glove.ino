/*
  ASL Glove - ESP32 Firmware
  Sends: thumb, index, middle, ring, pinky, ax, ay, az, gx, gy, gz
  Over Bluetooth Serial at 1500ms (1.5 second) intervals
*/

#include <Wire.h>
#include <MPU6050.h>
#include "BluetoothSerial.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled!
#endif

BluetoothSerial SerialBT;
MPU6050 mpu;

// Flex sensor pins
const int thumbPin  = 36;
const int indexPin  = 39;
const int middlePin = 34;
const int ringPin   = 35;
const int pinkyPin  = 32;

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout detector

  Serial.begin(115200);
  Wire.begin(21, 22);
  mpu.initialize();

  // Configure flex sensor pins as inputs
  pinMode(thumbPin, INPUT);
  pinMode(indexPin, INPUT);
  pinMode(middlePin, INPUT);
  pinMode(ringPin, INPUT);
  pinMode(pinkyPin, INPUT);

  SerialBT.begin("ASL_Glove");
  Serial.println("Bluetooth Started! Ready to pair...");
}

void loop() {
  // Read flex sensors
  int thumb  = analogRead(thumbPin);
  int index  = analogRead(indexPin);
  int middle = analogRead(middlePin);
  int ring   = analogRead(ringPin);
  int pinky  = analogRead(pinkyPin);

  // Read all 6 axes from MPU6050
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Format: 11 comma-separated values
  String dataString = String(thumb)  + "," +
                      String(index)  + "," +
                      String(middle) + "," +
                      String(ring)   + "," +
                      String(pinky)  + "," +
                      String(ax)     + "," +
                      String(ay)     + "," +
                      String(az)     + "," +
                      String(gx)     + "," +
                      String(gy)     + "," +
                      String(gz);

  SerialBT.println(dataString);
  Serial.println(dataString);

  delay(50); 
}