# ASL Glove — Wearable Sign Language Recognition

A wearable glove that reads hand shape via flex sensors and motion via an IMU, streams the data over Bluetooth from an ESP32, and uses a trained ML model to recognize static ASL letters (A–Z) and numbers (0–9) in real time.

> **Status: Work in progress.** Core pipeline (firmware → data collection → training → live prediction) is built and has run end-to-end. Sensor calibration and hardware reliability are still being refined. See [Known Issues](#known-issues--next-steps) below.

---

## Overview

- **Input:** 5 flex sensors (one per finger) + 6-axis IMU (accelerometer + gyroscope)
- **Transport:** ESP32 streams 11 comma-separated values over Bluetooth Serial at ~20Hz
- **Pipeline:** collect labeled samples → per-session calibration (open-hand / fist baseline) → feature engineering → Random Forest / SVM classifier → live smoothed prediction

```
Glove (flex sensors + MPU6050)
        │
   ESP32 (Bluetooth Serial)
        │
 collect_data.py  ──────────►  asl_data_calibrated.csv
        │                             │
        │                       train_model.py
        │                             │
        │                    asl_model.pkl / scaler / encoder
        │                             │
 predict_live.py  ◄───────────────────┘
        │
  Live prediction with confidence smoothing
```

## Hardware

| Component | Notes |
|---|---|
| ESP32 DevKit (Bluetooth + WiFi) | Main microcontroller |
| MPU6050 IMU | 6-axis accel + gyro, I2C |
| Spectra Symbol 2.2" flex sensors (×5) | One per finger |
| CD74HC4067 analog multiplexer | For scaling analog inputs |
| 47kΩ resistors | Flex sensor voltage dividers |
| TP4056 charging module | LiPo charge management |
| 3.7V 1000mAh LiPo battery | Portable power |
| Cotton glove + jumper wires | Sensor mounting (prototype wiring — no conductive thread yet) |

*(Add a wiring diagram / photo of the assembled glove here — this is the single highest-value addition for a hardware repo.)*

## Repository Structure

```
├── firmware/
│   └── ASL_Glove.ino        # ESP32 firmware — reads sensors, streams over Bluetooth
├── scripts/
│   ├── flex_sensor.py       # Quick serial monitor for raw flex sensor debugging
│   ├── collect_data.py      # Guided, calibrated data collection with synthetic augmentation
│   ├── train_model.py       # Trains Random Forest + SVM, group-aware split by session
│   └── predict_live.py      # Real-time inference with per-session calibration
├── requirements.txt
└── README.md
```

## How It Works

**1. Firmware (`ASL_Glove.ino`)**
Reads 5 analog flex sensor pins and the MPU6050 over I2C, packages `thumb, index, middle, ring, pinky, ax, ay, az, gx, gy, gz` into one line, and sends it over both Bluetooth Serial and USB Serial every 50ms.

**2. Data Collection (`collect_data.py`)**
For each gesture label, the script walks you through an OPEN-hand and FIST baseline calibration, checks that each finger moved enough to be trustworthy (per-finger thresholds), then records multiple sessions per gesture. Each session gets its own `session_id` so training can split by session rather than by row — this avoids leaking near-duplicate frames between train/test.
Includes synthetic data augmentation (small jitter on flex/IMU values) to expand the dataset.

**3. Training (`train_model.py`)**
Loads the calibrated CSV, engineers a few derived features (`flex_sum`, `flex_spread`, `accel_mag`), and trains both a Random Forest and an SVM. Evaluation uses a **group-aware** train/test split (`GroupShuffleSplit` / `GroupKFold` by `session_id`) so a whole recording session — not just a row — stays entirely on one side of the split. Saves the best model, scaler, and label encoder.

**4. Live Prediction (`predict_live.py`)**
Loads the saved model, runs a fresh OPEN/FIST calibration for the current wearing session (accounts for the glove fitting slightly differently each time it's put on), then classifies live sensor streams with a rolling smoothing window and confidence threshold before showing a prediction.

## Setup

```bash
pip install -r requirements.txt
```

Flash `firmware/ASL_Glove.ino` to the ESP32 via Arduino IDE (requires the `MPU6050` and `BluetoothSerial` libraries).

Update the `PORT` constant at the top of each Python script to match your OS's serial/Bluetooth port.

```bash
python scripts/collect_data.py     # record gestures
python scripts/train_model.py      # train + save model
python scripts/predict_live.py     # run live recognition
```

## Known Issues / Next Steps

- Flex sensor readings currently need improved wiring/shielding for consistent calibration across sessions
- Gyroscope data is collected but not yet used in the model (static letter poses don't need angular velocity; would matter for motion-based signs like J and Z)
- Wiring is currently jumper cables sewn into the glove rather than conductive thread — a durability/comfort improvement for a v2 build
- Model currently only handles static single-frame poses; motion letters (J, Z) aren't supported
- No enclosure/PCB yet — everything is breadboard + jumper wire prototype

## Future Ideas

- Switch flex sensor wiring to conductive thread for durability
- Add motion-letter support using the IMU gyro data
- On-device inference (TinyML) instead of streaming to a laptop
- Mobile app for live gloss-to-text output

## License

*(Choose one — MIT is common for hobby hardware/ML repos. Add a `LICENSE` file.)*