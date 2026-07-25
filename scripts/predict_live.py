"""
predict_live.py
----------------
Loads the trained model and runs real-time ASL prediction from the glove,
using per-session calibration.

USAGE:
    python predict_live.py
"""

import serial
import time
import numpy as np
import joblib
import os
from collections import deque, Counter

# ── CONFIG ──────────────────────────────────────────────────────────────────
PORT = '/dev/cu.SLAB_USBtoUART' # macOS example — Windows: 'COM3', 'COM4', etc. Linux: '/dev/ttyUSB0'
BAUD_RATE = 115200
MODEL_FILE = 'asl_model.pkl'
SCALER_FILE = 'asl_scaler.pkl'
ENCODER_FILE = 'asl_label_encoder.pkl'
NUM_FEATURES = 11 

USE_ACCEL = True
USE_GYRO = False

SMOOTHING_WINDOW = 10 
CONFIDENCE_THRESHOLD = 0.55 

# Calibration Constants
POSE_SETTLE_SECONDS = 5 # fixed window to get into position before trusting data
CALIBRATION_FRAMES = 15
FLEX_IDX = range(0, 5)
ACCEL_IDX = range(5, 8)
EPS = 1e-6
FLEX_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

# Per-finger thresholds — remeasured after swapping sensors (see the
# OPEN vs FIST readings: thumb ~110, index ~329, ring ~390, pinky ~572
# apart). Thresholds set at roughly half the observed span.
MIN_CALIBRATION_SPAN = {
    'thumb': 50,
    'index': 150,
    'middle': 300, # not actually enforced — see KNOWN_FAULTY_FINGERS below
    'ring': 180,
    'pinky': 280,
}

# Raw ADC saturation limits for a 12-bit ADC (ESP32 default).
# A reading pinned exactly at 0 or 4095 means a loose wire/bad contact,
# not a real sensor value. This is flagged both during calibration
# and during live prediction.
ADC_MIN, ADC_MAX = 0, 4095
SATURATION_WARNING_COOLDOWN = 2.0 # seconds between repeated live warnings, to avoid spam

# WORKAROUND: middle finger has a wiring fault — reads a genuine 0 when
# straight, normal values once bent. Exempt it from saturation rejection
# (see collect_data.py for the full explanation) or every straight-middle
# frame gets rejected forever and the read loop hangs. Remove 'middle' once
# the connection is physically fixed.
KNOWN_FAULTY_FINGERS = {'middle'}
# ────────────────────────────────────────────────────────────────────────────

def load_models():
    for f in [MODEL_FILE, SCALER_FILE, ENCODER_FILE]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"'{f}' not found. Run train_model.py first.")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)
    print(f" Model loaded. Classes: {list(encoder.classes_)}")
    return model, scaler, encoder


def read_clean_line(ser, saturation_counter=None):
    """
    Reads one line from serial. If a flex value is pinned exactly at 0
    or 4095 (ADC rails), that indicates a loose wire/bad contact rather
    than a real gesture value. Those lines are rejected (same as
    collect_data.py), and if saturation_counter is provided, it's
    updated so the caller can warn the user about which finger(s).
    """
    raw = ser.readline().decode('utf-8', errors='ignore').strip()
    if not raw:
        return None
    parts = raw.split(',')
    if len(parts) == NUM_FEATURES:
        try:
            values = [float(p) for p in parts]
            flex_values = values[:5]
            saturated_now = False
            for name, v in zip(FLEX_NAMES, flex_values):
                if name in KNOWN_FAULTY_FINGERS:
                    continue # known fault — 0/4095 may be a genuine reading here
                if v == ADC_MIN or v == ADC_MAX:
                    saturated_now = True
                    if saturation_counter is not None:
                        saturation_counter['count'] += 1
                        saturation_counter['fingers'].add(name)
            if saturated_now:
                return None
            return values
        except ValueError:
            return None
    return None


def wait_and_discard(ser, seconds):
    """
    Gives the user a fixed window of time to physically move into
    position, discarding everything received during that window, so
    capture afterward is guaranteed to reflect the NEW pose rather than
    leftover readings from whatever pose came before.
    """
    print(f" ⏳ You have {seconds} seconds to get into position "
          f"(ignoring all sensor data until then)...")
    start = time.time()
    while (time.time() - start) < seconds:
        ser.readline()
    print(f" Time's up — now reading live data.")


def capture_calibration_pose(ser, pose_name, n_frames):
    print(f"\n Get ready to hold your hand {pose_name}...")
    wait_and_discard(ser, POSE_SETTLE_SECONDS)

    print(f" Capturing {pose_name} baseline...")
    sat_counter = {'count': 0, 'fingers': set()}
    samples = []
    while len(samples) < n_frames:
        values = read_clean_line(ser, saturation_counter=sat_counter)
        if values:
            samples.append(values)

    if sat_counter['count'] >= n_frames:
        print(f" WARNING: {sat_counter['count']} readings during this pose were "
              f"pinned at 0 or 4095 on: {', '.join(sorted(sat_counter['fingers']))}. "
              f"This usually means a loose connection, not a real gesture value — "
              f"check the wiring/solder joints before continuing.")

    avg = [sum(s[i] for s in samples) / len(samples) for i in range(NUM_FEATURES)]
    print(f" {pose_name} baseline captured.")
    return avg


def get_valid_calibration(ser):
    while True:
        open_baseline = capture_calibration_pose(ser, "OPEN (relaxed, flat)", CALIBRATION_FRAMES)
        fist_baseline = capture_calibration_pose(ser, "in a FIST (fully closed)", CALIBRATION_FRAMES)

        bad_fingers = []
        print(" Measured spans:")
        for i, name in zip(FLEX_IDX, FLEX_NAMES):
            span = abs(fist_baseline[i] - open_baseline[i])
            if name in KNOWN_FAULTY_FINGERS:
                print(f" {name:8s}: open={open_baseline[i]:7.1f} fist={fist_baseline[i]:7.1f} "
                      f"span={span:7.1f} [SKIPPED — known hardware fault]")
                continue
            threshold = MIN_CALIBRATION_SPAN[name]
            status = "OK" if span >= threshold else "TOO WEAK"
            print(f" {name:8s}: open={open_baseline[i]:7.1f} fist={fist_baseline[i]:7.1f} "
                  f"span={span:7.1f} (need >= {threshold}) [{status}]")
            if span < threshold:
                bad_fingers.append((name, span, threshold))

        if not bad_fingers:
            return open_baseline, fist_baseline

        print(f"\n Calibration too weak on: "
              + ", ".join(f"{n} (moved {s:.0f}, needed {t})" for n, s, t in bad_fingers))
        print(" Let's redo OPEN and FIST.")


def calibrate(values, open_baseline, fist_baseline):
    cal = []
    for i in FLEX_IDX:
        lo, hi = open_baseline[i], fist_baseline[i]
        span = hi - lo
        if abs(span) < EPS:
            cal.append(0.0) 
        else:
            cal.append((values[i] - lo) / span)
    for i in ACCEL_IDX:
        cal.append(values[i] - open_baseline[i])
    return cal


def engineer_features_single(cal_flex, cal_accel, raw_gyro):
    """
    Constructs the feature vector using calibrated flex/accel data and raw gyro, 
    matching the training script perfectly.
    """
    feature_vec = list(cal_flex)
    if USE_ACCEL:
        feature_vec += list(cal_accel)
    if USE_GYRO:
        feature_vec += list(raw_gyro)

    flex_sum = sum(cal_flex)
    flex_spread = max(cal_flex) - min(cal_flex)
    feature_vec += [flex_sum, flex_spread]

    if USE_ACCEL:
        ax, ay, az = cal_accel
        accel_mag = (ax**2 + ay**2 + az**2) ** 0.5
        feature_vec.append(accel_mag)
    if USE_GYRO:
        gx, gy, gz = raw_gyro
        gyro_mag = (gx**2 + gy**2 + gz**2) ** 0.5
        feature_vec.append(gyro_mag)

    return np.array(feature_vec).reshape(1, -1)


def confidence_bar(prob, width=20):
    filled = int(prob * width)
    return '█' * filled + '░' * (width - filled)


def main():
    print("Loading model...")
    model, scaler, encoder = load_models()

    print(f"\nConnecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(" Connected!\n")
    except serial.SerialException as e:
        print(f"[ERROR] {e}")
        return

    # ── 1. LIVE CALIBRATION PHASE ─────────────────────────────────────────
    print("=" * 55)
    print(" LIVE CALIBRATION REQUIRED")
    print("=" * 55)
    open_baseline, fist_baseline = get_valid_calibration(ser)

    # ── 2. PREDICTION LOOP ────────────────────────────────────────────────
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    last_shown = None

    print("\n" + "=" * 55)
    print(" REAL-TIME ASL PREDICTION")
    print(f" Smoothing window : {SMOOTHING_WINDOW} frames")
    print(f" Confidence needed: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(" Press Ctrl+C to quit")
    print("=" * 55 + "\n")

    try:
        debug_counter = 0
        live_sat_counter = {'count': 0, 'fingers': set()}
        last_sat_warning_time = 0

        # Clear the serial buffer from any stale data during calibration
        ser.reset_input_buffer()
        
        while True:
            values = read_clean_line(ser, saturation_counter=live_sat_counter)

            # If a finger is reading 0/4095 during live prediction, that's
            # a wiring fault happening in real time (e.g. a wire came
            # loose while wearing the glove) — surface it, but only once
            # every SATURATION_WARNING_COOLDOWN seconds so it doesn't spam
            # the console while predictions are still being shown.
            if live_sat_counter['count'] > 0:
                now = time.time()
                if (now - last_sat_warning_time) >= SATURATION_WARNING_COOLDOWN:
                    print(f"\n WIRING WARNING: reading 0 or 4095 on "
                          f"{', '.join(sorted(live_sat_counter['fingers']))} — "
                          f"check the connection on that sensor. Predictions may "
                          f"be unreliable until this is fixed.")
                    last_sat_warning_time = now
                    live_sat_counter['count'] = 0
                    live_sat_counter['fingers'] = set()

            if values is None:
                continue

            debug_counter += 1
            
            # 1. Calibrate the raw data
            calibrated = calibrate(values, open_baseline, fist_baseline)
            cal_flex = calibrated[:5]
            cal_accel = calibrated[5:]
            raw_gyro = values[8:11]

            # 2. Engineer features using calibrated data
            features = engineer_features_single(cal_flex, cal_accel, raw_gyro)
            features_scaled = scaler.transform(features)
            
            # Show debug info every 30 frames
            if debug_counter == 1 or debug_counter % 30 == 0:
                print(f"\n[CALIBRATED SENSORS] Flex: {[f'{v:.2f}' for v in cal_flex]} | Accel: {[f'{v:.0f}' for v in cal_accel]}")

            # 3. Get class probabilities
            probs = model.predict_proba(features_scaled)[0]
            top_idx = np.argmax(probs)
            top_prob = probs[top_idx]
            prediction = encoder.classes_[top_idx]

            # Add to smoothing buffer
            prediction_buffer.append((prediction, top_prob))

            # Only act when buffer is full
            if len(prediction_buffer) < SMOOTHING_WINDOW:
                continue

            # Majority vote across the window
            votes = [p for p, _ in prediction_buffer]
            avg_probs = {}
            for cls in encoder.classes_:
                indices = [i for i, (p, _) in enumerate(prediction_buffer) if p == cls]
                if indices:
                    avg_probs[cls] = np.mean([prediction_buffer[i][1] for i in indices])

            winner = Counter(votes).most_common(1)[0][0]
            winner_conf = avg_probs.get(winner, 0.0)
            
            # Only display if above confidence threshold
            if winner_conf >= CONFIDENCE_THRESHOLD:
                bar = confidence_bar(winner_conf)
                print(f"\r [{bar}] {winner_conf*100:5.1f}% → '{winner}' ", end='', flush=True)
                last_shown = winner
            else:
                bar = confidence_bar(winner_conf)
                print(f"\r ? [{bar}] {winner_conf*100:5.1f}% → (uncertain) ", end='', flush=True)
                last_shown = None

    except KeyboardInterrupt:
        print("\n\nStopping prediction. Goodbye!")
    finally:
        if ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()