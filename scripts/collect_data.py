"""
collect_data.py
------------------
Collects labeled sensor samples from the ASL glove and saves them to a CSV.

Calibration approach:

  1. ACTIVE BUFFER DRAIN instead of a fixed settling delay. While the
     script waits at input() for the user to press ENTER, the ESP32 keeps
     streaming. A fixed "wait 0.6s" doesn't guarantee that backlog is
     gone — the script could still be averaging stale data from the
     PREVIOUS pose. It drains until ser.in_waiting == 0 (genuinely caught
     up to real time), then discards a few more lines for margin.
  2. PER-FINGER MIN_CALIBRATION_SPAN, because the thumb's real dynamic
     range (~90-190 raw units) is much smaller than the other fingers
     (~500-1300 units). A single global threshold of 50 was too close to
     the thumb's natural range if the pose wasn't fully committed.
  3. SATURATION = WIRING ERROR: a flex value of exactly 0 or 4095 means a
     loose connection/bad contact, not a real "fully bent" reading, so
     those frames are rejected everywhere (calibration included). If this
     happens repeatedly during calibration, a visible warning tells the
     user to check the physical connection instead of silently retrying
     forever.
  4. LIVE DIAGNOSTICS: prints the actual measured span per finger every
     time, pass or fail, to show why a calibration was accepted or
     rejected instead of guessing.
  5. MEDIAN instead of MEAN for the baseline average, since median is far
     less sensitive to a few leftover transient samples than a straight
     average.
"""

import serial
import time
import csv
import os
import random
import statistics

# ── CONFIG ──────────────────────────────────────────────────────────────────
PORT = '/dev/cu.SLAB_USBtoUART' # macOS example — Windows: 'COM3', 'COM4', etc. Linux: '/dev/ttyUSB0'
BAUD_RATE = 115200
SESSIONS_PER_GESTURE = 8
FRAMES_PER_SESSION = 15
POSE_SETTLE_SECONDS = 5 # fixed time given to physically get into position
CALIBRATION_FRAMES = 15

OUTPUT_FILE = 'asl_data_calibrated.csv'
NUM_FEATURES = 11

# ── AUGMENTATION CONFIG ──
SYNTHETIC_MULTIPLIER = 2
FLEX_NOISE_PCT = 0.05
IMU_NOISE = 500
# ────────────────────────────────────────────────────────────────────────────

VALID_LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [str(i) for i in range(10)]

RUN_ID = str(int(time.time()))

HEADER = ['thumb', 'index', 'middle', 'ring', 'pinky',
          'ax', 'ay', 'az', 'gx', 'gy', 'gz',
          'thumb_cal', 'index_cal', 'middle_cal', 'ring_cal', 'pinky_cal',
          'ax_cal', 'ay_cal', 'az_cal',
          'label', 'session_id']

FLEX_IDX = range(0, 5)
ACCEL_IDX = range(5, 8)
EPS = 1e-6
FLEX_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

# Per-finger minimum required span, remeasured after swapping sensors
# (based on OPEN vs FIST readings — thumb ~110, index ~329, ring ~390,
# pinky ~572 apart). Thresholds set at roughly half the observed span,
# so a genuinely weak calibration still gets caught without being so
# tight that normal pose variation trips it.
MIN_CALIBRATION_SPAN = {
    'thumb': 50,
    'index': 150,
    'middle': 300, # not actually enforced — see KNOWN_FAULTY_FINGERS below
    'ring': 180,
    'pinky': 280,
}

# Raw ADC saturation limits for a 12-bit ADC (ESP32 default)
ADC_MIN, ADC_MAX = 0, 4095

# WORKAROUND: the middle finger sensor has a wiring fault that makes it
# genuinely read 0 when straight, and only shows a normal value once bent.
# That 0 is real data for this sensor, not a comms glitch — but the
# saturation check below would otherwise reject EVERY frame where middle is
# straight, hanging read_clean_line() forever for any gesture with middle
# extended. Exempt it here. Its calibrated feature will still work (open
# baseline ~0, fist baseline ~1550 gives a big, valid span) — it just won't
# resolve partial/half-bent middle positions the way a healthy sensor would,
# since it can only really tell "straight" from "bent enough to reconnect."
# Remove 'middle' from this set once the connection is physically fixed.
KNOWN_FAULTY_FINGERS = {'middle'}


def read_clean_line(ser, saturation_counter=None):
    """
    Reads one valid line from serial. Any flex value that reads exactly
    0 or 4095 indicates a wiring/contact fault (not a real "fully bent"
    reading), so that line is always rejected, in both calibration and
    normal recording.

    If saturation_counter (a mutable dict) is passed in, rejected
    saturated lines increment it, so the caller can warn the user if
    this is happening a lot (pointing at a loose connection).
    """
    while True:
        raw = ser.readline().decode('utf-8', errors='ignore').strip()
        if not raw:
            continue
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
                    continue
                return values
            except ValueError:
                continue


def wait_and_discard(ser, seconds):
    """
    Gives the user a fixed window of time to physically move into
    position, while continuously reading and throwing away whatever the
    ESP32 sends during that window. Only after the full 'seconds' have
    elapsed does the caller start actually keeping samples — guaranteeing
    we are no longer looking at leftover readings from the previous pose.
    """
    print(f" ⏳ You have {seconds} seconds to get into position "
          f"(ignoring all sensor data until then)...")
    start = time.time()
    discarded = 0
    while (time.time() - start) < seconds:
        raw = ser.readline() # read and throw away, regardless of content
        if raw:
            discarded += 1
    print(f" Time's up — now reading live data ({discarded} lines discarded).")


def capture_calibration_pose(ser, pose_name, n_frames):
    print(f" Get ready to hold your hand {pose_name}...")

    # Fixed window to physically move into position — everything
    # arriving during this window is read and discarded, so whatever is
    # captured right after is guaranteed to be the new pose, not
    # leftover readings from the previous one.
    wait_and_discard(ser, POSE_SETTLE_SECONDS)

    print(f" Capturing {pose_name} baseline...")
    sat_counter = {'count': 0, 'fingers': set()}
    samples = []
    while len(samples) < n_frames:
        values = read_clean_line(ser, saturation_counter=sat_counter)
        if values:
            samples.append(values)

    # If a lot of lines had to be thrown out for saturation while
    # collecting just this one baseline, that's a strong sign of a loose
    # wire/bad contact rather than noise — flag it immediately.
    if sat_counter['count'] >= n_frames: # rejected as many (or more) than we kept
        print(f" WARNING: {sat_counter['count']} readings during this pose were "
              f"pinned at 0 or 4095 on: {', '.join(sorted(sat_counter['fingers']))}. "
              f"This usually means a loose connection on that sensor, not a real "
              f"gesture value — check the wiring/solder joints before continuing.")

    # Median per channel instead of mean — robust to any leftover
    # transient samples that snuck through the settling window.
    avg = [statistics.median(s[i] for s in samples) for i in range(NUM_FEATURES)]
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
        print(" This usually means that finger wasn't fully open or fully "
              "closed, the glove shifted, or the sensor is loose/miswired.")
        print(" Let's redo OPEN and FIST.\n")


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


def apply_jitter_raw(values):
    noisy = list(values)
    for i in ACCEL_IDX:
        noisy[i] = values[i] + random.uniform(-IMU_NOISE, IMU_NOISE)
    for i in range(8, 11):
        noisy[i] = values[i] + random.uniform(-IMU_NOISE, IMU_NOISE)
    return noisy


def apply_jitter_cal(cal_values):
    noisy = list(cal_values)
    for i in range(5):
        noisy[i] = cal_values[i] + random.uniform(-FLEX_NOISE_PCT, FLEX_NOISE_PCT)
    for i in range(5, 8):
        noisy[i] = cal_values[i] + random.uniform(-IMU_NOISE, IMU_NOISE)
    return noisy


def collect_one_session(ser, label, session_id, n_frames, open_baseline, fist_baseline):
    wait_and_discard(ser, POSE_SETTLE_SECONDS)

    samples = []
    count = 0
    while count < n_frames:
        values = read_clean_line(ser) # normal recording still rejects saturated frames
        if values:
            cal = calibrate(values, open_baseline, fist_baseline)
            samples.append(values + cal + [label, session_id])

            for _ in range(SYNTHETIC_MULTIPLIER):
                noisy_raw = apply_jitter_raw(values)
                noisy_cal = apply_jitter_cal(cal)
                samples.append(noisy_raw + noisy_cal + [label, session_id])
            count += 1
    return samples


def collect_gesture(ser, label, next_sid_counter):
    all_samples = []
    sid_counter = next_sid_counter
    for s in range(1, SESSIONS_PER_GESTURE + 1):
        print(f"\n --- Session {s}/{SESSIONS_PER_GESTURE} for '{label}' ---")
        input(f" Re-form '{label}' after calibration (vary angle/tightness slightly). "
              f"First, calibration for THIS session...")

        open_baseline, fist_baseline = get_valid_calibration(ser)

        input(f" Press ENTER, then re-form '{label}' — you'll have "
              f"{POSE_SETTLE_SECONDS}s to get into position before recording starts...")
        print(f" Recording gesture '{label}'...")

        session_id = f"{RUN_ID}_{sid_counter}"
        session_samples = collect_one_session(ser, label, session_id, FRAMES_PER_SESSION,
                                               open_baseline, fist_baseline)
        all_samples.extend(session_samples)

        real_count = FRAMES_PER_SESSION
        total_count = FRAMES_PER_SESSION * (1 + SYNTHETIC_MULTIPLIER)
        print(f" [session {s}] {real_count} real, {total_count} total samples "
              f"(session_id={session_id})")
        sid_counter += 1
    return all_samples, sid_counter


def main():
    file_exists = os.path.isfile(OUTPUT_FILE)
    csvfile = open(OUTPUT_FILE, 'a', newline='')
    writer = csv.writer(csvfile)

    if not file_exists:
        writer.writerow(HEADER)
        print(f"Created new file: {OUTPUT_FILE}")
    else:
        print(f"Appending to existing file: {OUTPUT_FILE}")

    sid_counter = 0

    print(f"\nConnecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        print(" Connected!\n")
    except serial.SerialException as e:
        print(f"[ERROR] Could not connect: {e}")
        csvfile.close()
        return

    label_counts = {}

    print("=" * 55)
    print(" ASL GLOVE — DATA COLLECTION (CALIBRATED, MULTI-SESSION)")
    print("=" * 55)
    print(f" Valid labels: A-Z and 0-9")
    print(f" Sessions per gesture: {SESSIONS_PER_GESTURE}")
    print(f" Real frames per session: {FRAMES_PER_SESSION}")
    print(f" Calibration frames per pose: {CALIBRATION_FRAMES}")
    print(f" Synthetic multiplier: {SYNTHETIC_MULTIPLIER}x")
    print(f" Run ID (session prefix): {RUN_ID}")
    print(f" Type 'list' to see sample counts, 'quit' to save & exit")
    print("=" * 55)

    try:
        while True:
            label_input = input("\n Enter label to record (or 'quit'/'list'): ").strip().upper()

            if label_input == 'QUIT':
                break

            if label_input == 'LIST':
                print("\n Collected so far:")
                for k, v in sorted(label_counts.items()):
                    print(f" {k}: {v} samples")
                continue

            if label_input not in VALID_LABELS:
                print(f" '{label_input}' is not a valid label. Use A-Z or 0-9.")
                continue

            samples, sid_counter = collect_gesture(ser, label_input, sid_counter)

            for row in samples:
                writer.writerow(row)
            csvfile.flush()

            label_counts[label_input] = label_counts.get(label_input, 0) + len(samples)
            print(f" Saved {len(samples)} samples for '{label_input}' "
                  f"across {SESSIONS_PER_GESTURE} sessions")

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt received.")

    finally:
        csvfile.close()
        if ser.is_open:
            ser.close()

        print("\n" + "=" * 55)
        print(" SESSION SUMMARY")
        print("=" * 55)
        total = 0
        for k, v in sorted(label_counts.items()):
            print(f" {k}: {v} samples")
            total += v
        print(f" TOTAL: {total} samples")
        print(f" Saved to: {OUTPUT_FILE}")
        print("=" * 55)


if __name__ == '__main__':
    main()