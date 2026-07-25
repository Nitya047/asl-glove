"""
train_model.py
---------------
Loads asl_data.csv, trains a Random Forest classifier (+ optional SVM),
evaluates it, and saves the model + scaler to disk.

Train/test split and cross-validation are GROUP-AWARE (grouped by
session_id, if present). Randomly shuffling rows into train/test lets
synthetic clones / frames from the same continuous hold appear on both
sides of the split, which makes accuracy look great without the model
actually generalizing. Splitting by session_id means an entire recording
session lives entirely in train OR entirely in test — never both.

USAGE:
    python train_model.py

REQUIREMENTS:
    pip install pandas scikit-learn matplotlib seaborn joblib

OUTPUT:
    asl_model.pkl — trained Random Forest model
    asl_scaler.pkl — fitted StandardScaler (MUST be used at prediction time)
    confusion_matrix.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_FILE = 'asl_data_calibrated.csv'
MODEL_FILE = 'asl_model.pkl'
SCALER_FILE = 'asl_scaler.pkl'
ENCODER_FILE = 'asl_label_encoder.pkl'

# Feature set toggles — flip these to test which sensors actually help.
# Flex sensors are always used. Gyro (angular velocity) is the noisiest
# signal for STATIC holds — try USE_GYRO = False first if there's
# a lot of "uncertain" results. Accel (orientation vs gravity) is usually
# worth keeping since it can be the only thing separating two letters
# that have identical finger curl but different wrist tilt.
USE_ACCEL = True
USE_GYRO = False # toggled off — gyro fluctuates too much for static holds

# Training on the calibrated, drift-free sensor data.
# The sensor swap / middle-finger fault needs no special handling here:
# the calibration formula in collect_data.py maps open->0, fist->1
# regardless of whether raw values increase or decrease with bending, and
# regardless of a sensor's absolute raw scale — so middle_cal behaves like
# any other calibrated flex feature to this script, just with less
# resolution on partial bends (see collect_data.py's KNOWN_FAULTY_FINGERS
# comment for why).
FLEX_COLS = ['thumb_cal', 'index_cal', 'middle_cal', 'ring_cal', 'pinky_cal']
ACCEL_COLS = ['ax_cal', 'ay_cal', 'az_cal'] if USE_ACCEL else []
GYRO_COLS = ['gx', 'gy', 'gz'] if USE_GYRO else [] # (Gyro remains raw as we didn't baseline it)
FEATURE_COLS = FLEX_COLS + ACCEL_COLS + GYRO_COLS

LABEL_COL = 'label'
GROUP_COL = 'session_id' # falls back gracefully if this column is absent
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ────────────────────────────────────────────────────────────────────────────


def load_and_inspect(filepath):
    print(f"Loading data from '{filepath}'...")
    df = pd.read_csv(filepath)
    print(f" Shape: {df.shape}")
    print(f"\n Samples per label:")
    counts = df[LABEL_COL].value_counts().sort_index()
    for label, count in counts.items():
        bar = '█' * (count // 5)
        print(f" {label:>3}: {count:>4} {bar}")

    low_labels = counts[counts < 20].index.tolist()
    if low_labels:
        print(f"\n Labels with < 20 samples (consider collecting more): {low_labels}")

    if GROUP_COL not in df.columns:
        print(f"\n No '{GROUP_COL}' column found — this file was likely collected "
              f"with the OLD single-hold collect_data.py. Falling back to a plain "
              f"random split, which is prone to leakage between near-duplicate "
              f"frames. Re-collect with collect_data.py for a fair split.")
        df[GROUP_COL] = np.arange(len(df)) # every row its own group == old behavior
    else:
        sessions_per_label = df.groupby(LABEL_COL)[GROUP_COL].nunique()
        print(f"\n Sessions per label:")
        for label, n in sessions_per_label.sort_index().items():
            flag = " only 1 session — no real variance to learn from!" if n <= 1 else ""
            print(f" {label:>3}: {n} sessions{flag}")

    return df


def engineer_features(df):
    """
    Add a few derived features to help the model.
    These are simple statistics that capture overall hand shape.
    Only computes accel/gyro-derived features for sensors that are
    actually enabled via USE_ACCEL / USE_GYRO above.
    """
    df = df.copy()

    df['flex_sum'] = df[FLEX_COLS].sum(axis=1)
    df['flex_spread'] = df[FLEX_COLS].max(axis=1) - df[FLEX_COLS].min(axis=1)
    extra_cols = ['flex_sum', 'flex_spread']

    if USE_ACCEL:
        df['accel_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        extra_cols.append('accel_mag')
    if USE_GYRO:
        df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
        extra_cols.append('gyro_mag')

    return df, extra_cols


def train_random_forest(X_train, y_train):
    print("\n Training Random Forest...")
    # Lightly regularized vs. the old unlimited-depth forest: capping depth
    # and raising min_samples_leaf trades a bit of train-set accuracy for
    # much better generalization on genuinely new hand positions.
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def train_svm(X_train, y_train):
    print(" Training SVM (this may take a moment)...")
    svm = CalibratedClassifierCV(SVC(kernel='rbf', C=1, gamma='scale'), cv=3)
    svm.fit(X_train, y_train)
    return svm


def evaluate(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n [{model_name}] Test Accuracy: {acc*100:.2f}%")

    classes = [str(c) for c in label_encoder.classes_]
    present = sorted(set(y_test) | set(y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred,
                                 labels=present,
                                 target_names=[classes[i] for i in present]))

    return y_pred, acc


def plot_confusion_matrix(y_test, y_pred, classes, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f" Confusion matrix saved to '{filename}'")
    plt.close()


def plot_feature_importance(model, feature_names, filename='feature_importance.png'):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f" Feature importances saved to '{filename}'")
    plt.close()


def main():
    if not os.path.isfile(DATA_FILE):
        print(f"[ERROR] '{DATA_FILE}' not found. Run collect_data.py first.")
        return

    df = load_and_inspect(DATA_FILE)
    df, extra_cols = engineer_features(df)

    all_feature_cols = FEATURE_COLS + extra_cols
    print(f"\n Using feature columns: {all_feature_cols}")

    X = df[all_feature_cols].values
    raw_labels = df[LABEL_COL].values
    groups = df[GROUP_COL].values

    le = LabelEncoder()
    y = le.fit_transform(raw_labels)
    print(f"\n Classes ({len(le.classes_)}): {list(le.classes_)}")

    # ── Group-aware train/test split ────────────────────────────────────
    # An entire session_id (real frame + its synthetic clones, or an entire
    # old-style continuous hold) goes entirely to train or entirely to test.
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\n Train: {len(X_train)} samples ({len(set(groups[train_idx]))} sessions) | "
          f"Test: {len(X_test)} samples ({len(set(groups[test_idx]))} sessions)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 55)
    print(" TRAINING")
    print("=" * 55)

    rf_model = train_random_forest(X_train_scaled, y_train)
    svm_model = train_svm(X_train_scaled, y_train)

    print("\n" + "=" * 55)
    print(" EVALUATION")
    print("=" * 55)

    rf_preds, rf_acc = evaluate(rf_model, X_test_scaled, y_test, le, "Random Forest")
    svm_preds, svm_acc = evaluate(svm_model, X_test_scaled, y_test, le, "SVM")

    # ── Group-aware cross-validation ────────────────────────────────────
    n_groups = len(set(groups))
    n_splits = min(5, n_groups) if n_groups >= 2 else 1
    if n_splits >= 2:
        print(f"\n Running {n_splits}-fold GROUP cross-validation on Random Forest...")
        gkf = GroupKFold(n_splits=n_splits)
        cv_scores = cross_val_score(rf_model, scaler.transform(X), y,
                                     groups=groups, cv=gkf, scoring='accuracy', n_jobs=-1)
        print(f" CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    else:
        print("\n Skipping cross-validation — not enough distinct sessions/groups.")

    best_model = rf_model if rf_acc >= svm_acc else svm_model
    best_name = "Random Forest" if rf_acc >= svm_acc else "SVM"
    best_preds = rf_preds if rf_acc >= svm_acc else svm_preds
    print(f"\n Best model: {best_name} ({max(rf_acc, svm_acc)*100:.2f}% accuracy)")

    present = sorted(set(y_test) | set(best_preds))
    plot_confusion_matrix(y_test, best_preds, [le.classes_[i] for i in present],
                          f'Confusion Matrix — {best_name}',
                          'confusion_matrix.png')
    plot_feature_importance(rf_model, all_feature_cols)

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)

    print(f"\n Saved model → '{MODEL_FILE}'")
    print(f" Saved scaler → '{SCALER_FILE}'")
    print(f" Saved encoder → '{ENCODER_FILE}'")
    print("\n Run predict_live.py to use the model in real time!")


if __name__ == '__main__':
    main()
