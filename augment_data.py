"""
augment_data.py — Augment existing reps + generate synthetic training data.

Reads:  lateral_raise_tcn_dataset.csv   (140 reps, 35/class)
Writes: lateral_raise_tcn_augmented.csv (~1380 reps, ~345/class)

After this runs:
  1. Open config.py
  2. Change CSV_PATH to point at lateral_raise_tcn_augmented.csv
  3. Run: python train_model.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)

BASE     = Path(__file__).parent
CSV_IN   = BASE / "lateral_raise_tcn_dataset.csv"
CSV_OUT  = BASE / "lateral_raise_tcn_augmented.csv"

SEQ_LEN    = 30
N_FEATURES = 6

# Feature indices within each frame's 6 values
F_SHOULDER = 0   # shoulder_abduction (degrees)
F_ELBOW    = 1   # elbow_flexion      (degrees, straight ≈ 155-180)
F_TORSO    = 2   # torso_lean         (degrees, upright ≈ 169-175)
F_WRIST_H  = 3   # wrist_height       (normalised 0-1)
F_VELOCITY = 4   # shoulder_velocity  (deg/frame, can be negative)
F_SYMMETRY = 5   # arm_symmetry       (degrees difference)

LABELS = ["correct", "elbow_bent", "not_high_enough", "torso_lean"]


# ─── Augmentation helpers ─────────────────────────────────────────────────────

def add_noise(seq: np.ndarray) -> np.ndarray:
    sigma = rng.uniform(0.8, 2.5)
    noise = rng.normal(0, sigma, seq.shape)
    noise[:, F_WRIST_H]  *= 0.015   # wrist_height is 0-1 scale — tiny sigma
    noise[:, F_VELOCITY] *= 1.8     # velocity is naturally noisy
    return seq + noise


def time_warp(seq: np.ndarray) -> np.ndarray:
    shift = int(rng.integers(-2, 3))
    return np.roll(seq, shift, axis=0)


def scale_amplitude(seq: np.ndarray) -> np.ndarray:
    out    = seq.copy()
    factor = rng.uniform(0.92, 1.08)
    out[:, [F_SHOULDER, F_ELBOW, F_TORSO, F_VELOCITY, F_SYMMETRY]] *= factor
    return out


def augment_rep(seq: np.ndarray, n: int = 5) -> list:
    results = []
    for _ in range(n):
        s = add_noise(seq.copy())
        if rng.random() > 0.4:
            s = time_warp(s)
        if rng.random() > 0.4:
            s = scale_amplitude(s)
        results.append(s)
    return results


# ─── Synthetic data generators ────────────────────────────────────────────────

def _smooth(start: float, peak: float, end: float, n: int = SEQ_LEN) -> np.ndarray:
    """Smooth up→hold→down curve (mimics a real raise)."""
    up   = int(n * 0.40)
    hold = int(n * 0.15)
    down = n - up - hold
    return np.concatenate([
        np.linspace(start, peak, up),
        np.full(hold, peak),
        np.linspace(peak, end, down),
    ])


def _velocity(shoulder: np.ndarray) -> np.ndarray:
    v = np.diff(shoulder, prepend=shoulder[0])
    return v + rng.normal(0, 0.25, SEQ_LEN)


def _synth_correct(n: int) -> list:
    rows = []
    for _ in range(n):
        peak    = rng.uniform(75, 93)
        start   = rng.uniform(20, 35)
        sh      = _smooth(start, peak, start + rng.uniform(-3, 3))
        elbow   = rng.uniform(158, 179) + rng.normal(0, 1.5, SEQ_LEN)
        torso   = rng.uniform(169, 174) + rng.normal(0, 0.5, SEQ_LEN)
        wrist_h = _smooth(rng.uniform(0.05, 0.10), rng.uniform(0.72, 0.92), rng.uniform(0.05, 0.10))
        sym     = rng.uniform(0, 8) + np.abs(rng.normal(0, 1.5, SEQ_LEN))
        rows.append(("correct", np.stack([sh, elbow, torso, wrist_h, _velocity(sh), sym], axis=1)))
    return rows


def _synth_elbow_bent(n: int) -> list:
    """Elbow clearly bent (110-145°) throughout the raise."""
    rows = []
    for _ in range(n):
        peak    = rng.uniform(72, 92)
        start   = rng.uniform(18, 33)
        sh      = _smooth(start, peak, start + rng.uniform(-3, 3))
        elbow   = rng.uniform(112, 145) + rng.normal(0, 2.5, SEQ_LEN)
        torso   = rng.uniform(169, 174) + rng.normal(0, 0.5, SEQ_LEN)
        wrist_h = _smooth(rng.uniform(0.05, 0.10), rng.uniform(0.60, 0.85), rng.uniform(0.05, 0.10))
        sym     = rng.uniform(2, 12) + np.abs(rng.normal(0, 1.5, SEQ_LEN))
        rows.append(("elbow_bent", np.stack([sh, elbow, torso, wrist_h, _velocity(sh), sym], axis=1)))
    return rows


def _synth_not_high_enough(n: int) -> list:
    """Arms peak at 35-62° — well below the 70° threshold."""
    rows = []
    for _ in range(n):
        peak    = rng.uniform(35, 62)
        start   = rng.uniform(15, 28)
        sh      = _smooth(start, peak, start + rng.uniform(-3, 3))
        elbow   = rng.uniform(155, 178) + rng.normal(0, 1.5, SEQ_LEN)
        torso   = rng.uniform(169, 174) + rng.normal(0, 0.5, SEQ_LEN)
        wrist_h = _smooth(rng.uniform(0.05, 0.10), rng.uniform(0.32, 0.52), rng.uniform(0.05, 0.10))
        sym     = rng.uniform(0, 8) + np.abs(rng.normal(0, 1.2, SEQ_LEN))
        rows.append(("not_high_enough", np.stack([sh, elbow, torso, wrist_h, _velocity(sh), sym], axis=1)))
    return rows


def _synth_torso_lean(n: int) -> list:
    """Torso tilts — angle drops to 155-167° instead of staying ≥168°."""
    rows = []
    for _ in range(n):
        peak    = rng.uniform(68, 90)
        start   = rng.uniform(18, 33)
        sh      = _smooth(start, peak, start + rng.uniform(-3, 3))
        elbow   = rng.uniform(155, 178) + rng.normal(0, 1.5, SEQ_LEN)
        torso   = rng.uniform(155, 167) + rng.normal(0, 1.2, SEQ_LEN)
        wrist_h = _smooth(rng.uniform(0.05, 0.10), rng.uniform(0.62, 0.85), rng.uniform(0.05, 0.10))
        sym     = rng.uniform(3, 15) + np.abs(rng.normal(0, 2.0, SEQ_LEN))
        rows.append(("torso_lean", np.stack([sh, elbow, torso, wrist_h, _velocity(sh), sym], axis=1)))
    return rows


# ─── Row serialisation ────────────────────────────────────────────────────────

def seq_to_row(label: str, seq: np.ndarray) -> dict:
    row = {"label": label}
    for t in range(SEQ_LEN):
        for f in range(N_FEATURES):
            row[f"f{t}_{f}"] = float(seq[t, f])
    return row


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} original reps")
    print(df["label"].value_counts().to_string())

    feature_cols = [c for c in df.columns if c != "label"]
    rows = []

    # 1. Keep all originals
    for _, row in df.iterrows():
        rows.append(row.to_dict())

    # 2. Augment each original × 5  → 140 × 5 = 700 extra
    for _, row in df.iterrows():
        label = row["label"]
        seq   = row[feature_cols].values.astype(np.float32).reshape(SEQ_LEN, N_FEATURES)
        for aug in augment_rep(seq, n=5):
            rows.append(seq_to_row(label, aug))

    print(f"\nAfter augmentation: {len(rows)} reps")

    # 3. Synthetic data — 135 per class → keeps class balance with augmented
    n_syn = 135
    synthetic = (
        _synth_correct(n_syn)
        + _synth_elbow_bent(n_syn)
        + _synth_not_high_enough(n_syn)
        + _synth_torso_lean(n_syn)
    )
    for label, seq in synthetic:
        rows.append(seq_to_row(label, seq))

    # 4. Shuffle and save
    out = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"After synthetic:   {len(out)} reps total")
    print(f"\nFinal label distribution:")
    print(out["label"].value_counts().to_string())

    out.to_csv(CSV_OUT, index=False)
    print(f"\nSaved: {CSV_OUT}")
    print()
    print("Next steps:")
    print("  1. Open config.py")
    print(f"  2. Change CSV_PATH to: {CSV_OUT}")
    print("  3. Run: python train_model.py")


if __name__ == "__main__":
    main()
