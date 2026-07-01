import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CONFIG
# ==========================================
CSV_FILE = "4.1FifthTryNormalization.csv"
OUTPUT_FOLDER = "ProcessedNPYdataNEW"

NUM_AUGMENTATIONS = 3

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================
print("Loading dataset...", flush=True)
df = pd.read_csv(CSV_FILE)

exclude = ["frame", "video", "label"]
feature_cols = [c for c in df.columns if c not in exclude]

print(f"Total features: {len(feature_cols)}")

# ==========================================
# GROUP INTO SEQUENCES
# ==========================================
print("Grouping sequences...", flush=True)

X, y = [], []

for video_name, group in df.groupby("video"):
    group = group.sort_values("frame")

    label = group["label"].iloc[0]
    seq = group[feature_cols].values.astype(np.float32)

    X.append(seq)
    y.append(label)

X = np.array(X, dtype=object)
y = np.array(y)

print(f"Found {len(X)} sequences\n")

# ==========================================
# LABEL ENCODING
# ==========================================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

np.save(f"{OUTPUT_FOLDER}/labels.npy", encoder.classes_)

# ==========================================
# FEATURE GROUPS
# ==========================================
def get_feature_indices(columns):
    face_idx, pose_idx, left_idx, right_idx = [], [], [], []

    for i, col in enumerate(columns):
        if col.startswith("face_"):
            face_idx.append(i)
        elif col.startswith("pose_"):
            pose_idx.append(i)
        elif col.startswith("left_hand_"):
            left_idx.append(i)
        elif col.startswith("right_hand_"):
            right_idx.append(i)

    return face_idx, pose_idx, left_idx, right_idx


face_idx, pose_idx, left_idx, right_idx = get_feature_indices(feature_cols)

print("Feature groups:")
print("Face:", len(face_idx))
print("Pose:", len(pose_idx))
print("Left:", len(left_idx))
print("Right:", len(right_idx))

# ==========================================
# AUGMENTATION FUNCTION (CORRECT VERSION)
# ==========================================
def augment_sequence(seq):
    seq = seq.copy()

    # preserve missing landmarks
    mask = (seq != 0)

    # FACE noise (very small)
    seq[:, face_idx] += np.random.normal(
        0, 0.001, seq[:, face_idx].shape
    )

    # POSE noise (small)
    seq[:, pose_idx] += np.random.normal(
        0, 0.005, seq[:, pose_idx].shape
    )

    # LEFT HAND noise (stronger)
    seq[:, left_idx] += np.random.normal(
        0, 0.020, seq[:, left_idx].shape
    )

    # RIGHT HAND noise (stronger)
    seq[:, right_idx] += np.random.normal(
        0, 0.020, seq[:, right_idx].shape
    )

    # restore missing values
    seq *= mask

    return seq

# ==========================================
# CREATE AUGMENTED DATASET
# ==========================================
print("\nCreating augmented dataset...\n")

X_aug, y_aug = [], []

total = len(X)

for i, (seq, label) in enumerate(zip(X, y_encoded), start=1):

    print(f"[{i}/{total}] Processing sequence")

    # original
    X_aug.append(seq)
    y_aug.append(label)

    print("   -> original added")

    # augmented copies
    for aug in range(NUM_AUGMENTATIONS):

        noisy = augment_sequence(seq)

        X_aug.append(noisy)
        y_aug.append(label)

        print(f"   -> augmentation {aug+1}/{NUM_AUGMENTATIONS}")

# ==========================================
# SAVE
# ==========================================
X_aug = np.array(X_aug, dtype=object)
y_aug = np.array(y_aug)

np.save(f"{OUTPUT_FOLDER}/X_aug.npy", X_aug)
np.save(f"{OUTPUT_FOLDER}/y_aug.npy", y_aug)

# ==========================================
# SUMMARY
# ==========================================
print("\n===================================")
print("AUGMENTATION COMPLETE")
print("===================================")
print(f"Original sequences : {len(X)}")
print(f"Final sequences    : {len(X_aug)}")
print(f"Augmentations      : {NUM_AUGMENTATIONS}")
print(f"Saved to           : {OUTPUT_FOLDER}")
print("===================================")
