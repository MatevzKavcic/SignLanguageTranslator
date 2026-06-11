import pandas as pd # type: ignore
import numpy as np
from sklearn.preprocessing import LabelEncoder# type: ignore

CSV_FILE = "../TestingCleaned/testing_cleanedFULLY.csv" 

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE)

exclude = ["frame", "video", "label"]
feature_cols = [c for c in df.columns if c not in exclude]

print("Total features:", len(feature_cols))

# =========================
# GROUP INTO SEQUENCES
# =========================
X = []
y = []

grouped = df.groupby("video")

for video_name, group in grouped:
    group = group.sort_values("frame")

    label = group["label"].iloc[0]
    seq = group[feature_cols].values.astype(np.float32)

    X.append(seq)
    y.append(label)

X = np.array(X, dtype=object)
y = np.array(y)

# =========================
# LABEL ENCODING
# =========================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

np.save("labels.npy", encoder.classes_)

# =========================
# FIND COLUMN INDEX GROUPS
# =========================
def get_feature_indices(columns):
    face_idx = []
    pose_idx = []
    left_idx = []
    right_idx = []

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

print("Face features:", len(face_idx))
print("Pose features:", len(pose_idx))
print("Left hand features:", len(left_idx))
print("Right hand features:", len(right_idx))

# =========================
# AUGMENTATION FUNCTION
# =========================
def augment_sequence(seq):
    seq = seq.copy()

    # FACE → VERY SMALL NOISE
    seq[:, face_idx] += np.random.normal(0, 0.001, seq[:, face_idx].shape)

    # POSE → SMALL NOISE
    seq[:, pose_idx] += np.random.normal(0, 0.005, seq[:, pose_idx].shape)

    # LEFT HAND → STRONGER NOISE
    seq[:, left_idx] += np.random.normal(0, 0.02, seq[:, left_idx].shape)

    # RIGHT HAND → STRONGER NOISE
    seq[:, right_idx] += np.random.normal(0, 0.02, seq[:, right_idx].shape)

    return seq


# =========================
# CREATE AUGMENTED DATASET
# =========================
X_aug = []
y_aug = []

for seq, label in zip(X, y_encoded):

    # original
    X_aug.append(seq)
    y_aug.append(label)

    # 3 augmented versions per sample
    for _ in range(3):
        X_aug.append(augment_sequence(seq))
        y_aug.append(label)

X_aug = np.array(X_aug, dtype=object)
y_aug = np.array(y_aug)

# =========================
# SAVE
# =========================
np.save("X_aug.npy", X_aug)
np.save("y_aug.npy", y_aug)

print("\n✅ DONE")
print("Original samples:", len(X))
print("Augmented samples:", len(X_aug))
print("Saved X_aug.npy + y_aug.npy")