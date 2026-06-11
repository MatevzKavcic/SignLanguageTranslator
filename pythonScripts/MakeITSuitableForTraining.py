import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ---------------- CONFIG ----------------
CSV_FILE = "TestingCleaned/testing_cleanedFULLY.csv"

X_OUTPUT = "X.npy"
Y_OUTPUT = "y.npy"
LABEL_ENCODER_OUTPUT = "label_classes.npy"

# ---------------- LOAD CSV ----------------
print("Loading CSV...")
df = pd.read_csv(CSV_FILE)

# ---------------- FEATURE COLUMNS ----------------
exclude = ["frame", "video", "label"]

feature_cols = [
    col for col in df.columns
    if col not in exclude
]

print(f"Found {len(feature_cols)} feature columns")

# ---------------- GROUP VIDEOS ----------------
X = []
y = []

grouped = df.groupby("video")

for video_name, group in grouped:
    print(f"Processing {video_name}")

    # sort frames
    group = group.sort_values("frame")

    # label (same for whole video)
    label = group["label"].iloc[0]

    # get only landmark features
    sequence = group[feature_cols].values

    X.append(sequence)
    y.append(label)

# ---------------- LABEL ENCODING ----------------
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# save label names
np.save(LABEL_ENCODER_OUTPUT, encoder.classes_)

# ---------------- SAVE ----------------
# dtype=object allows different sequence lengths
X = np.array(X, dtype=object)
y_encoded = np.array(y_encoded)

np.save(X_OUTPUT, X)
np.save(Y_OUTPUT, y_encoded)

print("\n✅ Done!")
print(f"Saved X -> {X_OUTPUT}")
print(f"Saved y -> {Y_OUTPUT}")
print(f"Classes -> {LABEL_ENCODER_OUTPUT}")

print("\nDataset info:")
print(f"Videos: {len(X)}")
print(f"Classes: {len(encoder.classes_)}")
