
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import os

# =========================
# LOAD DATA
# =========================
X = np.load("X.npy", allow_pickle=True)
y = np.load("y.npy", allow_pickle=True)

print(f"Loaded {len(X)} sequences")

# =========================
# PAD SEQUENCES
# =========================
# Pads to longest sequence automatically
X_padded = pad_sequences(
    X,
    padding="post",
    dtype="float32"
)

print("Padded shape:", X_padded.shape)

# =========================
# ENCODE LABELS
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label mapping for later inference
np.save("labels.npy", label_encoder.classes_)

print("Labels:", label_encoder.classes_)

# =========================
# SAVE PROCESSED DATA
# =========================
os.makedirs("processed", exist_ok=True)

np.save("processed/X_padded.npy", X_padded)
np.save("processed/y_encoded.npy", y_encoded)

print("Saved padded dataset to /processed/")