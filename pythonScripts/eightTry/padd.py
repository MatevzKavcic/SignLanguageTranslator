import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# CONFIG
# ==========================================

INPUT_FILE = "SplitBySigns/X_test.npy"
OUTPUT_FILE = "SplitBySigns/X_test_padded.npy"

MAX_SEQUENCE_LENGTH = 205

# ==========================================
# LOAD
# ==========================================

print("Loading data...")

X = np.load(INPUT_FILE, allow_pickle=True)

print("Original number of sequences:", len(X))

# ==========================================
# PAD
# ==========================================

print("Padding sequences...")

X_padded = pad_sequences(
    X,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post",
    dtype="float32"
)

print("New shape:", X_padded.shape)

# ==========================================
# SAVE
# ==========================================

np.save(OUTPUT_FILE, X_padded)

print("\nDone.")
print("Saved to:", OUTPUT_FILE)
