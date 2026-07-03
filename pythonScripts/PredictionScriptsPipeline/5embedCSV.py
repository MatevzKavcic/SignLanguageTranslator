import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# CONFIG
# ==========================================================

CSV_FILE = "CleanedSequence.csv"
MODEL_PATH = "../embedding_model.keras"

OUTPUT_FILE = "single_video_embedding.npy"

# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading embedding model...\n")
model = load_model(MODEL_PATH)

# ==========================================================
# LOAD CSV
# ==========================================================

print("\nLoading CSV...\n")

df = pd.read_csv(CSV_FILE)

exclude = ["frame", "video", "label"]
feature_cols = [c for c in df.columns if c not in exclude]

print("Total features:", len(feature_cols))

# ==========================================================
# BUILD SINGLE SEQUENCE
# ==========================================================

print("\nBuilding sequence...\n")

df = df.sort_values("frame")

sequence = df[feature_cols].values.astype(np.float32)

print("Raw shape:", sequence.shape)

# ==========================================================
# PAD SEQUENCE
# ==========================================================
MAX_SEQUENCE_LENGTH = 163  # same value used during training

sequence = pad_sequences(
    [sequence],
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post",
    dtype="float32"
)

# ==========================================================
# EMBEDDING
# ==========================================================

print("\nComputing embedding...\n")

embedding = model.predict(sequence, verbose=0)[0]

print("Embedding shape:", embedding.shape)
print("Embedding vector:\n", embedding)

# ==========================================================
# SAVE
# ==========================================================

np.save(OUTPUT_FILE, embedding)

print("\nSaved:", OUTPUT_FILE)
