import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# CHANGE THESE PATHS
# =====================================================

X_FILE = "ProcessedNPYdataNEW/X_aug.npy"

EMBEDDINGS_FILE = "embedingsNpyFiles/embeddings.npy"

MODEL_FILE = "embedding_model.keras"

SEQUENCE_INDEX = 0      # choose any sequence

# =====================================================
# LOAD DATA
# =====================================================

print("Loading sequences...")

X = np.load(
    X_FILE,
    allow_pickle=True
)

print("Padding...")

X = pad_sequences(
    X,
    padding="post",
    dtype="float32"
)

print("Shape:", X.shape)

print("Loading saved embeddings...")

saved_embeddings = np.load(EMBEDDINGS_FILE)

print(saved_embeddings.shape)

print("Loading embedding model...")

model = load_model(
    MODEL_FILE,
    compile=False
)

# =====================================================
# COMPUTE EMBEDDING AGAIN
# =====================================================

sequence = X[SEQUENCE_INDEX:SEQUENCE_INDEX+1]

new_embedding = model.predict(
    sequence,
    verbose=0
)[0]

saved_embedding = saved_embeddings[SEQUENCE_INDEX]

# =====================================================
# COMPARE
# =====================================================

difference = np.linalg.norm(
    new_embedding - saved_embedding
)

maximum = np.max(
    np.abs(new_embedding - saved_embedding)
)

print("\n==============================")
print("Sequence:", SEQUENCE_INDEX)
print("==============================")

print("Euclidean difference :", difference)
print("Maximum difference   :", maximum)

print()

print("First 10 values")

for i in range(10):
    print(
        f"{i:2d}: "
        f"saved={saved_embedding[i]: .6f}   "
        f"new={new_embedding[i]: .6f}"
    )

print()

if difference < 1e-6:
    print("✅ PERFECT MATCH")
elif difference < 1e-3:
    print("✅ Almost identical (floating point error)")
else:
    print("❌ Different embeddings!")
