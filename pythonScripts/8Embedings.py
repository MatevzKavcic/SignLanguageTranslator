import numpy as np

from tensorflow.keras.models import load_model  # type: ignore 
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore 
import sys

Try_number = sys.argv[1]

# ==========================================
# CONFIG
# ==========================================

DATA_FOLDER = f"eightTry/SplitBySigns/"   #{Try_number}/{Try_number}_Testing

MODEL_FILE = f"eightTry/embedding_model_UNSEEN.keras"

OUTPUT_EMBEDDINGS = "embeddings_UNSEEN_test.npy"
OUTPUT_LABELS = "embedding_labels_UNSEEN_test.npy"

# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading sequences...\n")

X = np.load(
    f"{DATA_FOLDER}/X_test_padded.npy",
    allow_pickle=True
)

y = np.load(
    f"{DATA_FOLDER}/y_test.npy"
)

labels = np.load(
    f"{DATA_FOLDER}/labels.npy",
    allow_pickle=True
)

print("Sequences :", len(X))
print("Labels    :", len(y))
print("Classes   :", len(labels))

# ==========================================
# PAD TO SAME LENGTH
# ==========================================

print("\nPadding sequences...\n")

X = pad_sequences(
    X,
    padding="post",
    dtype="float32"
)

print("Shape:", X.shape)

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading embedding model...\n")

embedding_model = load_model(
    MODEL_FILE,
    compile=False
)

embedding_model.summary()

# ==========================================
# COMPUTE EMBEDDINGS
# ==========================================

print("\nGenerating embeddings...\n")

embeddings = embedding_model.predict(
    X,
    batch_size=32,
    verbose=1
)

print()

print("Embedding shape:", embeddings.shape)

# ==========================================
# SAVE
# ==========================================

np.save(
    OUTPUT_EMBEDDINGS,
    embeddings
)

np.save(
    OUTPUT_LABELS,
    y
)

print("\n====================================")
print("Finished")
print("====================================")
print("Embeddings saved :", OUTPUT_EMBEDDINGS)
print("Labels saved     :", OUTPUT_LABELS)
print("Number of vectors:", len(embeddings))
print("Vector dimension :", embeddings.shape[1])
print("====================================")
