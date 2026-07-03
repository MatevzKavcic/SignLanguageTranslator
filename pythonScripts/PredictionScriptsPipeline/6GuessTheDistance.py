import numpy as np

TOP_K = 120
# ==========================================================
# CONFIG
# ==========================================================
LABEL_NAMES = np.load("../ProcessedNPYdataNEW/labels.npy", allow_pickle=True)
DATABASE_EMBEDDINGS = "../embedingsNpyFiles/embeddings.npy"
DATABASE_LABELS = "../embedingsNpyFiles/embedding_labels.npy"

UNKNOWN_EMBEDDING = "single_video_embedding.npy"


# ==========================================================
# LOAD
# ==========================================================

print("Loading embeddings...")

db_embeddings = np.load(DATABASE_EMBEDDINGS)
db_labels = np.load(DATABASE_LABELS, allow_pickle=True)

unknown = np.load(UNKNOWN_EMBEDDING)

print("Database:", db_embeddings.shape)
print("Unknown :", unknown.shape)

# Make sure unknown has shape (128,)
unknown = unknown.reshape(-1)

# ==========================================================
# DISTANCES
# ==========================================================

distances = []

for embedding, label in zip(db_embeddings, db_labels):

    distance = np.linalg.norm(embedding - unknown)

    distances.append((distance, label))

# ==========================================================
# SORT
# ==========================================================

distances.sort(key=lambda x: x[0])

# ==========================================================
# RESULTS
# ==========================================================

for i in range(min(TOP_K, len(distances))):

    d, class_id = distances[i]

    sign_name = LABEL_NAMES[int(class_id)]

    print(
        f"{i+1}. {sign_name:20} distance = {d:.4f}"
    )

print("\nPrediction:")

best_class = int(distances[0][1])

print(LABEL_NAMES[best_class])
