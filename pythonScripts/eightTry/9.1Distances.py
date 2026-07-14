import numpy as np


# ==========================================================
# CONFIG
# ==========================================================

TRAIN_EMBEDDINGS = "embeddings_UNSEEN_train.npy"
TRAIN_LABELS = "embedding_labels_UNSEEN_train.npy"

TEST_EMBEDDINGS = "embeddings_UNSEEN_test.npy"
TEST_LABELS = "embedding_labels_UNSEEN_test.npy"



# ==========================================================
# LOAD
# ==========================================================

print("Loading embeddings...\n")

train_embeddings = np.load(TRAIN_EMBEDDINGS)
train_labels = np.load(TRAIN_LABELS)

test_embeddings = np.load(TEST_EMBEDDINGS)
test_labels = np.load(TEST_LABELS)

print("Training embeddings :", train_embeddings.shape)
print("Testing embeddings  :", test_embeddings.shape)

# ==========================================================
# KNOWN DISTANCES
# ==========================================================

print("\nComputing known distances...")

known_distances = []

for i in range(len(train_embeddings)):

    d = np.linalg.norm(
        train_embeddings - train_embeddings[i],
        axis=1
    )

    # ignore itself
    d[i] = np.inf

    known_distances.append(np.min(d))

known_distances = np.array(known_distances)

# ==========================================================
# UNKNOWN DISTANCES
# ==========================================================

print("Computing unknown distances...")

unknown_distances = []

for emb in test_embeddings:

    d = np.linalg.norm(
        train_embeddings - emb,
        axis=1
    )

    unknown_distances.append(np.min(d))

unknown_distances = np.array(unknown_distances)

# ==========================================================
# STATISTICS
# ==========================================================

print("\n==============================")
print("KNOWN")
print("==============================")

print("Samples :", len(known_distances))
print("Mean    :", known_distances.mean())
print("Median  :", np.median(known_distances))
print("Std     :", known_distances.std())
print("Min     :", known_distances.min())
print("Max     :", known_distances.max())

print("\n==============================")
print("UNKNOWN")
print("==============================")

print("Samples :", len(unknown_distances))
print("Mean    :", unknown_distances.mean())
print("Median  :", np.median(unknown_distances))
print("Std     :", unknown_distances.std())
print("Min     :", unknown_distances.min())
print("Max     :", unknown_distances.max())

# ==========================================================
# THRESHOLD SEARCH
# ==========================================================

print("\n==============================")
print("Threshold evaluation")
print("==============================")

all_distances = np.concatenate(
    [known_distances, unknown_distances]
)

thresholds = np.linspace(
    all_distances.min(),
    all_distances.max(),
    50
)

best_acc = 0
best_threshold = 0

for threshold in thresholds:

    # Known should be BELOW threshold
    tp = np.sum(known_distances <= threshold)

    # Known classified as unknown
    fn = np.sum(known_distances > threshold)

    # Unknown correctly rejected
    tn = np.sum(unknown_distances > threshold)

    # Unknown incorrectly accepted
    fp = np.sum(unknown_distances <= threshold)

    accuracy = (tp + tn) / (
        tp + tn + fp + fn
    )

    if accuracy > best_acc:
        best_acc = accuracy
        best_threshold = threshold

print("\nBest threshold :", best_threshold)
print("Best accuracy  :", best_acc)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

tp = np.sum(known_distances <= best_threshold)
fn = np.sum(known_distances > best_threshold)

tn = np.sum(unknown_distances > best_threshold)
fp = np.sum(unknown_distances <= best_threshold)

print("\n==============================")
print("Confusion matrix")
print("==============================")

print(f"TP : {tp}")
print(f"FN : {fn}")
print(f"FP : {fp}")
print(f"TN : {tn}")

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("\nPrecision :", precision)
print("Recall    :", recall)
print("F1-score  :", f1)

print("\nDone.")
