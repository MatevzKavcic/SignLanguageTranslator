import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# CONFIG
# ==========================================================


MODEL = "SPLITED/embedding_model_SPLITED.keras"

TRAIN_X = "TrySeven_Testing/X_train.npy"
TRAIN_Y = "TrySeven_Testing/y_train.npy"

TEST_X = "TrySeven_Testing/X_test.npy"
TEST_Y = "TrySeven_Testing/y_test.npy"

LABELS = "TrySeven_Testing/labels.npy"

MAX_K = 5

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("Loading data...")
print("=" * 60)

X_train = np.load(TRAIN_X, allow_pickle=True)
y_train = np.load(TRAIN_Y)

X_test = np.load(TEST_X, allow_pickle=True)
y_test = np.load(TEST_Y)

label_names = np.load(LABELS, allow_pickle=True)

print(f"Training sequences : {len(X_train)}")
print(f"Testing sequences  : {len(X_test)}")
print(f"Number of classes  : {len(label_names)}")

# ==========================================================
# PAD SEQUENCES
# ==========================================================

print("\nPadding sequences...")

X_train = pad_sequences(
    X_train,
    padding="post",
    dtype="float32"
)

X_test = pad_sequences(
    X_test,
    maxlen=X_train.shape[1],
    padding="post",
    dtype="float32"
)

print("Training shape :", X_train.shape)
print("Testing shape  :", X_test.shape)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading embedding model...")

embedding_model = load_model(
    MODEL,
    compile=False
)

print("Model loaded.")

# ==========================================================
# CREATE TRAIN EMBEDDINGS
# ==========================================================

print("\nEmbedding training sequences...")

train_embeddings = embedding_model.predict(
    X_train,
    batch_size=32,
    verbose=1
)

print("Training embeddings:", train_embeddings.shape)

# ==========================================================
# CREATE TEST EMBEDDINGS
# ==========================================================

print("\nEmbedding testing sequences...")

test_embeddings = embedding_model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

print("Testing embeddings:", test_embeddings.shape)

# ==========================================================
# TOP-K EVALUATION USING UNIQUE SIGNS
# ==========================================================

print("\n")
print("=" * 60)
print("Evaluating Top-K (unique sign labels)")
print("=" * 60)

accuracies = []

for K in range(1, MAX_K + 1):

    correct = 0

    for emb, true_label in zip(test_embeddings, y_test):

        distances = np.linalg.norm(
            train_embeddings - emb,
            axis=1
        )

        sorted_idx = np.argsort(distances)

        unique_predictions = []

        seen = set()

        for idx in sorted_idx:

            lbl = int(y_train[idx])

            if lbl not in seen:
                seen.add(lbl)
                unique_predictions.append(lbl)

            if len(unique_predictions) == K:
                break

        if int(true_label) in unique_predictions:
            correct += 1

    accuracy = correct / len(y_test)

    accuracies.append(accuracy)

    print(
        f"Top-{K:<2} | "
        f"{correct:4d}/{len(y_test):4d} | "
        f"{accuracy*100:6.2f}%"
    )

    if accuracy == 1.0:
        print(f"\nReached 100% at Top-{K}.")
        break

# ==========================================================
# EXAMPLE PREDICTIONS
# ==========================================================

print("\n")
print("=" * 60)
print("Example predictions")
print("=" * 60)

NUM_EXAMPLES = min(10, len(test_embeddings))

for example in range(NUM_EXAMPLES):

    emb = test_embeddings[example]
    true_label = int(y_test[example])

    distances = np.linalg.norm(
        train_embeddings - emb,
        axis=1
    )

    sorted_idx = np.argsort(distances)

    unique_predictions = []
    seen = set()

    for idx in sorted_idx:

        lbl = int(y_train[idx])

        if lbl not in seen:
            seen.add(lbl)
            unique_predictions.append((lbl, distances[idx]))

        if len(unique_predictions) == 5:
            break

    print("\n------------------------------------------------")

    print(
        "True label :",
        label_names[true_label]
    )

    print("Top-5 predictions:")

    for rank, (lbl, dist) in enumerate(unique_predictions, start=1):

        print(
            f"{rank}. "
            f"{label_names[lbl]:25}"
            f"{dist:.5f}"
        )

print("\n")
print("=" * 60)
print("Finished")
print("=" * 60)
