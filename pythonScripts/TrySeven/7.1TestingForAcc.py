import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# CONFIG
# =====================================================

MODEL = "SPLITED/embedding_model_SPLITED.keras"

TRAIN_X = "TrySeven_Testing/X_train.npy"
TRAIN_Y = "TrySeven_Testing/y_train.npy"

TEST_X = "TrySeven_Testing/X_test.npy"
TEST_Y = "TrySeven_Testing/y_test.npy"

LABELS = "TrySeven_Testing/labels.npy"

# =====================================================
# LOAD
# =====================================================

print("Loading data...")

X_train = np.load(TRAIN_X, allow_pickle=True)
y_train = np.load(TRAIN_Y)

X_test = np.load(TEST_X, allow_pickle=True)
y_test = np.load(TEST_Y)

label_names = np.load(LABELS, allow_pickle=True)

# =====================================================
# PAD
# =====================================================

print("Padding...")

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

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading embedding model...")

model = load_model(MODEL, compile=False)

# =====================================================
# EMBED
# =====================================================

print("Embedding training set...")

train_embeddings = model.predict(
    X_train,
    batch_size=32,
    verbose=1
)

print("Embedding testing set...")

test_embeddings = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

# =====================================================
# NEAREST NEIGHBOUR TEST
# =====================================================

correct = 0

print("\n=====================================\n")

for i in range(len(test_embeddings)):

    dists = np.linalg.norm(
        train_embeddings - test_embeddings[i],
        axis=1
    )

    nearest = np.argmin(dists)

    predicted = y_train[nearest]
    truth = y_test[i]

    if predicted == truth:
        correct += 1
        result = "✓"
    else:
        result = "✗"

    print(
        f"{i+1:4d}: "
        f"GT={label_names[truth]:20}"
        f"PRED={label_names[predicted]:20}"
        f"Dist={dists[nearest]:.4f} {result}"
    )

# =====================================================
# RESULTS
# =====================================================

accuracy = correct / len(y_test)

print("\n=====================================")
print(f"Correct : {correct}")
print(f"Total   : {len(y_test)}")
print(f"Accuracy: {accuracy*100:.2f}%")
print("=====================================")
