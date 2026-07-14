import numpy as np

X = np.load("eightTry/SplitBySigns/train_classes.npy", allow_pickle=True)
print(len(X))
import numpy as np

# ==========================================
# CHANGE THESE PATHS
# ==========================================

X_FILE = "eightTry/SplitBySigns/X_test.npy"
Y_FILE = "eightTry/SplitBySigns/y_test.npy"
LABELS_FILE = "eightTry/SplitBySigns/labels.npy"

# ==========================================
# LOAD
# ==========================================

print("Loading files...\n")

X = np.load(X_FILE, allow_pickle=True)
y = np.load(Y_FILE, allow_pickle=True)
labels = np.load(LABELS_FILE, allow_pickle=True)

# ==========================================
# BASIC INFO
# ==========================================

print("=" * 60)
print("X")
print("=" * 60)

print("type      :", type(X))
print("dtype     :", X.dtype)
print("shape     :", X.shape)
print("length    :", len(X))

print()

print("=" * 60)
print("y")
print("=" * 60)

print("type      :", type(y))
print("dtype     :", y.dtype)
print("shape     :", y.shape)
print("length    :", len(y))

print()

print("=" * 60)
print("labels")
print("=" * 60)

print("type      :", type(labels))
print("dtype     :", labels.dtype)
print("shape     :", labels.shape)
print("length    :", len(labels))

print()


# ==========================================
# CHECK THAT EVERY SEQUENCE HAS SAME FEATURES
# ==========================================

print("\nChecking feature dimensions...")

feature_dims = set()

for seq in X:
    feature_dims.add(seq.shape[1])

print("Feature dimensions found:", feature_dims)

# ==========================================
# CHECK SEQUENCE LENGTHS
# ==========================================

lengths = [len(seq) for seq in X]

print("\nShortest sequence :", min(lengths))
print("Longest sequence  :", max(lengths))
print("Average length    :", np.mean(lengths))

# ==========================================
# CHECK LABELS
# ==========================================

print("\nUnique labels:", len(np.unique(y)))
print("Min label    :", np.min(y))
print("Max label    :", np.max(y))

