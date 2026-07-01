# ==========================================================
# SIAMESE LSTM FOR SIGN LANGUAGE
#
# Input:
#   X_aug.npy
#   y_aug.npy
#   labels.npy
#
# Output:
#   embedding_model.keras
#   siamese_model.keras
#
# ==========================================================

import time
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Masking,
    Lambda
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FOLDER = "ProcessedNPYdataNEW"

X_FILE = f"{DATA_FOLDER}/X_aug.npy"
Y_FILE = f"{DATA_FOLDER}/y_aug.npy"
LABEL_FILE = f"{DATA_FOLDER}/labels.npy"

EMBEDDING_SIZE = 128

BATCH_SIZE = 16

EPOCHS = 100

RANDOM_STATE = 42

# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading data...\n")

X = np.load(
    X_FILE,
    allow_pickle=True
)

y = np.load(Y_FILE)

labels = np.load(
    LABEL_FILE,
    allow_pickle=True
)

print("Number of sequences :", len(X))
print("Number of labels    :", len(y))
print("Classes             :", len(labels))




# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

print()

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))



# ==========================================================
# PAD SEQUENCES
# ==========================================================

print("\nPadding sequences...\n")

X_train = pad_sequences(
    X_train,
    padding="post",
    dtype="float32"
)

X_test = pad_sequences(
    X_test,
    padding="post",
    dtype="float32"
)

print("Training shape :", X_train.shape)
print("Testing shape  :", X_test.shape)


# ==========================================================
# BUILD CLASS INDEX
# ==========================================================

from collections import defaultdict

print("\nBuilding class index...\n")

class_indices = defaultdict(list)

for idx, label in enumerate(y_train):

    class_indices[label].append(idx)

print("Classes found :", len(class_indices))

for label in sorted(class_indices.keys())[:5]:
    print(
        f"Class {label}:",
        len(class_indices[label]),
        "samples"
    )



# ==========================================================
# CREATE SIAMESE PAIRS
# ==========================================================

def create_pairs(X, y):

    print("Unique labels:", np.unique(y))
    print("Number of classes:", len(np.unique(y)))

    left = []
    right = []
    targets = []

    class_indices = defaultdict(list)

    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    labels = list(class_indices.keys())

    print("\nGenerating pairs...\n")

    pair_counter = 0

    for label in labels:

        indices = class_indices[label]

        # Need at least two examples
        if len(indices) < 2:
            continue

        # -----------------------------
        # POSITIVE PAIRS
        # -----------------------------

        for i in range(len(indices)):

            for j in range(i + 1, len(indices)):

                left.append(X[indices[i]])
                right.append(X[indices[j]])
                targets.append(1)

                pair_counter += 1

        # -----------------------------
        # NEGATIVE PAIRS
        # -----------------------------

        negative_labels = [
            l for l in labels
            if l != label
        ]

        for idx_positive in indices:

            random_label = random.choice(
                negative_labels
            )

            random_index = random.choice(
                class_indices[random_label]
            )

            left.append(X[idx_positive])
            right.append(X[random_index])
            targets.append(0)

            pair_counter += 1

        print(
            f"Processed class {label} "
            f"({pair_counter} pairs)"
        )

    print("\nFinished creating pairs.\n")

    return (
        np.array(left),
        np.array(right),
        np.array(targets).astype(np.float32)
    )



# ==========================================================
# TRAINING PAIRS
# ==========================================================

train_left, train_right, train_targets = create_pairs(
    X_train,
    y_train
)

print()

print("Training pairs :", len(train_targets))


# ==========================================================
# VALIDATION PAIRS
# ==========================================================

test_left, test_right, test_targets = create_pairs(
    X_test,
    y_test
)

print()

print("Validation pairs :", len(test_targets))


# ==========================================================
# SHUFFLE PAIRS
# ==========================================================

train_perm = np.random.permutation(
    len(train_targets)
)

train_left = train_left[train_perm]
train_right = train_right[train_perm]
train_targets = train_targets[train_perm]

test_perm = np.random.permutation(
    len(test_targets)
)

test_left = test_left[test_perm]
test_right = test_right[test_perm]
test_targets = test_targets[test_perm]

print("\nPairs shuffled.\n")

# ==========================================================
# CHECK
# ==========================================================

print("Left train :", train_left.shape)
print("Right train:", train_right.shape)
print("Targets    :", train_targets.shape)

print()

print("Left test  :", test_left.shape)
print("Right test :", test_right.shape)
print("Targets    :", test_targets.shape)





# ==========================================================
# EMBEDDING NETWORK
# ==========================================================

print("\nBuilding embedding network...\n")

input_shape = (
    X_train.shape[1],
    X_train.shape[2]
)

inputs = Input(shape=input_shape)

# Ignore padded frames
x = Masking(mask_value=0.0)(inputs)

x = LSTM(
    128,
    return_sequences=True
)(x)

x = Dropout(0.3)(x)

x = LSTM(
    64
)(x)

x = Dropout(0.3)(x)

x = Dense(
    128,
    activation="relu"
)(x)

embedding = Dense(
    EMBEDDING_SIZE,
    activation=None,
    name="embedding"
)(x)

embedding_model = Model(
    inputs,
    embedding,
    name="EmbeddingModel"
)

embedding_model.summary()


# ==========================================================
# DISTANCE LAYER
# ==========================================================

def euclidean_distance(vectors):

    x, y = vectors

    return tf.sqrt(
        tf.reduce_sum(
            tf.square(x - y),
            axis=1,
            keepdims=True
        ) + 1e-8
    )


# ==========================================================
# SIAMESE MODEL
# ==========================================================

left_input = Input(shape=input_shape)

right_input = Input(shape=input_shape)

left_embedding = embedding_model(left_input)

right_embedding = embedding_model(right_input)

distance = Lambda(
    euclidean_distance
)([
    left_embedding,
    right_embedding
])

siamese_model = Model(
    inputs=[left_input, right_input],
    outputs=distance
)

siamese_model.summary()


# ==========================================================
# CONTRASTIVE LOSS
# ==========================================================

def contrastive_loss(
    y_true,
    y_pred,
    margin=1.0
):

    y_true = tf.cast(y_true, tf.float32)

    square_pred = tf.square(y_pred)

    margin_square = tf.square(
        tf.maximum(
            margin - y_pred,
            0
        )
    )

    return tf.reduce_mean(

        y_true * square_pred +

        (1 - y_true) * margin_square

    )



# ==========================================================
# COMPILE
# ==========================================================

siamese_model.compile(

    optimizer="adam",

    loss=contrastive_loss

)


# ==========================================================
# CALLBACKS
# ==========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=100,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "best_siamese.keras",

    monitor="val_loss",

    save_best_only=True

)



# ==========================================================
# TRAIN
# ==========================================================

print("\nStarting training...\n")

history = siamese_model.fit(

    [train_left, train_right],

    train_targets,

    validation_data=(

        [test_left, test_right],

        test_targets

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[

        early_stop,

        checkpoint

    ]

)


# ==========================================================
# SAVE
# ==========================================================

embedding_model.save(

    "embedding_model.keras"

)

siamese_model.save(

    "siamese_model.keras"

)

print("\nDone.\n")
