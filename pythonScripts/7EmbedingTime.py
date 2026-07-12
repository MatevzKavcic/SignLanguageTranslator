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
# =====================================================


import time
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import (Input,LSTM,Dense,Dropout,Masking,Lambda) # type: ignore
from tensorflow.keras.callbacks import (EarlyStopping,ModelCheckpoint) # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import sys

Try_number = sys.argv[1]

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FOLDER = f"{Try_number}/{Try_number}_npyData"

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

class SiameseGenerator(Sequence):

    def __init__(
        self,
        X,
        y,
        batch_size=16,
        steps_per_epoch=1000,
        shuffle=True
    ):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.shuffle = shuffle

        self.class_indices = defaultdict(list)

        for i, label in enumerate(y):
            self.class_indices[label].append(i)

        self.labels = list(self.class_indices.keys())

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):

        left_batch = []
        right_batch = []
        target_batch = []

        for _ in range(self.batch_size):

            # ---------------------------------
            # Positive or negative pair
            # ---------------------------------

            positive = random.random() < 0.5

            label = random.choice(self.labels)

            indices = self.class_indices[label]

            if positive and len(indices) >= 2:

                i1, i2 = random.sample(indices, 2)

                left_batch.append(self.X[i1])
                right_batch.append(self.X[i2])
                target_batch.append(1.0)

            else:

                label2 = random.choice(self.labels)

                while label2 == label:
                    label2 = random.choice(self.labels)

                i1 = random.choice(indices)
                i2 = random.choice(self.class_indices[label2])

                left_batch.append(self.X[i1])
                right_batch.append(self.X[i2])
                target_batch.append(0.0)

        return (
            (
                np.asarray(left_batch, dtype=np.float32),
                np.asarray(right_batch, dtype=np.float32),
            ),
            np.asarray(target_batch, dtype=np.float32),
        )


print("\nCreating generators...\n")

train_generator = SiameseGenerator(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    steps_per_epoch=1000
)

validation_generator = SiameseGenerator(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    steps_per_epoch=200
)
# ==========================================================
# CHECK
# ==========================================================

# print("Left train :", train_left.shape)
# print("Right train:", train_right.shape)
# print("Targets    :", train_targets.shape)

# print()

# print("Left test  :", test_left.shape)
# print("Right test :", test_right.shape)
# print("Targets    :", test_targets.shape)





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

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stop,
        checkpoint
    ]

)

# ==========================================================
# SAVE
# ==========================================================

embedding_model.save(

    f"{Try_number}_embedding_model.keras"

)

siamese_model.save(

    f"{Try_number}_siamese_model.keras"

)

print("\nDone.\n")

