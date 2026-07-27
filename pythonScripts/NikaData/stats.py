import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ==========================================================
# CONFIG
# ==========================================================

# Your test videos
X_FILE = "NikaDataX.npy"
Y_FILE = "NikaDatay.npy"
LABELS_FILE = "NikaDatalabel_classes.npy"


# Embedding database
DATABASE_EMBEDDINGS = "../TrySeven/embeddingsNpy/embeddings.npy"
DATABASE_LABELS = "../TrySeven/embeddingsNpy/embedding_labels.npy"


# Trained embedding network
MODEL_FILE = "../TrySeven/models/embedding_model.keras"


OUTPUT_CSV = "embedding_top5_results.csv"


# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading data...\n")


X = np.load(
    X_FILE,
    allow_pickle=True
)


y = np.load(
    Y_FILE
)


label_names = np.load(
    LABELS_FILE,
    allow_pickle=True
)


database_embeddings = np.load(
    DATABASE_EMBEDDINGS
)


database_labels = np.load(
    DATABASE_LABELS
)


print("Test sequences:", X.shape)
print("Database embeddings:", database_embeddings.shape)



# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading embedding model...\n")


embedding_model = load_model(
    MODEL_FILE
)



# ==========================================================
# CREATE TEST EMBEDDINGS
# ==========================================================

print("\nCreating embeddings...\n")


test_embeddings = embedding_model.predict(
    X,
    batch_size=32,
    verbose=1
)


print(
    "Embedding shape:",
    test_embeddings.shape
)



# ==========================================================
# EVALUATION
# ==========================================================


top1 = 0
top2 = 0
top3 = 0
top4 = 0
top5 = 0


results = []


missing_from_top5 = []



for i, (embedding, true_label) in enumerate(
        zip(test_embeddings, y)
):


    # ------------------------------------------
    # calculate distances
    # ------------------------------------------

    distances = np.linalg.norm(
        database_embeddings - embedding,
        axis=1
    )


    sorted_indices = np.argsort(
        distances
    )


    sorted_labels = database_labels[
        sorted_indices
    ]


    sorted_distances = distances[
        sorted_indices
    ]



    # ------------------------------------------
    # find correct sign rank
    # ------------------------------------------

    correct_positions = np.where(
        sorted_labels == true_label
    )[0]


    if len(correct_positions) > 0:

        correct_rank = (
            correct_positions[0] + 1
        )

        correct_distance = (
            sorted_distances[
                correct_positions[0]
            ]
        )

    else:

        correct_rank = -1
        correct_distance = np.nan



    # ------------------------------------------
    # Top K
    # ------------------------------------------

    top5_labels = sorted_labels[:5]


    if true_label in sorted_labels[:1]:
        top1 += 1

    if true_label in sorted_labels[:2]:
        top2 += 1

    if true_label in sorted_labels[:3]:
        top3 += 1

    if true_label in sorted_labels[:4]:
        top4 += 1

    if true_label in sorted_labels[:5]:
        top5 += 1

    else:

        missing_from_top5.append(i)



    # ------------------------------------------
    # closest wrong sign
    # ------------------------------------------

    wrong_mask = (
        sorted_labels != true_label
    )


    wrong_index = np.where(
        wrong_mask
    )[0][0]


    wrong_label = (
        sorted_labels[wrong_index]
    )


    wrong_distance = (
        sorted_distances[wrong_index]
    )



    margin = (
        wrong_distance -
        correct_distance
    )



    # ------------------------------------------
    # SAVE RESULT
    # ------------------------------------------

    results.append({

        "sample": i,

        "true_sign":
            label_names[int(true_label)],

        "nearest_prediction":
            label_names[
                int(sorted_labels[0])
            ],


        "correct_rank":
            correct_rank,


        "correct_distance":
            correct_distance,


        "nearest_wrong_sign":
            label_names[
                int(wrong_label)
            ],


        "nearest_wrong_distance":
            wrong_distance,


        "margin":
            margin,


        "top5_found":
            true_label in sorted_labels[:5]

    })



# ==========================================================
# PRINT RESULTS
# ==========================================================


total = len(y)


print("\n==============================")
print("TOP K RESULTS")
print("==============================")


print(
    f"Top-1 accuracy: "
    f"{100*top1/total:.2f}%"
)


print(
    f"Top-2 accuracy: "
    f"{100*top2/total:.2f}%"
)


print(
    f"Top-3 accuracy: "
    f"{100*top3/total:.2f}%"
)


print(
    f"Top-4 accuracy: "
    f"{100*top4/total:.2f}%"
)


print(
    f"Top-5 accuracy: "
    f"{100*top5/total:.2f}%"
)



print("\n==============================")
print("FAILED TOP-5 CASES")
print("==============================")


print(
    f"Signs not found in top 5: "
    f"{len(missing_from_top5)}"
)


for idx in missing_from_top5[:20]:

    r = results[idx]

    print("\nSample:", idx)

    print(
        "True:",
        r["true_sign"]
    )

    print(
        "Closest:",
        r["nearest_prediction"]
    )

    print(
        "Correct sign rank:",
        r["correct_rank"]
    )

    print(
        "Correct distance:",
        r["correct_distance"]
    )

    print(
        "Closest wrong distance:",
        r["nearest_wrong_distance"]
    )

    print(
        "Margin:",
        r["margin"]
    )



# ==========================================================
# SAVE CSV
# ==========================================================


df = pd.DataFrame(
    results
)


df.to_csv(
    OUTPUT_CSV,
    index=False
)


print(
    "\nDetailed results saved:",
    OUTPUT_CSV
)