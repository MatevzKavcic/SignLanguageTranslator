#!/bin/bash

read -p "Try number: " TRY

echo "making the embedding model for try number: $TRY"

python 7EmbedingTime.py "$TRY"


echo "embedding model is saved in the folder: ${TRY}/models as embedding_model.keras"


echo "embeding the data for try number: $TRY"

python 8Embedings.py "$TRY"

echo "embeddings are saved in the folder: ${TRY}/models as embeddings.npy"

mkdir -p "${TRY}/embeddingsNpy"

mv embeddings.npy "${TRY}/embeddingsNpy"
mv embedding_labels.npy "${TRY}/embeddingsNpy"

echo "embeddings are saved in the folder: ${TRY}/embeddingsNpy as embeddings.npy and embedding_labels.npy"

