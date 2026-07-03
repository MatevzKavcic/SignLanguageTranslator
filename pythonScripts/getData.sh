#!/bin/bash



read -p "Video path: " VIDEO
read -p "Try number: " TRY

python 1Extraction.py "$VIDEO" "$TRY"
python 2MakeItUTF.py "$TRY"
python 3CleanTheData.py "$TRY"
python 4Normalization.py "$TRY"
python 5Augmentation.py "$TRY"


echo  "Dataset is saved in the folder: ${TRY}/${TRY}_npyData"

mkdir -p "${TRY}"

mkdir -p "${TRY}/csvFiles"

mkdir -p "${TRY}/models"


mv *.csv "${TRY}/csvFiles"

mv  "${TRY}_npyData" "${TRY}"




echo "If you want to run the embedding model, please run the following command:"    

echo " .EmbeddingModel.sh $TRY"

echo "If you want to run the LSTM model, please run the following command:"

echo " .LSTMModel.sh $TRY"



