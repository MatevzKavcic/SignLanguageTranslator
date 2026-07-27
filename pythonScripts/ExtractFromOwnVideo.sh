#!/bin/bash

VIDEO=$1
TRY=$2

python 1.1ExtractWithNoAugmentation.py "$VIDEO" "$TRY"
python 2makeItUTF8.py "$TRY"
python 3CleanTheData.py "$TRY"
python 4Normalization.py "$TRY"
python HelperScripts/2MakeITSuitableForTraining.py "$TRY"

echo  "normalized dataset of your videos is save in the folder: ${TRY}/${TRY}_npyData"

mkdir -p "${TRY}"

mkdir -p "${TRY}/csvFiles"

mkdir -p "${TRY}/models"


mv *.csv "${TRY}/csvFiles"

mv  "${TRY}_npyData" "${TRY}"



echo "If you want to test the embedding model predictions, please run the following command:"

echo " popravi tako do bo path do prave scripte"





