#!/bin/bash

TRY=$1

echo "runing training for LSTM model"

python 6TrainingTesting_LSTM.py "$TRY"



mv *.h5 "${TRY}/models"


echo "model is saved in the folder: ${TRY}/models as lstm_model.h5"


