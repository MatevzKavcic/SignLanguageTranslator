#!/bin/bash

read -p "Try number: " TRY


python 6TrainingTesting_LSTM.py "$TRY"



mv *.h5 "${TRY}/models"


echo "model is saved in the folder: ${TRY}/models as lstm_model.h5"


