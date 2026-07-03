#!/bin/bash

set -e

echo "====================================="
echo " Sign Language Prediction Pipeline"
echo "====================================="

echo
echo "[1/6] Extracting landmarks..."
python 1ExtractFromVideos.py

echo
echo "[2/6] Converting CSV to UTF-8..."
python 2MakeUTF.py

echo
echo "[3/6] Cleaning sequence..."
python 3CleanVideo.py

echo
echo "[4/6] Normalizing sequence..."
python 4Normalize.py

echo
echo "[5/6] Computing embedding..."
python 5embedCSV.py

echo
echo "[6/6] Predicting sign..."
python 6GuessTheDistance.py

echo
echo "====================================="
echo " Prediction finished!"
echo "====================================="


echo "Organizing output files..."

mkdir -p csvFiles
mkdir -p npyFiles

# Move CSV files
mv -f ./*.csv csvFiles/ 2>/dev/null || true

# Move NPY files
mv -f ./*.npy npyFiles/ 2>/dev/null || true

echo
echo "====================================="
echo " Prediction finished!"
echo " CSV files -> csvFiles/"
echo " NPY files -> npyFiles/"
echo "====================================="
