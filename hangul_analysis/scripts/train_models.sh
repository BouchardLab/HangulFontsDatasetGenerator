#!/bin/bash -l
conda activate hangul_analysis

HANGUL_SAVE=$HOME/hangul_save
H5_DIR=$HOME/hangul/h5s

mkdir -p $HANGUL_SAVE/output

for fold in {0..9}; do
  for kind in "i" "m" "f"; do
    python -u train_dense.py $H5_DIR $HANGUL_SAVE test $fold 200 --device cpu &
  done
done
wait
