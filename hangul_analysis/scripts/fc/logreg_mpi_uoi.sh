#!/bin/bash -l

conda activate hangul_analysis
HANGUL_SAVE=$HOME/hangul_save

for imf in "i" "m" "f"; do
  for var in {0..8}; do
    for fold in {0..6};do
      for layer in {0..6};do
        python -u logreg_mpi_uoi.py $HANGUL_SAVE test $imf $fold $layer $var
      done
    done
  done
done
