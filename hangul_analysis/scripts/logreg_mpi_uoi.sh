#!/bin/bash -l
conda activate hangul_analysis

HANGUL_SAVE=$HOME/hangul_save

for imf in "i" "m" "f"; do
  for var in {6..8}; do
    python -u logreg_mpi_uoi.py $HANGUL_SAVE test $imf 0 0 $var &
  done
done
wait
