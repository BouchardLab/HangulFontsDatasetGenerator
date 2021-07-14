#!/bin/bash -l

conda activate hangul_analysis
HANGUL_SAVE=$HOME/bvae

for imf in "i" "m" "f" "trav" "trav2" "vae" "nvae"; do
  for var in {0..8}; do
    for fold in {0..6};do
      python -u vae_logreg_uoi.py $HANGUL_SAVE $imf $fold $var
    done
  done
done
