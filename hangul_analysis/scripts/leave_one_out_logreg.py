import numpy as np
import os, pickle, sys

from hangul.read_data import load_data
from hangul.ml import logreg_model
from hangul.ml import LeaveOneFontOutCV


def leave_one_out_logreg(h5_folder, fontsize):

    fontsize = int(fontsize)
    Cs = np.logspace(-2, 2, 9)
    n_folds = 7
    results = np.full((n_folds, 3, Cs.size, 3), np.nan)
    chance = np.zeros_like(results) + np.nan
    locs = os.walk(h5_folder)
    files = []
    for base, _, fnames in locs:
        for f in fnames:
            if '{}.h5'.format(fontsize) in f:
                files.append(os.path.join(base, f))

    for fold in range(n_folds):
        for jj, imf in enumerate(['i', 'm', 'f']):
            print(fold, imf)
            ds = LeaveOneFontOutCV(files, fold, imf=imf, ravel=True)
            X, y = ds.training_set()
            keep_feat = X.std(axis=0) > 0.
            X = X[:, keep_feat]
            Xv, yv = ds.validation_set()
            Xv = Xv[:, keep_feat]
            Xt, yt = ds.test_set()
            Xt = Xt[:, keep_feat]
            for kk, C in enumerate(Cs):
                model = logreg_model(X, y, C=C)
                train_score = model.score(X, y)
                val_score = model.score(Xv, yv)
                test_score = model.score(Xt, yt)
                results[fold, jj, kk, 0] = train_score
                results[fold, jj, kk, 1] = val_score
                results[fold, jj, kk, 2] = test_score

                model = logreg_model(X, np.random.permutation(y), C=C)
                train_score = model.score(X, np.random.permutation(y))
                val_score = model.score(Xv, np.random.permutation(yv))
                test_score = model.score(Xt, np.random.permutation(yt))
                chance[fold, jj, kk, 0] = train_score
                chance[fold, jj, kk, 1] = val_score
                chance[fold, jj, kk, 2] = test_score

    np.savez('leave_one_out_logreg_{}.npz'.format(fontsize),
             results=results, chance=chance, fonts=files, fontsize=fontsize,
             Cs=Cs)


if __name__ == "__main__":
    h5_folder = sys.argv[1]
    fontsize = sys.argv[2]
    leave_one_out_logreg(h5_folder, fontsize)
