import numpy as np
import os, pickle, sys

from hangul.read_data import load_data
from hangul.utils import resize
from hangul.ml import logreg
from hangul.fontslist import fontnames, n_blocks


def pairwise_logreg(base_path, fontsize):

    fontsize = int(fontsize)
    n_fonts = len(fontnames)
    Cs = np.logspace(-1.5, 1.5, 5)
    results = np.full((n_fonts, n_fonts, Cs.size, 3), np.nan)
    correlation = np.full((n_fonts, n_fonts), np.nan)

    with open('median_fontsize.pkl', 'rb') as f:
        median_h, median_w = pickle.load(f)[fontsize]

    all_x = np.full((n_fonts, n_blocks, median_h * median_w), np.nan)
    all_y = np.full((n_fonts, n_blocks, 3), np.nan)

    for ii, fontname in enumerate(fontnames):
        fontname = os.path.splitext(fontname)[0]
        fname = os.path.join(base_path, 'h5s', fontname,
                             '{}_{}.h5'.format(fontname, fontsize))
        imgs, y, _, _, _ = load_data(fname)
        all_x[ii] = resize(imgs,
                           (n_blocks, median_h, median_w)).reshape(n_blocks, -1)
        all_y[ii] = y

    for ii in range(n_fonts):
        for jj in range(n_fonts):
            train_x = all_x[ii]
            train_y = all_y[ii]

            init_train_y = train_y[:, 0]
            med_train_y = train_y[:, 1]
            fin_train_y = train_y[:, 2]

            test_x = all_x[jj]
            test_y = all_y[jj]

            init_test_y = test_y[:, 0]
            med_test_y = test_y[:, 1]
            fin_test_y = test_y[:, 2]

            x0 = train_x - train_x.mean(axis=0, keepdims=True)
            x0 = x0 / x0.std(axis=0, keepdims=True)
            x1 = test_x - test_x.mean(axis=0, keepdims=True)
            x1 = x1 / x1.std(axis=0, keepdims=True)
            correlation[ii, jj] = np.mean(x0 * x1)

            for kk, C in enumerate(Cs):
                init_score, _ = logreg(train_x, init_train_y,
                                       test_x, init_test_y, C=C)
                med_score, _ = logreg(train_x, med_train_y,
                                      test_x, med_test_y, C=C)
                fin_score, _ = logreg(train_x, fin_train_y,
                                      test_x, fin_test_y, C=C)

                results[ii, jj, kk, 0] = init_score
                results[ii, jj, kk, 1] = med_score
                results[ii, jj, kk, 2] = fin_score

    np.savez('pairwise_logreg_{}.npz'.format(fontsize),
             results=results, correlation=correlation,
             fontnames=fontnames, fontsize=fontsize, Cs=Cs)


if __name__ == "__main__":
    base_folder = sys.argv[1]
    fontsize = sys.argv[2]
    pairwise_logreg(base_folder, fontsize)
