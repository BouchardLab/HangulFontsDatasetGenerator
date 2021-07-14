# Logistic Regression for single fontname, single fontsize

import numpy as np
import sys, os

from hangul.read_data import load_data
from hangul.ml import logreg
from hangul.fontslist import fontnames, fontsizes


def single_logreg(base_path, seed=20180810):

    rng = np.random.RandomState(seed)
    n_fonts = len(fontnames)
    n_sizes = len(fontsizes)
    cv = 5
    Cs = np.logspace(-1.5, 1.5, 5)
    results = np.full((n_fonts, n_sizes, 3, Cs.size, cv), np.nan)

    for ii, fname in enumerate(fontnames):
        base_fname = os.path.splitext(fname)[0]
        for jj, fsize in enumerate(fontsizes):
            fname = os.path.join(base_path, 'h5s', base_fname,
                                 '{}_{}.h5'.format(base_fname, fsize))
            print(fname)
            imgs, labels, _, _, _ = load_data(fname)
            imgs = imgs.reshape(imgs.shape[0], -1)
            for kk in range(cv):
                idx = rng.permutation(len(imgs))
                imgs, labels = imgs[idx], labels[idx]
                n_test = len(imgs) // 10

                initial_labels = labels[:, 0]
                medial_labels = labels[:, 1]
                final_labels = labels[:, 2]

                test_x = imgs[:n_test]
                train_x = imgs[n_test:]

                init_test_y = initial_labels[:n_test]
                init_train_y = initial_labels[n_test:]

                med_test_y = medial_labels[:n_test]
                med_train_y = medial_labels[n_test:]

                fin_test_y = final_labels[:n_test]
                fin_train_y = final_labels[n_test:]

                for ll, C in enumerate(Cs):
                    # initial log_reg
                    init_score, _ = logreg(train_x, init_train_y,
                                           test_x, init_test_y, C)
                    # medial log_reg
                    med_score, _ = logreg(train_x, med_train_y,
                                          test_x, med_test_y, C)
                    # final log_reg
                    fin_score, _ = logreg(train_x, fin_train_y,
                                          test_x, fin_test_y, C)

                    results[ii, jj, 0, ll, kk] = init_score
                    results[ii, jj, 1, ll, kk] = med_score
                    results[ii, jj, 2, ll, kk] = fin_score

    np.savez('single_logreg.npz',
             results=results, fontnames=fontnames, fontsizes=fontsizes,
             Cs=Cs, seed=seed)


if __name__ == "__main__":
    base_folder = sys.argv[1]
    single_logreg(base_folder)
