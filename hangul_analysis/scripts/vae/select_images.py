from hangul.ml import LeaveOneFontOutCV
from hangul.label_mapping import imf2idx, idx2imf
import os
import numpy as np
import argparse
import pickle

def main(path_h5, nfolds):
    root = os.path.abspath(os.getcwd() + path_h5)
    locs = os.walk(root)
    files = []
    for base, _, fnames in locs:
        for f in fnames:
            if '{}.h5'.format(24) in f:
                files.append(os.path.join(base, f))

    ds = LeaveOneFontOutCV(
        files, 0, imf='i', ravel=False, mean_center=False, n_folds=nfolds)
    X_train, _ = ds.training_set()
    X_valid, yv = ds.validation_set()
    X_test, yt = ds.test_set()
    Xt = np.concatenate((X_train, X_valid, X_test), axis=0)
    print(len(Xt))
    data = [imf2idx(0, 14, 25), imf2idx(1, 15, 26), imf2idx(2, 14, 27), imf2idx(3, 15, 24),
            imf2idx(4, 16, 23), imf2idx(5, 17, 22), imf2idx(6, 16, 21), imf2idx(7, 4, 20),
            imf2idx(8, 10, 19), imf2idx(9, 18, 18), imf2idx(10, 3, 8), imf2idx(11, 21, 7),
            imf2idx(12, 20, 6), imf2idx(13, 9, 5), imf2idx(14, 13, 4), imf2idx(15, 0, 3),
            imf2idx(16, 18, 2), imf2idx(17, 7, 1), imf2idx(18, 16, 0)]

    
    for set_num in range(len(Xt)//11172):
        print(set_num)
        pics = [Xt[i + 11172 * set_num] for i in data]
        indices = [idx2imf(i) for i in data]
        new_data = [pics, indices]
        with open(os.path.abspath(os.path.join(os.getcwd(), f'test_sets_font_{set_num}.pkl')), 'wb') as f:
            pickle.dump(new_data, f)
        if set_num == 0:
            with open(os.path.abspath(os.path.join(os.getcwd(), f'test_sets.pkl')), 'wb') as f:
                pickle.dump(new_data, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfolds', default=7, type=int,
                        help="number of folds")
    parser.add_argument('--path_h5', type=str,
                        help="path to h5 files")
    parser = parser.parse_args()
    main(parser.path_h5, parser.nfolds)