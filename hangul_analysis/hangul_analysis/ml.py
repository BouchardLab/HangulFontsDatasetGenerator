import os, glob

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from hangul_analysis.read_data import load_data


def normalized_acc_chance(acc, chance):
    num = acc - chance
    den = 1. - chance
    return num / den


def logreg_model(train_x, train_y, C=1.):
    model = LogisticRegression(C=C, multi_class='multinomial',
                               solver='lbfgs')
    model.fit(train_x, train_y)
    return model


def logreg(train_x, train_y, test_x, test_y, C=1.):
    model = logreg_model(train_x, train_y, C=C)
    score = model.score(test_x, test_y)
    return score, model.coef_


class LeaveOneFontOutCV(object):
    def __init__(self, font_files, fold, imf, seed=20190114, ravel=False,
                 n_folds=7, mean_center=True):
        imf_list = ['i', 'm', 'f']
        if imf not in imf_list:
            raise ValueError
        self.imf = imf_list.index(imf)
        self.fold = fold
        if isinstance(seed, int):
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = seed
        self.font_files = self.rng.permutation(font_files)
        self.ravel = ravel
        self.n_folds = n_folds
        self.mean_center = mean_center
        if fold >= self.n_folds or fold < 0:
            raise ValueError

    def training_set(self):
        splits = np.array_split(self.font_files, self.n_folds)
        test_idx = self.fold
        val_idx = (self.fold + 1) % self.n_folds
        del splits[test_idx]
        # Shift val_idx for removed test_idx
        if val_idx > test_idx:
            val_idx -= 1
        del splits[val_idx]
        train_files = np.concatenate(splits)
        ims = []
        labels = []
        for tf in train_files:
            im, label, _, _, _ = load_data(tf, median_shape=True)
            ims.append(im)
            labels.append(label[:, self.imf])
        ims = np.concatenate(ims).astype('float32')
        ims /= 255.
        if self.mean_center:
            self.train_mean = ims.mean(axis=0, keepdims=True)
            ims -= self.train_mean
        if self.ravel:
            ims = ims.reshape(ims.shape[0], -1)
        labels = np.concatenate(labels)
        return ims, labels.astype(int)

    def validation_set(self):
        splits = np.array_split(self.font_files, self.n_folds)
        valid_files = splits[(self.fold + 1) % self.n_folds]
        ims = []
        labels = []
        for vf in valid_files:
            im, label, _, _, _ = load_data(vf, median_shape=True)
            ims.append(im)
            labels.append(label[:, self.imf])
        ims = np.concatenate(ims).astype('float32')
        ims /= 255.
        if self.mean_center:
            ims -= self.train_mean
        if self.ravel:
            ims = ims.reshape(ims.shape[0], -1)
        labels = np.concatenate(labels)
        return ims, labels.astype(int)

    def test_set(self):
        splits = np.array_split(self.font_files, self.n_folds)
        test_files = splits[self.fold]
        ims = []
        labels = []
        for tf in test_files:
            im, label, _, _, _ = load_data(tf, median_shape=True)
            ims.append(im)
            labels.append(label[:, self.imf])
        ims = np.concatenate(ims).astype('float32')
        ims /= 255.
        if self.mean_center:
            ims -= self.train_mean
        if self.ravel:
            ims = ims.reshape(ims.shape[0], -1)
        labels = np.concatenate(labels)
        return ims, labels.astype(int)

def best_imf_networks(save_folder, exp_name, n_models, n_tasks=3, n_folds=7):
    saved_data = os.walk(save_folder)
    values = []
    for cur_path, dirs, files in saved_data:
        parent, cur_dir = os.path.split(cur_path)
        if exp_name in cur_dir:
            imf, fold, seed = cur_dir.split('_')[-3:]
            fold = int(fold)
            seed = int(seed)
            folders = glob.glob(os.path.join(parent, '{}_*_{}'.format(exp_name, seed)))
            if len(folders) < n_tasks * n_folds:
                continue
            for folder in folders:
                model_file = glob.glob(os.path.join(folder, '*.pth'))
                if len(model_file) > 1:
                    raise ValueError(cur_path)
                elif len(model_file) == 0:
                    continue
            model_file = glob.glob(os.path.join(cur_path, '*.pth'))
            model_file = model_file[0]
            val_value = os.path.splitext(model_file)[0].split('=')[1]
            values.append((imf, int(seed), int(fold), float(val_value), model_file))
    index = pd.MultiIndex.from_tuples([tup[:3] for tup in values], names=['imf', 'seed', 'fold'])
    df = pd.DataFrame(values,
                      index=index,
                      columns=['imf', 'seed', 'fold', 'validation value', 'model file'])
    df.sort_index(inplace=True)
    model_files = []
    for imf in ['i', 'm', 'f']:
        vals = []
        seeds = sorted(set(df['seed']))
        if len(seeds) < n_models:
            raise ValueError('{} models availble, {} requested'.format(len(seeds), n_models))
        seeds = seeds[:n_models]
        for s in seeds:
            vals.append(df.loc[imf, s]['validation value'].mean())
        idx = np.argmax(vals)
        si = seeds[idx]
        model_files.append(df.loc[imf, si][['model file']])
    return model_files

def load_cv_data(h5_folder, fold, imf, mean_center=True, fontsize=24):
    if fold not in np.arange(7):
        raise ValueError('fold should be in 0-6.')
    locs = os.walk(h5_folder)
    files = []
    for base, _, fnames in locs:
        for f in fnames:
            if '{}.h5'.format(fontsize) in f:
                files.append(os.path.join(base, f))
    ds = LeaveOneFontOutCV(files, fold, imf=imf, ravel=True, mean_center=mean_center)
    return ds.training_set(), ds.validation_set(), ds.test_set()
