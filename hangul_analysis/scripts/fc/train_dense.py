import argparse, os, time

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from ignite.metrics import Accuracy, Loss

from hangul_analysis.nets.model_creator import train_loop
from hangul_analysis.nets.hp_space import DenseHyperparameterSpace
from hangul_analysis.ml import LeaveOneFontOutCV
from hangul_analysis.variables import n_initial, n_medial, n_final


def main(h5_folder, save_folder, exp_name, fold, fontsize, imf, device, info=False,
         numpy_seed=201904221):
    rng = np.random.RandomState(numpy_seed)
    int_info = np.iinfo(int)
    torch.manual_seed(rng.randint(int_info.min, int_info.max))
    if imf == 'i':
        output_dim = n_initial
    elif imf == 'm':
        output_dim = n_medial
    elif imf == 'f':
        output_dim = n_final
    else:
        raise ValueError
    print(h5_folder, save_folder, exp_name, fold, fontsize, imf)
    locs = os.walk(h5_folder)
    files = []
    for base, _, fnames in locs:
        for f in fnames:
            if '{}.h5'.format(fontsize) in f:
                files.append(os.path.join(base, f))
    start = time.time()

    ds = LeaveOneFontOutCV(files, fold, imf=imf, ravel=True)
    X, y = ds.training_set()
    _, yv = ds.validation_set()
    _, yt = ds.test_set()

    print('Dataset loaded: {}'.format(time.time() - start))

    hp_sp = DenseHyperparameterSpace(X.shape[1], output_dim, info=info, seed=rng)
    hp_sp.create_hp_space()
    params = hp_sp.random_params()
    params['n_dense_layers'] = 4
    settings = {'loss': 'xent',
                'info': info}

    funcs = {'accuracy': Accuracy(),
             'loss': Loss(F.cross_entropy)}

    ds_train = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.training_set()]),
                          batch_size=params['batch_size'], shuffle=True)
    ds_valid = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.validation_set()]),
                            batch_size=len(yv))
    ds_test = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.test_set()]),
                         batch_size=len(yt))

    save_name = '{}_{}_{}_{}'.format(exp_name, imf, fold, numpy_seed)
    train_loop([ds_train, ds_valid, ds_test], params, settings,
               save_folder, save_name,
               funcs, 'accuracy', n_classes=output_dim, device=device)
    print('Training ended: {}'.format(time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('h5_folder', type=str,
                        help='Path to Hangul h5 folder.')
    parser.add_argument('save_folder', type=str,
                        help='Path to save the model output.')
    parser.add_argument('exp_name', type=str,
                        help='Experiment name.')
    parser.add_argument('fold', type=int,
                        help='Fold.')
    parser.add_argument('seed', type=int,
                        help='Seed for rng.')
    parser.add_argument('--fontsize', type=int, default=24,
                        help='Fontsize for training.')
    parser.add_argument('--imf', type=str, default='i',
                        help='Train on initial, medial, or final.')
    parser.add_argument('--info', default=False, action='store_true',
                        help='Train with Achille and Soatto Info layers.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
    args = parser.parse_args()
    main(args.h5_folder, args.save_folder, args.exp_name, args.fold,
         args.fontsize, args.imf, args.device, args.info, args.seed)
