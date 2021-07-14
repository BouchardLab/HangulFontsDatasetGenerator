import argparse, os, time, pickle
import torch
import numpy as np
from functions import (find_mean, trav, path_maker, generate_noise)
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from hangul_analysis.ml import LeaveOneFontOutCV
from hangul_analysis.nets.hp_space import VAEHyperparameterSpace
from hangul_analysis.vaes.vae_train_loop import vae_train_loop

def main():
    """
    Training script

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int,
                        help="Experiment number")
    parser.add_argument('--nfolds', default=7, type=int,
                        help="number of folds")
    parser.add_argument('--fold', type=int,
                        help='fold iteration')
    parser.add_argument('--path_h5', type=str,
                        help="path to h5 files")
    parser.add_argument('--seed', type=int,
                        help="numpy and pytorch seed")
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--beta', type=str,
                        help='normal or beta')
    parser.add_argument('--pretrained', type=str, default='False',
                        help='pretrained model')
    parser.add_argument('-f', '--ff', help="Dummy arg")
    args = vars(parser.parse_args())

    # Some hyperparameters
    experiment = args['experiment']
    nfolds = args['nfolds']
    path_h5 = args['path_h5']
    seed = args['seed']
    fold = args['fold']
    device = args['device']
    beta = args['beta']
    pretrained = args['pretrained']

    print(f'pretrained: {pretrained}, {type(pretrained)}')
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if beta == "beta":
        beta = True
    elif beta == "normal":
        beta = False
    else:
        raise ValueError(f"beta: {beta} should be beta or normal")
    print(f'{seed} {fold} beta: {beta}')
    rng = np.random.RandomState(seed)
    int_info = np.iinfo(int)
    torch.manual_seed(rng.randint(int_info.min, int_info.max))

    root = os.path.abspath(path_h5)
    print("root: " + str(root))
    locs = os.walk(root)
    files = []
    for base, _, fnames in locs:
        for f in fnames:
            if '{}.h5'.format(24) in f:
                files.append(os.path.join(base, f))


    start = time.time()

    base, base_valid, base_data, base_trav, base_random = path_maker(experiment, nfolds)

    ds = LeaveOneFontOutCV(files, fold, imf='i', ravel=False, mean_center=False, n_folds=nfolds)
    X_train, y = ds.training_set()
    X_valid, yv = ds.validation_set()
    X_test, yt = ds.test_set()

    print('Dataset loaded: {}'.format(time.time() - start))

    si = torch.tensor(X_train[0]).unsqueeze(0).size()
    print(si)
    hp_sp = VAEHyperparameterSpace(si, rng)
    hp_sp.create_hp_space()
    params = hp_sp.random_params()

    params['pool_size'] = None

    gam = params['gamma']
    learn = params['lr']
    params = params #if have trained model with gamma < 1, replace with dictionary

    if pretrained == 'True':
        pretrained = True
        gamh = gam[-1]
        o_gam = 1e-4 #replace with trained model's gamma < 1
        gam = np.geomspace(o_gam, gamh, num=26)
        params['lr'] = learn
    else:
        pretrained = False
    params['gamma'] = gam

    if device != 'cpu':
        ds_train = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.training_set()]),
                              batch_size=params['batch_size'], shuffle=True, num_workers=0)
        ds_valid = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.validation_set()]),
                              batch_size=params['batch_size'], num_workers=0)
        ds_test = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.test_set()]),
                             batch_size=params['batch_size'], num_workers=0)
    else:
        ds_train = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.training_set()]),
                          batch_size=params['batch_size'], shuffle=True)
        ds_valid = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.validation_set()]),
                              batch_size=len(yv))
        ds_test = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.test_set()]),
                             batch_size=len(yt))

    model_id = '{}_{}_{}'.format(experiment, fold, seed)

    root = os.path.abspath(os.path.join(os.getcwd(), 'test_sets.pkl'))
    pixels = 0
    with open(root, 'rb') as f:
        pixels = pickle.load(f)
    recon = DataLoader(TensorDataset(*[torch.tensor(t) for t in pixels]), batch_size=len(pixels[1]))

    vae_train_loop([ds_train, ds_valid, ds_test], params, recon,
                    base, base_valid, base_data, model_id, fold,
                    beta, device, pretrained, experiment, seed)

    end_time = time.time() - start
    print("Training ended: {}".format(end_time))
    if end_time < 10:
        raise RuntimeError("training failed")
    photos = pixels[0]
    model_path = 0
    ps = list(Path(os.path.join(base_data, model_id)).rglob('c*.pt'))
    if len(ps) > 1:
        print('Multiple final c models')
        raise ValueError(ps)
    elif len(ps) == 0:
        raise ValueError("Something went wrong")
    model_path = ps[0]

    h_dim = params['h_dim']
    # Generate random samples
    generate_noise(model_path, model_id, si, ds, params, base_random, seed)

    # Find personalized range for samples
    low_mean, high_mean, low_sample, high_sample, indexes = find_mean(
        model_path, base_trav, params, nfolds, [ds_train, ds_valid, ds_test],
        [X_train, X_valid, X_test], device=device, seed=seed, fold=fold)

    # Static range for samples; traversal
    trav(model_path, params, indexes, base_trav, si=si, low=np.array(
        [-25] * h_dim), high=np.array([25] * h_dim), mean_sample='mean',
         trav_steps=20, seed=seed, fold=fold, device=device)


if __name__ == "__main__":
    main()
