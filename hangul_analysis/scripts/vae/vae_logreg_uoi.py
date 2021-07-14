import argparse, os, h5py, pickle, time
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold

from mpi4py import MPI

from pyuoi import UoI_L1Logistic
from pyuoi.mpi_utils import Bcast_from_root

from hangul_analysis.variables import label_df, n_blocks


exp_names = {'i': (5, 48656),
             'm': (5, 15285),
             'f': (5, 46148),
             'imf': (5, 46119),
             'trav': (6, 3868),
             'trav2': (26, 120001),
             'vae': (2300, 4),
             'nvae': (2500, 10005)}

parser = argparse.ArgumentParser(description='UoI LogisticRegression analysis.')
parser.add_argument('save_folder', type=str,
                    help='Folder for saved networks/representations.')
parser.add_argument('imf', type=str, choices=list(exp_names.keys()),
        help='Task for network')
parser.add_argument('fold', type=int, help='CV fold.')
parser.add_argument('variable_idx', type=int, choices=np.arange(9).tolist(),
        help='Variable index for logistic regression.')

args = parser.parse_args()

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

variables = np.array(['initial', 'medial', 'final',
    'initial_geometry', 'medial_geometry', 'final_geometry', 'all_geometry',
    'atom_bof', 'atom_mod_rotations_bof'])

imf = args.imf
num, exp_num = exp_names[imf]
fold = args.fold
variable = variables[args.variable_idx]

rng = np.random.RandomState(20210430)
n_cv = 2
kwargs = {'stability_selection': 0.95,
          'estimation_score': 'BIC',
          'shared_support': False,
          'tol': 1e-2,
          'n_boots_sel': 20,
          'n_boots_est': 20,
          'n_C': 24,
          'random_state': rng}

save_folder = args.save_folder
if rank == 0:
    print(save_folder, num, exp_num)

n_samples = None
n_tasks = None
if rank == 0:
    # Load representations
    print('Loading representations')
    subfolder = '{}_{}_{}'.format(num, fold, exp_num)
    path = os.path.join(save_folder, subfolder)
    h5_name = os.path.join(path, 'layer_reps.h5')
    with h5py.File(h5_name, 'r') as f:
        X = f['test/encoding'][:]
    if variable == 'atom_bof':
        scores = np.zeros((n_cv, 24, 3))
    elif variable == 'atom_mod_rotations_bof':
        scores = np.zeros((n_cv, 16, 3))
    else:
        scores = np.zeros((n_cv, 3))
    start = time.time()
    n_samples = X.shape[0]
    if 'bof' in variable:
        y = np.tile(np.stack(label_df[variable]), (n_samples // n_blocks, 1))
        n_tasks = y.shape[1]
    else:
        y = np.tile(label_df[variable], n_samples // n_blocks)
    n_samples = len(X)

# Only rank 0 actually splits for CV, make fake data for other ranks
n_samples = comm.bcast(n_samples, root=0)

if 'bof' in variable:
    n_tasks = comm.bcast(n_tasks, root=0)
    if rank == 0:
        coefs = {}
    else:
        X = np.zeros(n_samples, dtype=bool)
        y = np.zeros((n_samples, n_tasks), dtype=bool)
    for jj in range(y.shape[1]):
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=rng)
        X_train = None
        y_train = None
        for ii, (train_idx, test_idx) in enumerate(skf.split(X, y[:, jj])):
            if rank == 0:
                if not (ii in coefs):
                    coefs[ii] = []
                X_train = X[train_idx]
                X_test = X[test_idx]
                std = X_train.std(axis=0)
                y_train = y[train_idx, jj]
                y_test = y[test_idx, jj]
                mean = X_train.mean(axis=0, keepdims=True)
                std = X_train.std(axis=0, keepdims=True)
                X_train -= mean
                X_train /= std
                X_test -= mean
                X_train /= std
                print('Training networks', X.shape)
                print(variable)
            X_train = Bcast_from_root(X_train, comm)
            y_train = Bcast_from_root(y_train, comm)
            LR = UoI_L1Logistic(comm=comm, **kwargs)
            LR.fit(X_train, y_train)
            if rank == 0:
                scores[ii, jj, 0] = LR.score(X_train, y_train)
                scores[ii, jj, 1] = LR.score(X_test, y_test)
                scores[ii, jj, 2] = np.mean(np.equal(y_test, mode(y_train, axis=None)[0][0]))
                coefs[ii].append(LR.coef_)
                print(np.count_nonzero(LR.coef_), LR.coef_.size, scores[ii, jj, 1])
                print(time.time() - start)
else:
    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=rng)
    X_train = None
    y_train = None
    if rank == 0:
        coefs = {}
    else:
        X = np.zeros(n_samples, dtype=bool)
        y = np.zeros(n_samples, dtype=bool)
    for ii, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if rank == 0:
            coefs[ii] = []
            X_train = X[train_idx]
            X_test = X[test_idx]
            std = X_train.std(axis=0)
            y_train = y[train_idx]
            y_test = y[test_idx]
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True)
            X_train -= mean
            X_train /= std
            X_test -= mean
            X_train /= std
            print('Training networks', X.shape)
            print(variable)
        X_train = Bcast_from_root(X_train, comm)
        y_train = Bcast_from_root(y_train, comm)

        LR = UoI_L1Logistic(comm=comm, **kwargs)
        LR.fit(X_train, y_train)
        if rank == 0:
            scores[ii, 0] = LR.score(X_train, y_train)
            scores[ii, 1] = LR.score(X_test, y_test)
            scores[ii, 2] = np.mean(np.equal(y_test, mode(y_train, axis=None)[0][0]))
            coefs[ii].append(LR.coef_)
            print(np.count_nonzero(LR.coef_), LR.coef_.size, scores[ii, 1])
            print(time.time() - start)

if rank == 0:
    save_file = os.path.join(save_folder, subfolder, 'LR_UoI_{}_{}_{}.pkl'.format(imf, fold, variable))
    with open(save_file, 'wb') as f:
        pickle.dump((scores, coefs), f)
comm.Barrier()
if rank == 0:
    print('Done')
