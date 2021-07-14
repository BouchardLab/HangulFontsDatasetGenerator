import argparse, os, time, pickle, h5py

import torch
from torch.utils.data import DataLoader, TensorDataset

from hangul_analysis.nets.model_creator import make_dense_model
from hangul_analysis.ml import LeaveOneFontOutCV, best_imf_networks


def main(h5_folder, save_folder, exp_name, fontsize, device, n_models):
    print(h5_folder, save_folder, exp_name, fontsize)
    locs = os.walk(h5_folder)
    h5_files = []
    for base, _, fnames in locs:
        for f in fnames:
            if '{}.h5'.format(fontsize) in f:
                h5_files.append(os.path.join(base, f))

    model_files = best_imf_networks(save_folder, exp_name, n_models)
    print(model_files)
    assert False

    for mfs in model_files:
        for fold, fpath in mfs.iterrows():
            fpath = fpath['model file']
            folder, fname = os.path.split(fpath)
            imf, fold, seed = folder.split('_')[-3:]
            fold = int(fold)
            seed = int(seed)
            with open(os.path.join(folder, 'model_params.pkl'), 'rb') as f:
                args = pickle.load(f)
            input_dim, output_dim, params, settings = args
            model = make_dense_model(input_dim, output_dim, params, settings)
            model.load_state_dict(torch.load(fpath))
            model.eval()
            reps = {'train': [], 'valid': [], 'test': []}
            ds_type = 'train'
            layer_names = []
            def hook(module, input, output):
                reps[ds_type].append(output.detach().cpu().numpy())
            for layer in model.children():
                if not isinstance(layer, torch.nn.Dropout):
                    layer_names.append(layer.__str__())
                    layer.register_forward_hook(hook)

            ds = LeaveOneFontOutCV(h5_files, fold, imf=imf, ravel=True)
            X, y = ds.training_set()
            _, yv = ds.validation_set()
            _, yt = ds.test_set()

            ds_train = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.training_set()]),
                                  batch_size=len(y))
            ds_valid = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.validation_set()]),
                                  batch_size=len(yv))
            ds_test = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.test_set()]),
                                 batch_size=len(yt))
            ds_type = 'train'
            for bX, by in ds_train:
                yhat = model(bX)
            ds_type = 'valid'
            for bX, by in ds_valid:
                y_hat = model(bX)
            ds_type = 'test'
            for bX, by in ds_test:
                y_hat = model(bX)
            with h5py.File(os.path.join(folder, 'layer_reps.h5'), 'w') as f:
                for ds_type in ['train', 'valid', 'test']:
                    grp = f.create_group(ds_type)
                    for ii, lname in enumerate(layer_names):
                        grp.create_dataset('{}: {}'.format(ii, lname), data=reps[ds_type][ii])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('h5_folder', type=str,
                        help='Path to Hangul h5 folder.')
    parser.add_argument('save_folder', type=str,
                        help='Path to save the model output.')
    parser.add_argument('exp_name', type=str,
                        help='Experiment name.')
    parser.add_argument('--fontsize', type=int, default=24,
                        help='Fontsize for training.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--n_models', type=int, default=100,
                        help='How many models to look at')
    args = parser.parse_args()
    main(args.h5_folder, args.save_folder, args.exp_name,
         args.fontsize, args.device, args.n_models)
