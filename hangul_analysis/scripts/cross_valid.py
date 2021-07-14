import json, argparse, time, os, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from hangul_analysis.ml import LeaveOneFontOutCV
from hangul_analysis.vaes.functions import cross_validate_vec
from hangul_analysis.nets.model_creator import make_vae_model
from subprocess import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int,
                        help="Experiment number")
    parser.add_argument('--nfolds', default=7, type=int,
                        help="number of folds")
    parser.add_argument('--path_h5', type=str,
                        help="path to h5 files")
    parser.add_argument('--path_output', type=str,
                        help="path to hangul outputs")
    parser.add_argument('--seed', type=int,
                        help="numpy and pytorch seed")
    parser.add_argument('--num_fonts', type=int, 
                        help="num of fonts in validation set")
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--knn', type=int,
                        help='k nearest neighbors')
    parser.add_argument('--imf', type=str, default='f',
                        help='img glyphs for cv')
    parser.add_argument('--path_model', type=str, default=None,
                        help='path to models if cannot ' + 
                        'fit all in memory')
    parser.add_argument('-f', '--ff', help="Dummy arg")
    args = vars(parser.parse_args())
    
    experiment = args['experiment']
    nfolds = args['nfolds']
    path_h5 = args['path_h5']
    path_output = args['path_output']
    seed = args['seed']
    num_fonts = args['num_fonts']
    k = args['knn']
    device = args['device']
    imf = args['imf']
    path_model = args['path_model']
    
    rng = np.random.RandomState(seed)
    int_info = np.iinfo(int)
    torch.manual_seed(rng.randint(int_info.min, int_info.max))
    
    locs = os.walk(path_h5)
    files = []
    for base, _, fnames in locs:
        for f in fnames:
            if '{}.h5'.format(24) in f:
                files.append(os.path.join(base, f))
    
    start = time.time()
    
    yv = 0
    X_valid = 0
    ds = 0
    for j, i in enumerate(imf):
        ds = LeaveOneFontOutCV(files, 0, imf=i, ravel=False,
                               mean_center=False, n_folds=nfolds)
        X_valid, y = ds.validation_set()
        if j == 0:
            yv = y
        elif j == 1:
            yv = np.stack((yv, y), axis=1)
        else:
            y = y[..., None]
            yv = np.concatenate((yv, y), axis=1)

    if type(yv) == np.ndarray:
        yv = list(yv)
        if type(yv[0]) == np.ndarray:
            yv = [y.tolist() for y in yv]
    ds_valid = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.validation_set()]),
                              batch_size=int(len(X_valid)/num_fonts))
    cv_info = {}
    cv_info['knn'] = k
    cv_info['imf'] = imf
    with torch.no_grad():
        for fold in range(nfolds):
            print(f"Fold: {fold}")
            if path_model is not None:
                mod_path = os.path.join(path_model,
                                        f'experiment_{experiment}/{experiment}_{fold}_{seed}/')
                args = ['mv', mod_path, f'outputs/experiment_{experiment}/data']
                run(args)
            root = f'{path_output}/experiment_{experiment}/data/{experiment}_{fold}_{seed}'
            data = 0
            with open(f'{root}/model_params.pkl', 'rb') as f:
                data = pickle.load(f)
            params = data[1]
            h_dim = params['h_dim']
            batch_size = params['batch_size']

            ps = list(Path(root).rglob('c*.pt'))
            if len(ps) != 1:
                print(f"No final model exists for fold {fold}")
                continue
            mod = make_vae_model(data[0], params).to(device)
            mod.load_state_dict(torch.load(ps[0], map_location=device))
            mod.eval()
            ds_val = torch.tensor([])
            labels = []
            for i, d in enumerate(ds_valid):
                d, _ = d
                curr = d.to(device).unsqueeze(1)
                enc, _, _ = mod.encode(curr)
                if i == 0:
                    ds_val = enc.cpu()
                elif i == 1:
                    ds_val = torch.stack((ds_val, enc.cpu()))
                else:
                    ds_val = torch.cat((ds_val, enc.cpu().unsqueeze(0)))
                labels.append(yv[(i*11172):((i+1)*11172)])
            acc = cross_validate_vec(h_dim, ds_val, labels, k, False)
            print(acc)
            cv_info[f"fold: {fold}"] = {"accuracy": acc}
            with open(f'{path_output}/experiment_{experiment}/data/{experiment}_{seed}_{imf}_cross_validation.json',
                      'w+') as f:
                json.dump(cv_info, f)
            if path_model is not None:
                mod_path = os.path.join(path_model,
                                        f'experiment_{experiment}/')
                args = ['mv', f'outputs/experiment_{experiment}/data/{experiment}_{fold}_{seed}/', mod_path]
                run(args)
    print(f"Done with {experiment}_{fold}_{seed} {imf}")
if __name__ == "__main__":
    main()