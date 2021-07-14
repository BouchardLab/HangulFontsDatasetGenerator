import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
import os, pickle
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import cv2

from hangul_analysis.nets.model_creator import make_vae_model



def kl_div(h_dim, sample, q=[], hp=[], require_grad=True, individual_elements=False,
           device='cpu'):
    """
    Calculates kl-divergence for multiple scenarios:
    I. No q (prior) passed in:
        Assume unit Gaussian prior
        a. If require_grad:
            Calculate kl-divergence manually
            i. If individual_elements:
                Return kl-divergence for each latent distribution
            ii. Else:
                Calculates kl-divergence averaged across distributions
                and batch
                1. If hp passed in:
                    Return Beta-VAE kl-loss
                2. If 1 hp passed in:
                    Return VAE with gamma
                3. Else:
                    Return averaged loss
        b. else:
            Use torch kl_divergence with torch Normal distributions
            i. ii. same as for Ia.
    II. q (prior) passed in:
        Return torch kl_divergence with torch Normal distributions
    """
    if len(sample.size()) == 1:
        sample = sample.unsqueeze(0)
    mu, logvar = sample[:, :h_dim], sample[:, h_dim:]
    d1 = Normal(mu, torch.sqrt(torch.exp(logvar)))
    if len(q) == 0:
        k = 0
        if require_grad:
            k = -0.5  * (1 + logvar - mu.pow(2) - logvar.exp())
        else:
            muq, logvarq = torch.zeros(mu.size()).to(device), torch.ones(logvar.size()).to(device)
            d2 = Normal(muq, logvarq)
            k = kl_divergence(d1, d2)
        if individual_elements:
            return k
        if len(hp) == 0:
            kl =  torch.mean(k, dim=1)
            kld = torch.mean(kl, dim=0)
            return kld
        if len(hp) == 1:
            kl = torch.mean(k, dim=1)
            kld = torch.mean(kl, dim=0)
            return hp[0] * torch.abs(kld)
        else:
            gamma, c = hp
            kl = torch.mean(k, dim=1)
            kld = torch.mean(kl-c, dim=0)
            return gamma * torch.abs(kld)
    elif type(q) == torch.Tensor:
        if len(q.size()) == 1:
            q = q.unsqueeze(0)
        q = q.to(device)
        muq, logvarq = q[:, :h_dim], q[:, h_dim:]
        d2 = Normal(muq, torch.sqrt(torch.exp(logvarq)))
        kld = kl_divergence(d1, d2)
        kld = torch.mean(kld, dim=-1)
        return kld
    else:
        raise ValueError("q must be tensor [batch, mu/logvar]")




def find_mean(model_path, trav_path, params, nfolds, dataset, Xs, device, seed, fold, mod=''):
    """
    Plots histogram of mu's generated from a fold's worth of characters.
    Plots histogram of samples generated from a fold's worth of characters.
    Returns 1st percentile mean and 99th percentile mean to use for more accurate
    traversals.
    """

    X_train, X_valid, X_test = Xs
    ds_train, ds_valid, ds_test = dataset
    if type(mod) == str:
        model = make_vae_model(torch.tensor(X_train[0]).unsqueeze(0).size(),
                               params).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = mod
    h_dim = model.h_dim

    mu_list = torch.tensor(
        [[0.] * h_dim] * int(np.ceil(X_train.shape[0]/ds_train.batch_size)\
                             + np.ceil(X_valid.shape[0]/ds_valid.batch_size)\
                             + np.ceil(X_test.shape[0]/ds_test.batch_size))).to(device)
    std_list = torch.zeros(mu_list.shape).to(device)
    print(mu_list.shape)
    model.eval()
    with torch.no_grad():
        l = 0
        for i, s in enumerate(zip(dataset, Xs)):
            ds, X = s
            for j, d in tqdm(enumerate(ds),
                                total=X.shape[0] / ds.batch_size):
                data, _ = d

                data = data.unsqueeze(1).to(device)
                reconstruction, mu, logvar = model.forward(data)
                mu_list[l + j] = torch.mean(mu.detach(), dim=0)
                std_list[l + j] = torch.mean(logvar.detach(), dim=0)
            l += len(ds)
            print(l)
    std = torch.exp(0.5 * std_list.cpu())
    eps = torch.randn_like(std)
    sample = mu_list.cpu() + std * eps
    sample = sample.numpy()
    mu_list = mu_list.cpu().numpy()
    std_list = std_list.cpu().numpy()
    flatten_mu_list = np.reshape(mu_list, -1)
    flatten_sample = np.reshape(sample, -1)
    flatten_std = np.reshape(std_list, -1)
    fig, axis = plt.subplots(1, 3, constrained_layout=True, figsize=(20, 10))
    axis[0].hist(flatten_sample, bins=200)
    axis[0].set_title("Sample Histogram")
    axis[0].set_xlabel("Sample Values")
    axis[0].set_ylabel("Frequency")
    axis[1].hist(flatten_mu_list, bins=200)
    axis[1].set_title("Mu Histogram")
    axis[1].set_xlabel("Mu Values")
    axis[1].set_ylabel("Frequency")
    axis[2].hist(np.exp(flatten_std * 0.5), bins=200)
    axis[2].set_title("Sigma Histogram")
    axis[2].set_xlabel("Sigma Values")
    axis[2].set_ylabel("Frequency")
    fig.savefig(os.path.join(trav_path, f"{seed} {fold} samples visualized.png"))
    mu_max = np.quantile(mu_list, 0.99, axis=0)
    # Use samples instead of means and percentile 97.5/2.5
    mu_min = np.quantile(mu_list, 0.01, axis=0)
    sample_max = np.quantile(sample, 0.99, axis=0)
    sample_min = np.quantile(sample, 0.01, axis=0)

    mu_mean = torch.tensor(np.mean(mu_list, axis=0))
    std_mean = torch.tensor(np.mean(std_list, axis=0))

    kld = kl_div(h_dim, torch.cat((mu_mean, std_mean)), individual_elements=True)
    sorted_kld, indexes = torch.sort(kld, descending=True)
    np.savez(os.path.join(trav_path, f"{seed}_{fold}_samples"), mu=mu_list,
             std=std_list, sample=sample, kld=kld, kl_indices=indexes)  # Saved as (num_samples/batch, h_dim)

    return mu_min, mu_max, sample_min, sample_max, indexes


def generate_noise(model_path, model_id, si, ds, params, trav_path, seed):
    model = make_vae_model(si, params)
    model.load_state_dict(torch.load(model_path))
    h_dim = model.h_dim
    model.eval()
    data_valid = DataLoader(TensorDataset(*[torch.tensor(t) for t in ds.validation_set()]),
                          batch_size=100)
    data_loader_iter = iter(data_valid)
    x, y = next(data_loader_iter)
    x = x.unsqueeze(1)
    with torch.no_grad():
        _, indices, sizes = model.encode(x)
        inputs = torch.tensor(np.random.normal(0, 1, (100, h_dim)))
        sample = model.decode(inputs.float(), indices, sizes)
        sample = sample.view(-1, *si)
        save_image(sample, os.path.join(trav_path, f"{seed}_{model_id}_random.png"), nrow=10)
        save_image(x, os.path.join(trav_path, f"{seed}_{model_id}_random_input.png"), nrow=10)

    """ Creates and saves latent space traversals based on either means or samples

    Parameters
    ----------
    model : torch model
    h_dim : int
        size of mu/logvar in latent space
    kl_indexes : list
        how to order the latent traversals (decreasing KL divergence)
    pixels : np.array
        the image to traverse across
    trav_path : str
        path to traversal directory
    number : int
        image number out of entire sequence
    low : np.array
        low end of traversal for each latent distribution
    high : np.array
        high end of traversal for each latent distribution
    trav_steps : int
        number of steps to take in traversal
    mean_sample : string
        traversing by mean or sample
    """
def traverse(model, h_dim, kl_indexes, pixels, trav_path, number, si, low, high, trav_steps, mean_sample, seed, fold):

    with torch.no_grad():
        pixels = torch.tensor([pixels] * h_dim).unsqueeze(1).float()
        x, m_indices, sizes = model.encode(pixels)
        mu, logvar = x[:, :h_dim], x[:, h_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        actual, _, _ = model.forward(pixels)
        actual = actual.view(-1, *si)
        pixels = pixels[kl_indexes]
        actual = actual[kl_indexes]
        indices = torch.tensor(np.linspace(low, high, trav_steps))
        image = torch.tensor([])
        if mean_sample == 'mean':
            for i in range(len(indices)):
                mu_copy = mu.clone()
                step = indices[i]
                for j in range(h_dim):
                    mu_copy[j, j] = step[j]

                sample = mu_copy + std * eps
                sample = model.decode(sample, m_indices.copy(), sizes.copy())
                sample = sample.view(-1, *si)
                sample = sample[kl_indexes]
                image = torch.cat((image, sample))
        elif mean_sample == 'sample':
            for i in range(len(indices)):
                step = indices[i]
                sample = mu + std * eps
                for j in range(h_dim):
                    sample[j, j] = step[j]
                sample = model.decode(sample, m_indices.copy(), sizes.copy())

                sample = sample.view(-1, *si)
                sample = sample[kl_indexes]
                image = torch.cat((image, sample))
        else:
            raise ValueError

        both = torch.cat((pixels.view(-1, *si),
                          actual.view(-1, *si),
                          image.view(-1, *si)))
        save_image(both, os.path.join(trav_path,
                                      f"{seed}_{number}_{fold}_{mean_sample}_{high[0]}_traversal.png"),
                   nrow=h_dim)


def trav(model_path, params, kl_indexes, trav_path, mean_sample, si, low=-3,
         high=3, trav_steps=10, seed=0, fold=0, device='cpu'):
    root = os.path.abspath(os.path.join(os.getcwd(), 'test_sets.pkl'))
    pixels = 0
    with open(root, 'rb') as f:
        pixels = pickle.load(f)[0]

    num_samples = len(pixels)

    model = make_vae_model(si, params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    h_dim = model.h_dim
    for i in range(num_samples):
        traverse(model, h_dim, kl_indexes, pixels[i], trav_path,
                str(i), si=si, low=low, high=high,
                trav_steps=trav_steps, mean_sample=mean_sample, seed=seed, fold=fold)

def visualization(file_name, num):
    root = os.path.abspath(os.getcwd() + '/' +
                           file_name + '_cv_' + str(num) + '.npz')
    pixels = np.load(root)

    val_loss = pixels['val_total']
    val_bce = pixels['val_bce']
    val_kl = pixels['val_kld']
    train_loss = pixels['train_total']
    train_bce = pixels['train_bce']
    train_kl = pixels['train_kld']
    n_samples = np.arange(len(val_loss))

    figure, axis = plt.subplots(constrained_layout=True, figsize=(10, 10))
    ax2 = axis.twinx()
    axis.plot(n_samples, val_loss, color='blue', label='Val Loss')
    axis.plot(n_samples, val_bce, color='green', label='Val Recon Loss')
    ax2.plot(n_samples, val_kl, color='red', label='Val KL Loss')
    axis.plot(n_samples, train_loss, linestyle='--',
              color='blue', label='Train Loss')
    axis.plot(n_samples, train_bce, linestyle='--',
              color='green', label='Train Recon Loss')
    axis.set_ylabel("Total/Recon Loss")

    ax2.plot(n_samples, train_kl, linestyle='--',
             color='red', label='Train KL Loss')
    axis.set_xlabel('Num epochs')
    ax2.set_ylabel("KL Loss")
    figure.legend(loc="upper right")
    axis.set_title(file_name + 'Loss')
    figure.savefig(file_name + '_cv_' + str(num) + '_visualization.png')


def averaged_vis(file_name, num):
    val_loss = []
    bce = []
    kl = []
    train_loss = []
    train_bce = []
    train_kl = []
    n_samples = 0
    for i in range(num):
        root = os.path.abspath(os.getcwd() + '/' +
                               file_name + '_cv_' + str(i) + '.npz')
        pixels = np.load(root)
        val = pixels['val_total']
        bce_loss = pixels['val_bce']
        kld = pixels['val_kld']
        t_loss = pixels['train_total']
        t_bce = pixels['train_bce']
        t_kl = pixels['train_kld']
        n_samples = len(kld)
        val_loss.append(val)
        bce.append(bce_loss)
        kl.append(kld)
        train_loss.append(t_loss)
        train_bce.append(t_bce)
        train_kl.append(t_kl)
    val_loss = np.mean(val_loss, axis=0)
    bce = np.mean(bce, axis=0)
    kl = np.mean(kl, axis=0)
    train_loss = np.mean(train_loss, axis=0)
    train_bce = np.mean(train_bce, axis=0)
    train_kl = np.mean(train_kl, axis=0)
    n_samples = np.arange(n_samples)
    figure, axis = plt.subplots(constrained_layout=True, figsize=(10, 10))
    ax2 = axis.twinx()
    axis.plot(n_samples, val_loss, color='blue', label='Val Loss')
    axis.plot(n_samples, bce, color='green', label='Val Recon Loss')
    ax2.plot(n_samples, kl, color='red', label='Val KL Loss')
    axis.plot(n_samples, train_loss, linestyle='--',
              color='blue', label='Train Loss')
    axis.plot(n_samples, train_bce, linestyle='--',
              color='green', label='Train Recon Loss')
    axis.set_ylabel("Total/Recon Loss")

    ax2.plot(n_samples, train_kl, linestyle='--',
             color='red', label='Train KL Loss')
    ax2.set_ylabel("KL Loss")
    figure.legend(loc="upper right")
    axis.set_title(file_name + ' Loss')
    figure.savefig(file_name + '_' + 'average_visualization.png')


def vis(fname, nfolds, averaged=True):
    if averaged:
        averaged_vis(fname, nfolds)
    else:
        for i in range(nfolds):
            visualization(fname, i)


def path_maker(experiment, nfolds):
    """ Creates the directories used by training script.

    Returns
    -------
    base : str
        experiment base path
    base_valid : str
        save directory for validation images
    base_data : str
        save directory for logging
    """
    base = f"outputs/experiment_{experiment}"
    Path(base).mkdir(exist_ok=True)
    base_valid = f"{base}/valid_images"
    Path(base_valid).mkdir(exist_ok=True)
    for fold in range(nfolds):
        Path(f"{base}/valid_images/cv_{fold}").mkdir(exist_ok=True)
        for epoch in range(25):
            Path(f"{base}/valid_images/cv_{fold}/C_{epoch}").mkdir(exist_ok=True)

    base_trav = f"{base}/traversal"
    base_data = f"{base}/data"
    base_random = f"{base}/random_generated"
    Path(base_trav).mkdir(exist_ok=True)
    Path(base_data).mkdir(exist_ok=True)
    Path(base_random).mkdir(exist_ok=True)
    return base, base_valid, base_data, base_trav, base_random

def cross_validate_vec(h_dim, ds_valid, labels, knn, randomize):
    """ vectorized semi-supervised cross validation for vae latent
    encodings

    h_dim : int
        number of latent distributions
    ds_valid : torch.tensor
        dataset containing encodings
        encodings -> font x block x (h_dim * 2)
    labels : list
        contains labels for each encoding
        labels -> font x block x 1
    knn : int
        number of nearest neighbors to include
    randomize : boolean
        shuffle beta_labels

    Returns:
    accuracy : int
    top_labels : dict
        layer a -> label a -> layer b -> top k labels
    """

    hits = [0] * knn
    total = 0
    for i, (font, a_label) in tqdm(enumerate(zip(ds_valid[:-1], labels[:-1])),
                                   total=len(ds_valid[:-1]), position=0, leave=True):
        for j, (za, a_block_label) in tqdm(enumerate(zip(font, a_label)),
                                           total=len(font), position=0, leave=True):
            try:
                iter(a_block_label)
            except:
                a_block_label = [a_block_label]
            for beta in range(i + 1, len(ds_valid)):
                beta_labels = labels[beta]
                if randomize:
                    np.random.shuffle(beta_labels)
                beta_font = ds_valid[beta]
                m = 0.5 * (za + beta_font)
                kla = kl_div(h_dim, za, q=m)
                klb = kl_div(h_dim, beta_font, q=m)
                neighbors = (kla + klb) / 2
                nn, nn_indices = torch.topk(neighbors, knn, largest=False, sorted=True)
                for k in range(1, knn + 1):
                    b_labels = [beta_labels[lab_num] for lab_num in nn_indices[:k]]
                    if type(b_labels[0]) == list:
                        b_labels = [i for j in b_labels for i in j]
                    if bool(set(a_block_label) & set(b_labels)):
                        hits[k-1] += 1
                total += 1
        print(f"set: {i+1} out of {len(ds_valid)-1}")
    return [hit/total for hit in hits]

def prepare_templates(root, scale):
    template_folder = Path(root).rglob('*.png')
    unwanted = list(Path(root).rglob('.*/*'))
    files = [str(x) for x in template_folder if x not in unwanted]
    files.sort()
    print(files)
    templates = []
    names = []
    for f in files:
        temp = cv2.imread(f, 0)/255
        scale_percent = scale # percent of original size
        width = int(temp.shape[1] * scale_percent / 100)
        height = int(temp.shape[0] * scale_percent / 100)
        res_temp = cv2.resize(temp, (width, height), interpolation=cv2.INTER_AREA)
        res_temp = 255 - (res_temp*255).astype(np.uint8)
        templates.append(res_temp)
        names.append(f.split("/")[-1])

    return templates, names
