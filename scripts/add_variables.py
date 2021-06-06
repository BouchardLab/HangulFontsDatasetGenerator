import numpy as np
import argparse, os, sys, h5py

from hfd.variables import label_df

parser = argparse.ArgumentParser(description='Add latent annotations to h5s.')
parser.add_argument('folder', type=str, help='Folder to search for h5 files.')
parser.add_argument('fontsize', type=int, help='Fontsize.')

args = parser.parse_args()
folder = args.folder
fontsize =args.fontsize

labels = ['initial_geometry', 'medial_geometry', 'final_geometry', 'all_geometry']
bof = ['atom_bof', 'atom_mod_rotations_bof']
files = []
for d, _, files in os.walk(folder):
    for fname in files:
        if '{}.h5'.format(fontsize) in fname:
            with h5py.File(os.path.join(d, fname), 'a') as f:
                for l in labels:
                    try:
                        del f[l]
                    except KeyError:
                        pass
                    f.create_dataset(l, data=label_df[l].values)
                for l in bof:
                    try:
                        del f[l]
                    except KeyError:
                        pass
                    f.create_dataset(l, data=np.stack([*label_df[l].values]))
