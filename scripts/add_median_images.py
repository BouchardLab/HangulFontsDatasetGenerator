import numpy as np
import argparse, h5py, os, sys

from hfd.read_data import load_image_shape
from hfd.utils import resize
from hfd.variables import n_blocks


parser = argparse.ArgumentParser(description='Compute median image shape and add median images to h5.')
parser.add_argument('folder', type=str, help='Folder to search for h5 files.')
parser.add_argument('fontsize', type=int, help='Fontsize.')

args = parser.parse_args()
folder = args.folder
fontsize =args.fontsize

files_for_median = []
for d, _, files in os.walk(folder):
    for fname in files:
        if '{}.h5'.format(fontsize) in fname:
            files_for_median.append(os.path.join(d, fname))

shapes = np.array([load_image_shape(f) for f in files_for_median]).T
shape = [n_blocks]
for dim in shapes:
    sh, counts = np.unique(dim, return_counts=True)
    shape.append(sh[counts.argmax()])

for fname in files_for_median:
    with h5py.File(fname) as f:
        try:
            del f['images_median_shape']
        except KeyError:
            pass
        median_images = resize(f['images'].value, shape)
        f.create_dataset('images_median_shape', data=median_images)
