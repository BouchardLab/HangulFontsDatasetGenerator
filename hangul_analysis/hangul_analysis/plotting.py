# Plotting function for hangul fonts

import h5py, os
import numpy as np

from hangul_analysis.tile_raster_images import tile_raster_images

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .read_data import load_data  # noqa: E402


def plot_clustering_comparison(mean, error, fax=None):
    if fax is None:
        fax = plt.subplots(1, figsize=(6, 6))
    f, ax = fax
    ax.bar(np.arange(7), mean, yerr = error, color='gray')
    ax.set_ylabel('Fold over Chance', fontsize=10)
    ax.set_ylim(0,13)
    ax.set_title('{} Representation vs Labels'.format(name), fontsize=10)
    ax._xticks(np.arange(7), ('Initial', 'Medial', 'Final', 'Initial Geom', 'Medial Geom', 'Final Geom', 'All Geom'),
               rotation=65, fontsize=8)

def plot_km_comparison(cost_matrix, xlabel, title, y, new_y, fax=None):
    if fax is None:
        fax = plt.subplots(1)
    f, ax = fax
    ax.imshow(cost_matrix, cmap='gray', vmin=-1, vmax=0)
    ax.set_ylabel('k-means label')
    ax.set_xlabel(xlabel)
    ax.set_title(title)


def plot_page(ims, title=None, label_mf=False,
              pdf=None):
    shape = ims.shape[1:]
    f, ax = plt.subplots(1)
    n_ims = ims.shape[0]
    img = tile_raster_images(ims.reshape(n_ims, -1),
                             shape,
                             (int(np.ceil(n_ims / 28)), 28),
                             (2, 2),
                             output_pixel_vals=False,
                             initial_value=1)
    ax.imshow(img, cmap='gray_r', interpolation='nearest',
              vmin=0, vmax=1)
    ax.set_title('{}, {} by {} px'.format(title, shape[0], shape[1]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if label_mf:
        ax.set_ylabel('medial index')
        ax.set_xlabel('final index')
    if pdf is not None:
        pdf.savefig(dpi=1000)
    return f, ax


def plot_single_font(base_folder, font_folder, fontsize):
    print('plot_pdf: {}, {}'.format(font_folder, fontsize))
    h5_path = os.path.join(base_folder, 'h5s', font_folder,
                           '{}_{}.h5'.format(font_folder, fontsize))
    save_path = os.path.join(base_folder, 'pdfs', font_folder,
                             '{}_{}.pdf'.format(font_folder, fontsize))
    with h5py.File(h5_path)as f:
        images = f['images'].value.astype('float')
        initial = f['initial'].value.astype('float')
        medial = f['medial'].value.astype('float')
        final = f['final'].value.astype('float')
    with PdfPages(save_path) as pdf:
        # initial
        std = np.std(images, axis=0)
        std /= std.max()
        f, ax = plt.subplots(1, figsize=(3, 3 * std.shape[1] / std.shape[0]))
        ax.imshow(std, cmap='gray_r', interpolation='nearest',
                  vmin=0, vmax=1)
        ax.set_title('Image Standard Deviation')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        pdf.savefig(dpi=1000)

        plt.close(f)

        f, ax = plot_page(initial, 'initial', pdf=pdf)
        plt.close(f)

        f, ax = plot_page(medial, 'medial', pdf=pdf)
        plt.close(f)

        shape = final.shape[1:]
        final = np.concatenate([np.zeros((1,) + shape), final])
        f, ax = plot_page(final, 'final', pdf=pdf)
        plt.close(f)

        idx = 0
        for ii in range(19):
            f, ax = plot_page(images[idx:idx + 21 * 28],
                              title='initial index: {}'.format(ii),
                              pdf=pdf,
                              label_mf=True)
            idx += 21 * 28
            plt.close(f)
    return


def _plot_single_font(args):
    plot_single_font(*args)
    return


def plot_diff_char_same_font(indices, files, fname):
    img, lab, init, med, fin = load_data(files)
    fig, axes = plt.subplots(3, 3)
    axes = axes.ravel()
    for idx, ii in enumerate(indices):
        if 'single' in fname:
            data = init
        else:
            data = img
        axes[ii].imshow(data[idx], cmap='gray')
        axes[ii].axis('off')
    return


def plot_same_char_diff_font(num, files, fname):
    fig, ax = plt.subplots()
    for ii, f in enumerate(files):
        img, lab, init, med, fin = load_data(f)
        plt.subplot(3, 3, ii + 1)
        if 'single' in fname:
            data = init
        else:
            data = img
        plt.imshow(data[num], cmap='gray')
        plt.axis('off')
    return


def plot_diff_char_diff_font(files, indices, fname):
    fig, ax = plt.subplots()
    for file, ind, i in zip(files, indices, range(9)):
        img, lab, init, med, fin = load_data(file)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img[ind], cmap='gray')
        plt.axis('off')
    return
