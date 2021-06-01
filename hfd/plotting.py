# Plotting function for hangul fonts

import h5py, os
import numpy as np

from .tile_raster_images import tile_raster_images

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .read_data import load_data  # noqa: E402


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
        images = f['images'][:].astype('float')
        initial = f['initial'][:].astype('float')
        medial = f['medial'][:].astype('float')
        final = f['final'][:].astype('float')
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
