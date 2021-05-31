import glob, h5py, os, subprocess

import numpy as np
from matplotlib.pyplot import imread

from .utils import resize
from .label_mapping import int2imf
from .fontslist import fonts_with_imf
from .cropping import load_crops500


def txt2png(base_path, font_file, fontsize):
    """Save pngs of text files in one or many fonts and fontsizes.

    Parameters
    ----------
    base_path : str
        Path to hangul folder
    font_file : str
        Name of font file (ttf or otf).
    fontsize : int or list of ints
        Fontsize(s) to output.
    """

    print('txt2png: {}, {}'.format(font_file, fontsize))
    font_path = os.path.join(base_path, 'all_fonts', font_file)
    texts_path = os.path.join(base_path, 'texts')
    if isinstance(fontsize, list):
        fontsizes = fontsize
    else:
        fontsizes = [fontsize]

    for fontsize in fontsizes:
        _, tail = os.path.split(font_path)
        font_name, _ = os.path.splitext(tail)

        base_path, _ = os.path.split(texts_path)
        images_path = os.path.join(base_path, 'pngs', font_name, str(fontsize))

        os.makedirs(images_path, exist_ok=True)

        # convert using font
        files = glob.glob(os.path.join(texts_path, '*.txt'))
        for filename in files:
            name, ext = os.path.splitext(filename)
            input_txt = os.path.join(texts_path, filename)
            _, name = os.path.split(name)
            output_png = os.path.join(images_path,
                                      '{}_{}.png'.format(name, fontsize))
            subprocess.run(['convert', '-font', font_path,
                            '-pointsize', str(fontsize),
                            '-background', 'rgba(0,0,0,0)',
                            'label:@' + input_txt, output_png])


def _txt2png(args):
    txt2png(*args)


def stack_same_size(images, max_h, max_w):
    output = np.zeros((len(images), max_h, max_w), dtype=np.uint8)
    for ii, im in enumerate(images):
        output[ii] = resize(im, (max_h, max_w))
    return output


def png2h5(base_folder, font_name, fontsize):
    print('png2h5: {}, {}'.format(font_name, fontsize))
    h5_folder = os.path.join(base_folder, 'h5s', font_name)
    image_folder = os.path.join(base_folder, 'pngs', font_name, str(fontsize))

    try:
        os.mkdir(h5_folder)
    except FileExistsError:
        pass

    images = []
    labels = []
    initial = []
    medial = []
    final = []

    starts = [44032, 4352, 4449, 4520]
    ends = [55204, 4371, 4470, 4547]

    size_array = []

    def read_im(num, folder, fontsize, size_array, images):
        h = hex(num)
        name = h[2:].upper()
        filename = os.path.join(image_folder,
                                '{}_{}.png'.format(name, fontsize))
        im = imread(filename)
        size_array.append(im.shape)
        images.append(im[..., 1])

    for num in range(starts[0], ends[0]):
        read_im(num, image_folder, fontsize, size_array, images)
        mapping_numbers = int2imf(num)
        labels.append(mapping_numbers)

    sizeSet = set(size_array)
    max_h, max_w, _ = max(sizeSet)

    images = stack_same_size(images, max_h, max_w)
    labels = np.stack(labels)

    if font_name in fonts_with_imf:
        for start, end, im_list in zip(starts[1:], ends[1:],
                                       [initial, medial, final]):
            for num in range(start, end):
                read_im(num, image_folder, fontsize, size_array, im_list)
    else:
        if fontsize == 500:
            shape500 = images.shape[1:]
        else:
            shape500 = None
        initial, medial, final = load_crops500(font_name, shape500=shape500)

    initial = stack_same_size(initial, max_h, max_w)
    medial = stack_same_size(medial, max_h, max_w)
    final = stack_same_size(final, max_h, max_w)

    sum_image = images.sum(axis=0)
    if not np.allclose(initial.std(axis=0), 0):
        sum_image += (initial.sum(axis=0) + medial.sum(axis=0) +
                      final.sum(axis=0))

    keep_r = np.nonzero(sum_image.sum(axis=1) > 0)[0]
    keep_c = np.nonzero(sum_image.sum(axis=0) > 0)[0]

    images = np.pad(images[:, keep_r][:, :, keep_c], ((0, 0), (2, 2), (2, 2)),
                    mode='constant')
    initial = np.pad(initial[:, keep_r][:, :, keep_c], ((0, 0), (2, 2), (2, 2)),
                     mode='constant')
    medial = np.pad(medial[:, keep_r][:, :, keep_c], ((0, 0), (2, 2), (2, 2)),
                    mode='constant')
    final = np.pad(final[:, keep_r][:, :, keep_c], ((0, 0), (2, 2), (2, 2)),
                   mode='constant')

    fname = os.path.join(h5_folder, '{}_{}.h5'.format(font_name, fontsize))
    with h5py.File(fname, 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('labels', data=labels)
        f.create_dataset('initial', data=initial)
        f.create_dataset('medial', data=medial)
        f.create_dataset('final', data=final)


def _png2h5(args):
    png2h5(*args)
