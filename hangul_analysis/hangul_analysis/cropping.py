import os, glob

import numpy as np
from scipy.misc import imread

from hangul_analysis.utils import resize, bounding_box
from hangul_analysis.read_data import image_shape


def paste(newimage, dims, character_type_ratio, num):
    sh = newimage.shape
    minh = character_type_ratio[num][0]
    minw = character_type_ratio[num][1]
    y0 = int(np.floor(minh * dims[0]))
    x0 = int(np.floor(minw * dims[1]))

    blank = np.zeros((max(y0 + sh[0], dims[0]), max(x0 + sh[1], dims[1])),
                     dtype='uint8')
    blank[y0:y0 + sh[0], x0:x0 + sh[1]] = newimage

    return resize(blank, dims)


def load_crops500(fontname, shape500=None):
    """Load png crops and write them into imf arrays.

    Parameters
    ----------
    fontname : str
        Font name.
    shape500 : tuple
        Shape for 500pt images. If None, loads from h5 file.
    """

    base_path = os.path.join(os.environ['HOME'], 'data/hangul')
    if shape500 is None:
        h5_path = os.path.join(base_path,
                               'h5s/{}/{}_500.h5'.format(fontname, fontname))
        dims = image_shape(h5_path)
    else:
        dims = shape500
    filename = os.path.join(base_path, 'crops/{}/*.png'.format(fontname))
    initarr = 19 * [None]
    medarr = 21 * [None]
    finarr = 27 * [None]
    try:
        crop_ratio_data = np.load('crop_ratios.npz')
    except FileNotFoundError:
        raise FileNotFoundError('Run crop_ratios.py first.')
    for x in glob.glob(filename):
        char = x.split('_')[1][0]
        num = int(x.split('_')[1].split('.')[0][1:]) - 1
        if char == 'i':
            character_type_ratio = crop_ratio_data['initial']
        elif char == 'm':
            character_type_ratio = crop_ratio_data['medial']
        elif char == 'f':
            num -= 1
            if num < 0:
                continue
            character_type_ratio = crop_ratio_data['final']
        else:
            raise ValueError
        img = imread(os.path.join(filename, x))
        newimage = bounding_box(img)
        blank = paste(newimage, dims, character_type_ratio, num)

        if char == 'i':
            initarr[num] = blank
        elif char == 'm':
            medarr[num] = blank
        elif char == 'f':
            finarr[num] = blank
        else:
            raise ValueError

    med2 = np.stack(medarr)
    fin2 = np.stack(finarr)
    init2 = np.stack(initarr)
    return init2, med2, fin2
