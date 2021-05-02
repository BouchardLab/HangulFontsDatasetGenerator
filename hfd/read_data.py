import h5py


def load_data(filename, median_shape=False):
    with h5py.File(filename, 'r') as f:
        if median_shape:
            img = f['images_median_shape'][:]
        else:
            img = f['images'][:]
        lab = f['labels'][:]
        init = f['initial'][:]
        med = f['medial'][:]
        fin = f['final'][:]
    return img, lab, init, med, fin


def load_imf_images(filename):
    with h5py.File(filename, 'r') as f:
        init = f['initial'][:]
        med = f['medial'][:]
        fin = f['final'][:]
    return init, med, fin


def load_image_shape(filename, median_shape=False):
    with h5py.File(filename, 'r') as f:
        if median_shape:
            shape = f['images_median_shape'].shape[1:]
        else:
            shape = f['images'].shape[1:]
    return shape


def load_images(filename, median_shape=False):
    with h5py.File(filename, 'r') as f:
        if median_shape:
            img = f['images_median_shape'][:]
        else:
            img = f['images'][:]
    return img


def load_imf_labels(filename):
    with h5py.File(filename, 'r') as f:
        imf = f['labels'][:]
    return imf


def load_all_labels(filename):
    with h5py.File(filename, 'r') as f:
        imf = f['labels'][:]
        imf_style_labels = f['imf_style_labels'][:]
    return imf, imf_style_labels
