import numpy as np
from skimage.transform import resize as sk_resize
from skimage import img_as_ubyte


def bounding_box(img):
    keep_r = np.nonzero(img.sum(axis=1) > 0)[0]
    keep_c = np.nonzero(img.sum(axis=0) > 0)[0]
    return img[keep_r][:, keep_c]


def resize(img, dims):
    return img_as_ubyte(sk_resize(img, dims, anti_aliasing=True))
