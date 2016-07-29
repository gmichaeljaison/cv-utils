import numpy as np
import cv2 as cv

from cv_utils import DeepNet

import skimage.feature

_DEF_HOG_OPTS = dict(cell_size=(8, 8), orientations=8, block_size=(1, 1))

_deepnet = None


def factory(feature):
    """
    Factory to choose feature extractor

    :param feature: name of the feature
    :return: Feature extractor function
    """
    if feature == 'hog':
        return hog
    elif feature == 'deep':
        return deep
    elif feature == 'gray':
        return gray
    elif feature == 'lab':
        return lab
    elif feature == 'luv':
        return luv
    elif feature == 'hsv':
        return hsv
    elif feature == 'hls':
        return hls
    else:
        return rgb


def deep(img, op=None):
    if op is None or op['prototxt'] is None or op['caffemodel'] is None:
        raise Exception('Insufficient options. prototxt and caffemodel required')

    global _deepnet
    if _deepnet is None:
        _deepnet = DeepNet(op['prototxt'], op['caffemodel'], op['gpu'], op['gpu_device_id'])

    img_f = _deepnet.extract_feature(img, op['layer'])

    return img_f


def hog(img, options=None):
    """
    HOG feature extractor.

    :param img:
    :param options:
    :return: HOG Feature for given image
        The output will have channels same as number of orientations.
        Height and Width will be reduced based on block-size and cell-size
    """
    op = _DEF_HOG_OPTS.copy()
    if options is not None:
        op.update(options)

    img = gray(img)
    img_fd = skimage.feature.hog(img,
                                 orientations=op['orientations'],
                                 pixels_per_cell=op['cell_size'],
                                 cells_per_block=op['block_size'],
                                 visualise=False)
    h, w = img.shape

    cx, cy = op['cell_size']
    n_cellsx, n_cellsy = w // cx, h // cy

    bx, by = op['block_size']
    n_blksx, n_blksy = (n_cellsx - bx) + 1, (n_cellsy - by) + 1

    hog_shape = n_blksy * by, n_blksx * bx, op['orientations']

    image_hog = np.reshape(img_fd, hog_shape)
    return image_hog


def gray(img, op=None):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def lab(img, op=None):
    return cv.cvtColor(img, cv.COLOR_BGR2LAB)


def luv(img, op=None):
    return cv.cvtColor(img, cv.COLOR_BGR2Luv)


def hsv(img, op=None):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)


def hls(img, op=None):
    return cv.cvtColor(img, cv.COLOR_BGR2HLS)


def rgb(img, op=None):
    return img
