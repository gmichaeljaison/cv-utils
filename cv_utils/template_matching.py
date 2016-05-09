from __future__ import division

import numpy as np
import cv2 as cv
import scipy.spatial

from cv_utils import img_utils, feature_extractor as fe
from cv_utils.bbox import Box


_DEF_TM_OPT = dict(feature='rgb',
                   distance='correlation',
                   normalize=True,
                   retain_size=True)


def match_one(template, image, options=None):
    """
    Match template and find exactly one match in the Image using specified features.

    :param template: Template Image
    :param image: Search Image
    :param options: Options include
        - features: List of options for each feature
    :return: (Box, Score) Bounding box of the matched object, Heatmap value
    """
    heatmap, scale = multi_feat_match(template, image, options)

    min_val, _, min_loc, _ = cv.minMaxLoc(heatmap)
    top_left = tuple(scale * x for x in min_loc)
    score = min_val

    h, w = template.shape[:2]
    return Box(top_left[0], top_left[1], w, h), score


def multi_feat_match(template, image, options=None):
    """
    Match template and image by extracting multiple features (specified) from it.

    :param template: Template image
    :param image:  Search image
    :param options: Options include
        - features: List of options for each feature
    :return:
    """
    h, w = image.shape[:2]
    scale = 1
    if options is not None and 'features' in options:
        heatmap = np.zeros_like(image)
        for foptions in options['features']:
            f_hmap, _ = feature_match(template, image, foptions)
            heatmap += cv.resize(f_hmap, (w, h), interpolation=cv.INTER_AREA)
        heatmap /= len(options['features'])
    else:
        heatmap, scale = feature_match(template, image, options)
    return heatmap, scale


def feature_match(template, image, options=None):
    """
    Match template and image by extracting specified feature

    :param template: Template image
    :param image: Search image
    :param options: Options include
        - feature: Feature extractor to use. Default is 'rgb'. Available options are:
            'hog', 'lab', 'rgb', 'gray'
    :return: Heatmap
    """
    op = _DEF_TM_OPT.copy()
    if options is not None:
        op.update(options)

    feat = fe.factory(op['feature'])
    tmpl_f = feat(template, op)
    img_f = feat(image, op)

    scale = image.shape[0] / img_f.shape[0]
    heatmap = match_template(tmpl_f, img_f, op)
    return heatmap, scale


def match_template(template, image, options=None):
    """
    Multi channel template matching using simple correlation distance

    :param template: Template image
    :param image: Search image
    :param options: Other options:
        - distance: Distance measure to use. Default: 'correlation'
        - normalize: Heatmap values will be in the range of 0 to 1. Default: True
        - retain_size: Whether to retain the same size as input image. Default: True
    :return: Heatmap
    """
    # If the input has max of 3 channels, use the faster OpenCV matching
    if len(image.shape) <= 3 and image.shape[2] <= 3:
        return match_template_opencv(template, image, options)

    op = _DEF_TM_OPT.copy()
    if options is not None:
        op.update(options)

    template = img_utils.gray3(template)
    image = img_utils.gray3(image)

    h, w, d = template.shape
    im_h, im_w = image.shape[:2]

    template_v = template.flatten()

    heatmap = np.zeros((im_h - h, im_w - w))
    for col in range(0, im_w - w):
        for row in range(0, im_h - h):
            cropped_im = image[row:row + h, col:col + w, :]
            cropped_v = cropped_im.flatten()

            if op['distance'] == 'euclidean':
                heatmap[row, col] = scipy.spatial.distance.euclidean(template_v, cropped_v)
            elif op['distance'] == 'correlation':
                heatmap[row, col] = scipy.spatial.distance.correlation(template_v, cropped_v)

    # normalize
    if op['normalize']:
        heatmap /= heatmap.max()

    # size
    if op['retain_size']:
        hmap = np.ones(image.shape[:2]) * heatmap.max()
        h, w = heatmap.shape
        hmap[:h, :w] = heatmap
        heatmap = hmap

    return heatmap


def match_template_opencv(template, image, options):
    """
    Match template using OpenCV template matching implementation.
        Limited by number of channels as maximum of 3.
        Suitable for direct RGB or Gray-scale matching

    :param options: Other options:
        - distance: Distance measure to use. (euclidean | correlation | ccoeff).
            Default: 'correlation'
        - normalize: Heatmap values will be in the range of 0 to 1. Default: True
        - retain_size: Whether to retain the same size as input image. Default: True
    :return: Heatmap
    """
    # if image has more than 3 channels, use own implementation
    if len(image.shape) > 3:
        return match_template(template, image, options)

    op = _DEF_TM_OPT.copy()
    if options is not None:
        op.update(options)

    method = cv.TM_CCORR_NORMED
    if op['normalize'] and op['distance'] == 'euclidean':
        method = cv.TM_SQDIFF_NORMED
    elif op['distance'] == 'euclidean':
        method = cv.TM_SQDIFF
    elif op['normalize'] and op['distance'] == 'ccoeff':
        method = cv.TM_CCOEFF_NORMED
    elif op['distance'] == 'ccoeff':
        method = cv.TM_CCOEFF
    elif not op['normalize'] and op['distance'] == 'correlation':
        method = cv.TM_CCORR

    heatmap = cv.matchTemplate(image, template, method)

    # make minimum peak heatmap
    if method not in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        if op['normalize']:
            heatmap = 1 - heatmap
        else:
            heatmap = heatmap.max() - heatmap

    # size
    if op['retain_size']:
        hmap = np.ones(image.shape[:2]) * heatmap.max()
        h, w = heatmap.shape
        hmap[:h, :w] = heatmap
        heatmap = hmap

    return heatmap
