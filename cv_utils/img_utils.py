from __future__ import division

import cv2 as cv
import math
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from cv_utils.constants import *
from cv_utils import Box


LABEL_COLORS = [COL_WHITE, COL_GREEN, COL_YELLOW, COL_MAGENTA]

_DEF_PLOT_OPTS = dict(cmap='jet', vmin=0, vmax=1)


def remove_bg(img, th=(240, 255)):
    """
    Removes similar colored background in the given image.

    :param img: Input image
    :param th: Tuple(2)
        Background color threshold (lower-limit, upper-limit)
    :return: Background removed image as result
    """
    if img.size == 0:
        return img

    img = gray3(img)

    # delete rows with complete background color
    h, w = img.shape[:2]
    i = 0
    while i < h:
        mask = np.logical_or(img[i, :, :] < th[0], img[i, :, :] > th[1])
        if not mask.any():
            img = np.delete(img, i, axis=0)
            i -= 1
            h -= 1
        i += 1

    # if image is complete background only
    if img.size == 0:
        return img

    # delete columns with complete background color
    h, w = img.shape[:2]
    i = 0
    while i < w:
        mask = np.logical_or(img[:, i, :] < th[0], img[:, i, :] > th[1])
        if not mask.any():
            img = np.delete(img, i, axis=1)
            i -= 1
            w -= 1
        i += 1

    return img


def add_bg(img, padding, color=COL_WHITE):
    """
    Adds a padding to the given image as background of specified color

    :param img: Input image.
    :param padding: constant padding around the image.
    :param color: background color that needs to filled for the newly padded region.
    :return: New image with background.
    """
    img = gray3(img)
    h, w, d = img.shape
    new_img = np.ones((h + 2*padding, w + 2*padding, d)) * color[:d]
    new_img = new_img.astype(np.uint8)
    set_img_box(new_img, (padding, padding, w, h), img)
    return new_img


def img_box(img, box):
    """
    Selects the sub-image inside the given box

    :param img: Image to crop from
    :param box: Box to crop from. Box can be either Box object or array of [x, y, width, height]
    :return: Cropped sub-image from the main image
    """
    if isinstance(box, tuple):
        box = Box.from_tup(box)

    if len(img.shape) == 3:
        return img[box.y:box.y + box.height, box.x:box.x + box.width, :]
    else:
        return img[box.y:box.y + box.height, box.x:box.x + box.width]


def set_img_box(img, box, value):
    """
    Updates the given value inside the specified box of the input image

    :param img: Input image
    :param box: Box dimension
    :param value: new image value
    :return: Update image
    """
    if isinstance(box, tuple):
        box = Box.from_tup(box)

    if len(img.shape) == 3:
        img[box.y:box.y + box.height, box.x:box.x + box.width, :] = value
    else:
        img[box.y:box.y + box.height, box.x:box.x + box.width] = value


def add_text_img(img, text, pos, box=None, color=None, thickness=1, scale=1, vertical=False):
    """
    Adds the given text in the image.

    :param img: Input image
    :param text: String text
    :param pos: (x, y) in the image or relative to the given Box object
    :param box: Box object. If not None, the text is placed inside the box.
    :param color: Color of the text.
    :param thickness: Thickness of the font.
    :param scale: Font size scale.
    :param vertical: If true, the text is displayed vertically. (slow)
    :return:
    """
    if color is None:
        color = COL_WHITE

    text = str(text)
    top_left = pos
    if box is not None:
        top_left = box.move(pos).top_left()
        if top_left[0] > img.shape[1]:
            return

    if vertical:
        if box is not None:
            h, w, d = box.height, box.width, 3
        else:
            h, w, d = img.shape
        txt_img = np.zeros((w, h, d), dtype=np.uint8)
        # 90 deg rotation
        top_left = h - pos[1], pos[0]
        cv.putText(txt_img, text, top_left, cv.FONT_HERSHEY_PLAIN, scale, color, thickness)

        txt_img = ndimage.rotate(txt_img, 90)
        mask = txt_img > 0
        if box is not None:
            im_box = img_box(img, box)
            im_box[mask] = txt_img[mask]
        else:
            img[mask] = txt_img[mask]
    else:
        cv.putText(img, text, top_left, cv.FONT_HERSHEY_PLAIN, scale, color, thickness)


def add_rect(img, box, color=None, thickness=1):
    """
    Draws a bounding box inside the image.

    :param img: Input image
    :param box: Box object that defines the bounding box.
    :param color: Color of the box
    :param thickness: Thickness of line
    :return: Rectangle added image
    """
    if color is None:
        color = COL_GRAY

    cv.rectangle(img, box.top_left(), box.bottom_right(), color, thickness)


def add_view_box(img, vbox):
    if vbox.color is None:
        vbox.color = COL_GRAY

    add_rect(img, vbox, vbox.color, vbox.thickness)

    for i, label in enumerate(vbox.labels):
        vertical = False
        if label.angle == 90:
            vertical = True
        add_text_img(img, label.text, label.pos, vbox, label.color, vertical=vertical)


def show_img(img, options=None):
    if is_gray(img):
        op = _DEF_PLOT_OPTS.copy()
        op.update(options)
        plt.imshow(img, cmap=op['cmap'], vmin=op['vmin'], vmax=op['vmax'])
    else:
        # convert from BGR to RGB before plotting
        plt.imshow(img[:, :, ::-1])


def imshow(*imgs, **options):
    """
    Plots multiple images using matplotlib
        by dynamically finding the required number of rows and cols.
    :param imgs: Images as any number of arguments
    :param options: Dict of options
        - cmap: Color map for gray scale images
        - vmin: Minimum value to be used in color map
        - vmax: Maximum value to be used in color map
    """
    n = len(imgs)
    nrows = int(math.ceil(math.sqrt(n)))
    ncols = int(math.ceil(n / nrows))
    for row in range(nrows):
        for col in range(ncols):
            i = row * ncols + col
            if i >= n:
                break
            plt.subplot(nrows, ncols, i+1)
            show_img(imgs[i], options)
    plt.show()


def collage(imgs, size, padding=10):
    """
    Constructs a collage of same-sized images with specified padding.

    :param imgs: Array of images. Either 1d-array or 2d-array.
    :param size: (no. of rows, no. of cols)
    :param padding: Padding space between each image
    :return: New collage
    """
    # make 2d array
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    h, w = imgs[0][0].shape[:2]
    nrows, ncols = size
    nr, nc = nrows * h + (nrows-1) * padding, ncols * w + (ncols-1) * padding

    res = np.zeros((nr, nc, 3), dtype=np.uint8)

    for r in range(nrows):
        for c in range(ncols):
            img = imgs[r][c]

            if is_gray(img):
                img = gray3ch(img)

            rs = r * (h + padding)
            re = rs + h

            cs = c * (w + padding)
            ce = cs + w

            res[rs:re, cs:ce, :] = img

    return res


def is_gray(img):
    return len(img.shape) == 2


def gray3(img):
    return img[:, :, np.newaxis] if is_gray(img) else img


def gray3ch(img):
    return cv.cvtColor(img, cv.COLOR_GRAY2BGR) if is_gray(img) else img
