import cv2 as cv

import cv_utils
from cv_utils import img_utils


def test_remove_bg():
    """
    Test: remove white background
    """
    img = cv.imread('tests/resources/with-bg.jpg')
    img_res = img_utils.remove_bg(img)
    assert img_res.shape == (428, 332, 3)


def test_remove_bg_black():
    """
    Test: remove black background
    """
    img = cv.imread('tests/resources/with-black-bg.jpg')
    img_res = img_utils.remove_bg(img, (0, 10))
    assert img_res.shape == (262, 254, 3)


def test_remove_empty():
    """
    Test image with complete background
    """
    img = cv.imread('tests/resources/with-black-bg.jpg')
    img_res = img_utils.remove_bg(img, (0, 255))
    assert img_res.size == 0


def test_add_bg():
    img = cv.imread('tests/resources/without-black-bg.jpg')
    img_res = img_utils.add_bg(img, 50, cv_utils.COL_YELLOW)
    h, w, d = img.shape
    assert img_res.shape == (h + 2*50, w + 2*50, d)
