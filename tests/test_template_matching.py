import cv2 as cv

from cv_utils import template_matching as tm


template = cv.imread('tests/resources/kelloggs-red-fruit.jpg')
image = cv.imread('tests/resources/sch-image.jpg')


def test_match_one():
    box, _ = tm.match_one(template, image)

    assert box.height == template.shape[0]
    assert box.width == template.shape[1]


def test_match_template_euclidean():
    hmap = tm.match_template(template, image, dict(distance='euclidean', retain_size=False))

    h, w = image.shape[:2]
    th, tw = template.shape[:2]

    print(hmap.shape, (h-th, w-tw))
    assert hmap.shape == (h - th + 1, w - tw + 1)


def test_feature_match():
    hmap, scale = tm.feature_match(template, image, dict(feature='hog'))
    assert round(scale) == 8


def test_multi_feat_match():
    options = dict()
    options['features'] = [dict(feature='hog'), dict(feature='rgb')]
    hmap, scale = tm.multi_feat_match(template, image, options)

    assert scale == 1
    assert hmap.shape == image.shape[:2]
