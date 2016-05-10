Computer Vision utility functions
=================================
A library of Computer Vision and OpenCV utility functions. It also includes basic image processing and common utility functions.

### Two major contributions are:
1. Template matching
2. Image Utilities

## Requirements:
- Numpy
- OpenCV

## Template Matching utils
One drawback with OpenCV template matching module is that it is limited to RGB or similar 3-channel feature representations only. However, more meaningful representations like HOG or DeepNetwork features are represented with more number of channels.

### Usage
The package is pip installable. The easiest way to install is by using pip.

``` $pip install specularity_removal ```

You can also download the git repo and install by running the below command.

``` $python setup.py ```

#### HOG template matching
An example for template matching with HOG features. It returns both the heatmap and the scale factor by which the feature representation reduced the image dimension.
##### Other supported features are: 
1. HOG
2. Color spaces: gray, rgb, lab, luv, hsv, hls

```python
import cv2 as cv
from cv_utils import template_matching as tm

template = cv.imread('tests/resources/kelloggs-red-fruit.jpg')
image = cv.imread('tests/resources/sch-image.jpg')

options = dict(feature='hog')
heatmap, scale = tm.feature_match(template, image, options)
```

#### Multi features template matching
In some cases, using one feature is not enough to represent the entire image. Combining RGB with HOG might perform better in some cases.
```python
import ...

template = cv.imread('tests/resources/kelloggs-red-fruit.jpg')
image = cv.imread('tests/resources/sch-image.jpg')

options = dict()
options['features'] = [dict(feature='hog'), dict(feature='rgb')]
heatmap, scale = tm.multi_feat_match(template, image, options)
```

#### Normal template matching to find single instance
In most cases, template matching is used to find exactly one instance of an object in the search image. A special utility for that is given below. It returns the bounding box object and the chosen minimum heatmap score.
```python
import ...

template = cv.imread('tests/resources/kelloggs-red-fruit.jpg')
image = cv.imread('tests/resources/sch-image.jpg')

box, score = tm.match_one(template, image, dict(feature='rgb'))
```
