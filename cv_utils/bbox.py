from __future__ import division

import math

from cv_utils.constants import *


class Box(object):
    """
        To represent a bounding box.

        (x,y), width, and height
    """

    # primary constructor
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    # Other constructors
    @classmethod
    def from_xy(cls, x, y, x2, y2):
        return cls(x, y, x2-x, y2-y)

    @classmethod
    def from_tup(cls, tup):
        return cls(tup[0], tup[1], tup[2], tup[3])
    # end constructors

    @staticmethod
    def enclosing_box(boxes):
        """
        Finds a new box that exactly encloses all the given boxes.

        :param boxes: Array of Box objects
        :return: Box object that encloses all boxes
        """
        x = max(0, min([box.x for box in boxes]))
        y = max(0, min([box.y for box in boxes]))
        x2 = max([box.bottom_right()[0] for box in boxes])
        y2 = max([box.bottom_right()[1] for box in boxes])
        return Box.from_xy(x, y, x2, y2)

    @staticmethod
    def left_most(boxes):
        """
        Finds the left most box out of the given boxes.

        :param boxes: Array of Box objects
        :return: The left-most Box object
        """
        x_list = [(box.x, box) for box in boxes]
        x_list.sort()
        return x_list[0][1]

    @staticmethod
    def right_most(boxes):
        """
        Finds the right most box out of the given boxes.

        :param boxes: Array of Box objects
        :return: The right-most Box object
        """
        x_list = [(box.x, box) for box in boxes]
        x_list.sort()
        return x_list[-1][1]

    @staticmethod
    def intersection_box(box1, box2):
        """
        Finds an intersection box that is common to both given boxes.

        :param box1: Box object 1
        :param box2: Box object 2
        :return: None if there is no intersection otherwise the new Box
        """
        b1_x2, b1_y2 = box1.bottom_right()
        b2_x2, b2_y2 = box2.bottom_right()

        x, y = max(box1.x, box2.x), max(box1.y, box2.y)
        x2, y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
        w, h = x2-x, y2-y
        if w <= 0 or h <= 0:
            return None
        else:
            return Box(x, y, w, h)

    @staticmethod
    def iou(box1, box2):
        """
        Find intersection over union-area for given two boxes.
            It the ratio of intersection-area / union-area

        :param box1: Box object 1
        :param box2: Box object 2
        :return: Intersection-over-union value
        """
        int_box = Box.intersection_box(box1, box2)
        int_area = int_box.area()
        union_area = box1.area() + box2.area() - int_area

        result = 0.0
        if union_area > 0:
            result = float(int_area) / float(union_area)

        return result

    def area(self):
        """
        Area of the current box object. A = width * height
        """
        return self.width * self.height

    def smaller(self, box):
        """
        Checks whether this box is smaller than given box.

        :returns: True if this is smaller than given box by area
        """
        return True if self.area() < box.area() else False

    def overlaps(self, box, th=0.0001):
        """
        Check whether this box and given box overlaps at least by given threshold.

        :param box: Box to compare with
        :param th: Threshold above which overlapping should be considered
        :returns: True if overlaps
        """
        int_box = Box.intersection_box(self, box)
        small_box = self if self.smaller(box) else box
        return True if int_box.area() / small_box.area() >= th else False

    def expand(self, percentage):
        """
        Expands the box co-ordinates by given percentage on four sides. Ignores negative values.

        :param percentage: Percentage to expand
        :return: New expanded Box
        """
        ex_h = math.ceil(self.height * percentage / 100)
        ex_w = math.ceil(self.width * percentage / 100)

        x = max(0, self.x - ex_w)
        y = max(0, self.y - ex_h)
        x2 = self.x + self.width + ex_w
        y2 = self.y + self.height + ex_h
        return Box.from_xy(x, y, x2, y2)

    def padding(self, px):
        """
        Add padding around four sides of box

        :param px: padding value in pixels.
            Can be an array in the format of [top right bottom left] or single value.
        :return: New padding added box
        """
        # if px is not an array, have equal padding all sides
        if not isinstance(px, list):
            px = [px] * 4

        x = max(0, self.x - px[3])
        y = max(0, self.y - px[0])
        x2 = self.x + self.width + px[1]
        y2 = self.y + self.height + px[2]
        return Box.from_xy(x, y, x2, y2)

    def split(self, box_size):
        """
        Splits the box object into many boxes each of given size.
        Assumption: the entire box can be split into rows and columns with the given size.

        :param box_size: (width, height) of the new tiny boxes
        :return: Array of tiny boxes
        """
        w, h = box_size
        rows = round(self.height / h)
        cols = round(self.width / w)

        boxes = []
        for row in range(rows):
            for col in range(cols):
                box = Box(self.x + (col * w), self.y + (row * h), w, h)
                boxes.append(box)
        return boxes

    def pos_by_percent(self, x_percent, y_percent):
        """
        Finds a point inside the box that is exactly at the given percentage place.

        :param x_percent: how much percentage from left edge
        :param y_percent: how much percentage from top edge
        :return: A point inside the box
        """
        x = round(x_percent * self.width)
        y = round(y_percent * self.height)
        return x, y

    def move(self, point, reverse=False):
        """
        Translates the box by given (tx, ty)

        :param point: (tx, ty)
        :param reverse: If true, the translation direction is reversed. ie. (-tx, -ty)
        :return: New translated box
        """
        if reverse:
            point = [-1 * i for i in point]
        return Box(self.x + point[0], self.y + point[1], self.width, self.height)

    def xy_coord(self):
        return self.x, self.y, self.x + self.width, self.y + self.height

    def top_left(self):
        return self.x, self.y

    def bottom_right(self):
        return self.x + self.width, self.y + self.height

    def top_right(self):
        return self.x + self.width, self.y

    def bottom_left(self):
        return self.x, self.y + self.height

    def __str__(self):
        return '(x: {0.x}, y: {0.y}, w: {0.width}, h: {0.height})'.format(self)


class ViewBox(Box):

    def __init__(self, box, color=None, labels=None, thickness=1):
        super(ViewBox, self).__init__(box.x, box.y, box.width, box.height)
        if labels is None:
            labels = []
        self.color = color
        self.labels = labels
        self.thickness = thickness


class Label:

    def __init__(self, pos, text, angle=0, color=None):
        if color is None:
            color = COL_WHITE
        self.pos = pos
        self.text = text
        self.angle = angle
        self.color = color
