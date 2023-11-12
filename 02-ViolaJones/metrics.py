import numpy as np

import sklearn.metrics as metrics


def union_bounding_box(a: tuple, b: tuple) -> tuple:
    """
    Get the bounding box resulting of the union of two bounding boxes.

    :param a: Tuple defining a bounding box using the format (x, y, width, height)
    :param b: Tuple defining a bounding box using the format (x, y, width, height)
    :return:
    """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return x, y, w, h


def intersection_bounding_box(a, b) -> tuple:
    """
    Get the bounding box resulting of the intersection of two bounding boxes.
    If no intersection return a tuple with (0, 0, 0, 0).

    :param a: Tuple defining a bounding box using the format (x, y, width, height)
    :param b: Tuple defining a bounding box using the format (x, y, width, height)
    :return:
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w < 0 or h < 0:
        return 0, 0, 0, 0
    else:
        return x, y, w, h


def IoU_bounding_box(bb1, bb2) -> float:
    """
    Get the intersection over union (IoU) metric value of two given bounding boxes.

    :param bb1: Tuple defining a bounding box using the format (x, y, width, height)
    :param bb2: Tuple defining a bounding box using the format (x, y, width, height)
    :return:
    """
    x, y, w, h = union_bounding_box(bb1, bb2)
    area_union = w * h

    x, y, w, h = intersection_bounding_box(bb1, bb2)
    area_intersection = w*h

    if area_union == 0:
        return 0
    else:
        return area_intersection / area_union
