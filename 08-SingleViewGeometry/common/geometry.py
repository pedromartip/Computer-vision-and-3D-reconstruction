# -*- coding: utf-8 -*-
""" Module containing a set of function to work with descriptors and geometry
"""
import cv2
import numpy as np
from typing import Tuple

ORG_2_HMG = 1
HMG_2_ORG = 2


def convert(coordinates, img: np.ndarray, code: int):
    """ Convert coordinates from image system to homogenous system and vice
    versa.

    The function converts an input coordinate from coordinate system to another.
    To do so we need the shape of the images.

    Args:
        coordinates:
        img:
        code:

    Returns:

    """
    if code != ORG_2_HMG and code != HMG_2_ORG:
        raise TypeError("Unknown operation")

    if code == ORG_2_HMG:
        return __to_homogenous(coordinates, (img.shape[1], img.shape[0]))
    elif code == HMG_2_ORG:
        return __to_original(coordinates, (img.shape[1], img.shape[0]))


def __to_homogenous(coordinates: Tuple[int, int], size: Tuple[int, int]):
    """ Converts coordinates from an image to homogenous system.

    Args:
        coordinates:
        size:

    Returns:

    """
    semi_axis = np.array(size) / 2

    w = ((size[0] + size[1]) / 4)

    homogenous = np.array([coordinates[0] - semi_axis[0], coordinates[1] - semi_axis[1]])
    homogenous = np.append(homogenous, [w])

    return tuple(homogenous)


def __to_original(coordinates: Tuple[int, int], size: Tuple[int, int]):
    semi_axis = np.array(size) / 2

    w = ((size[0] + size[1]) / 4)

    original_x = int(((coordinates[0] / coordinates[-1]) * w) + semi_axis[1])
    original_y = int(((coordinates[1] / coordinates[-1]) * w) + semi_axis[0])

    return original_x, original_y


def draw_epipolar_lines(img1: np.ndarray, img2: np.ndarray, lines, pts1, pts2):
    """ We draw a set of lines and points on the images passed as parameter.

    Args:
        img1 (np.ndarray): First image to draw the epipolar images
        img2 (np.ndarray): Second image to draw the epipolar images
        lines (List): A set of epipolar lines
        pts1 (List[Tuple[int,int]]): A list of points in the first image. The
            first part of a matching
        pts2 (List[Tuple[int,int]]): A list of points in the second image. The
            second part of a matching

    Returns:

    """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


def get_kp_desc(method: str, img: np.ndarray, **kwargs):
    """ Gets the keypoints and the descriptors from an image.

    Calculates the keypoints of an image and its descriptors. To do so we use
    different well-known algorithms. The method used is passed as parameter.
    The results are always a tuple with two elements, the keypoints and its
    descriptors


    Args:
        method (str): See below for more information
        img (np.ndarray): Image to extract the descriptors
        **kwargs: Extra arguments for the methods
    Methods:
        "O" (ORB): Oriented FAST and rotated BRIEF descriptors
        "sift": Scale-Invariant Feature Transform descriptors
    Returns:
        Tuple with the descriptors and the descriptions
    """
    method_call = None

    if method == "O":
        orb = cv2.ORB_create(**kwargs)
        method_call = orb.detectAndCompute
    elif method == "sift":
        sift = cv2.xfeatures2d.SIFT_create(**kwargs)
        method_call = sift.detectAndCompute

    kp, descs = method_call(img, mask=None, **kwargs)

    return kp, descs


def match_descriptors(method: str, desc1, desc2, **kwargs):
    """ Search matches between two sets of descriptors

    Search matches between two set of descriptors. Multiple methods are
    available. The usefulness of the methods depend on the format of the
    descriptors.

    Args:
        method (str): See options below
        desc1: List of descriptors
        desc2: List of descriptors
    Methods:
        "D": Bruteforce matcher with Hamming.
        "F": Bruteforce without Hamming
    Returns:
        List of matches
    """
    matches = None
    if method == "D":
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(desc1, desc2, **kwargs)
    elif method == "F":
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(desc1, desc2, **kwargs)

    return matches


def filter_matches(method: str, matches, min_distance: int = None,
                   proportion: float = None):
    """ Filter matches by multiple conditions.

    Args:
        method (str): See below list of methods available
        matches: List of matches
        min_distance (int): Minimum distance to accept a mathc (DIST)
        proportion (float): Proportion between the second and first match
                            distances (KNN)

    Methods:
        "DIST": Select the matches with a distance higher than a minimum
        "KNN": Select the matches with a small distance on the first match and
                with a high value on the second. Defined by proportion

    Returns:
        List of filtered matches
    """
    if method == "DIST":
        matches = list(filter(lambda m: m.distance < min_distance, matches))
    elif method == "KNN":
        matches = list(
            filter(lambda m: m[0].distance < m[1].distance * proportion,
                   matches))

    return matches
