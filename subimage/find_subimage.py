"""
Module: find_subimage.py
Desc: find instances of an image in another image
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Saturday, Sept 21st 2014

  Given two image inputs, find instances of one image in the other.
  This ought not account for scaling or rotation.
"""

import numpy as np
import scipy.ndimage as sp
import cv2


def find_subimages(image, subimage, confidence=0.80):
    sup_edges = cv2.Canny(image, 32, 128, apertureSize=3)
    sub_edges = cv2.Canny(subimage, 32, 128, apertureSize=3)

    result = cv2.matchTemplate(sup_edges, sub_edges, cv2.TM_CCOEFF_NORMED)
    result = np.where(result > confidence, 1.0, 0.0)

    ccs = get_connected_components(result)
    return correct_bounding_boxes(subimage, ccs)  # [(x1, y1, x2, y2), ...]


def get_connected_components(image):
    s = sp.morphology.generate_binary_structure(2, 2)
    labels, _ = sp.measurements.label(image)
    objects = sp.measurements.find_objects(labels)
    return objects


def correct_bounding_boxes(subimage, connected_components):
    h, w = subimage.shape[:2]
    corrected = []
    for cc in connected_components:
        x = (cc[1].start + cc[1].stop) // 2
        y = (cc[0].start + cc[0].stop) // 2
        corrected.append((x, y, x + w, y + h))
    return corrected


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('subimage')
    parser.add_argument('--confidence', type=float, default=0.80)

    args = parser.parse_args()

    assert os.path.isfile(args.image) and os.path.isfile(args.subimage)

    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    subimage = cv2.imread(args.subimage, cv2.IMREAD_GRAYSCALE)
    locations = find_subimages(image, subimage, args.confidence)
    print(locations)
