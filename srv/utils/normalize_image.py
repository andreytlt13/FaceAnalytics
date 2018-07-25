import cv2

import numpy as np

# Below is the output of calibrate.py script
DIM = (1280, 720)
K = np.array(
    [[601.406657865378, 0.0, 714.6361088321798], [0.0, 605.7953276079065, 316.1816796984329], [0.0, 0.0, 1.0]])
D = np.array([[0.05540844317604619], [-0.8784316408407845], [2.7661761909290625], [-1.5287598287880562]])


def fisheye_to_flat(original_frame):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    return cv2.remap(
        original_frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
