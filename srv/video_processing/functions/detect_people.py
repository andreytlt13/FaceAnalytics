import cv2
import imutils
import numpy as np

from imutils.object_detection import non_max_suppression

MIN_CONFIDENCE_LEVEL = 0.5
SAFE_MARGIN = 10

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_people(frame, img_w):
    scale_factor = 0.5
    resized = imutils.resize(frame, width=int(img_w * scale_factor))
    objects, weights = hog.detectMultiScale(
        resized, winStride=(8, 8), padding=(24, 24), scale=1.15
    )
    scale_back_factor = int(1 / scale_factor)
    rects = np.array(
        [[scale_back_factor * x - SAFE_MARGIN, scale_back_factor * y - SAFE_MARGIN,
          scale_back_factor * (x + w) + SAFE_MARGIN, scale_back_factor * (y + h) + SAFE_MARGIN]
         for (x, y, w, h) in objects]
    )
    people = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    people = [p for i, p in enumerate(people) if weights[i] > MIN_CONFIDENCE_LEVEL]
    return people
