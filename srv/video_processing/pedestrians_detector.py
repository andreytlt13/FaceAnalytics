import cv2
import imutils
import numpy as np

from imutils.object_detection import non_max_suppression

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_people(frame, img_w):
    resized = imutils.resize(frame, width=int(img_w * 0.5))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    objects, weights = hog.detectMultiScale(
        gray, winStride=(8, 8), padding=(24, 24), scale=1.15
    )
    rects = np.array([[2 * x - 10, 2 * y - 10, 2 * (x + w) + 10, 2 * (y + h) + 10] for (x, y, w, h) in objects])
    pedestrians = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    pedestrians = [p for i, p in enumerate(pedestrians) if weights[i] > 0.5]
    return pedestrians
