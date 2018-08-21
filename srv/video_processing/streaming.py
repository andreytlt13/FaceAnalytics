import argparse
import datetime
import json
import os
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

from srv.video_processing.common.draw_label import draw_label
from srv.video_processing.common.tracker import CentroidTracker

FRAME_WIDTH = 400
RED_COLOR = (0, 0, 255)
SCALE_FACTOR = 1.0

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False, default='models/deploy.prototxt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, default='models/res10_300x300_ssd_iter_140000.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.75,
                help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", required=False, default='video_processing/tmp/photo',
                help="path to output directory")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()  # 'rtsp://admin:admin@10.101.106.12:554/ch01/0' tNgB4SZD
time.sleep(2.0)

# initialize the FPS throughput estimator
fps = None

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=FRAME_WIDTH)

    fps = FPS().start()

    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, SCALE_FACTOR, (W, H),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            box_coordinates = box.astype("int")
            rects.append(box_coordinates)

            (startX, startY, endX, endY) = box_coordinates

    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid), (x, y, w, h) in zip(objects.items(), rects):
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)

        if datetime.datetime.now().second % 10 == 0:
            imgCrop = frame[y:h, x:w]
            p = os.path.sep.join(
                [args["output"], "id_{}.png".format(str(objectID))]
            )
            cv2.imwrite(p, imgCrop)

        description_path = 'video_processing/tmp/description/id_{}.json'.format(objectID)
        if os.path.exists(description_path):
            description = json.loads(open(description_path).read())
            draw_label(frame, description['gender'], (centroid[0] - 100, centroid[1] + 20))
            draw_label(frame, str(description['age']), (centroid[0] - 100, centroid[1]))
            draw_label(frame, description['name'], (centroid[0] - 100, centroid[1] - 20))

        draw_label(frame, text, (centroid[0], centroid[1] + 75))

    fps.update()
    fps.stop()

    info = [
        ("FPS", "{:.2f}".format(fps.fps())),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        draw_label(image=frame, label=text, point=(10, H - ((i * 20) + 20)), color=RED_COLOR)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF1

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
