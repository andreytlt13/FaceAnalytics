import datetime
import dlib
import json
import os

import cv2
import imutils
import numpy as np

from common import config_parser
from common.on_frame_drawer import draw_label
from frame_processing.object_tracker import CentroidTracker, TrackableObject, CentroidTracker2

FRAME_WIDTH = 400
SCALE_FACTOR = 1.0

CONFIG = config_parser.parse()

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


class FrameProcessor:
    def __init__(self, confidence=CONFIG['confidence'], descriptions_dir=CONFIG['descriptions_dir'],
                 detected_faces_dir=CONFIG['detected_faces_dir'], model=CONFIG['caffe_model'],
                 prototxt=CONFIG['prototxt'],
                 prototxt2=CONFIG['prototxt2'], model2=CONFIG['caffe_model2'], classes=CLASSES, trackableObjects={},
                 trackers=[]) -> None:

        self.confidence = float(confidence)
        self.ct = CentroidTracker()
        self.description_pattern = descriptions_dir + '/id_{}.json'
        self.detected_face_img_pattern = detected_faces_dir + '/id_{}.png'
        (self.H, self.W) = (None, None)
        self.trackableObjects = trackableObjects
        self.trackers = trackers

        # load our serialized model from disk
        print('[INFO] loading model 1...')
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        print('[INFO] loading model 2...')
        self.net2 = cv2.dnn.readNetFromCaffe(prototxt2, model2)
        self.classes = classes

        detected_faces_dir_path = os.path.dirname(self.detected_face_img_pattern)
        if not os.path.exists(detected_faces_dir_path):
            os.makedirs(detected_faces_dir_path)

    def process_next_frame(self, vs, totalFrames=0, totalDown=0, totalUp=0):
        """
        Detects, tracks and stores people's faces as images so that another
        asynchronous process (`face_descriptor.py`) can catch it up and describe face features
        Also draws info onto frame for already completed face descriptions

        :param vs: video stream from camera
        """
        ct = CentroidTracker2(maxDisappeared=40, maxDistance=50)
        # trackers = []
        # trackableObjects = {}

        frame = vs.read()
        frame = imutils.resize(frame, width=FRAME_WIDTH)

        # if the frame dimensions are None, grab them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, SCALE_FACTOR, (self.W, self.H),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        rects = []
        status = "Waiting"

        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > self.confidence:
                box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                box_coordinates = box.astype('int')
                rects.append(box_coordinates)
        objects = self.ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid), (x, y, w, h) in zip(objects.items(), rects):
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = 'ID {}'.format(objectID)

            if datetime.datetime.now().second % 3 == 0:
                imgCrop = frame[y:h, x:w]
                cv2.imwrite(self.detected_face_img_pattern.format(str(objectID)), imgCrop)

            description_path = self.description_pattern.format(objectID)
            if os.path.exists(description_path):
                description = json.loads(open(description_path).read())
                draw_label(frame, description['gender'], (centroid[0] - 100, centroid[1] + 20))
                draw_label(frame, str(description['age']), (centroid[0] - 100, centroid[1]))
                draw_label(frame, description['person_name'], (centroid[0] - 100, centroid[1] - 20))

            draw_label(frame, text, (centroid[0], centroid[1] + 75))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # args["skip_frames"]
        if totalFrames % 30 == 0:
            status = "Detecting"

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
            self.net2.setInput(blob)
            detections = self.net2.forward()

            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > self.confidence:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype('int')

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    self.trackers.append(tracker)
                # otherwise, we should utilize our object *trackers* rather than
                # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in self.trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 0), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = self.trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < self.H // 2:
                        totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > self.H // 2:
                        totalDown += 1
                        to.counted = True

            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            # cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        totalFrames += 1

        info = [
            ("Enter", totalUp),
            ("Exit", totalDown),
            ("TotalFrames", totalFrames),
        ]

        return frame, self.H, status, totalFrames, totalDown, totalUp
