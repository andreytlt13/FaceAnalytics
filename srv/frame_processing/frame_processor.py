import datetime
import dlib
import json
import os

import cv2
import imutils
import numpy as np
#import matplotlib.pyplot as plt

from datetime import datetime
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
                 trackers=[], path_for_image=None, table=None) -> None:

        self.confidence = float(confidence)
        self.ct = CentroidTracker()
        self.description_pattern = descriptions_dir + '/id_{}.json'
        self.detected_face_img_pattern = detected_faces_dir + '/id_{}.png'
        (self.H, self.W) = (None, None)
        self.trackableObjects = trackableObjects
        self.trackers = trackers
        # load our serialized model from disk
        #print('[INFO] loading model 1...')
        #self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        print('[INFO] loading model for person detection...')
        self.net = cv2.dnn.readNetFromCaffe(prototxt2, model2)
        self.classes = classes
        self.path_for_image = path_for_image
        self.table = table

    def process_next_frame(self, vs, totalFrames=0, totalDown=0, totalUp=0, connection=None, camera_url=None):
        frame = vs.read()

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        status = "Waiting"
        rects = []

        if totalFrames % int(CONFIG['skip_frames']) == 0:
            status = 'Detecting'
            self.trackers =[]

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence:
                    idx = int(detections[0, 0, i, 1])

                    if self.classes[idx] != 'person':
                        continue

                    box = detections[0, 0, i, 3:7]  * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype('int')

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    self.trackers.append(tracker)
        else:

            for tracker in self.trackers:

                status = 'Tracking'
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                cropped = frame[startY:endY, startX:endX]
                cv2.imwrite('{0}/{1}_{2}.png'.format(self.path_for_image,
                                                     self.trackableObjects.__len__(),
                                                     datetime.now().strftime("%H:%M:%S")), cropped)
                event = {
                    'event_time': datetime.now(),
                    'object_id': self.trackableObjects.__len__(),
                    'enter': totalUp,
                    'exit': totalDown,
                    'y': round((startY+endY)/2, 0),
                    'x': round((startX+endX)/2, 0)
                }

                connection.insert(self.table, event)#self.trackableObjects.__len__())

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 255), 2)

        objects = self.ct.update(rects)

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
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        totalFrames += 1

        info = [
            ('Enter', totalUp),
            ('Exit', totalDown),
            ('TotalFrames', totalFrames),
            ('Status', status),
            ('Count People',  self.trackableObjects.__len__())
        ]

        return frame,  self.H, info


# Debugging middleware caught exception in streamed response at a point where response headers were already sent.
# Traceback (most recent call last):
#   File "/Users/andrey/anaconda3/envs/py37/lib/python3.7/site-packages/werkzeug/wsgi.py", line 870, in __next__
#     return self._next()
#   File "/Users/andrey/anaconda3/envs/py37/lib/python3.7/site-packages/werkzeug/wrappers.py", line 82, in _iter_encoded
#     for item in iterable:
#   File "/Users/andrey/PycharmProjects/FaceAnalytics/srv/flask_api/flask_streaming_api.py", line 77, in stream
#     frame, _, info = frame_processor.process_next_frame(vs, totalFrames, totalDown, totalUp, connection, camera_url)
#   File "/Users/andrey/PycharmProjects/FaceAnalytics/srv/frame_processing/frame_processor.py", line 121, in process_next_frame
#     objects = self.ct.update(rects)
#   File "/Users/andrey/PycharmProjects/FaceAnalytics/srv/frame_processing/object_tracker.py", line 46, in update
#     for objectID in self.disappeared.keys():
# RuntimeError: OrderedDict mutated during iteration
