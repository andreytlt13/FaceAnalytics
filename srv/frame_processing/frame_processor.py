import datetime
import dlib
import json
import os
import numpy as np
import cv2
import imutils
import numpy as np
#import matplotlib.pyplot as plt
import cv2
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


def in_polygon(x, y, xp, yp):
    c = 0
    for i in range(len(xp)):
        if (((yp[i] <= y and y < yp[i - 1]) or (yp[i - 1] <= y and y < yp[i])) and \
                (x > (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) + xp[i])): c = 1 - c
    return c


class FrameProcessor:
    def __init__(self, confidence=CONFIG['confidence'], descriptions_dir=CONFIG['descriptions_dir'],
                 detected_faces_dir=CONFIG['detected_faces_dir'], model=CONFIG['caffe_model'],
                 prototxt=CONFIG['prototxt'],
                 prototxt2=CONFIG['prototxt_person_detection'], model2=CONFIG['caffe_model_person_detection'], classes=CLASSES, trackableObjects={},
                 trackers=[], path_for_image=None, table=None, contours=None) -> None:

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
        self.contours = np.array(contours)
        self.x = [item[0] for item in self.contours]
        self.y = [item[1] for item in self.contours]

    def fill(self, img, points):
        filter = cv2.convexHull(points)
        cv2.fillConvexPoly(img, filter, 255)
        return img

    def process_next_frame(self, vs, info, connection=None, camera_url=None):
        frame = vs.read()

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        info['status'] = "Waiting"
        rects = []

        if info['TotalFrames'] % int(CONFIG['skip_frames']) == 0:
            info['status'] = 'Detecting'
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

                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype('int')

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    self.trackers.append(tracker)
        else:

            for tracker in self.trackers:

                info['status'] = 'Tracking'
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

                #!!!!!need to rewrite this for correct write enter and exit events
                event = {
                    'event_time': datetime.now(),
                    'object_id': self.trackableObjects.__len__(),
                    'enter': info['Enter'],
                    'exit': info['Exit'],
                    'y': round((startY+endY)/2, 0),
                    'x': round((startX+endX)/2, 0)
                }

                connection.insert(self.table, event)#self.trackableObjects.__len__())

                rects.append((startX, startY, endX, endY))

        #cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 255), 2)

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
                    include_centroid = bool(in_polygon(centroid[0], centroid[1], self.x, self.y))
                    exclude_centroid = bool(in_polygon(centroid[0], centroid[1], self.x, self.y)) == False
                    if direction < 0 and include_centroid:
                        info['Enter'] += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and exclude_centroid:
                        info['Exit'] += 1
                        to.counted = True

            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info['TotalFrames'] += 1
        info['Count People'] = self.trackableObjects.__len__()

        overlay = frame.copy()
        #output = frame.copy()
        alpha = 0.3
        cv2.fillPoly(overlay, pts=[self.contours], color=(0, 0, 255))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        #frame = cv2.fillPoly(frame, pts=[self.contours], color=(255, 255, 255))
        #frame = self.fill(frame, self.contours)

        return frame,  self.H, info






