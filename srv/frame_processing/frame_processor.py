import datetime
import dlib
import imutils
import numpy as np
import cv2
from datetime import datetime
from common import config_parser
from common.on_frame_drawer import draw_label
from face_description.best_face_selector import select_best_face_cascades
from face_description.face_recognition import face_recognizer
from frame_processing.object_tracker import CentroidTracker, TrackableObject
import collections

from frame_processing.deep_sort import preprocessing, nn_matching
from frame_processing.deep_sort.detection import Detection
from frame_processing.deep_sort.tracker import Tracker
from frame_processing.tools import generate_detections as gdet


FRAME_WIDTH = 400
SCALE_FACTOR = 1.0

CONFIG = config_parser.parse()


def in_polygon(x, y, xp, yp):
    c = 0
    for i in range(len(xp)):
        if (((yp[i] <= y and y < yp[i - 1]) or (yp[i - 1] <= y and y < yp[i])) and \
                (x > (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) + xp[i])): c = 1 - c
    return c


class FrameProcessor:
    def __init__(self,
                 confidence=CONFIG['confidence'],
                 detection_model=CONFIG['{}_detection'.format(str(CONFIG['detection_mode']))],
                 contours=None,
                 table=None):

        model, prototxt = detection_model.split(',')

        self.confidence = float(confidence)
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        (self.H, self.W) = (None, None)
        self.ct = CentroidTracker()
        self.trackableObjects = {}
        self.contours = np.array(contours)
        self.table = table

    def process_next_frame(self,
                           vs,
                           faces_sequence,
                           known_face_encodings,
                           known_face_names,
                           info,
                           connection=None
                           ):

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

            if CONFIG['detection_mode'] == 'face':
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            else:
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)

            self.net.setInput(blob)
            detections = self.net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence:
                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype('int')

                    # person box visualization
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)

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

                #!!!!! need to rewrite this for correct write enter and exit events
                event = {
                    'event_time': datetime.now(),
                    'object_id': self.trackableObjects.__len__(),
                    'enter': info['Enter'],
                    'exit': info['Exit'],
                    'y': round((startY+endY)/2, 0),
                    'x': round((startX+endX)/2, 0)
                }

                connection.insert(self.table, event)

                rects.append((startX, startY, endX, endY))

        #cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 255), 2)

        objects = self.ct.update(rects)

        for (objectID, centroid), (x_,y_,w_,h_) in zip(objects.items(), rects):
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

                # y = [c[1] for c in to.centroids]
                # direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)

            if objectID not in faces_sequence:
                faces_sequence[objectID] = collections.deque(maxlen=int(CONFIG['analyze_frames']))

            pad_x, pad_y = 30, 30
            imgCrop = frame[y_ - pad_y : h_ + pad_y, x_ - pad_x : w_ + pad_x]

            if all(imgCrop.shape) > 0:
                faces_sequence[objectID].append(imgCrop)

            if objectID not in faces_sequence:
                info['status'] = 'Recognizing'
                recognized_face_label = face_recognizer(imgCrop, known_face_encodings, known_face_names)
                cv2.putText(frame, recognized_face_label, (centroid[0], centroid[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            else:
                if (len(faces_sequence[objectID]) > 0) and (info['TotalFrames'] % int(CONFIG['analyze_frames']) == 0):
                    info['status'] = 'Recognizing'
                    best_detected_face = select_best_face_cascades(faces_sequence[objectID])

                    # analyzing the best face from stream and return a match with db faces
                    recognized_face_label = face_recognizer(best_detected_face, known_face_encodings, known_face_names)

                    cv2.putText(frame, recognized_face_label, (centroid[0], centroid[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info['TotalFrames'] += 1
        info['Count People'] = self.trackableObjects.__len__()

        # overlay = frame.copy()
        # alpha = 0.1
        # cv2.fillPoly(overlay, pts=[self.contours], color=(255, 255, 0))
        # cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame,  self.H, info, faces_sequence

    def process_next_frame_deepsort_tracker(self,
                                           vs,
                                           faces_sequence,
                                           tracker, encoder,
                                           known_face_encodings,
                                           known_face_names,
                                           info,
                                           connection=None
                                           ):

        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        info['status'] = "Waiting"

        if info['TotalFrames'] % int(CONFIG['skip_frames']) == 0:
            info['status'] = 'Detecting'

        if CONFIG['detection_mode'] == 'face':
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        else:
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)

        self.net.setInput(blob)
        detections = self.net.forward()

        boxs = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence:
                box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                (startX, startY, endX, endY) = box.astype('int')

                # person box visualization
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
                boxs.append(box)

        features = encoder(frame, boxs)

        # score to 1.0 here
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression
        nms_max_overlap = 1.0
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update > 1:
                continue
            bbox = track.to_face()
            objectID = track.track_id

            cv2.putText(frame, str(objectID), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 255), 2)

            to = self.trackableObjects.get(track.track_id, None)
            centroid = (int(0.5*(bbox[0]+bbox[2])), int(0.5*(bbox[1]+bbox[3])))
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                to.centroids.append(centroid)

            self.trackableObjects[objectID] = to
            text = "ID {}".format(objectID)

            if objectID not in faces_sequence:
                faces_sequence[objectID] = collections.deque(maxlen=int(CONFIG['analyze_frames']))

            pad_x, pad_y = 30, 30
            x_,y_,w_,h_ = bbox
            x_, y_, w_, h_ = [int(elem) for elem in [x_,y_,w_,h_]]
            imgCrop = frame[y_ - pad_y : h_ + pad_y, x_ - pad_x : w_ + pad_x]

            if all(imgCrop.shape) > 0:
                faces_sequence[objectID].append(imgCrop)

            if objectID not in faces_sequence:
                info['status'] = 'Recognizing'
                recognized_face_label = face_recognizer(imgCrop, known_face_encodings, known_face_names)
                cv2.putText(frame, recognized_face_label, (centroid[0], centroid[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                if (len(faces_sequence[objectID]) > 0) and (info['TotalFrames'] % int(CONFIG['analyze_frames']) == 0):
                    info['status'] = 'Recognizing'

                    best_detected_face = select_best_face_cascades(faces_sequence[objectID])

                    # analyzing the best face from stream and return a match with db faces
                    recognized_face_label = face_recognizer(best_detected_face, known_face_encodings, known_face_names)

                    cv2.putText(frame, recognized_face_label, (centroid[0], centroid[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            for det in detections:
                bbox = det.to_face()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

                startX = int(int(bbox[0]))
                startY = int(int(bbox[1]))
                endX = int(int(bbox[2]))
                endY = int(int(bbox[3]))

                event = {
                    'event_time': datetime.now(),
                    'object_id': self.trackableObjects.__len__(),
                    'enter': info['Enter'],
                    'exit': info['Exit'],
                    'y': round((startY+endY)/2, 0),
                    'x': round((startX+endX)/2, 0)
                }

                connection.insert(self.table, event)

        info['TotalFrames'] += 1
        info['Count People'] = self.trackableObjects.__len__()

        return frame,  self.H, info, faces_sequence