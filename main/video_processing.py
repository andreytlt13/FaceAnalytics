#!/usr/bin/env python
import cv2
import numpy as np
from common.deep_sort import preprocessing, nn_matching
from common.deep_sort.detection import Detection
from common.deep_sort.tracker import Tracker
from common.tools import generate_detections as gdet
from common.object_tracker import TrackableObject

PROTOTXT = "/Users/andrey/PycharmProjects/FaceAnalytics/main/model/MobileNetSSD_deploy.prototxt"
MODEL = "/Users/andrey/PycharmProjects/FaceAnalytics/main/model/MobileNetSSD_deploy.caffemodel"

# deep sort implementation
# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None

ENCODER_PATH = "/Users/andrey/PycharmProjects/FaceAnalytics/main/model/mars-small128.pb"
#metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#tracker = Tracker(metric)

class VideoStream():
    def __init__(self, camera_url=0):
        self.camera_url = camera_url
        self.vs = cv2.VideoCapture(self.camera_url)
        self.W, self.H = int(self.vs.get(3)), int(self.vs.get(4))
        self.fps = self.vs.get(5)
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

        self.encoder = gdet.create_box_encoder(ENCODER_PATH, batch_size=1)
        self.metrics = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metrics)

        self.trackableObjects = {}

        self.info = {
            'status': None
        }
        #self.start_video_processing = self.start_video_processing

    # def start_video_processing(self):
    #     #vs = VideoStream(src=self.camera_url)
    #     while True:
    #         ret, frame = self.vs.read()
    #
    #         return frame

    def preocess_next_frame(self):

        ret, frame = self.vs.read()
        frame, detections = self.detecting(frame)
        frame = self.tracking(frame, detections)

        return frame

    def detecting(self, frame):

        self.info['status'] = 'Detecting'
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)

        self.net.setInput(blob)
        detections = self.net.forward()

        boxs = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.75:
                box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                (startX, startY, endX, endY) = box.astype('int')

                # person box visualization
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
                boxs.append(box)

        features = self.encoder(frame, boxs)

        # score to 1.0 here
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression
        nms_max_overlap = 1.0
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        return frame, detections

    def tracking_by_nn(self, frame, detections):

        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:
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

        return frame

    def tracking_by_centroid(self, frame, detections):




        return "kokoko"





if __name__ == "__main__":
    url = 0
    cam = VideoStream(url)

    while True:
        frame = cam.preocess_next_frame()
        cv2.imshow('Frame', frame)
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()