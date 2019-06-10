#!/usr/bin/env python
import cv2
import numpy as np
import dlib
from datetime import datetime
from main.common.deep_sort import preprocessing, nn_matching
from main.common.deep_sort.detection import Detection
from main.common.deep_sort.tracker import Tracker
from main.common.tools import generate_detections as gdet
from main.common.object_tracker import TrackableObject, CentroidTracker

import nets.resnet_v1_50 as model
import heads.fc1024 as head
import tensorflow as tf

from imutils.video import FPS
import multiprocessing
import numpy as np
import argparse
import imutils
import dlib
import cv2

# path to PycharmProjects
root_path = '/Users/andrey/PycharmProjects/'

PROTOTXT = root_path + "FaceAnalytics/main/model/MobileNetSSD_deploy.prototxt"
MODEL = root_path + "FaceAnalytics/main/model/MobileNetSSD_deploy.caffemodel"

# deep sort implementation
# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None

ENCODER_PATH = root_path + "FaceAnalytics/main/model/mars-small128.pb"
#metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#tracker = Tracker(metric)


class VideoStream():
    def __init__(self, camera_url=0):
        self.camera_url = 0 if camera_url == '0' else camera_url
        self.vs = cv2.VideoCapture(self.camera_url)
        self.W, self.H = int(self.vs.get(3)), int(self.vs.get(4))
        self.fps = self.vs.get(5)
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        self.ct = CentroidTracker()
        self.trackableObjects = {}
        self.trackers = []
        self.embeding_list = []

        self.imgVectorizer, self.endpoints, self.images = self.start_img_vectorizer()

        self.info = {
            'status': None,
            'FPS': 0,
            'Enter': 0,
            'Exit': 0,
            'TotalFrames': 0,
            'Status': 'start',
            'Count People': 0
        }

    def process_next_frame(self):
        ret, frame = self.vs.read()

        if not ret:
            return None

        orig_frame = frame.copy()
        frame = imutils.resize(frame, width=600)
        self.H, self.W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = []
        objects = ()
        if self.info['TotalFrames'] % 30 == 0:
            frame = self.detecting(frame, rgb)
        else:
            rects = self.tracking(rgb, rects)

        if len(rects) > 0:
            objects, M = self.ct.update(rects, orig_frame, frame, self.trackableObjects, self.embeding_list)
            frame = self.draw_labels(frame, objects)

            #Upddate Trackable objects
            self.update_trackable_objects(objects)

            #Face recognition


        self.info['TotalFrames'] += 1
        return frame

    def draw_labels(self, frame, objects):

        for (objectID, info) in objects.items():

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (info['centroid'][0] - 10, info['centroid'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (info['centroid'][0], info['centroid'][1]), 4, (0, 255, 0), -1)

        return frame

    def tracking(self, rgb, rects):
        self.info['status'] = 'Tracking'

        for tracker in self.trackers:

            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

        return rects

    def detecting(self, frame, rgb):
        self.info['status'] = 'Detecting'
        self.trackers = []
        self.embeding_list = []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            idx = int(detections[0, 0, i, 1])

            if self.classes[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
            (startX, startY, endX, endY) = box.astype('int')

            # --- person box visualization
            #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            if startY < 0:
                startY = 0

            imgCrop = frame[startY:endY, startX:endX]

            resize_img = cv2.resize(imgCrop, (128, 256))
            resize_img = np.expand_dims(resize_img, axis=0)
            emb = self.imgVectorizer.run(self.endpoints['emb'], feed_dict={self.images: resize_img})
            self.embeding_list.append(emb)

            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)
            self.trackers.append(tracker)

        return frame

    def start_img_vectorizer(self):
        tf.Graph().as_default()
        sess = tf.Session()
        images = tf.zeros([1, 256, 128, 3], dtype=tf.float32)
        endpoints, body_prefix = model.endpoints(images, is_training=False)
        with tf.name_scope('head'):
            endpoints = head.head(endpoints, 128, is_training=False)
        tf.train.Saver().restore(sess, root_path+'FaceAnalytics/main/model/checkpoint-25000')
        return sess, endpoints, images

    def update_trackable_objects(self, objects):

        for (objectID, info) in objects.items():

            # if M.size == 0:
            #     objectID = self.trackableObjects.__len__()
            # elif M[np.argmin(M[list(objects.keys()).index(objectID)])] < 20:
            #     objectID = np.argmin(M[list(objects.keys()).index(objectID)])
            # else:
            #     objectID = self.trackableObjects.__len__()


            to = self.trackableObjects.get(objectID, None)


            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, info["centroid"], info["embeding"], info["rect"], info["img"])
            else:
                to.centroids.append(info["centroid"])
                to.embeding.append(info["embeding"])
                to.rect = info["rect"]
                to.img = info["img"]

            self.trackableObjects[objectID] = to


