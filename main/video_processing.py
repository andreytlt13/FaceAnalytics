#!/usr/bin/env python
import cv2
import numpy as np
import dlib
import datetime
from common.deep_sort import preprocessing, nn_matching
from common.deep_sort.detection import Detection
from common.deep_sort.tracker import Tracker
from common.tools import generate_detections as gdet
from common.object_tracker import TrackableObject, CentroidTracker

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
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

        ##for nn
        self.encoder = gdet.create_box_encoder(ENCODER_PATH, batch_size=1)
        self.metrics = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metrics)

        #for centroid
        self.trackers = []
        self.ct = CentroidTracker()
        self.trackableObjects = {}
        self.trackers = []
        self.objects = 0
        self.EmbTrackers = {}
        #self.rects = []


        #nn
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
        #self.start_video_processing = self.start_video_processing

    # def start_video_processing(self):
    #     #vs = VideoStream(src=self.camera_url)
    #     while True:
    #         ret, frame = self.vs.read()
    #
    #         return frame

    def start_img_vectorizer(self):
        tf.Graph().as_default()
        sess = tf.Session()
        images = tf.zeros([1, 256, 128, 3], dtype=tf.float32)
        endpoints, body_prefix = model.endpoints(images, is_training=False)
        with tf.name_scope('head'):
            endpoints = head.head(endpoints, 128, is_training=False)
        tf.train.Saver().restore(sess, 'model/checkpoint-25000')
        return sess, endpoints, images

    def person_distance(self, person_encodings, person_to_compare):
        if len(person_encodings) == 0:
            return np.empty((0))
        return np.linalg.norm(person_encodings - person_to_compare, axis=1)

    def compare_persons(self, known_person_encodings, person_encoding_to_check, tolerance):
        print(self.person_distance(known_person_encodings, person_encoding_to_check))
        return list(self.person_distance(known_person_encodings, person_encoding_to_check) <= tolerance)

    def person_recognizer(self, new_person_vector, known_person_encodings, known_person_names):
        #new_person_vector = api.human_vector(new_person_image)[0]

        matches = self.compare_persons(known_person_encodings, new_person_vector, tolerance=20)

        name = 'unknown_person'

        # Or instead, use the known face with the smallest distance to the new face
        person_distances = self.person_distance(known_person_encodings, new_person_vector)
        best_match_index = np.argmin(person_distances)
        if matches[best_match_index]:
            name = known_person_names[best_match_index]

        print('linalg.norm the smallest distance result match:{}'.format(name))

        return known_person_names[best_match_index], name

    def process_next_frame(self):

        ret, frame = self.vs.read()
        frame = imutils.resize(frame, width=500)
        self.H, self.W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = []

        if self.info['TotalFrames'] % 30 == 0:
            frame, detections = self.detecting(frame)
        else:
            rects = self.tracking(rgb, rects)

        objects = self.ct.update(rects)
        frame = self.draw_labels(frame, objects)

        self.info['TotalFrames'] += 1
        return frame

    def detecting_by_nn(self, frame):

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
        self.info['status'] = 'Detecting'
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

    def detecting(self, frame):
        self.info['status'] = 'Detecting'
        self.trackers = []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.75:
                idx = int(detections[0, 0, i, 1])

                if self.classes[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                (startX, startY, endX, endY) = box.astype('int')

                # --- person box visualization
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                if startY < 0:
                    startY = 0

                imgCrop = frame[startY:endY, startX:endX]

                resize_img = cv2.resize(imgCrop, (128, 256))
                resize_img = np.expand_dims(resize_img, axis=0)
                emb = self.imgVectorizer.run(self.endpoints['emb'], feed_dict={self.images: resize_img})

                name = "unknown_person"
                for n in list(self.EmbTrackers.keys()):
                    tmp, name = self.person_recognizer(emb[0], self.EmbTrackers[n], [n])
                    print("kokok")

                if name == "unknown_person" or len(self.EmbTrackers.keys()) == 0 :
                    #objectID = self.objects + 1

                    self.objects = self.objects + 1

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(frame, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    self.trackers.append(tracker)
                    #to = self.trackableObjects.get(objectID, None)

                    cv2.imwrite("photo/ID_{}.jpeg".format(self.objects), imgCrop)

                    self.EmbTrackers[self.objects] = emb


        return frame, detections

    def tracking(self, rgb, rects):

        self.info['status'] = 'Tracking'

        for tracker in self.trackers:

            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # !!!!!need to rewrite this for correct write enter and exit events

            if self.trackableObjects.__len__() > 0:
                event = {
                    #'event_time': datetime.now(),
                    'object_id': self.trackableObjects.__len__() - 1,
                    'enter': self.info['Enter'],
                    'exit': self.info['Exit'],
                    'y': round((startY + endY) / 2, 0),
                    'x': round((startX + endX) / 2, 0)#,
                    #'names': self.trackableObjects[int(self.trackableObjects.__len__() - 1)].names,
                    #'name': self.trackableObjects[int(self.trackableObjects.__len__() - 1)].name,
                    #'stars': self.trackableObjects[int(self.trackableObjects.__len__() - 1)].stars,
                    #'description': self.trackableObjects[int(self.trackableObjects.__len__() - 1)].description
                }

                #connection.insert(self.table, event)  # self.trackableObjects.__len__())

            rects.append((startX, startY, endX, endY))

        return rects


    def draw_labels(self, frame, objects):

        for (objectID, centroid) in objects.items():

            to = self.trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                #need for next step
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

            self.trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        return frame




    def write_person(self, frame):


        return "done"



class VideoStream2():

    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.vs = cv2.VideoCapture(self.camera_url)
        self.inputQueues = []
        self.outputQueues = []
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        self.fps = FPS().start()
        (self.h, self.w) = None, None
        self.W, self.H = int(self.vs.get(3)), int(self.vs.get(4))
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

        #for centroid
        self.trackers = []
        self.ct = CentroidTracker()
        self.trackableObjects = {}
        #self.contours = np.array(contours)

    def process_next_frame(self):
        (grabbed, frame) = self.vs.read()
        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = self.detecting(frame)

        # if len(self.inputQueues) == 0:
        #     #detections = self.detecting(frame)
        #     frame = self.create_queues(detections, frame, rgb)
        # else:
        #     frame = self.queues(frame, rgb)

        return frame

    def queues(self, frame, rgb):
        for iq in self.inputQueues:
            iq.put(rgb)

        # loop over each of the output queues
        for oq in self.outputQueues:
            # grab the updated bounding box coordinates for the
            # object -- the .get method is a blocking operation so
            # this will pause our execution until the respective
            # process finishes the tracking update
            (label, (startX, startY, endX, endY)) = oq.get()

            # draw the bounding box from the correlation object
            # tracker
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        return frame

    def create_queues(self, detections, frame, rgb):

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.75:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])
                label = self.classes[idx]

                # if the class label is not a person, ignore it
                if self.classes[idx] != "person":
                    continue
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                (startX, startY, endX, endY) = box.astype("int")
                bb = (startX, startY, endX, endY)

                # create two brand new input and output queues,
                # respectively
                iq = multiprocessing.Queue()
                oq = multiprocessing.Queue()
                self.inputQueues.append(iq)
                self.outputQueues.append(oq)

                # spawn a daemon process for a new object tracker
                p = multiprocessing.Process(
                    target=self.start_tracker,
                    args=(bb, label, rgb, iq, oq))
                p.daemon = True
                p.start()

                # grab the corresponding class label for the detection
                # and draw the bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        return frame

    def detecting(self, frame):
        (self.h, self.w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.w, self.h), 127.5)

        # pass the blob through the network and obtain the detections
        # and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.75:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])
                label = self.classes[idx]

                # if the class label is not a person, ignore it
                if self.classes[idx] != "person":
                    continue
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                (startX, startY, endX, endY) = box.astype("int")
                bb = (startX, startY, endX, endY)

                # create two brand new input and output queues,
                # respectively
                iq = multiprocessing.Queue()
                oq = multiprocessing.Queue()
                self.inputQueues.append(iq)
                self.outputQueues.append(oq)

        return detections



    def start_tracker(self, box, label, rgb, inputQueue, outputQueue):
        # construct a dlib rectangle object from the bounding box
        # coordinates and then start the correlation tracker
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        t.start_track(rgb, rect)

        # add the tracker to our list of trackers so we can
        # utilize it during skip frames
        self.trackers.append(t)

        # loop indefinitely -- this function will be called as a daemon
        # process so we don't need to worry about joining it
        while True:
            # attempt to grab the next frame from the input queue
            rgb = inputQueue.get()

            # if there was an entry in our queue, process it
            if rgb is not None:
                # update the tracker and grab the position of the tracked
                # object
                t.update(rgb)
                pos = t.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the label + bounding box coordinates to the output
                # queue
                outputQueue.put((label, (startX, startY, endX, endY)))


if __name__ == "__main__":
    url = "rtsp://user:Hneu74k092@10.101.106.104:554/live/main"  #"/Users/andrey/PycharmProjects/multiobject-tracking-dlib/race.mp4" #0
    url = "/Users/andrey/Downloads/Telegram Desktop/vlc_record_2019_05_24_15h29m07s.mp4"
    cam = VideoStream(url)

    while True:
        frame = cam.process_next_frame()
        cv2.imshow('Frame', frame)
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()