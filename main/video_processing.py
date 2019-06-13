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

from face_processing.best_face_selector import select_best_face

# dlib recognizer
from face_processing.face_recognition import recognize_face, load_known_face_encodings, get_cropped_person

# VGG regonizer
# from keras_vggface.vggface import VGGFace
# from face_processing.face_recognition import load_face_models, load_known_face_encodings_vgg, recognize_face_vgg


# path to PycharmProjects
root_path = '/Users/andrey/PycharmProjects/'

# person detection model
PROTOTXT = root_path + "FaceAnalytics/main/model/MobileNetSSD_deploy.prototxt"
MODEL = root_path + "FaceAnalytics/main/model/MobileNetSSD_deploy.caffemodel"

ENCODER_PATH = root_path + "FaceAnalytics/main/model/mars-small128.pb"

# face detection model
PROTOTXT_FACE = root_path + "FaceAnalytics/main/face_processing/models/deploy.prototxt"
MODEL_FACE = root_path + "FaceAnalytics/main/face_processing/models/res10_300x300_ssd_iter_140000.caffemodel"

# faces base
DB_PATH = root_path + "FaceAnalytics/main/face_processing/known_faces/"

class VideoStream():
    def __init__(self, camera_url=0):
        self.camera_url = 0 if camera_url == '0' else camera_url
        self.vs = cv2.VideoCapture(self.camera_url)
        self.W, self.H = int(self.vs.get(3)), int(self.vs.get(4))
        self.fps = self.vs.get(5)
        # person detection model
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        # face models
        self.net_face_detector = cv2.dnn.readNetFromCaffe(PROTOTXT_FACE, MODEL_FACE)

        # dlib approach
        self.known_face_encodings, self.known_face_names = load_known_face_encodings(DB_PATH)

        # VGG approach
        # self.net_face_emb = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        # self.known_face_encodings, self.known_face_names = load_known_face_encodings_vgg(self.net_face_emb,
        #                                                                                  self.net_face_detector,
        #                                                                                  DB_PATH)

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
            objects, embeding_matrix = self.ct.update(rects, orig_frame, frame, self.trackableObjects, self.embeding_list)
            frame = self.draw_labels(frame, objects)

            #Upddate Trackable objects
            objects = self.ct.check_embeding(embeding_matrix, self.trackableObjects)
            self.update_trackable_objects(objects)

            #Face recognition
            frame = self.face_recognition(frame, orig_frame)


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
            self.ct.nextObjectID = self.trackableObjects.__len__()

    def face_recognition(self, frame, orig_frame):

        if len(self.trackableObjects.items()) > 0:
            for tr_indx, tr_obj in enumerate(self.trackableObjects.items()):

                if tr_obj[1].names[0] is None:
                    # detect face from orig size person
                    frame, detected_face = self.face_detection(frame, orig_frame, tr_obj[1].rect, tr_obj[1].img)

                    # print('---'*5, tr_indx, tr_obj[1].rect)

                    # add cropped_face to face_sequence
                    if detected_face is not None and all(detected_face.shape) > 0:
                        # add detected_face to faces_sequence_for_person
                        tr_obj[1].face_seq.append(detected_face)
                        print('added detected_face to faces_sequence_for_trackableObject')

                    if len(tr_obj[1].face_seq) > 0:
                        # select the best face from face_sequence
                        best_detected_face = select_best_face(tr_obj[1].face_seq)
                        # cv2.imwrite('best_detected_face.jpg', best_detected_face)

                        # recognize best_face
                        self.info['status'] = 'Recognizing face'
                        names, best_face_emb = recognize_face(best_detected_face, self.known_face_encodings, self.known_face_names)
                        print('names', names)

                        if len(names) > 0:
                            tr_obj[1].names = names
                        if len(best_face_emb) > 0:
                            tr_obj[1].face_emb = best_face_emb

        return frame

    def face_detection(self, frame, orig_frame, person_box, person_im):

        # crop person by person_box coords
        # person_im = get_cropped_person(orig_frame=orig_frame, resized_frame=frame, resized_box=person_box)
        # cv2.imwrite('cropped_person.jpg', person_im)

        # detect face from cropped person
        self.info['status'] = 'Detecting face'

        if person_im is not None and all(person_im.shape) > 0:
            H, W = person_im.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(person_im, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net_face_detector.setInput(blob)
            detections = self.net_face_detector.forward()

            face_im = None
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.75:
                    # face box in person_im coordinates
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype('int')
                    pad_y, pad_x = 0, 0
                    face_im = person_im[startY - pad_y: endY + pad_y, startX - pad_x: endX + pad_x]
                    # cv2.imwrite('cropped_face.jpg', face_im)

                    # reconstruction face box coordinates for visualization on frame
                    # person box coordinates
                    sX, sY, eX, eY = person_box
                    rel_sX = int( sX / frame.shape[1] * orig_frame.shape[1] )
                    rel_sY = int( sY / frame.shape[0] * orig_frame.shape[0] )
                    # rel_eX = int( eX / frame.shape[1] * orig_frame.shape[1] )
                    # rel_eY = int( eY / frame.shape[0] * orig_frame.shape[0] )

                    # face box visualization in resized frame coords
                    x_ = int((startX + rel_sX) / orig_frame.shape[1] * frame.shape[1])
                    y_ = int((startY + rel_sY) / orig_frame.shape[0] * frame.shape[0])
                    w_ = int((endX + rel_sX) / orig_frame.shape[1] * frame.shape[1])
                    h_ = int((endY + rel_sY) / orig_frame.shape[0] * frame.shape[0])
                    cv2.rectangle(frame, (x_, y_), (w_, h_), (255, 0, 0), 2)
                    # cv2.imwrite('frame_vis.jpg', frame)
        else:
            face_im = None

        return frame, face_im
