#!/usr/bin/env python
import numpy as np
import imutils
import dlib
import cv2
import os
import time
import json
import datetime

from main.common import config_parser
from main.common.object_tracker import TrackableObject, CentroidTracker
import model.person_processing.nets.resnet_v1_50 as model
import model.person_processing.heads.fc1024 as head
import tensorflow as tf

from face_processing.best_face_selector import select_best_face
from face_processing.face_recognition import recognize_face, load_known_face_encodings

from keras.applications.resnet50 import preprocess_input

from rest_api.db.event_db_logger import EventDBLogger


CONFIG = config_parser.parse()

# path to PycharmProject
print('root_path: ', CONFIG["root_path"])

# person detection model
person_models = os.path.join(CONFIG["root_path"], CONFIG["person_models"])
PROTOTXT = os.path.join(person_models, CONFIG["person_deploy"])
MODEL = os.path.join(person_models, CONFIG["person_detector"])

# face detection model
face_models = os.path.join(CONFIG["root_path"], CONFIG["face_models"])
PROTOTXT_FACE = os.path.join(face_models, CONFIG["face_deploy"])
MODEL_FACE = os.path.join(face_models, CONFIG["face_detector"])

# faces base
DB_PATH = os.path.join(CONFIG["root_path"], CONFIG["known_faces_db"])

# dir for saving testing images [optional]
save_img = True

for d in ['face_processing/tmp_faces', 'data/db']:
    os.makedirs(os.path.join(CONFIG["root_path"], d), exist_ok=True)


class VideoStream():
    def __init__(self, camera_url=0):
        self.camera_url = 0 if camera_url == '0' else camera_url
        self.vs = cv2.VideoCapture(self.camera_url)
        self.W, self.H = int(self.vs.get(3)), int(self.vs.get(4))
        self.fps = self.vs.get(5)

        t_nets_initialization = time.monotonic()
        # person detection model
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

        # face models
        if CONFIG['face_detection'] == 'True':
            self.net_face_detector = cv2.dnn.readNetFromCaffe(PROTOTXT_FACE, MODEL_FACE)
            print('[TIME LOG] t_nets_initialization_elapsed:', time.monotonic() - t_nets_initialization)

            t_loading_embs = time.monotonic()
            # dlib embeddings
            self.known_face_encodings, self.known_face_names = load_known_face_encodings(DB_PATH)
            print('[TIME LOG] t_loading_embs_elapsed:', time.monotonic() - t_loading_embs)
        else:
            print('[TIME LOG] t_nets_initialization_elapsed:', time.monotonic() - t_nets_initialization)

        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        self.ct = CentroidTracker()
        self.trackableObjects = {}
        self.trackers = []
        self.embeding_list = []

        t_vectorizer_initialization = time.monotonic()
        self.imgVectorizer, self.endpoints, self.images = self.start_img_vectorizer()
        t_vectorizer_initialization_elapsed = time.monotonic() - t_vectorizer_initialization
        print('[TIME LOG] t_vectorizer_initialization_elapsed:', t_vectorizer_initialization_elapsed)

        self.info = {
            'status': None,
            'FPS': 0,
            'Enter': 0,
            'Exit': 0,
            'TotalFrames': 0,
            'Status': 'start',
            'Count People': 0
        }

        # extracting camera name from json
        with open('../rest_api/cam_info.json') as json_file:
            data = json.load(json_file)
        for elem in data:
            if elem["camera_url"] == self.camera_url:
                self.db_name = elem["name"].replace(' ', '_')
                break

        # creating db and connection
        self.connection = EventDBLogger(db_name=self.db_name)
        # creating table in db
        self.table_event_log = self.connection.create_table_event_logger(cam_name = self.db_name)

    def process_next_frame(self):
        ret, frame = self.vs.read()

        if not ret:
            return None, [None], self.trackableObjects

        orig_frame = frame.copy()
        frame = imutils.resize(frame, width=600)
        self.H, self.W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = []
        objects = ()
        if self.info['TotalFrames'] % 30 == 0:
            t_detecting = time.monotonic()
            frame = self.detecting(frame, rgb)
            t_detecting_elapsed = time.monotonic() - t_detecting
            t_tracking_elapsed = None

            if len(self.trackableObjects.items()) > 0:
                for tr_indx, tr_obj in self.trackableObjects.items():
                    event = {
                        'object_id': int(tr_indx),
                        'event_time': datetime.datetime.now(),
                        'centroid_x': int(int(tr_obj.centroids[-1][0])),
                        'centroid_y': int(tr_obj.centroids[-1][1])
                    }
                    self.connection.insert(self.table_event_log, event)

                    person_save_path = os.path.join(CONFIG["root_path"],
                                                    'data/photo/{}/id_{}/person/'.format(self.db_name, tr_indx))
                    os.makedirs(person_save_path, exist_ok=True)

                    if save_img:
                        if tr_obj.img is not None and all(tr_obj.img.shape) > 0:
                            cv2.imwrite(person_save_path + '{}_{}.jpeg'.format(tr_indx, datetime.datetime.now()), tr_obj.img)

        else:
            t_tracking = time.monotonic()
            rects = self.tracking(rgb, rects)
            t_tracking_elapsed = time.monotonic() - t_tracking
            t_detecting_elapsed = None

        if len(rects) > 0:
            t_emb_matrix = time.monotonic()
            objects, embeding_matrix = self.ct.update(rects, orig_frame, frame, self.trackableObjects, self.embeding_list)
            t_emb_matrix_elapsed = time.monotonic() - t_emb_matrix

            frame = self.draw_labels(frame, orig_frame, objects)

            # Update Trackable objects
            t_updating_trObj = time.monotonic()
            objects = self.ct.check_embeding(embeding_matrix, self.trackableObjects)
            self.update_trackable_objects(objects)
            t_updating_trObj_elapsed = time.monotonic() - t_updating_trObj

            # Face recognition
            if CONFIG['face_detection'] == 'True':
                t_face_recognition = time.monotonic()
                frame = self.face_recognition(frame, orig_frame)
                t_face_recognition_elapsed = time.monotonic() - t_face_recognition
            else:
                t_face_recognition_elapsed = None

        else:
            t_emb_matrix_elapsed = None
            t_face_recognition_elapsed = None
            t_updating_trObj_elapsed = None

        self.info['TotalFrames'] += 1

        time_log = [len(self.trackableObjects), t_detecting_elapsed, t_tracking_elapsed,
                    t_emb_matrix_elapsed, t_updating_trObj_elapsed,
                    t_face_recognition_elapsed]

        return frame, time_log, self.trackableObjects

    def draw_labels(self, frame, orig_frame, objects):

        for (objectID, info) in objects.items():

            text = "ID {}".format(objectID)
            sX, sY, eX, eY = info['rect']
            output = frame.copy()

            # # --- visualization - centroid point and label "ID <objectID>"
            # cv2.putText(frame, text, (info['centroid'][0] - 10-5, info['centroid'][1] - 10-5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.circle(frame, (info['centroid'][0], info['centroid'][1]), 4, (0, 255, 0), -1)
            # ---

            # # --- visualization - text in the rectangle above the head
            # line_s, line_e = (info['centroid'][0]-20, info['centroid'][1]-80-10), (info['centroid'][0]-20, info['centroid'][1]-80)
            # rect_s, rect_e = (line_s[0]-30, line_s[1]-30), (line_s[0]+30, line_s[1])
            # text_s = (rect_s[0]+10, int(0.5*(rect_e[1]+rect_s[1])))
            # overlay = frame.copy()
            # cv2.line(overlay, line_s, line_e, (255,255,255), 2)
            # cv2.rectangle(overlay, rect_s, rect_e, (255, 255, 255), -1)
            # cv2.putText(overlay, text, text_s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # cv2.addWeighted(overlay, 0.8, output, 1.0-0.8, 0, output)
            # # ---

            # --- visualization - blurred ellipse
            mask = np.zeros(frame.shape, dtype=np.uint8)
            color_ellipse = (0, 255, 0)
            mask = cv2.ellipse(mask, (info['centroid'][0], info['centroid'][1]),
                                     (int(0.5 * (eX - sX)), int(0.5 * (eY - sY))), 5, 0, 360, color_ellipse, -1)
            mask = cv2.ellipse(mask, (info['centroid'][0], info['centroid'][1]),
                                     (int(0.5 * (eX - sX)), int(0.5 * (eY - sY))), 5, 0, 360, (255, 255, 255), 5)
            mask = cv2.GaussianBlur(mask, (11, 11), 0)
            output = cv2.addWeighted(mask, 0.22, output, 1.0, 0, output)
            cv2.putText(output, text, (info['centroid'][0] - 10 - 5, info['centroid'][1] - 10 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # ---

            frame = output.copy()

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
            confidence = detections[0, 0, i, 2]

            if idx < 0:
                idx = 1

            if self.classes[idx] != "person":
                continue
            else:
                if confidence > 0.75:
                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype('int')

                    # person box visualization
                    # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    if startY < 0:
                        startY = 0

                    imgCrop = frame[startY:endY, startX:endX]

                    resize_img = cv2.resize(imgCrop, (128, 256))
                    resize_img = np.expand_dims(resize_img, axis=0)
                    # resize_img = preprocess_input(resize_img.reshape(1, 256,128,3))
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
        tf.train.Saver().restore(sess, os.path.join(person_models, CONFIG["person_vectorizer_weights"]))
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
            for tr_indx, tr_obj in self.trackableObjects.items():

                if tr_obj.names[0] is None:
                    # detect face from orig size person
                    frame, detected_face = self.face_detection(frame, orig_frame, tr_obj.rect, tr_obj.img, tr_indx)

                    face_save_path = os.path.join(CONFIG["root_path"], 'data/photo/{}/id_{}/face/'.format(self.db_name,
                                                                                                          tr_indx))
                    os.makedirs(face_save_path, exist_ok=True)

                    # add cropped_face to face_sequence
                    if detected_face is not None and all(detected_face.shape) > 0:
                        # add detected_face to faces_sequence_for_person
                        tr_obj.face_seq.append(detected_face)
                        # save face to folder
                        if save_img:

                            cv2.imwrite(face_save_path + '{}_detected_{}.jpg'.format(tr_indx,
                                                                                     datetime.datetime.now()),
                                                                                     detected_face)

                    if len(tr_obj.face_seq) > 0:
                        # select the best face from face_sequence
                        best_detected_face = select_best_face(tr_obj.face_seq)
                        if save_img:
                            cv2.imwrite(face_save_path + '{}_best_detected_face_{}.jpg'.format(tr_indx,
                                                                                               datetime.datetime.now()),
                                                                                               best_detected_face)

                        # recognize best_face
                        self.info['status'] = 'Recognizing face'
                        names, best_face_emb = recognize_face(best_detected_face, self.known_face_encodings, self.known_face_names)
                        print('This person looks like:', names)

                        if len(names) > 0:
                            tr_obj.names = names
                        if len(best_face_emb) > 0:
                            tr_obj.face_emb = best_face_emb

        return frame

    def face_detection(self, frame, orig_frame, person_box, person_im, tr_indx):

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

                    pad_y, pad_x = 30, 30
                    if startY - pad_y < 0:
                        pad_y = 0
                    if startX - pad_x < 0:
                        pad_y = 0

                    face_im = person_im[startY - pad_y: endY + pad_y, startX - pad_x: endX + pad_x]

                    # if save_img:
                    #     cv2.imwrite(os.path.join(CONFIG["root_path"],
                    #                 os.path.join(CONFIG["tmp_face_tests"], 'tmp_cropped_face_{}.jpg'.format(tr_indx))),
                    #                 face_im)

                    # reconstruction face box coordinates for visualization on frame
                    # person box coordinates
                    sX, sY, eX, eY = person_box
                    rel_sX = int( sX / frame.shape[1] * orig_frame.shape[1] )
                    rel_sY = int( sY / frame.shape[0] * orig_frame.shape[0] )

                    # face box visualization in resized frame coords
                    x_ = int((startX + rel_sX) / orig_frame.shape[1] * frame.shape[1])
                    y_ = int((startY + rel_sY) / orig_frame.shape[0] * frame.shape[0])
                    w_ = int((endX + rel_sX) / orig_frame.shape[1] * frame.shape[1])
                    h_ = int((endY + rel_sY) / orig_frame.shape[0] * frame.shape[0])
                    cv2.rectangle(frame, (x_, y_), (w_, h_), (255, 0, 0), 2)

        else:
            face_im = None

        return frame, face_im