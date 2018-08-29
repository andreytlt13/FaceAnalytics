import datetime
import json
import os

import cv2
import imutils
import numpy as np
from imutils.video import FPS

from srv.common.config import config_parser
from srv.video_processing.common.draw_label import draw_label
from srv.video_processing.common.tracker import CentroidTracker
from video_processing.common import enhasher

FRAME_WIDTH = 400
SCALE_FACTOR = 1.0

CONFIG = config_parser.parse_default()


class FrameProcessor:

    def __init__(self, camera_url=0, confidence=CONFIG['confidence'], detected_faces_dir=CONFIG['detected_faces_dir'],
                 model=CONFIG['model'], prototxt=CONFIG['prototxt']) -> None:
        self.camera_url = camera_url  # 'rtsp://admin:admin@10.101.106.12:554/ch01/0' tNgB4SZD
        self.confidence = float(confidence)
        self.ct = CentroidTracker()

        camera_url_hash = enhasher.hash_string(camera_url)
        self.description_pattern = CONFIG['descriptions_dir'] + '/' + camera_url_hash + '/id_{}.json'
        self.detected_face_img_pattern = detected_faces_dir + '/' + camera_url_hash + '/id_{}.png'
        (self.H, self.W) = (None, None)

        # load our serialized model from disk
        print('[INFO] loading model...')
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        detected_faces_dir_path = os.path.dirname(self.detected_face_img_pattern)
        if not os.path.exists(detected_faces_dir_path):
            os.makedirs(detected_faces_dir_path)

    def process_next_frame(self, vs):

        frame = vs.read()
        frame = imutils.resize(frame, width=FRAME_WIDTH)

        fps = FPS().start()

        # if the frame dimensions are None, grab them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, SCALE_FACTOR, (self.W, self.H),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        rects = []

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

            if datetime.datetime.now().second % 10 == 0:
                imgCrop = frame[y:h, x:w]
                cv2.imwrite(self.detected_face_img_pattern.format(str(objectID)), imgCrop)

            description_path = self.description_pattern.format(objectID)
            if os.path.exists(description_path):
                description = json.loads(open(description_path).read())
                draw_label(frame, description['gender'], (centroid[0] - 100, centroid[1] + 20))
                draw_label(frame, str(description['age']), (centroid[0] - 100, centroid[1]))
                draw_label(frame, description['person_name'], (centroid[0] - 100, centroid[1] - 20))

            draw_label(frame, text, (centroid[0], centroid[1] + 75))

        fps.update()
        fps.stop()

        return frame, fps, self.H
