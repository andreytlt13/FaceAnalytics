import json
import os
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join

import cv2
import dlib
import numpy as np
from imutils.face_utils import FaceAligner

from common import config_parser
from db import event_db_logger
from face_description import dlib_api
from face_description.network_loader import load_network, load_known_face_encodings

detected_faces_dir = "../face_description/known_faces"
descriptions_dir = "../face_description/tmp/faces"

class Descriptor:

    def __init__(self, source="face_description/known_faces",
                 descriptions_dir="face_description/tmp/faces",
                 model_dir="models/"):
        self.source = source
        self.descriptions_dir = descriptions_dir
        self.model_dir = model_dir



    def load_network(self):
        sess, age, gender, train_mode, images_pl = load_network(self.model_dir)


        return sess, age, gender, train_mode, images_pl




    def load_faces(self):

        return "kokokok"



