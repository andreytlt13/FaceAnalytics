import json
import os
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join

import cv2

import srv.common as face_recognition
from db import event_db_logger
from srv.common.config import config_parser
from srv.video_processing.common.known_face_encodings import known_face_encoding
from srv.video_processing.face_feature_detector import load_network, detect_faces
from video_processing.common import enhasher

CONFIG = config_parser.parse_default()

sess, age, gender, train_mode, images_pl = load_network(CONFIG['models_dir'])
known_face_encodings, known_face_names = known_face_encoding(CONFIG['known_people_dir'])


def measure_performance(t, pattern):
    elapsed_time = (time.process_time() - t) * 24
    print(pattern.format(elapsed_time))
    return time.process_time()


def describe(camera_url, source=CONFIG['detected_faces_dir']):
    description_file_pattern = CONFIG['descriptions_dir'] + '/' + enhasher.hash_string(camera_url) + '/id_{}.json'
    descriptions_dir = os.path.dirname(description_file_pattern)
    if not os.path.exists(descriptions_dir):
        os.makedirs(descriptions_dir)
    db_logger = event_db_logger.EventDBLogger()
    camera_source = source + '/' + enhasher.hash_string(camera_url)
    while True:
        img_sources = [f for f in listdir(camera_source) if isfile(join(camera_source, f))]

        for img_source in img_sources:

            t = time.process_time()
            frame = cv2.imread('{0}/{1}'.format(camera_source, img_source), cv2.IMREAD_UNCHANGED)
            t = measure_performance(t, '1. Read image: {}')

            detected, faces = detect_faces(frame)
            t = measure_performance(t, '2. Face detect: {}')

            if len(faces) > 0:
                ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
                t = measure_performance(t, '3. Estimate age and gender: {}')

                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                t = measure_performance(t, '4. Recognize face: {}')

                if len(detected) > 1 or len(ages) > 1 or len(genders) > 1 or len(face_encodings) > 1:
                    raise RuntimeError('There should be exactly one face per img_source')

                _gender = 'Female' if genders[0] == 0 else 'Male'
                _age = int(ages[0])

                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                name = 'Unknown'

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                t = measure_performance(t, '5. Name detected: {}')

                _id = img_source.replace('.png', '').replace('id_', '')
                description = {
                    'id': _id, 'person_name': name, 'age': _age, 'gender': _gender,
                    'log_time': int(datetime.timestamp(datetime.now())),
                    'camera_url': camera_url
                }
                db_logger.log(description)
                with open(description_file_pattern.format(_id), 'w+') as f:
                    json.dump(description, f)

        time.sleep(10)
