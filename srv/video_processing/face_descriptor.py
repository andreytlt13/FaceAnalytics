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

CONFIG = config_parser.parse_default()

sess, age, gender, train_mode, images_pl = load_network(CONFIG['models_dir'])
known_face_encodings, known_face_names = known_face_encoding(CONFIG['known_people_dir'])

db_logger = event_db_logger.EventDBLogger()


def measure_performance(t, pattern):
    elapsed_time = (time.process_time() - t) * 24
    print(pattern.format(elapsed_time))
    return time.process_time()


def start(source=CONFIG['detected_faces_dir'], descriptions_dir=CONFIG['descriptions_dir']):
    description_file_pattern = descriptions_dir + '/id_{}.json'
    descriptions_dir_path = os.path.dirname(description_file_pattern)
    if not os.path.exists(descriptions_dir_path):
        os.makedirs(descriptions_dir_path)

    while True:
        if os.path.exists(source):
            img_sources = [f for f in listdir(source) if isfile(join(source, f))]

            for img_source in img_sources:

                t = time.process_time()
                frame = cv2.imread('{0}/{1}'.format(source, img_source), cv2.IMREAD_UNCHANGED)
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

                    person_id = img_source.replace('.png', '').replace('id_', '')
                    description = {
                        'person_id': person_id, 'person_name': name, 'age': _age, 'gender': _gender,
                        'log_time': int(datetime.timestamp(datetime.now()))
                    }
                    db_logger.log(description)
                    with open(description_file_pattern.format(person_id), 'w+') as f:
                        json.dump(description, f)

            time.sleep(10)
