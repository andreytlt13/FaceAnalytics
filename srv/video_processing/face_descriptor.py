import json
import os
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join

import cv2

import srv.common as face_recognition
from srv.common.config import config_parser
from srv.video_processing.face_feature_detector import load_network, detect_faces

CONFIG = config_parser.parse_default()

sess, age, gender, train_mode, images_pl = load_network(CONFIG['models_dir'])

andrey_image = face_recognition.load_image_file(CONFIG['known_people_dir'] + "/andrey.jpg")
andrey_face_encoding = face_recognition.face_encodings(andrey_image)[0]

simon_image = face_recognition.load_image_file(CONFIG['known_people_dir'] + "/simon.jpg")
simon_face_encoding = face_recognition.face_encodings(simon_image)[0]

misha_image = face_recognition.load_image_file(CONFIG['known_people_dir'] + "/misha.jpg")
misha_face_encoding = face_recognition.face_encodings(misha_image)[0]

known_face_encodings = [andrey_face_encoding, simon_face_encoding, misha_face_encoding]
known_face_names = ['Andrey', 'Simon', 'Misha']


def measure_performance(t, pattern):
    elapsed_time = (time.process_time() - t) * 24
    print(pattern.format(elapsed_time))
    return time.process_time()


def describe(source=CONFIG['detected_faces_dir']):
    description_file_pattern = CONFIG['descriptions_dir'] + '/id_{}.json'
    descriptions_dir = os.path.dirname(description_file_pattern)
    if not os.path.exists(descriptions_dir):
        os.makedirs(descriptions_dir)
    while True:
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

                _gender = "Female" if genders[0] == 0 else "Male"
                _age = int(ages[0])

                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                name = 'Unknown'

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                t = measure_performance(t, '5. Name detected: {}')

                _id = img_source.replace('.png', '').replace('id_', '')
                description = {
                    'id': _id, 'name': name, 'age': _age, 'gender': _gender,
                    'time': int(datetime.timestamp(datetime.now()))
                }
                with open(description_file_pattern.format(_id), 'w+') as f:
                    json.dump(description, f)

        time.sleep(10)
