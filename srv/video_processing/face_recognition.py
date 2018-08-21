import argparse
import json
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join

import cv2

import srv.common as face_recognition
from srv.video_processing.functions.face_feature_detector import load_network, detect_faces

SOURCE = 'source'

# construct the arguments and parse them
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False, default='video_processing/tmp/photo',
                help="path to output directory")
ap.add_argument("-o", "--output", required=False, default='video_processing/tmp/description',
                help="path to output directory")
args = vars(ap.parse_args())

sess, age, gender, train_mode, images_pl = load_network('models')

andrey_image = face_recognition.load_image_file("photo/andrey.jpg")
andrey_face_encoding = face_recognition.face_encodings(andrey_image)[0]

simon_image = face_recognition.load_image_file("photo/simon.jpg")
simon_face_encoding = face_recognition.face_encodings(simon_image)[0]

misha_image = face_recognition.load_image_file("photo/misha.jpg")
misha_face_encoding = face_recognition.face_encodings(misha_image)[0]

known_face_encodings = [andrey_face_encoding, simon_face_encoding, misha_face_encoding]
known_face_names = ['Andrey', 'Simon', 'Misha']

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

log_time = 0


def measure_performance(t, pattern):
    elapsed_time = (time.process_time() - t) * 24
    print(pattern.format(elapsed_time))
    return time.process_time()


while True:
    img_sources = [f for f in listdir(args[SOURCE]) if isfile(join(args[SOURCE], f))]

    for img_source in img_sources:

        t = time.process_time()
        frame = cv2.imread('{0}/{1}'.format(args[SOURCE], img_source), cv2.IMREAD_UNCHANGED)
        t = measure_performance(t, '1. Read image: {}')

        detected, faces = detect_faces(frame)
        t = measure_performance(t, '2. Face detect: {}')

        if len(detected) > 0:
            ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
        t = measure_performance(t, '3. Estimate age and gender: {}')

        for i, d in enumerate(detected):
            _gender = "Female" if genders[i] == 0 else "Male"
            _age = int(ages[i])
        t = measure_performance(t, '4. Draw age and gender: {}')

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = 'Unknown'

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)
        t = measure_performance(t, '5. Name detected: {}')

        if len(detected) > 0:
            _id = img_source.replace('.png', '').replace('id_', '')
            description = {
                'id': _id, 'name': name, 'age': _age, 'gender': _gender,
                'time': int(datetime.timestamp(datetime.now()))
            }
            with open('video_processing/tmp/description/id_{}.json'.format(_id), 'w+') as f:
                json.dump(description, f)

    time.sleep(10)
