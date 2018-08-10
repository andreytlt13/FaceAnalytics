import json
import argparse
import time
import os
import dlib
import cv2
import srv.common as face_recognition
import numpy as np

from os import listdir
from os.path import isfile, join
from imutils.face_utils import FaceAligner
from srv.video_processing.functions.face_feature_detector import load_network
#from video_processing.functions.detect_age import  detect_age
#from video_processing.functions.detect_gender import detect_gender
#from video_processing.functions.face_description import recognize_faces
#from video_processing.functions.face_feature_detector import detect_faces, load_network
#from models import inception_resnet_v1
# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False, default='video_processing/tmp/photo',
                help="path to output directory")
ap.add_argument("-o", "--output", required=False, default='video_processing/tmp/description',
                help="path to output directory")
args = vars(ap.parse_args())


sess, age, gender, train_mode, images_pl = load_network(
    'models'
)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=160)
img_size = 160


#1
andrey_image = face_recognition.load_image_file("photo/andrey.jpg")
andrey_face_encoding = face_recognition.face_encodings(andrey_image)[0]

simon_image = face_recognition.load_image_file("photo/simon.jpg")
simon_face_encoding = face_recognition.face_encodings(simon_image)[0]

misha_image = face_recognition.load_image_file("photo/misha.jpg")
misha_face_encoding = face_recognition.face_encodings(misha_image)[0]

known_face_encodings = [
    andrey_face_encoding,
    simon_face_encoding,
    misha_face_encoding
]

known_face_names = [
    'Andrey',
    'Simon',
    'Misha'
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

log_time = 0


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


while True:
    onlyfiles = [f for f in listdir(args['source']) if isfile(join(args['source'], f))]

    for file in onlyfiles:

        t = time.process_time()

        frame = cv2.imread('{0}/{1}'.format(args['source'], file), cv2.IMREAD_UNCHANGED)

        elapsed_time = (time.process_time() - t)*24
        print('Get Frame: {}'.format(elapsed_time))
        t = time.process_time()

        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)

        elapsed_time = (time.process_time() - t)*24
        print('1. Read image: {}'.format(elapsed_time))
        t = time.process_time()

        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        elapsed_time = (time.process_time() - t)*24
        print('2. Face detect: {}'.format(elapsed_time))
        t = time.process_time()

        for i, d in enumerate(detected):
            faces[i, :, :, :] = fa.align(input_img, gray, detected[i])

        elapsed_time = (time.process_time() - t) * 24
        print('3. Face numerate: {}'.format(elapsed_time))
        t = time.process_time()

        if len(detected) > 0:
            ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

        elapsed_time = (time.process_time() - t) * 24
        print('4. Estimate age and gender: {}'.format(elapsed_time))
        t = time.process_time()

        for i, d in enumerate(detected):
            _sex = "Female" if genders[i] == 0 else "Male"
            _age = int(ages[i])

        elapsed_time = (time.process_time() - t) * 24
        print('5. Draw age and gender: {}'.format(elapsed_time))
        t = time.process_time()

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

        elapsed_time = (time.process_time() - t) * 24
        print('6. Name detected: {}'.format(elapsed_time))
        t = time.process_time()

        if len(detected) > 0:
            description = {'sex': _sex, 'age': _age, 'name': name}
            id = file.replace('.png', '').replace('id_', '')
            with open('video_processing/tmp/description/id_{}.json'.format(id), 'w+') as f:
                json.dump(description, f)

    time.sleep(10)





