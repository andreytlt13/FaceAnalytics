#!/usr/bin/env python
from datetime import datetime

import cv2
import flask
import imutils
import numpy as np

from srv.video_processing.common.draw_label import draw_label
from srv.video_processing.common.log_faces import log
from srv.video_processing.functions.detect_age import detect_age
from srv.video_processing.functions.detect_gender import detect_gender
from srv.video_processing.functions.detect_people import detect_people
from srv.video_processing.functions.face_feature_detector import load_network, detect_faces
from srv.video_processing.functions.recognize_face import recognize_faces, UNKNOWN
from srv.video_processing.object_tracker import CentroidTracker
from srv.webapp.video_streaming.utils.normalize_image import fisheye_to_flat

LOG_PATH = '/tmp/faces_log.txt'
last_log_message = ''
detected_regions_count = 1

sess, age, gender, train_mode, images_pl = load_network(
    'srv/models'
)
ct = CentroidTracker()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template(
        'index.html',
        img_path='/static/images/noise.jpg'
    )


@app.route('/analyse', methods=['POST'])
def analyse():
    return flask.render_template(
        'index.html',
        img_path='video_stream?camera_url=' + flask.request.form.get('camera_url')
    )


@app.route('/video_stream', methods=['GET'])
def video_stream():
    try:
        camera_url = int(flask.request.args.get('camera_url'))
    except:
        camera_url = flask.request.args.get('camera_url')
    return flask.Response(
        generate_stream(camera_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_stream(camera_url):
    while True:
        _, frame = cv2.VideoCapture(camera_url).read()
        _, img_w, _ = np.shape(frame)

        frame = fisheye_to_flat(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        people = detect_people(gray, img_w)
        if len(people) > 0:
            for i, (x, y, w, h) in enumerate(people):
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

                cropped = frame[y:h, x:w]
                crop_h, crop_w = np.shape(frame)
                print('Detected region: ' + str(crop_w) + ', ' + str(crop_h))

                global detected_regions_count
                cv2.imwrite('/tmp/images/frame' + str(detected_regions_count) + '.jpg', cropped)
                detected_regions_count += 1

                process_frame(cropped, frame, 1, x, y)
        else:
            scale_factor = 0.25
            resized = imutils.resize(frame, width=int(img_w * scale_factor))
            process_frame(resized, frame, scale_factor, 0, 0)

        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def process_frame(cropped, frame, scale_factor, x, y):
    result = {}
    detected, faces = detect_faces(cropped)
    detected_faces_count = len(detected)
    if detected_faces_count > 0:
        print('we have detected somebody!')
        objects = ct.update(detected)
        ages = detect_age(faces, sess, age, train_mode, images_pl)
        genders = detect_gender(faces, sess, gender, train_mode, images_pl)
        face_locations, face_names = recognize_faces(cropped)

        min_horizontal_center = int(min([d.right() - d.left() for d in detected]) / 2)
        min_vertical_center = int(min([d.top() - d.bottom() for d in detected]) / 2)
        scale_back_factor = int(1 / scale_factor)
        for (top, right, bottom, left), name, (i, (obj_id, centroid)), age_i, gender_i in zip(face_locations,
                                                                                              face_names,
                                                                                              enumerate(
                                                                                                  objects.items()),
                                                                                              ages, genders):
            # Scale back up face locations since the frame we detected in was scaled
            top = y + top * scale_back_factor
            right = x + right * scale_back_factor

            font = cv2.FONT_HERSHEY_DUPLEX
            if face_names[0] == UNKNOWN:
                cv2.putText(frame, name, (right + 6, top - 6), font, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, name, (right + 6, top - 6), font, 1.0, (0, 255, 0), 2)

            result.setdefault('name', []).append(name)

            if i >= detected_faces_count:
                break

            age_i = int(age_i)
            result.setdefault('id', []).append(obj_id)
            result.setdefault('time', []).append(datetime.now().strftime('%H:%M:%S')[0:8])
            result.setdefault('age', []).append(age_i)

            gender_i = "Female" if genders[i] == 0 else "Male"
            result.setdefault('time', []).append(datetime.now().strftime('%H:%M:%S')[0:8])
            result.setdefault('gender', []).append(gender_i)

            label = "ID={}, {}, age={}".format(obj_id, gender_i, age_i)
            draw_label(
                frame,
                (scale_back_factor * (centroid[0] - min_horizontal_center),
                 scale_back_factor * (centroid[1] - min_vertical_center)),
                label
            )

        if len(result) > 0:
            log(result)


@app.route('/text_stream', methods=['GET'])
def text_stream():
    try:
        faces = open(LOG_PATH, 'r')
    except IOError:
        faces = open(LOG_PATH, 'w+')

    objects_info = faces.readlines()
    faces.close()

    if not objects_info:
        msg = 'Logging started...'
    else:
        msg = objects_info[-1]

    global last_log_message
    if last_log_message != msg:
        last_log_message = msg
        return flask.Response(
            msg,
            mimetype='text/xml'
        )
    else:
        return flask.Response(
            'Too many similar requests',
            status=429,
            mimetype='text/xml'
        )


def run():
    app.run(port=9090, debug=True)
