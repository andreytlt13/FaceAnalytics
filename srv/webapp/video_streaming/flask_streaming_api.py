#!/usr/bin/env python

import cv2
import flask
import numpy as np

from srv.video_processing.functions.detect_face_features import detect_face_features
from srv.video_processing.functions.detect_people import detect_people
from srv.video_processing.functions.recognize_face import recognize_face
from srv.webapp.video_streaming.utils.normalize_image import fisheye_to_flat

LOG_PATH = '/tmp/faces_log.txt'
last_log_message = ''
detected_regions_count = 1

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
    img_size = 160

    while True:
        _, frame = cv2.VideoCapture(camera_url).read()
        img_h, img_w, _ = np.shape(frame)

        frame = fisheye_to_flat(frame)
        people = detect_people(frame, img_w)
        if len(people) > 0:
            for i, (x, y, w, h) in enumerate(people):
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

                cropped = frame[y:h, x:w, :]
                crop_h, crop_w, _ = np.shape(cropped)
                print('Detected region: ' + str(crop_w) + ', ' + str(crop_h))
                global detected_regions_count
                cv2.imwrite('/tmp/images/frame' + str(detected_regions_count) + '.jpg', cropped)
                detected_regions_count += 1

                detect_face_features(cropped, frame, img_size, x, y)
        else:
            detect_face_features(frame, frame, img_size, 0, 0)

        recognize_face(frame)
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


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
