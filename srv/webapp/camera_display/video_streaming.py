#!/usr/bin/env python
from datetime import datetime

import cv2
import flask
from flask_restful.representations import json

import srv.models as face_recognition
from srv.camera_stream.opencv_read_stream import Camera

andreym_image = face_recognition.load_image_file("srv/webapp/photo/simon.jpg")
andreym_face_encoding = face_recognition.face_encodings(andreym_image)[0]

known_face_encodings = [
    andreym_face_encoding
]

known_face_names = [
    "Is it Simon?"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

log_time = 0
last_read_row = ''

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
    return flask.Response(
        generate_stream(Camera(int(flask.request.args.get('camera_url')))),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_stream(camera):
    process_this_frame = True

    while True:
        _, frame = camera.get_frame()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

            # log results
            log_msg_builder = ''
            for name in face_names:
                log_msg_builder += name + ', '
            log_msg_builder = log_msg_builder[:-2]  # truncate last ', '
            log_faces(log_msg_builder)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # faces = FaceDetector.detect(frame)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def log_faces(msg):
    out = open('camera_display/log/faces_log.txt', 'a+')
    global log_time
    current_time = datetime.now()
    minutes_since_last_log = (current_time.timestamp() - log_time) / 10
    # log every 10 seconds
    if minutes_since_last_log >= 1:
        log_time = current_time.timestamp()
        are_many_people = len(msg.split(',')) > 10
        out.write(
            '\n' + msg + (' were ' if are_many_people else ' was ') + 'there at '
            + current_time.strftime('%H:%M:%S')[0:8]
            + ' on ' + str(current_time.date())
        )
    out.close()


@app.route('/text_stream', methods=['GET'])
def text_stream():
    faces = open('camera_display/log/faces_log.txt', 'r')
    objects_info = faces.readlines()[-1]
    faces.close()

    global last_read_row
    if last_read_row == objects_info:
        return json.dumps({'success': False}), 400, {'ContentType': 'application/json'}

    last_read_row = objects_info
    return flask.Response(
        objects_info,
        mimetype='text/xml'
    )


def run():
    if __name__ == "main":
        app.run(port=9090, debug=True)
