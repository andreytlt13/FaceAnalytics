#!/usr/bin/env python
import cv2
import flask
import srv.models as face_recognition

from srv.camera_stream.opencv_read_stream import Camera
from srv.video_processing.haar_cascade import FaceDetector

andreym_image = face_recognition.load_image_file("srv/webapp/photo/andrey_m.jpg")
andreym_face_encoding = face_recognition.face_encodings(andreym_image)[0]

known_face_encodings = [
    andreym_face_encoding
]

known_face_names = [
    "Andrey_M"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    return flask.render_template(
        'response.html',
        cam_url=flask.request.form.get('camera_url')
    )


@app.route('/video_stream', methods=['GET'])
def video_stream():
    return flask.Response(
        generate_stream(
            Camera(int(flask.request.args.get('url')))
        ),
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


def run():
    app.run(port=9090, debug=True)
