#!/usr/bin/env python
import cv2
import flask

from srv.camera_stream.opencv_read_stream import Camera
from srv.video_processing.haar_cascade import FaceDetector

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
    while True:
        _, frame = camera.get_frame()
        faces = FaceDetector.detect(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, img_encoded = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def run():
    app.run(port=9090, debug=True)
