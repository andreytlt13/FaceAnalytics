#!/usr/bin/env python
import cv2
import flask

from srv.camera_stream.opencv_read_stream import Camera
from srv.video_processing.haar_cascade import FaceDetector

CAMERA_URL_PARAMETER = 'camera_url'
FRAME_TYPE = '.jpg'

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


def generate_stream(camera):
    while True:
        _, frame = camera.get_frame()
        faces = FaceDetector.detect(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, img_encoded = cv2.imencode(FRAME_TYPE, frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return flask.Response(
        generate_stream(Camera("0")),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(port='9090', debug=True)
