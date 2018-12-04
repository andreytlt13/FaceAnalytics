import time

import cv2
import flask
from imutils.video import VideoStream

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
        get_stream(
            '/video_stream?camera_url=rtsp://admin:0ZKaxVFi@10.101.106.4:554/live/main'
        ),
        ramimetype='multipart/x-mixed-replace; boundary=frame'
    )


def get_stream(camera_url):
    vs = VideoStream(src=camera_url).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def run():
    app.run(port=5000, debug=True)
