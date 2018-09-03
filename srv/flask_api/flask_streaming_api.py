#!/usr/bin/env python
import time

import cv2
import flask
from imutils.video import VideoStream

from frame_processsing.frame_processor import FrameProcessor

app = flask.Flask(__name__,
                  instance_path='/home/andrey/PycharmProjects/FaceAnalytics/srv/common/config')  # '../common/config')


@app.route('/video_stream', methods=['GET'])
def video_stream():
    try:
        camera_url = int(flask.request.args.get('camera_url'))
    except:
        camera_url = flask.request.args.get('camera_url')

    return flask.Response(
        stream(camera_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def stream(camera_url):
    # initialize the video stream and allow the camera sensor to warmup
    print('[INFO] starting video stream...')
    vs = VideoStream(src=camera_url).start()
    time.sleep(2.0)
    frame_processor = FrameProcessor()
    while True:
        frame, _, _ = frame_processor.process_next_frame(vs)
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def run():
    app.run(port=9090, debug=True)
