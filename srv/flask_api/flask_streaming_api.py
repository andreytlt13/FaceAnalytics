#!/usr/bin/env python
import time

import cv2
import flask
from imutils.video import FPS
from imutils.video import VideoStream

from common import config_parser
from frame_processing.frame_processor import FrameProcessor

CONFIG = config_parser.parse()

app = flask.Flask(
    __name__,
    instance_path='/home/andrey/PycharmProjects/FaceAnalytics/srv/config'
)


# '/home/andrey/PycharmProjects/FaceAnalytics/srv/config'
# '/srv/config'
# '../common/config')
# https://blog.miguelgrinberg.com/post/video-streaming-with-flask


@app.route('/video_stream', methods=['GET'])
def video_stream():
    """
    Simple API for video streaming with face detection/recognition on top
    :return: generator object so that on-the-fly stream can be shown to user
    """
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
    vs = VideoStream(src=camera_url).start()  # src=camera_url
    time.sleep(2.0)
    frame_processor = FrameProcessor()
    fps = None
    (H, W) = (None, None)

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    trackableObjects = {}
    countObject = 0

    while True:
        fps = FPS().start()

        frame, _, status, totalFrames, totalDown, totalUp, countObject = frame_processor.process_next_frame(vs,
                                                                                                            totalFrames,
                                                                                                            totalDown,
                                                                                                            totalUp)

        fps.update()
        fps.stop()

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        info = [
            ('Enter', totalUp),
            ('Exit', totalDown),
            ('TotalFrames', totalFrames),
            ('Status', status),
            ('Count Person', countObject),
            ("FPS", "{:.2f}".format(fps.fps()))
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def run():
    app.run(host='0.0.0.0', port=9090, debug=True)
