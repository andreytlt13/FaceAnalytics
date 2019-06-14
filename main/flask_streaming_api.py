
import cv2, flask

from common import config_parser
from imutils import video

CONFIG = config_parser.parse()

app = flask.Flask(
    __name__,
    instance_path='/Users/andrey/PycharmProjects/FaceAnalytics/main/config'
)


@app.route('/video_stream', methods=['POST'])
def add_video_stream():
    camera_url = flask.request.args.get('camera_url')

    return  flask.Response(
        stream(camera_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def stream(camera_url):
    print('[INFO] starting video stream...')

    if camera_url == None:
        print('[ERROR] video .....')

    vs = video.VideoStream(camera_url).start()
    vd = stream_processing(vs)
    return vd


def stream_processing(stream):

    while True:
        print("kokok")


    return 1