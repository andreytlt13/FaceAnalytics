import cv2
import flask
from flask import Response, jsonify
from imutils.video import VideoStream, FPS
from main.common import config_parser
from main import video_processing

import json
import os

CONFIG = config_parser.parse()

app = flask.Flask(__name__)

cam_info_json = 'cam_info.json'

@app.route('/get_camera_url', methods=['GET'])
def get_camera_url():
    camera_url = str(flask.request.args.get('camera_url'))
    existed, camera_status, camera_indx = check_camera(camera_url)

    if not existed:
        print('cam doesnt exist in file')
        camera_status = "active"
        add_camera(camera_url, camera_status, camera_indx)
    else:
        print('indx:{} url:{} status:{}'.format(camera_indx, camera_url, camera_status))

    return jsonify({"camera_url":camera_url, "status":camera_status, "index":camera_indx})

def check_camera(camera_url):
    if not os.path.isfile(cam_info_json):
        with open(cam_info_json, 'w') as f:
            data = []
            json.dump(data, f, indent=4)
    else:
        with open(cam_info_json, 'r') as f:
            data = json.load(f)

    existed = False
    status = "active"
    camera_indx = len(data)

    for d in data:
        if d["camera_url"] == camera_url:
            existed = True
            status = d["status"]
            camera_indx = d["index"]
            break

    return existed, status, camera_indx

def add_camera(camera_url, status, camera_indx):
    with open(cam_info_json, 'r') as f:
        data = json.load(f)

    cam_info = {
        "camera_url" : camera_url,
        "status" : status,
        "index" : camera_indx
    }

    with open(cam_info_json, 'w') as f:
        if len(data) == 0:
            data = []
        data.append(cam_info)
        json.dump(data, f, indent=4)


@app.route('/stream', methods=['GET'])
def stream():
    camera_url = flask.request.args.get('camera_url')

    # webcam test hack:
    camera_url = int(camera_url)

    return flask.Response(
        get_stream(camera_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def get_stream(camera_url):
    print('[INFO] starting video stream...')

    vs = video_processing.VideoStream(camera_url)

    while True:
        frame = vs.preocess_next_frame()
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

def run():
    app.run(host='0.0.0.0', port=9090, debug=True, threaded=True)

if __name__ == '__main__':
    run()