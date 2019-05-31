import cv2
import flask
from flask import Response, jsonify
from main.common import config_parser
from main import video_processing

import json
import os
from datetime import datetime

from flask_cors import cross_origin
from db.event_db_logger import EventDBLogger

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
    print(camera_url)

    return flask.Response(
        get_stream(camera_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def get_stream(camera_url):
    print('[INFO] starting video stream...')

    if camera_url == '0':
        camera_url = int(camera_url)

    camera_url = 'vlc_record_2019_05_30_12h50m55s.mp4'

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('results/output_{}.mp4'.format(datetime.now()), fourcc, 20.0, (640, 360))

    connection = EventDBLogger()
    table = connection.create_table(camera_url)

    vs = video_processing.VideoStream(camera_url, table, connection)

    while True:
        frame = vs.process_next_frame()
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

    #     out.write(frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # vs.release()
    # out.release()
    # cv2.destroyAllWindows()

@app.route('/db_select', methods=['GET'])
@cross_origin()
def db_select():
    start_date = flask.request.args.get('start_date')
    end_date = flask.request.args.get('end_date')
    table_name = flask.request.args.get('camera_url')
    connection = EventDBLogger()
    table = connection.create_table(table_name)
    result = connection.select(table, start_date, end_date)
    return Response(result, mimetype='application/json')



def run():
    app.run(host='0.0.0.0', port=9090, debug=True, threaded=True)

if __name__ == '__main__':
    run()