#!/usr/bin/env python
import cv2, os, flask
from flask import Response, jsonify
from flask_cors import cross_origin
from imutils.video import FPS
from imutils.video import VideoStream
from datetime import datetime
from common import config_parser
from frame_processing.frame_processor import FrameProcessor
from db.event_db_logger import EventDBLogger
import collections
from face_description.network_loader import load_network, load_known_face_encodings
from flask import jsonify
import ast


#250X200
CONFIG = config_parser.parse()

app = flask.Flask(
    __name__,
    instance_path='/Users/andrey/PycharmProjects/FaceAnalytics/srv/config'
)

tasks = [
    {
        'table': 'rtsp://admin:0ZKaxVFi@10.101.106.4:554/live/main',
        'contours':[[0, 185], [0, 375], [500, 375], [500, 185]]
    },
    {
        'table': 'rtsp://admin:0ZKaxVFi@10.101.106.6:554/live/main',
        'contours':[[50, 50], [50, 150], [150, 150], [150, 50]]
    },
    {
        'table': 'rtsp://admin:0ZKaxVFi@10.101.106.8:554/live/main',
        'contours':[[50, 50], [50, 150], [150, 150], [150, 50]]
    },
    {
        'table': '0',
        'contours': [[50, 50], [50, 150], [150, 150], [150, 50]]
    }
]

#'rtsp://admin:0ZKaxVFi@10.101.106.4:554/live/main'
camera = {
    'camera_url': 'rstp://zal:ichehol800@46.0.193.39:8871/videoMain'
}

@app.route('/add_aim_region', methods=['POST'])
def add_aim_region():
    print(tasks)
    if flask.request.is_json:
        flask.abort(400)

    task = flask.request.get_json()
    if not flask.request.json or not 'table' in flask.request.json:
        flask.abort(400)
    for i, v in enumerate(tasks):
       if v['table'] == task['table']:
           tasks[i]['contours'] = task['contours']
       else:
           tasks.append(task)
    print(tasks)
    return jsonify({'task': task}), 201


@app.route('/video_stream', methods=['GET'])
def video_stream():
    """
    Simple API for video streaming with face detection/recognition on top
    :return: generator object so that on-the-fly stream can be shown to user
    """
    camera_url = flask.request.args.get('camera_url')

    return flask.Response(
        stream(camera_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def stream(camera_url):
    print('[INFO] starting video stream...')

    if camera_url == None:
        camera_url = camera['camera_url']

    vs = VideoStream(src="rstp://zal:ichehol800@46.0.193.39:8871/videoMain").start() #camera_url
    connection = EventDBLogger()
    table = connection.create_table(camera_url)

    for i, v in enumerate(tasks):
       if v['table'] == camera_url:
           contours = tasks[i]['contours']


    global frame_processor
    frame_processor = FrameProcessor(contours=contours, table=table)


    (H, W) = (None, None)

    info = {
        'FPS': 0,
        'Enter': 0,
        'Exit': 0,
        'TotalFrames': 0,
        'Status': 'start',
        'Count People': 0
    }

    DB_PATH = '../face_description/known_faces'
    known_face_encodings, known_face_names = load_known_face_encodings(DB_PATH)
    print('known_face_names: ', known_face_names)

    faces_sequence = collections.defaultdict()


    while True:

        fps = FPS().start()
        frame, _, info, faces_sequence = frame_processor.process_next_frame(vs,faces_sequence,
                                                                            known_face_encodings,
                                                                            known_face_names,
                                                                            info,
                                                                            connection
                                                                            )
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        fps.update()
        fps.stop()
        info['FPS'] = "{:.2f}".format(fps.fps())

        for i, (k, v) in enumerate(info.items()):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


@app.route('/db_select', methods=['GET'])
@cross_origin()
def db_select():
    start_date = flask.request.args.get('start_date')
    end_date = flask.request.args.get('end_date')
    table = flask.request.args.get('table')
    connection = EventDBLogger()
    table = connection.create_table(table)
    result = connection.select(table, start_date, end_date)
    return Response(result, mimetype='application/json')


@app.route('/get_person_id', methods=['GET'])
@cross_origin()
def get_person_id():
    global frame_processor

    ObjectID = list(frame_processor.trackableObjects.keys())
    Names = [frame_processor.trackableObjects[x].names for x in ObjectID]
    Name = [frame_processor.trackableObjects[x].name for x in ObjectID]

    result = {
        "ObjectID": ObjectID,
        "Names": Names,
        "Name": Name
    }
    return jsonify(result)


@app.route('/select_name', methods = ['PUT'])
@cross_origin()
def select_name():
    objectID = flask.request.args.get('object_id')
    name = flask.request.args.get('name')

    global frame_processor
    frame_processor.trackableObjects[int(objectID)].name = name
    return objectID


@app.route('/get_face', methods=['GET'])
def get_face_tmp():
    objectID = flask.request.args.get('object_id')
    path = '../face_description/tmp/faces/{}.jpg'.format(objectID)
    return flask.Response(
        read_face(path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_face_known', methods=['GET'])
def get_face_known():
    objectID = flask.request.args.get('name')
    path = '../face_description/known_faces/{}.jpg'.format(objectID)
    return flask.Response(
        read_face(path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/put_description', methods = ['PUT'])
@cross_origin()
def put_description():
    objectID = flask.request.args.get('object_id')
    description = flask.request.args.get('description')
    stars = flask.request.args.get('stars')

    global frame_processor
    frame_processor.trackableObjects[int(objectID)].description = description
    frame_processor.trackableObjects[int(objectID)].stars = stars

    result = {
        "objectID":objectID,
        "description":description,
        "stars":stars
    }

    return jsonify(result)


@app.route('/get_description', methods=['GET'])
@cross_origin()
def get_description():
    objectID = flask.request.args.get('object_id')

    global frame_processor
    description = frame_processor.trackableObjects[int(objectID)].description
    stars = frame_processor.trackableObjects[int(objectID)].stars
    name = frame_processor.trackableObjects[int(objectID)].name

    result = {
        "objectID":objectID,
        "name":name,
        "description":description,
        "stars":stars
    }

    return jsonify(result)


def read_face(path):
    img = cv2.imread(path, 0)
    _, img_encoded = cv2.imencode('.jpg', img)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


def run():
    app.run(host='0.0.0.0', port=9090, debug=True, threaded=True)


