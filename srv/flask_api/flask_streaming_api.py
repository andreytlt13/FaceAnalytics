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

from frame_processing.tools import generate_detections as gdet
from frame_processing.deep_sort import preprocessing, nn_matching
from frame_processing.deep_sort.tracker import Tracker

CONFIG = config_parser.parse()

app = flask.Flask(
    __name__,
    instance_path='/home/ekaterinaderevyanka/PycharmProjects/FaceAnalytics/srv/config'
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
    }
]

camera = {
    'camera_url': 'rtsp://admin:0ZKaxVFi@10.101.106.4:554/live/main'
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

    vs = VideoStream(src=0).start()  #camera_url
    connection = EventDBLogger()
    table = connection.create_table(camera_url)

    for i, v in enumerate(tasks):
       if v['table'] == camera_url:
           contours = tasks[i]['contours']

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

    known_face_encodings, known_face_names = load_known_face_encodings(CONFIG['known_people_dir'])
    print('known_face_names: ', known_face_names)

    faces_sequence = collections.defaultdict()

    # deep sort implementation
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    encoder = gdet.create_box_encoder('../frame_processing/mars-small128.pb', batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    while True:

        fps = FPS().start()

        if CONFIG['tracker_type'] == 'centroid':
            # centroid tracker
            frame, _, info, faces_sequence = frame_processor.process_next_frame(vs,
                                                                                faces_sequence,
                                                                                known_face_encodings,
                                                                                known_face_names,
                                                                                info,
                                                                                connection
                                                                                )

        if CONFIG['tracker_type'] == 'deepsort':
            # deep sort tracker
            frame, _, info, faces_sequence = frame_processor.process_next_frame_deepsort_tracker(
                                                                                vs,
                                                                                faces_sequence,
                                                                                tracker, encoder,
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


def run():
    app.run(host='0.0.0.0', port=9090, debug=True, threaded=True)