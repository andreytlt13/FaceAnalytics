#!/usr/bin/env python
import cv2, os, flask
import codecs, json
#from flask import request
import numpy as np
from flask import Response, jsonify
from flask_cors import cross_origin
from imutils.video import FPS
from imutils.video import VideoStream
from datetime import datetime
from common import config_parser
from frame_processing.frame_processor import FrameProcessor
from db.event_db_logger import EventDBLogger

import collections
import glob
import face_recognition

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


def load_known_face_encodings(path):
    photos = glob.glob(path + '/*.jpg')
    known_face_encodings = []
    known_face_names = []
    for photo in photos:
        image = face_recognition.load_image_file(photo)
        title = photo.split('/')[-1].split('.')[0]
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(title)
    return known_face_encodings, known_face_names


def stream(camera_url):
    print('[INFO] starting video stream...')

    if camera_url == None:
        camera_url = camera['camera_url']

    # hack_cam = camera_url
    # hack_cam = 0
    # test_vid_path = '/home/ekaterinaderevyanka/TESTS/FaceAnalytics/foscam_cctv/processed_videos/144555_6_persons.mp4'
    # test_vid_path = '/home/ekaterinaderevyanka/TESTS/FaceAnalytics/foscam_cctv/processed_videos/133747_8_persons.mp4'
    # test_vid_path = '/home/ekaterinaderevyanka/TESTS/FaceAnalytics/foscam_cctv/processed_videos/154522_3_persons.mp4'
    # test_vid_path = '/home/ekaterinaderevyanka/TESTS/FaceAnalytics/foscam_cctv/processed_videos/132636_4_persons.mp4'
    # test_vid_path = '/home/ekaterinaderevyanka/TESTS/FaceAnalytics/G20 leaders pose for family photo.mp4'
    test_vid_path = '/home/ekaterinaderevyanka/TESTS/FaceAnalytics/foscam_cctv/gyroscooter/144848.ts'

    TEST_SAVE_PATH = '/home/ekaterinaderevyanka/TESTS/FaceAnalytics/results'

    if CONFIG['webcam_mode'] == 'stream':
        vs = VideoStream(src=0).start() # hack with webcam --- instead it put the camera_url variable

    if CONFIG['webcam_mode'] == 'test':
        vs = cv2.VideoCapture(test_vid_path)
        out_vid_path = os.path.join(TEST_SAVE_PATH, 'res_{}_{}'.format(CONFIG['detection_mode'], test_vid_path.split('/')[-1]))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(out_vid_path, fourcc, 15, (500, 281))

    connection = EventDBLogger()
    table = connection.create_table(camera_url)

    image_camera_dir = '../image_for_processing/{}'.format(camera_url.replace('/', '_'))
    if not os.path.exists(image_camera_dir):
        os.makedirs(image_camera_dir)

    image_camera_dir = '{0}/{1}'.format(image_camera_dir, datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(image_camera_dir):
        os.makedirs(image_camera_dir)

    for i, v in enumerate(tasks):
       if v['table'] == camera_url:
           contours = tasks[i]['contours']

    frame_processor = FrameProcessor(path_for_image=image_camera_dir, table=table, contours=contours)

    (H, W) = (None, None)

    info = {
        'FPS': 0,
        'Enter': 0,
        'Exit': 0,
        'TotalFrames': 0,
        'Status': 'start',
        'Count People': 0
    }

    # ======
    DB_PATH = '../face_description/known_faces'
    known_face_encodings, known_face_names = load_known_face_encodings(DB_PATH)
    print('known_face_names: ', known_face_names)

    faces_sequence = collections.defaultdict()
    # ======

    while True:

        fps = FPS().start()

        if CONFIG['webcam_mode'] == 'stream':
            vs_frame = vs.read()

        if CONFIG['webcam_mode'] == 'test':
            ret, vs_frame = vs.read()
            if not ret:
                break

        # frame, _, info = frame_processor.process_next_frame(vs_frame, info, connection, camera_url)

        frame, _, info, faces_sequence = frame_processor.process_next_frame_WITH_RECOGNITION(vs_frame, faces_sequence,
                                                                                             known_face_encodings, known_face_names,
                                                                                             info, connection, camera_url)

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

        # save video into file
        if CONFIG['webcam_mode'] == 'test':
            out_video.write(frame)

    if CONFIG['webcam_mode'] == 'test':
        vs.release()
        out_video.release()


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

# if __name__ == '__main__':
def run():
    app.run(host='0.0.0.0', port=9090, debug=True, threaded=True)
