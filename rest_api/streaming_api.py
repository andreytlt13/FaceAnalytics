import glob
import json
import os
import pickle
import socket

import flask
from flask import Response, jsonify
from flask_cors import cross_origin

from main.common import config_parser
from rest_api.db.event_db_logger import EventDBLogger

CONFIG = config_parser.parse()
PORT = 14500

app = flask.Flask(__name__)

# sock = socket.socket()

root_path = os.path.dirname(os.getcwd())
cam_info_json = root_path + '/rest_api/cam_info.json'
db_faces = root_path + '/main/face_processing/known_faces/'
db_faces = root_path + '/main/data/photo/0/known_faces/'
db_objects = root_path + '/main/data/photo/'


@app.route('/camera', methods=['GET'])
@cross_origin()
def get_camers_list():
    if not os.path.isfile(cam_info_json):
        with open(cam_info_json, 'r') as f:
            data = []
            json.dump(data, f, indent=4)
    else:
        with open(cam_info_json, 'r') as f:
            data = json.load(f)

    return jsonify(data)


@app.route('/camera', methods=['POST'])
@cross_origin()
def add_camera():

    camera_url = flask.request.values['camera_url']
    status = "disabled"
    with open(cam_info_json, 'r') as f:
        data = json.load(f)

    cam_info = {
        "camera_url": camera_url,
        "status": status,
        "url_stream": None,
        "name": "4 floor"
    }

    with open(cam_info_json, 'w') as f:
        if len(data) == 0:
            data = []
        data.append(cam_info)
        json.dump(data, f, indent=4)

    return jsonify(cam_info)


@app.route('/camera/del', methods=['PUT'])
@cross_origin()
def del_camera():
    camera_url = flask.request.values['camera_url']

    with open(cam_info_json, 'r') as f:
        data = json.load(f)

    for idx, val in enumerate(data):
        if val['camera_url'] == camera_url:
            del data[idx]

    with open(cam_info_json, 'w') as f:
        if len(data) == 0:
            data = []
        json.dump(data, f, indent=4)

    return jsonify(data)


@app.route('/camera', methods=['PUT'])
@cross_origin()
def run_processing():
    camera_url = flask.request.values['camera_url']
    print('[INFO] starting video stream...')

    # script = "{} -src {}".format('main/run.py',camera_url)
    # os.system(script)
    # program = "{} -src {}".format('main/run.py', camera_url)

    with open(cam_info_json, 'r') as f:
        data = json.load(f)

    for idx, val in enumerate(data):
        if val['camera_url'] == camera_url:
            data[idx]['camera_url'] == 'active'

    with open(cam_info_json, 'w') as f:
        if len(data) == 0:
            data = []
        json.dump(data, f, indent=4)

    return Response({"starting":camera_url}, mimetype='application/json')


@app.route('/camera/objects', methods=['GET'])
@cross_origin()
def get_objects():
    camera_name = flask.request.args.get('camera_name')
    sock = socket.socket()
    sock.connect(('127.0.0.1', PORT))
    object_id_info = {
        "type": "get_objects"
    }
    b_message = pickle.dumps(object_id_info)
    sock.send(b_message)
    data = b""
    tmp = sock.recv(16384)
    while tmp:
        data += tmp
        tmp = sock.recv(16384)

    result = pickle.loads(data)
    print(result)
    sock.close()

    return jsonify(result)


@app.route('/camera/object/photo', methods=['GET'])
@cross_origin()
def get_object_photo():
    camera_name = flask.request.args.get('camera_name')
    object_id = flask.request.args.get('object_id')
    img_path = os.path.join(db_objects, '{}/objects/id_{}/face/*'.format(camera_name, object_id))
    img_path = glob.glob(img_path)
    img_path = max(img_path, key=os.path.getctime)
    if os.path.exists(img_path):
        return flask.send_file(img_path, mimetype='image/jpg')
    else:
        return 'file doesnt exist'


@app.route('/camera/name/photo', methods=['GET'])
@cross_origin()
def get_name_photo():
    name = flask.request.args.get('name')
    img_path = os.path.join(db_faces, '{}.jpg'.format(name))
    if os.path.exists(img_path):
        return flask.send_file(img_path, mimetype='image/jpg')
    else:
        return 'file doesnt exist'


@app.route('/camera/name/info', methods=['GET'])
@cross_origin()
def get_name_info():
    camera_url = flask.request.args.get('camera_url')
    name = flask.request.args.get('name')  # flask.request.values['name']
    # object_id = [0,1,2]#flask.request.args.get('object_id')
    sock = socket.socket()
    sock.connect(('127.0.0.1', PORT))
    message = {
        "type": "get_name_info",
        "camera_url": camera_url,
        "name": name
    }
    b_message = pickle.dumps(message)
    sock.send(b_message)

    data = b""
    tmp = sock.recv(16384)
    while tmp:
        data += tmp
        tmp = sock.recv(16384)

    result = pickle.loads(data)

    print(result)
    sock.close()

    return jsonify(result)


@app.route('/camera/object', methods=['PUT'])
@cross_origin()
def object_id():
    sock = socket.socket()
    camera_name = flask.request.values['camera_name']
    object_id = flask.request.values['object_id']
    name = flask.request.values['name']
    stars = flask.request.values['stars']
    description = flask.request.values['description']
    object_id_info = {
        "type": "set_name",
        "camera_url": camera_name,
        "object_id": object_id,
        "name": name,
        "stars": stars,
        "description": description
    }
    sock.connect(('127.0.0.1', PORT))
    b_message = pickle.dumps(object_id_info)
    sock.send(b_message)
    data = b""
    tmp = sock.recv(16384)
    while tmp:
        data += tmp
        tmp = sock.recv(16384)

    result = pickle.loads(data)
    print(result)
    sock.close()

    return jsonify(result)


def db_insert():
    start_date = flask.request.args.get('start_date')
    end_date = flask.request.args.get('end_date')
    name = flask.request.args.get('name')
    table = "event_logger"
    db_name = "0"
    connection = EventDBLogger(db_name)
    table = connection.create_table(table)
    result = connection.select(table, start_date, end_date)
    return Response(result, mimetype='application/json')


@app.route('/camera/get_object_img', methods=['GET'])
def get_object_img():
    object_id = flask.request.args.get('object_id')
    img_path = os.path.join(db_objects, 'ID_{}.jpeg'.format(object_id))
    if os.path.exists(img_path):
        return flask.send_file(img_path, mimetype='image/jpg')
    else:
        return 'file doesnt exist'


@app.route('/camera/get_name_img', methods=['GET'])
@cross_origin()
def get_name_img():
    name = flask.request.args.get('name')
    img_path = os.path.join(db_faces, '{}.jpg'.format(name))
    if os.path.exists(img_path):
        return flask.send_file(img_path, mimetype='image/jpg')
    else:
        return 'file doesnt exist'


@app.route('/camera/db_select', methods=['GET'])
@cross_origin()
def db_select():
    start_date = flask.request.args.get('start_date')
    end_date = flask.request.args.get('end_date')
    name = flask.request.args.get('name')
    table = "event_logger"
    db_name = "0"
    connection = EventDBLogger(db_name)
    table = connection.create_table(table)
    result = connection.select(table, start_date, end_date)
    return Response(result, mimetype='application/json')


@app.route('/camera/db_select_description', methods=['GET'])
@cross_origin()
def db_select_description():
    name = flask.request.args.get('name')
    table = "event_logger"
    db_name = "0"
    connection = EventDBLogger(db_name)
    table = connection.create_table_recognized_logger()
    result = connection.select_name_description(table, name)
    return Response(result, mimetype='application/json')


def run():
    app.run(host='0.0.0.0', port=9090, debug=True, threaded=True)


if __name__ == '__main__':
    run()
