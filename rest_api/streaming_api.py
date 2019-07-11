import flask
import json
import os
import pickle
import socket
from flask import Response, jsonify
from main.common import config_parser
from flask_cors import cross_origin
from rest_api.db.event_db_logger import EventDBLogger

CONFIG = config_parser.parse()
PORT = 14600

app = flask.Flask(__name__)

sock = socket.socket()
cam_info_json = 'rest_api/cam_info.json'
root_path = os.path.dirname(os.getcwd())
db_faces = root_path + '/main/data/known_faces/'
db_objects = root_path + '/main/photo/'


@app.route('/camera', methods=['GET'])
@cross_origin()
def get_camers_list():

    if not os.path.isfile(cam_info_json):
        with open(cam_info_json, 'r') as f:
            data = []
            json.dump(data, f, indent=4)
    else:
        with open(cam_info_json, 'r') as  f:
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
    program = "{} -src {}".format('main/run.py', camera_url)

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
    camera_url = flask.request.args.get('camera_url')
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
    object_id = flask.request.args.get('object_id')
    img_path = os.path.join(db_objects, 'ID_{}.jpg'.format(object_id))
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
    #object_id = [0,1,2]#flask.request.args.get('object_id')
    name = flask.request.args.get('name') #flask.request.values['name']
    stars = 3 #flask.request.values['stars']
    description = 'iphone' #flask.request.values['description']
    object_id_info = {
        "camera_url": camera_url,
        "name": name,
        "stars": stars,
        "description": description
    }

    return jsonify(object_id_info)


@app.route('/camera/object', methods=['PUT'])
@cross_origin()
def object_id():
    camera_url = flask.request.values['camera_url']
    object_id = flask.request.values['object_id']
    name = flask.request.values['name']
    stars = flask.request.values['stars']
    description = flask.request.values['description']
    object_id_info = {
        "type": "set_name",
        "camera_url": camera_url,
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

    return jsonify(object_id_info)


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
    start_date = flask.request.args.get('start_date') #2019-05-16 22:06:04.369857
    end_date = flask.request.args.get('end_date') #2019-05-16 22:06:04.369857
    table = "rtsp://admin:0ZKaxVFi@10.101.106.4:554/live/main"
    name = flask.request.args.get('name')
    connection = EventDBLogger()
    table = connection.create_table(table)
    result = connection.select(table, start_date, end_date)
    return Response(result, mimetype='application/json')


def run():
    app.run(host='0.0.0.0', port=9090, debug=True, threaded=True)


if __name__ == '__main__':
    run()
