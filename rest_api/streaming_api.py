import flask
import json
import os
from flask import Response, jsonify
from main.common import config_parser
from flask_cors import cross_origin
from db.event_db_logger import EventDBLogger

CONFIG = config_parser.parse()

app = flask.Flask(__name__)

cam_info_json = 'rest_api/cam_info.json'


@app.route('/camera/list', methods=['GET'])
def get_camers_list():

    if not os.path.isfile(cam_info_json):
        with open(cam_info_json, 'r') as f:
            data = []
            json.dump(data, f, indent=4)
    else:
        with open(cam_info_json, 'r') as  f:
            data = json.load(f)

    return jsonify(data)


@app.route('/camera/add', methods=['PUT'])
def add_camera():

    camera_url = flask.request.values['camera_url']
    status = "disabled"
    with open(cam_info_json, 'r') as f:
        data = json.load(f)

    cam_info = {
        "camera_url": camera_url,
        "status": status
    }

    with open(cam_info_json, 'w') as f:
        if len(data) == 0:
            data = []
        data.append(cam_info)
        json.dump(data, f, indent=4)

    return jsonify(cam_info)


@app.route('/camera/del', methods=['PUT'])
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


@app.route('/camera/db_select', methods=['GET'])
@cross_origin()
def db_select():
    start_date = flask.request.args.get('start_date')
    end_date = flask.request.args.get('end_date')
    table_name = flask.request.args.get('camera_url')
    connection = EventDBLogger()
    table = connection.create_table(table_name)
    result = connection.select(table, start_date, end_date)
    return Response(result, mimetype='application/json')


@app.route('/camera/start', methods=['PUT'])
def run_processing():
    camera_url = flask.request.values['camera_url']
    print('[INFO] starting video stream...')
    # script = "{} -src {}".format('main/run.py',camera_url)
    # os.system(script)

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


@app.route('/camera/object_id', methods=['PUT'])
def object_id():
    camera_url = flask.request.values['camera_url']
    object_id = flask.request.values['object_id']
    name = flask.request.values['name']
    stars = flask.request.values['stars']
    description = flask.request.values['description']
    object_id_info = {
        "camera_url": camera_url,
        "object_id": object_id,
        "name": name,
        "stars": stars,
        "description": description
    }

    return jsonify(object_id_info)


@app.route('/camera/get_object_img', methods=['GET'])
def get_object_img():
    start_date = flask.request.args.get('start_date')
    end_date = flask.request.args.get('end_date')
    table_name = flask.request.args.get('camera_url')
    connection = EventDBLogger()
    table = connection.create_table(table_name)
    result = connection.select(table, start_date, end_date)
    return Response(result, mimetype='application/json')

@app.route('/camera/get_name_img', methods=['GET'])
def get_name_img():
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