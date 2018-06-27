import configparser
import logging
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request

API_PORT_PROPERTY = 'camera_recorder_rest_api_port'
API_SUCC_RESPONSE = 'Success'
CONFIG_FILE_PATH = 'config.ini'
DEFAULT_SECTION = 'DEFAULT'
FRAMES_DIR_PROPERTY = 'frames_dir'
FRAME_TYPE_PROPERTY = 'frame_type'
LOG_LEVEL = logging.DEBUG
LOG_PATH_PROPERTY = 'log_path'

# Simple api that stores given frames on disk
app = Flask(__name__)

# Parse config
parser = configparser.ConfigParser()
parser.read(CONFIG_FILE_PATH)
config = parser[DEFAULT_SECTION]
logging.basicConfig(filename=config[LOG_PATH_PROPERTY], level=LOG_LEVEL)

api_port = config[API_PORT_PROPERTY]
frames_dir = config[FRAMES_DIR_PROPERTY]
frame_type = config[FRAME_TYPE_PROPERTY]


@app.route("/store/frames", methods=['POST'])
def post():
    # convert string of image data to uint8
    buffer = np.fromstring(request.data, np.uint8)

    # decode image
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # write to file
    frame_path = frames_dir + str(datetime.now().timestamp()) + frame_type
    cv2.imwrite(frame_path, frame)
    logging.debug('Successfully written %s', frame_path)

    return API_SUCC_RESPONSE


if __name__ == '__main__':
    app.run(port=api_port)
