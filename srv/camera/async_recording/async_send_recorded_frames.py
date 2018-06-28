import configparser
import glob
import logging
from datetime import datetime

import cv2
import requests

API_SUCC_RESPONSE_CODE = 200
API_URL_PROPERTY = 'api_url'
CONFIG_FILE_PATH = 'config.ini'
CONTENT_TYPE_PROPERTY = 'api_request_content_type'
DEFAULT_SECTION = 'DEFAULT'
FRAME_TYPE_PROPERTY = 'frame_type'
FRAMES_DIR_PROPERTY = 'frames_dir'

# Parse config
parser = configparser.ConfigParser()
parser.read(CONFIG_FILE_PATH)
config = parser[DEFAULT_SECTION]

api_url = config[API_URL_PROPERTY]
content_type = config[CONTENT_TYPE_PROPERTY]
headers = {'content-type': content_type}

frame_type = config[FRAME_TYPE_PROPERTY]
frames_dir = config[FRAMES_DIR_PROPERTY]

# Send frames
for path in glob.glob(frames_dir + '*' + frame_type):
    frame = cv2.imread(path)
    # encode frame as frame_type
    _, img_encoded = cv2.imencode(frame_type, frame)
    # Http POST
    response = requests.post(api_url, data=img_encoded.tostring(), headers=headers)
    if response.status_code == API_SUCC_RESPONSE_CODE:
        logging.debug('Successfully sent %s', str(datetime.now().timestamp()) + frame_type)
