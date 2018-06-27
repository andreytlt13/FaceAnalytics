import configparser
from datetime import datetime

import cv2

from srv.camera.static_stream.read_static_stream import CameraStreamReader

CONFIG_FILE = 'config.ini'
DEFAULT = 'DEFAULT'

parser = configparser.ConfigParser()
parser.read(CONFIG_FILE)
config = parser[DEFAULT]

reader = CameraStreamReader(CONFIG_FILE)
frames = reader.read_stream()
for frame in frames:
    # write to file
    cv2.imwrite('/tmp/frames/' + str(datetime.now().timestamp()) + '.bmp', frame)
