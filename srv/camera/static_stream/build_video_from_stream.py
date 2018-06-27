import configparser

import cv2

from srv.camera.static_stream.read_static_stream import CameraStreamReader

CAMERA_CODEC = 'camera_codec'
CAMERA_FPS = 'camera_fps'
CAMERA_RESOLUTION = 'camera_resolution'
CONFIG_FILE = 'config.ini'
DEFAULT = 'DEFAULT'

parser = configparser.ConfigParser()
parser.read(CONFIG_FILE)
config = parser[DEFAULT]
resolution = config[CAMERA_RESOLUTION].split('x')
video_codec = cv2.VideoWriter_fourcc(*config[CAMERA_CODEC])
out = cv2.VideoWriter('/tmp/sample.mp4', video_codec, int(config[CAMERA_FPS]), (int(resolution[0]), int(resolution[1])))

reader = CameraStreamReader(CONFIG_FILE)
frames = reader.read_stream()
for frame in frames:
    # write to file
    out.write(frame)

out.release()
