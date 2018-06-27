import configparser
import logging
from datetime import datetime

import cv2
import requests

API_SUCC_RESPONSE_CODE = 200
API_URL_PROPERTY = 'api_url'
CAMERA_ID_PROPERTY = 'camera_id'
CONTENT_TYPE_PROPERTY = 'api_request_content_type'
DEFAULT_SECTION = 'DEFAULT'
FRAME_INDEX_PROPERTY = 'frame_index'
FRAME_TYPE_PROPERTY = 'frame_type'
FRAMES_DIR_PROPERTY = 'frames_dir'
LOG_LEVEL = logging.DEBUG
LOG_PATH_PROPERTY = 'log_path'


class CameraStreamRemoteRecorder:
    """
    Reads every frame['frame_index'] of live video stream from given camera
     and sends it over the Network for processing
    """

    def __init__(self, config_path) -> None:
        parser = configparser.ConfigParser()
        parser.read(config_path)
        config = parser[DEFAULT_SECTION]

        self.api_url = config[API_URL_PROPERTY]
        content_type = config[CONTENT_TYPE_PROPERTY]
        self.headers = {'content-type': content_type}

        self.frame_index = int(config[FRAME_INDEX_PROPERTY])
        self.frame_type = config[FRAME_TYPE_PROPERTY]
        self.camera_id = config[CAMERA_ID_PROPERTY]
        self.capture = cv2.VideoCapture(int(self.camera_id))

        logging.basicConfig(filename=config[LOG_PATH_PROPERTY], level=LOG_LEVEL)

    def __del__(self):
        self.capture.release()

    def record_stream(self):
        frames_read_count = 0
        while self.capture.isOpened():
            # Capture frame-by-frame
            success, frame = self.capture.read()
            if success:
                frames_read_count += 1
                # No need to store all the frames hence choosing every N-th
                # which is specified within FRAME_INDEX property
                if frames_read_count % self.frame_index == 0:
                    # encode frame as frame_type
                    _, img_encoded = cv2.imencode(self.frame_type, frame)
                    # send
                    response = requests.post(self.api_url, data=img_encoded.tostring(), headers=self.headers)
                    if response.status_code == API_SUCC_RESPONSE_CODE:
                        logging.debug('Successfully sent %s', str(datetime.now().timestamp()) + self.frame_type)
            else:
                raise RuntimeError(str.format('Camera %d has been disconnected', self.camera_id))
