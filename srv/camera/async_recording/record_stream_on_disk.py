import configparser
import logging
from datetime import datetime

import cv2

CAMERA_ID_PROPERTY = 'camera_id'
DEFAULT_SECTION = 'DEFAULT'
FRAME_INDEX_PROPERTY = 'frame_index'
FRAME_TYPE_PROPERTY = 'frame_type'
FRAMES_DIR_PROPERTY = 'frames_dir'
LOG_LEVEL = logging.DEBUG
LOG_PATH_PROPERTY = 'log_path'


class CameraStreamOnDiskRecorder:
    """
    Reads every frame['frame_index'] of live video stream from given camera and writes it on disk
    so that another asynchronous task can pick it up later and send over the Network for processing
    """

    def __init__(self, config_path) -> None:
        parser = configparser.ConfigParser()
        parser.read(config_path)
        config = parser[DEFAULT_SECTION]

        self.frame_index = int(config[FRAME_INDEX_PROPERTY])
        self.frame_type = config[FRAME_TYPE_PROPERTY]
        self.frames_dir = config[FRAMES_DIR_PROPERTY]
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
                    # write to file
                    frame_path = self.frames_dir + str(datetime.now().timestamp()) + self.frame_type
                    cv2.imwrite(frame_path, frame)

                    logging.debug('Successfully written %s', frame_path)
            else:
                raise RuntimeError(str.format('Camera %d has been disconnected', self.camera_id))
