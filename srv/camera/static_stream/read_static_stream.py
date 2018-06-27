import configparser

import cv2

FILE_PATH = 'file_path'
DEFAULT = 'DEFAULT'
FRAME_INDEX = 'frame_index'


class CameraStreamReader:
    """
    Reads and collects every frame['frame_index'] of video stream from given static video
    """

    def __init__(self, config_path) -> None:
        parser = configparser.ConfigParser()
        parser.read(config_path)
        self.config = parser[DEFAULT]

        self.capture = cv2.VideoCapture(self.config[FILE_PATH])

    def __del__(self):
        self.capture.release()

    def read_stream(self):
        frames = []
        while self.capture.isOpened():
            # Capture frame-by-frame
            success, frame = self.capture.read()
            if success:
                frames.append(frame)
            else:
                # Video end
                break

        fi = int(self.config[FRAME_INDEX])
        return frames[::fi]
