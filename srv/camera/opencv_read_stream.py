import configparser

import cv2


class Camera:
    def __init__(self, config_path) -> None:
        parser = configparser.ConfigParser()
        parser.read(config_path)
        config = parser['DEFAULT']

        self.camera_id = config['camera_id']
        self.capture = cv2.VideoCapture(int(self.camera_id))
        self.frame_type = config['frame_type']

    def __del__(self):
        self.capture.release()

    def get_frame(self):
        success, frame = self.capture.read()
        if success:
            _, img_encoded = cv2.imencode(self.frame_type, frame)
            return img_encoded.tobytes()
        else:
            raise RuntimeError(str.format('Camera %d has been disconnected', self.camera_id))
